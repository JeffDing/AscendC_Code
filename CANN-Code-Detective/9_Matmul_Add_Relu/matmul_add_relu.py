"""Challenge 09：固定 256×256×256 的 Catlass Python 融合核骨架。"""

import catlass.tla as tla


M = N = K = 256
L0_K = 64
AIV_M = M // 2
X_CHUNK_M = 64
SIMD_LANES = 64  # 256 字节 SIMD 寄存器可容纳 64 个 FP32 元素。


@tla.kernel
def matmul_add_relu_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_x: tla.Tensor,
    gm_y: tla.Tensor,
) -> None:
    """仅计算 A(256,256) @ B(256,256) + X(256,256) 后的 ReLU。"""
    # mutex 管理核内缓冲的访问顺序；A/B 成组搬运和使用，共用一把锁。
    mutex_l1 = tla.mutex(resource="l1_ab", id=0)
    mutex_l0 = tla.mutex(resource="l0_ab", id=1)
    mutex_l0c = tla.mutex(resource="l0_c", id=2)
    mutex_acc = tla.mutex(resource="ub_acc", id=3)
    mutex_x = tla.mutex(resource="ub_x", id=4)

    # v2.1.0 的 mutex 不支持跨核同步，FIX → AIV 仍需跨核通知。
    fix_done = tla.cross_flag("fix_done")

    # L1: A/B 各 128 KiB；L0A/L0B: 各 32 KiB；L0C: 256 KiB。
    l1_a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l1, 512)
    l1_b_ptr = tla.allocate(K * N, tla.Float16, tla.AddressSpace.l1, 512)
    l0_a_ptr = tla.allocate(M * L0_K, tla.Float16, tla.AddressSpace.l0a, 512)
    l0_b_ptr = tla.allocate(L0_K * N, tla.Float16, tla.AddressSpace.l0b, 512)
    l0_c_ptr = tla.allocate(M * N, tla.Float32, tla.AddressSpace.l0c, 512)

    # 每个 AIV 的 UB: 累加结果 128 KiB + X 分片 64 KiB = 192 KiB。
    ub_acc_ptr = tla.allocate(AIV_M * N, tla.Float32, tla.AddressSpace.ub, 256)
    ub_x_ptr = tla.allocate(X_CHUNK_M * N, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        l1_a = tla.make_tensor_like(l1_a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1_b_ptr, gm_b, tla.arch.zN)
        l0_c = tla.make_tensor_like(l0_c_ptr, gm_y, tla.arch.L0Clayout)

        mutex_l1.lock(pipe=tla.arch.MTE2)
        # --------------------------------
        # TODO 1（GM → L1）：输入 gm_a/gm_b 是连续行优先的 FP16 矩阵。
        # 按 A、B 的顺序调用 tla.copy，分别搬到 L1 中 zN 布局的
        # l1_a/l1_b；mutex_l1 保证 MTE1 在两次搬运完成后读取。
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)
        # --------------------------------
        mutex_l1.unlock(pipe=tla.arch.MTE2)

        for k_tile in tla.range(0, K // L0_K, 1):
            l1_a_k = tla.tile_view(
                l1_a, tla.make_shape(M, L0_K), tla.make_coord(0, k_tile)
            )
            l1_b_k = tla.tile_view(
                l1_b, tla.make_shape(L0_K, N), tla.make_coord(k_tile, 0)
            )
            l0_a = tla.make_tensor_like(l0_a_ptr, l1_a_k, tla.arch.zN)
            l0_b = tla.make_tensor_like(l0_b_ptr, l1_b_k, tla.arch.nZ)

            mutex_l1.lock(pipe=tla.arch.MTE1)
            mutex_l0.lock(pipe=tla.arch.MTE1)
            # --------------------------------
            # TODO 2（L1 → L0）：输入为 L1 的 l1_a_k/l1_b_k。
            # 按 A、B 的顺序调用 tla.copy，分别搬到 L0A 的 l0_a（zN）
            # 和 L0B 的 l0_b（nZ）；mutex_l0 保证 Cube 读取前搬运完成，
            # 并保证下一次搬运在当前矩阵乘读完 L0A/L0B 后开始。
            tla.copy(l0_a, l1_a_k)
            tla.copy(l0_b, l1_b_k)
            # --------------------------------
            mutex_l0.unlock(pipe=tla.arch.MTE1)
            mutex_l1.unlock(pipe=tla.arch.MTE1)

            mutex_l0.lock(pipe=tla.arch.CUBE)
            mutex_l0c.lock(pipe=tla.arch.CUBE)
            # --------------------------------
            # TODO 3（矩阵乘累加）：用 tla.mmad 将 FP16 的 l0_a×l0_b
            # 累加到 L0C 中的 FP32 l0_c。k_tile=0 时 init_c=True，
            # 后续三次必须为 False，以保留前面 K 分片的累加结果。
            # 此处只计算矩阵乘；X 的相加和 ReLU 留给 Vector。
            init_c = k_tile == 0
            tla.mmad(l0_c, l0_a, l0_b, init_c=init_c)
            # --------------------------------
            mutex_l0c.unlock(pipe=tla.arch.CUBE)
            mutex_l0.unlock(pipe=tla.arch.CUBE)

        ub_acc_full = tla.make_tensor_like(ub_acc_ptr, gm_y, tla.arch.RowMajor)
        mutex_l0c.lock(pipe=tla.arch.FIX)
        # --------------------------------
        # TODO 4（L0C → UB）：输入是 l0_c 中完整的 FP32 乘积累加结果。
        # 用 tla.copy 搬到 ub_acc_full，并传入 CopyL0C2DstParams，
        # 将 l0c2ub_mode 设为 SPLIT_M，使两个 AIV 各收到 128×256。
        # 此时尚未加 X，不能在 FixPipe 中提前启用 ReLU。
        tla.copy(
            ub_acc_full,
            l0_c,
            tla.params.CopyL0C2DstParams(
                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
            ),
        )
        # --------------------------------
        mutex_l0c.unlock(pipe=tla.arch.FIX)
        tla.cross_core_set_flag(fix_done, tla.arch.FIX)

    with tla.vector():
        aiv_id = tla.arch.sub_block_idx()
        gm_x_aiv = tla.tile_view(
            gm_x, tla.make_shape(AIV_M, N), tla.make_coord(aiv_id, 0)
        )
        gm_y_aiv = tla.tile_view(
            gm_y, tla.make_shape(AIV_M, N), tla.make_coord(aiv_id, 0)
        )
        ub_acc_aiv = tla.make_tensor_like(ub_acc_ptr, gm_y_aiv, tla.arch.RowMajor)

        tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)
        for chunk in tla.range(0, AIV_M // X_CHUNK_M, 1):
            gm_x_chunk = tla.tile_view(
                gm_x_aiv, tla.make_shape(X_CHUNK_M, N), tla.make_coord(chunk, 0)
            )
            gm_y_chunk = tla.tile_view(
                gm_y_aiv, tla.make_shape(X_CHUNK_M, N), tla.make_coord(chunk, 0)
            )
            ub_acc_chunk = tla.tile_view(
                ub_acc_aiv, tla.make_shape(X_CHUNK_M, N), tla.make_coord(chunk, 0)
            )
            ub_x = tla.make_tensor_like(ub_x_ptr, gm_x_chunk, tla.arch.RowMajor)

            mutex_x.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_x, gm_x_chunk)
            mutex_x.unlock(pipe=tla.arch.MTE2)

            mutex_acc.lock(pipe=tla.arch.VECTOR)
            mutex_x.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                for row in tla.range(0, X_CHUNK_M, 1):
                    for col in tla.range(0, N // SIMD_LANES, 1):
                        acc_reg = tla.tile_view(
                            ub_acc_chunk,
                            tla.make_shape(1, SIMD_LANES),
                            tla.make_coord(row, col),
                        )
                        x_reg = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, SIMD_LANES),
                            tla.make_coord(row, col),
                        )
                        # --------------------------------
                        # TODO 5（Vector Add + ReLU）：从 UB 的 acc_reg/x_reg
                        # 载入 FP32 向量，先用 tla.add 相加，再用 tla.max
                        # 与 0.0 取最大值，最后原位写回 acc_reg。
                        # 每行固定 256 列，共 4 个完整 SIMD 向量，无需尾掩码。
                        # 下方 tla.copy 将处理后的 ub_acc_chunk 写到 GM 的 Y。
                        acc_data = acc_reg.load()
                        x_data = x_reg.load()
                        relu_data = tla.max(tla.add(acc_data, x_data), 0.0)
                        acc_reg.store(relu_data)
                        # --------------------------------

            mutex_x.unlock(pipe=tla.arch.VECTOR)
            mutex_acc.unlock(pipe=tla.arch.VECTOR)

            mutex_acc.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_y_chunk, ub_acc_chunk)
            mutex_acc.unlock(pipe=tla.arch.MTE3)
