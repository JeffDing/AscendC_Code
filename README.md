# AscendC

## 介绍
Ascend C是CANN针对算子开发场景推出的编程语言，原生支持C和C++标准规范，最大化匹配用户开发习惯；通过多层接口抽象、自动并行计算、孪生调试等关键技术，极大提高算子开发效率，助力AI开发者低成本完成算子开发和模型调优部署。

## 环境命令
### msopgen命令

```bash
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i sinh_custom.json -c ai_core-Ascend910A -lan cpp -out ./SinhCustom
```

### cmake安装
```bash
sed -i '$a export PATH=/tmp/code/cmake-3.29.0-linux-aarch64/bin:$PATH' ~/.bashrc
```

### 激活Ascend环境
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```