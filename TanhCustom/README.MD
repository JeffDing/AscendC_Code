## TanhCustom自定义算子样例说明
<!--注：该样例仅用于说明目的，不用作生产质量代码的示例-->

本样例基于Ascend C实现Tanh自定义算子，并通过aclnn单算子调用的方式验证算子功能。

| 目录  | 描述  |
|---|---|
| [TanhCustom](./TanhCustom)  | TanhCustom自定义算子工程，包含算子原型、Tiling实现与核函数实现。  |
| [AclNNInvocation](./AclNNInvocation)  | 通过aclnn调用的方式调用TanhCustom自定义算子。  |

<a id="operatorcompile"></a>
## 编译算子工程

运行此样例前，请先完成CANN开发套件的安装与环境变量配置，然后按以下步骤编译并部署算子。

1. 进入算子工程目录。
   ```bash
   cd ${path}/TanhCustom/TanhCustom
   ```
2. 编译算子工程，生成自定义算子安装包custom_opp_ubuntu_aarch64.run。
   ```bash
   bash build.sh
   ```
3. 安装算子工程，算子文件默认部署到ascend-toolkit的opp/vendors/customize目录。
   ```bash
   ./build_out/custom_opp_ubuntu_aarch64.run --install
   ```

   若使用默认安装路径，无需增加其他参数。安装完成后，请根据安装提示设置LD_LIBRARY_PATH环境变量。
