# AscendC

## msopgen命令

```bash
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i sinh_custom.json -c ai_core-Ascend910A -lan cpp -out ./SinhCustom
```

## cmake安装
```bash
sed -i '$a export PATH=/tmp/code/cmake-3.29.0-linux-aarch64/bin:$PATH' ~/.bashrc
```

## 激活Ascend环境
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```