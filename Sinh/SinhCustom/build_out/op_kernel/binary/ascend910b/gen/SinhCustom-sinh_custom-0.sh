#!/bin/bash
echo "[Ascend910B1] Generating SinhCustom_01ec8ddcf6437b12e69898b368caea08 ..."
opc $1 --main_func=sinh_custom --input_param=/tmp/code/SinhCustom/build_out/op_kernel/binary/ascend910b/gen/SinhCustom_01ec8ddcf6437b12e69898b368caea08_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.json ; then
  echo "$2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.json not generated!"
  exit 1
fi

if ! test -f $2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.o ; then
  echo "$2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating SinhCustom_01ec8ddcf6437b12e69898b368caea08 Done"
