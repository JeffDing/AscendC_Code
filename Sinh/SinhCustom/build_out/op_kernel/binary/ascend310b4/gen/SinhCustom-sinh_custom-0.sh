#!/bin/bash
echo "[Ascend310b4] Generating SinhCustom_01ec8ddcf6437b12e69898b368caea08 ..."
opc $1 --main_func=sinh_custom --input_param=/root/cann_camp_2024/SinhCustom/build_out/op_kernel/binary/ascend310b4/gen/SinhCustom_01ec8ddcf6437b12e69898b368caea08_param.json --soc_version=Ascend310b4 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.json ; then
  echo "$2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.json not generated!"
  exit 1
fi

if ! test -f $2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.o ; then
  echo "$2/SinhCustom_01ec8ddcf6437b12e69898b368caea08.o not generated!"
  exit 1
fi
echo "[Ascend310b4] Generating SinhCustom_01ec8ddcf6437b12e69898b368caea08 Done"
