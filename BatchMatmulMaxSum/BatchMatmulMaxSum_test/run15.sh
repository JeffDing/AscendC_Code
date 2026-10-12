#!/bin/bash
# 15-case correctness gate.  usage: run15.sh [reps] [cases...]
# 跑完自动生成/刷新 TEST_REPORT.md 的"正确性回归"章节。
set -u
REPS=${1:-5}
[ $# -gt 0 ] && shift
CASES=${@:-$(seq 1 15)}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/usr/local/python3.12.13/bin/python3
REPORT="$ROOT/TEST_REPORT.md"
cd "$ROOT" || exit 1
mkdir -p input output

pass=0; fail=0
rows=""
for cid in $CASES; do
  cp "input/case${cid}/x1.bin" input/x1.bin
  cp "input/case${cid}/x2.bin" input/x2.bin
  rm -f output/y.bin
  rc=0; total_us=""
  for r in $(seq 1 "$REPS"); do
    out=$(timeout 70 ./build/bmm_multi "$cid" 2>&1 >/dev/null) || rc=1
    total_us=$(printf '%s\n' "$out" | grep '\[timing\]' | tail -1 | sed -E 's/.*total=([0-9.]+)us.*/\1/')
  done
  if [ $rc -ne 0 ]; then
    echo "case${cid}: CRASH"; rows="${rows}| ${cid} | CRASH | - | - | - |\n"; fail=$((fail+1)); continue
  fi
  line=$($PY check_one.py "$cid" "$ROOT") && {
    pass=$((pass+1)); y0=$(printf '%s' "$line" | sed -E 's/.*y\[0\]=([-0-9.e+]+),.*/\1/')
    mrel=$(printf '%s' "$line" | sed -E 's/.*max_rel=([0-9.e+-]+).*/\1/')
    rows="${rows}| ${cid} | PASS | ${y0} | ${mrel} | ${total_us:-n/a} |\n"
  } || {
    fail=$((fail+1)); mrel=$(printf '%s' "$line" | sed -E 's/.*max_rel=([0-9.e+-]+).*/\1/')
    rows="${rows}| ${cid} | FAIL | - | ${mrel:-n/a} | ${total_us:-n/a} |\n"
  }
done
echo "---- $pass passed, $fail failed"

# ---- write markdown report section ----
{
  echo "# BatchMatmulMaxSum 测试报告"
  echo
  echo "- 生成时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "- 被测文件：kernel.asc（md5: $(md5sum kernel.asc | cut -d' ' -f1)）"
  echo "- 环境：910B2 (dav-2201)，CANN 9.1.0，24 cube / 48 vector cores"
  echo "- 判据：output/y.bin vs FP64 golden，rtol=atol=1e-3"
  echo "- 口径：每 case 以 ${REPS} 个独立进程各执行 1 次 launch；耗时为末次进程的单次 total（us，含 ACL/IO 进程固定开销）"
  echo
  echo "## 正确性回归（15 case）"
  echo
  echo "| Case | 结果 | y[0] | max_rel | 单次耗时 (us) |"
  echo "|---|---|---|---|---|"
  printf "$rows"
  echo
  echo "**汇总：${pass}/$(echo $CASES | wc -w) PASS**"
} > "$REPORT"
[ $fail -eq 0 ]
