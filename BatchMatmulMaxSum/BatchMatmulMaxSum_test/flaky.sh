#!/bin/bash
# usage: flaky.sh <caseId> <runs>  — each run is a fresh process (1 launch)
# 连跑结果以 markdown 表格追加到 TEST_REPORT.md 的"稳定性"章节。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
CID=${1:-12}; N=${2:-20}
REPORT="$ROOT/TEST_REPORT.md"
pass=0; fail=0
for i in $(seq 1 "$N"); do
  cp "input/case${CID}/x1.bin" input/x1.bin
  cp "input/case${CID}/x2.bin" input/x2.bin
  rm -f output/y.bin
  timeout 70 ./build/bmm_multi "$CID" > /dev/null 2>&1 || { echo "run $i: CRASH"; fail=$((fail+1)); continue; }
  if /usr/local/python3.12.13/bin/python3 check_one.py "$CID" "$ROOT" > "$ROOT/.flaky_out" 2>&1; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); echo "run $i: $(tail -1 "$ROOT/.flaky_out")"
  fi
done
echo "case${CID}: $pass/$N passed"

# ---- append markdown row ----
if [ ! -f "$REPORT" ]; then
  { echo "# BatchMatmulMaxSum 测试报告"; echo;
    echo "- 生成时间：$(date '+%Y-%m-%d %H:%M:%S')"; } > "$REPORT"
fi
if ! grep -q '^## 稳定性' "$REPORT"; then
  echo "" >> "$REPORT"
  echo "## 稳定性（独立进程连跑）" >> "$REPORT"
  echo >> "$REPORT"
  echo "| Case | 连跑 | 通过 | 失败 | 通过率 |" >> "$REPORT"
  echo "|---|---|---|---|---|" >> "$REPORT"
fi
rate=$(awk "BEGIN{printf \"%.0f%%\", ${pass}/${N}*100}")
echo "| ${CID} | ${N} | ${pass} | ${fail} | ${rate} |" >> "$REPORT"
rm -f "$ROOT/.flaky_out"
