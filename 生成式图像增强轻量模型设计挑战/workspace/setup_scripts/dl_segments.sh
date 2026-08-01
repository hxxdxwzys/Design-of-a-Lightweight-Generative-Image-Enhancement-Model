#!/bin/bash
# 分段并发下载 (curl, 断点续传 + 重试), 用于大文件慢速/不稳定源。
# 用法: bash dl_segments.sh <url> <out> <segments> <total_bytes>
set -u
URL="$1"; OUT="$2"; N="${3:-6}"; TOTAL="${4:-0}"
if [ "$TOTAL" -eq 0 ]; then TOTAL=$(curl -sIL "$URL" | grep -i content-length | tail -1 | tr -dc '0-9'); fi
echo "TOTAL=$TOTAL segments=$N"
SEG=$(( (TOTAL + N - 1) / N ))
SEGDIR="/tmp/dl_segs_$$"; mkdir -p "$SEGDIR"

for i in $(seq 0 $((N-1))); do
  (
    S=$((i*SEG)); E=$((S+SEG-1)); [ "$E" -ge "$TOTAL" ] && E=$((TOTAL-1))
    P="$SEGDIR/seg_$i.part"; touch "$P"
    want=$((E-S+1))
    while [ "$(stat -c%s "$P" 2>/dev/null || echo 0)" -lt "$want" ]; do
      # 纯 Range 一次性下载 (Windows curl 的 -C 与 -r 组合报错), 断连靠 --retry-all-errors
      curl -sL --retry 8 --retry-all-errors --connect-timeout 20 \
        -r "$S-$E" -o "$P" "$URL" 2>/dev/null
      sleep 2
    done
    echo "seg $i done ($(stat -c%s "$P")B)"
  ) &
done
wait
cat "$SEGDIR"/seg_*.part > "$OUT"
SZ=$(stat -c%s "$OUT")
rm -rf "$SEGDIR"
echo "DL_DONE size=$SZ expected=$TOTAL match=$([ "$SZ" -eq "$TOTAL" ] && echo YES || echo NO)"
