#!/usr/bin/env bash
# Download TabFM's regression weights (6.6 GB). huggingface_hub stalls at random offsets,
# so curl resumes with --continue-at in a loop that reconnects after every stall.
set -u
URL="https://huggingface.co/google/tabfm-1.0.0-pytorch/resolve/main/regression/model.safetensors"
OUT="$HOME/tabfm-regression-model.safetensors"
SIZE=6591243724

for i in $(seq 1 200); do
  have=$( [ -f "$OUT" ] && stat -c %s "$OUT" || echo 0 )
  if [ "$have" -ge "$SIZE" ]; then echo "done: $have bytes"; exit 0; fi
  awk -v h="$have" -v s="$SIZE" -v i="$i" \
    'BEGIN{printf "attempt %d: %.2f/%.2f GB (%.1f%%)\n", i, h/1e9, s/1e9, 100*h/s}'
  # Give up on a connection slower than 50 KB/s for 30 s, then try again.
  curl -L -C - --retry 5 --retry-delay 3 --retry-connrefused \
       --speed-limit 50000 --speed-time 30 \
       -o "$OUT" "$URL" 2>/dev/null
done
echo "out of attempts"
exit 1
