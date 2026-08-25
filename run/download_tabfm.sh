#!/usr/bin/env bash
# Otporno skidanje TabFM regresionih tezina (6,6 GB).
# huggingface_hub se ponavljano zaglavljuje na raznim ofsetima; curl sa --continue-at
# i petljom koja se rekonektuje na svaki zastoj to resava.
set -u
URL="https://huggingface.co/google/tabfm-1.0.0-pytorch/resolve/main/regression/model.safetensors"
OUT="$HOME/tabfm-regression-model.safetensors"
SIZE=6591243724

for i in $(seq 1 200); do
  have=$( [ -f "$OUT" ] && stat -c %s "$OUT" || echo 0 )
  if [ "$have" -ge "$SIZE" ]; then echo "GOTOVO: $have B"; exit 0; fi
  awk -v h="$have" -v s="$SIZE" -v i="$i" \
    'BEGIN{printf "pokusaj %d: %.2f/%.2f GB (%.1f%%)\n", i, h/1e9, s/1e9, 100*h/s}'
  # --speed-limit/--speed-time: prekini ako padne ispod 50 KB/s duze od 30 s, pa ponovo
  curl -L -C - --retry 5 --retry-delay 3 --retry-connrefused \
       --speed-limit 50000 --speed-time 30 \
       -o "$OUT" "$URL" 2>/dev/null
done
echo "iscrpljeno pokusaja"
exit 1
