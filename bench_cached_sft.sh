#!/usr/bin/env bash
# Run on a node with the 041 index and 008 tensor mounts.
set -euo pipefail
cd "$(dirname "$0")"
python_bin=${TRAIN_PYTHON:-/home/zkln/miniconda3/envs/fastvideo/bin/python}
gpu=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2
out=${OUT:-$(mktemp -d "$PWD/bench-XXXXXXXX")}
mkdir -p "$out"
printf 'Results: %s\n' "$out"
for frames in ${FRAMES:-97 77}; do
    for ((batch=${START_BATCH:-1}; ; batch++)); do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        if ((used > 512)); then
            echo "GPU $gpu already using ${used} MiB; stopping" >&2
            exit 1
        fi
        log="$out/f${frames}-b${batch}.log"
        if [[ -e "$log" ]]; then echo "Existing log: $log" >&2; exit 1; fi
        if "$python_bin" -u train_cached_sft.py --frames "$frames" --batch "$batch" "$@" >"$log" 2>&1; then
            tail -1 "$log"
        else
            status=$?
            tail -12 "$log"
            if ((status == 42)); then break; fi
            exit "$status"
        fi
    done
done
