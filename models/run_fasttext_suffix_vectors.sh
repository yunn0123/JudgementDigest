#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo 'Usage: bash run_fasttext_suffix_vectors.sh SHARED_DATA_DIR NEW_VECTOR_DIR REPORT.md' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

# Same fastText settings as the current baseline; train accusation/article separately.
python "$script_dir/fasttext_gpu.py" \
    --shared "$1" \
    --article-field relevant_articles_with_suffix \
    --tasks accusation,article --folds all \
    --dim 128 --buckets 1048576 --batch 512 --lr 0.01 \
    --epochs 15 --patience 4 --seed 0 \
    --export-vectors "$2" --out "$3"
