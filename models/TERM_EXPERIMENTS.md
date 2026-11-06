# ELECTRA 刑期與 fastText 罪法模型

從 repository 根目錄執行。`models/` 放訓練入口與共用評分，`model/judgment_bert/` 放模型架構、向量讀取與訓練元件。
本分支提供程式、設定與測試；資料、預訓練 encoder、向量與 checkpoint 需另行準備。

## 安裝與資料

```bash
python -m pip install -r model/requirements.txt -r models/requirements.txt
export PYTHONPATH="$PWD/model"
```

共用資料目錄須有 `shared_100k_with_article_suffix.jsonl`、`shared_fold_assignments.csv`、`fixed_classes_coarse.json`、`fixed_classes_suffix.json`。CSV 定義每個 outer fold 的 train/valid/test/holdout；不要重新隨機切分。罪法多標籤；刑期使用有期徒刑案件，拘役日數不視為零個月。

## 1. 單刑期 ELECTRA

```bash
python models/electra_term_gpu.py --shared /path/to/shared --folds all \
  --last-n 6 --epochs 5 --patience 2 --seed 42 --batch 4 --grad-accum 4 \
  --encoder-lr 2e-5 --head-lr 1e-3 \
  --checkpoint-dir model/runs/term_single --out model/runs/term_single.md
```

ELECTRA CLS → MLP → 7 區間分類與 log(月數＋1) 迴歸。以 VALID CAIL 選 epoch，之後評外層 TEST 與 2025 holdout。最多 5 epochs，可提早停止。
歷史 seed42 單任務對照為融合模型 `beta=0`，以維持該次初始化順序；上方無輔助入口是獨立單任務模型，不保證逐位重現該次分數。

## 2. fastText 預訓練與凍結融合

```bash
python models/fasttext_gpu.py --shared /path/to/shared \
  --article-field relevant_articles_with_suffix --tasks accusation,article --folds all \
  --dim 128 --buckets 1048576 --batch 512 --lr 0.01 --epochs 15 --patience 4 --seed 0 \
  --save-checkpoints --export-vectors model/runs/fasttext_vectors --out model/runs/fasttext.md

python models/electra_term_gpu.py --shared /path/to/shared --folds all \
  --last-n 6 --epochs 5 --patience 2 --seed 42 --batch 4 --grad-accum 4 \
  --fasttext-vectors model/runs/fasttext_vectors --beta-mode fixed --beta 0.2 \
  --checkpoint-dir model/runs/term_frozen --out model/runs/term_frozen.md
```

fastText 為 PyTorch EmbeddingBag 實作，罪法各自訓練。ELECTRA 向量作 query，罪法投影作 key/value；融合為 `h_term + beta * LayerNorm(h_aux)`，接刑期雙輸出頭。
預先產生的罪法向量不接收刑期梯度；投影、attention 與刑期模型會更新。`--beta-mode learnable` 可改學習全域 beta。

## 3. OOF 罪法機率向量

`fasttext_oof.py` 在每個 outer TRAIN 內再作 grouped inner folds，共 5 × 5 × 2 個模型。每筆 TRAIN 的向量由未用該筆訓練或選模的模型產生；inner checkpoint 仍由 outer VALID 選擇。VALID/TEST/2025 使用對應 outer TRAIN 完整訓練的原始 fastText。

先為上述原始權重建立雜湊清單（須確定權重實際由該份資料與分配產生；雜湊本身不能證明訓練來源）：

```bash
python - /path/to/shared model/runs/fasttext_vectors <<'PY'
import hashlib, json, sys
import numpy as np
from pathlib import Path
shared, weights = map(Path, sys.argv[1:])
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
files = [weights / f'fold_{i:02d}' / f'{task}.pt'
         for i in range(1, 6) for task in ('accusation', 'article')]
for checkpoint in list(files):
    with np.load(checkpoint.with_suffix('.npz'), allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
    metadata_path = checkpoint.with_suffix('.json')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    files.append(metadata_path)
manifest = {
    'data_sha256': {name: sha(shared / name) for name in
                   ('shared_100k_with_article_suffix.jsonl', 'shared_fold_assignments.csv')},
    'sha256': {str(path.relative_to(weights)): sha(path) for path in files},
}
(weights / 'fasttext_checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2))
PY

python models/fasttext_oof.py --shared /path/to/shared \
  --original-vectors model/runs/fasttext_vectors \
  --checkpoint-manifest model/runs/fasttext_vectors/fasttext_checkpoint_manifest.json \
  --output model/runs/fasttext_oof --inner-folds 5
```

再將凍結融合指令的 `--fasttext-vectors` 改成 `model/runs/fasttext_oof/vectors`，並指定新的輸出目錄。
OOF 使用固定類別座標的 sigmoid scores（罪名80／法條109維），不是原版128維 hidden vectors；這兩版同時改變 OOF 與表示方式，不能把全部差異歸因於 OOF。
另輸出 `in_sample_probability_control/`，供固定表示方式的後續對照。

## 4. 可更新 fastText 的 MTL

```bash
python models/fasttext_electra_mtl.py --shared /path/to/shared \
  --model hfl/chinese-legal-electra-base-discriminator --fold fold_01 \
  --fasttext-checkpoints model/runs/fasttext_vectors --term-cohort-only \
  --stop-term-gradient --beta 0.2 --seed 42 --last-n 6 --epochs 5 \
  --batch 4 --grad-accum 4 --output model/runs/mtl_stop/fold_01
```

逐折重複 `fold_01` 到 `fold_05`，每折使用自己的預訓練 checkpoint。預設罪法 loss 權重各0.1。
`--stop-term-gradient`：罪法只由自身分類 loss 更新，刑期仍可更新投影與 attention。
若要刑期梯度參與，移除該旗標並加 `--term-gradient-ratio 0.1`：在每個 fastText embedding 參數群，將刑期梯度範數調至其加權自身梯度的0.1倍，再相加。這不是更新量或準確率的10%；零梯度無法透過縮放變成非零。

## 本機測試（不下載 encoder）

```bash
PYTHONPATH=model HF_HUB_OFFLINE=1 python -m unittest discover -s model/tests -v
PYTHONPATH=models:model HF_HUB_OFFLINE=1 python -m unittest discover -s models -p 'test_fasttext*.py' -v
PYTHONPATH=model HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'test_model_*.py' -v
python -m unittest discover -s tests -p 'test_models_evaluate.py' -v
```
