# 先訓練 fastText 案件向量，再供刑期 ELECTRA 使用

本輪沿用現有 `JudgementDigest-pr16/models/fasttext_gpu.py`，沒有另換官方 fastText 套件，也沒有改成詞向量預訓練任務。罪名與法條仍各自用自己的標籤監督學習；輸出的是每個案件的隱向量，不是每個詞的向量或罪法機率。

## 固定設定

| 項目 | 設定 |
|---|---|
| 輸入 | fact，字元 1–2 gram；HashingVectorizer，alternate_sign=false，L2 正規化 |
| 編碼 | EmbeddingBag(sum, sparse=true)，1,048,576 buckets，128 維 |
| Dropout | 訓練 0.2；匯出 eval，關閉 dropout |
| 優化器 | EmbeddingBag 用 SparseAdam，分類層用 Adam；LR 都是 0.01 |
| 訓練 | batch=512，最多 15 epochs，patience=4，seed=0 |
| Loss | 各任務 BCE；pos_weight=sqrt(負例/正例)，截在 1–30 |
| 選模 | 各任務 VALID 的 (micro-F1+macro-F1)/2，還原最佳 epoch |
| 法條版本 | relevant_articles_with_suffix，固定 109 類，不改用粗條號 |
| 分折 | shared_fold_assignments.csv 的 5 folds，各自維持 TRAIN／VALID／TEST／holdout／purged |

罪法各自只使用本任務標籤有效的 TRAIN 訓練。法條先篩固定 suffix 名單，篩後沒有任何法條的案件不當成全負例；不重排來源行號。匯出仍涵蓋整個 100k 包的 fact，讓刑期任務不必因罪法標籤無效而缺少向量；purged 亦僅匯出，不參與訓練。

## 執行 fastText 訓練與匯出

需要同一目錄中的四個檔案：

- `JudgementDigest-pr16/models/fasttext_gpu.py`
- `JudgementDigest-pr16/models/fasttext_export.py`
- `JudgementDigest-pr16/models/evaluate.py`
- `JudgementDigest-pr16/models/run_fasttext_suffix_vectors.sh`

例如在 Kaggle（實際 input 路徑依上傳名稱替換）：

```sh
bash /kaggle/input/code/run_fasttext_suffix_vectors.sh \
  /kaggle/input/data/shared_100k_sentence_article_suffix \
  /kaggle/working/fasttext_suffix_vectors \
  /kaggle/working/fasttext_suffix.md
```

本機直接執行同一套設定（目前無 CUDA，會用 CPU；完整訓練與小型驗證不同）：

```sh
cd JudgementDigest-pr16/models
bash run_fasttext_suffix_vectors.sh \
  ../../deliverables/shared_100k_with_sentence_suffix_20261026/shared_100k_sentence_article_suffix \
  ../../modeldata/fasttext_suffix_vectors \
  ../../modeldata/fasttext_suffix_vectors_report.md
```

向量目錄必須尚不存在，避免混用或覆寫上一輪結果。要先跑單一 fold，可直接呼叫 Python，將 `--folds all` 改為 `--folds fold_01`；其他參數不變。不得將正式資料測試用的小 buckets／dim 結果當成基準設定。

輸出：

```text
fasttext_suffix_vectors/
  fold_01/accusation.npz
  fold_01/article.npz
  ...
  fold_05/accusation.npz
  fold_05/article.npz
```

每檔含 `vectors [N,128]`（float32）、`source_lines`、`jids`、`fact_sha256`、`roles` 以及 JSON `metadata`。metadata 記錄任務、fold、法條欄位、類別順序、VALID 選用 epoch、實際參數、實際訓練行號與向量 SHA-256。讀取不使用 pickle。

每個 fold 的兩個檔案必須成對使用；不同 fold 的 fastText 座標不保證一致，不把它們拼成同一個全域隱向量表。向量檔只供同一份案件資料使用，沒有保存可處理任意新案件的 fastText 權重。

## ELECTRA 讀取

`judgment_bert/fasttext_vectors.py` 提供 `FastTextVectorBank`。讀取時核對 fold、suffix 版本、向量 checksum、案件行號、文字雜湊與 split；若 prepared record 含 jid 也核對 jid。資料集即使過濾或 shuffle，也不按 batch 位置猜配對。

```python
from judgment_bert.fasttext_vectors import FastTextVectorBank
from judgment_bert.encoding import Collator

bank = FastTextVectorBank('/path/fasttext_suffix_vectors', 'fold_01')
# records 為同一 fold prepared TRAIN，須保留原始 source_line、fact。
collator = bank.collator(records, 'train', Collator(tokenizer.pad_token_id))
# DataLoader 回傳的 batch 會多出 accusation_vectors、article_vectors。
```

既有單次訓練入口已支援向量，於 `JudgementDigest/model/` 執行：

```sh
python -m judgment_bert.train \
  --data-dir /path/prepared_imprisonment/fold_01 \
  --output /path/new_term_fusion/fold_01 \
  --model hfl/chinese-legal-electra-base-discriminator \
  --tokenizer hfl/chinese-legal-electra-base-discriminator \
  --revision ffd97a559532e03a0073d2cdfede81a8a7f6d361 \
  --head mlp --finetune last_n --last-n 2 \
  --fasttext-vectors /path/fasttext_suffix_vectors \
  --fasttext-fold fold_01 --fusion-beta 0.1 --fusion-attention-dim 128
```

prepared 刑期資料必須沿用該共用包的來源行號與 fold 角色；不能另做年度切分後沿用舊向量角色。這個命令只示範已準備好的單 fold，沒有自動重建刑期分組或整套 CV；刑期分組仍以 prepared manifest 為準。

向量只讀取，fastText 不在 optimizer 中；FC、Q/K/V、刑期 head 與設定允許的 ELECTRA 層由刑期 loss 更新。所有非 encoder 可訓練參數都使用 head LR，沒有漏掉投影與 attention。訓練保存向量 metadata；獨立 TEST 評估讀取同一份並核對是否變更。

原 `train_model.py` 的整套 CV 入口尚未新增這些參數；目前使用上述單 fold 入口，逐折明確指定對應向量，避免所有 fold 誤用 fold_01。

## 驗證與執行狀態

已用合成小資料實際執行罪法 fastText 訓練、VALID 選模、向量匯出，以及本機 tiny ELECTRA 的檔案讀取與訓練測試。小測試刻意降低 dim/buckets 且不下載預訓練模型，不是正式實驗。

2026-10-28 已在本機 CPU 完成正式 100k × 5-fold 罪名／法條 fastText 訓練與向量匯出；執行紀錄與最佳 checkpoint 在 `modeldata/fasttext_suffix_vectors_local_20261028/`，給 Kaggle 的使用包與核對結果在 `deliverables/term_electra_fasttext_local_20261028/`。沒有啟動正式 ELECTRA 訓練，依使用者指定留待 Kaggle 執行。
