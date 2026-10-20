# Taiwan 判決預測模型

由 `BERT-base-prediction/Taiwan` 整理而來的程式快照（2026-10-20）。每次執行只訓練一個任務：罪名、法條或刑期；各自使用 encoder、分類頭與 checkpoint，沒有 MTL 權重共享。本目錄包含程式與設定，不包含資料集、預訓練權重、訓練結果或個案紀錄。

## 目前架構

現有 `configs/accusation_electra.json`、`configs/article_electra.json` 使用以下設定。雖然類別名稱是 `SingleTaskBert`，encoder 實際由設定決定，目前兩份設定均為法律 ELECTRA。

```text
判決事實 fact
  → 與 encoder 相同的 tokenizer
  → 前 512 tokens（含 CLS/SEP，超長截斷）
  → hfl/chinese-legal-electra-base-discriminator
  → CLS 向量（768 維）
  → Linear(768,512) → ReLU → Dropout(0.1)
  → Linear(512,256) → ReLU → Dropout(0.1)
  → Linear(256,類別數) → logits
```

凍結 embeddings 與前十層 encoder，只微調最後兩層及分類頭。類別數由資料的 `manifest.json` 讀取，不寫死。

| 任務 | 標籤與 loss | 推論 |
|---|---|---|
| 罪名 accusation | 多標籤；BCEWithLogitsLoss | sigmoid，門檻 0.5 |
| 法條 article | 多標籤；BCEWithLogitsLoss | sigmoid，門檻 0.5 |
| 刑期 imprisonment | 單標籤多分類；CrossEntropyLoss | argmax（softmax 可用於呈現機率） |

目前罪名／法條：batch 4、梯度累積 4（有效 batch 16）；AdamW，encoder LR 2e-5、分類頭 LR 1e-3、weight decay 0.01、梯度裁剪 1.0。最多 5 epochs，驗證 Macro-F1 連續 2 輪未改善即停止，保存最佳權重。warmup_ratio=0，未啟用 LR scheduler。多標籤 pos_weight 僅依訓練集計算，逐類上限 10。seed=42。

程式另外支援 BERT-base、CLS／mean／attention pooling、linear／MLP head、full／last_n／frozen 微調，以及全文 chunk 的 mean／attention 聚合；這些是可選變體，並非上述兩份設定的預設。直接執行 CLI 而未載入 config 時，預設為 BERT-base、linear head、full 微調。

## 檔案

| 路徑 | 用途 |
|---|---|
| `judgment_bert/model.py` | encoder、pooling、分類頭、凍結與分塊聚合 |
| `judgment_bert/encoding.py` | tokenization、Dataset、padding 與 chunk collator |
| `judgment_bert/train.py` | 設定讀取、訓練、驗證、選模與輸出 |
| `judgment_bert/data.py` | CAIL JSONL 前處理、年份切分、標籤篩選、刑期分組及去重 |
| `judgment_bert/analyze.py` | 前處理共用的資料讀取，以及 token 長度統計 |
| `configs/` | 現有罪名／法條實驗設定 |
| `tests/` | 架構、前處理、設定與刑期邊界測試，使用合成資料 |

## 安裝與執行

以下命令都在 `JudgementDigest/model/` 執行。模型相依套件獨立於爬蟲環境；Python 3.10+ 建議。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 只查看設定，不下載模型或讀取資料集
python -m judgment_bert.train --config configs/accusation_electra.json --print-config

# 已在本機準備好資料後才執行訓練
bash run_accusation_electra.sh
bash run_article_electra.sh

# 測試不下載預訓練權重；使用本機建立的 tiny encoder/tokenizer
python -m unittest discover -s tests -v
```

設定中的 `data_dir` 與 `output` 相對於執行時的工作目錄。罪名設定指向 `data/prepared_v1.2_closed/accusation`，法條指向 `data/prepared_v1.3_closed/article`；資料需自行準備或以 `--data-dir /本機路徑` 指定。CLI 同名參數優先於 config。

本份法條設定將來源的 `device=mps` 改為 `auto`，依 CUDA → MPS → CPU 選擇；將 `evaluate_test=true` 改為 `false`，測試集評估須明確加入 `--evaluate-test`。其餘設定沿用來源。正式訓練首次使用會下載 encoder 與 tokenizer。

## 資料與切分

來源是含 `fact` 與 `meta` 的 CAIL 格式 JSONL；須有 `meta.year`。罪名使用 `meta.accusation`，法條使用保留法律名稱的 `meta.relevant_articles`，兩者均為字串陣列。刑期使用 `term_task`、`sentence_kind`、`term_of_imprisonment`；拘役選項另讀取 `detention_days` 與 `reextraction` 的安全萃取資訊。

```bash
python -m judgment_bert.data \
  --data /本機路徑/CAIL資料.jsonl \
  --task accusation --output data/prepared_v1.2_closed/accusation
```

法條改用 `--task article` 與對應 output。前處理輸出 `train.jsonl`、`valid.jsonl`、`test.jsonl`、`manifest.json`；已存在的 split 不覆寫。預設按年份切分：2018–2023 訓練、2024 驗證、2025 測試（程式會將所有 ≤2023 的年份歸入訓練，來源資料需自行確認年份範圍）。沒有 10-fold；原始 `meta.fold` 不參與切分。

罪名／法條以訓練集頻率 ≥30 建立固定類別，所有 split 只保留這些標籤，移除標籤為空的案件；不要求另一任務標籤有效，也沒有額外事實長度門檻。同案跨集合時保留較早集合，預設亦移除與較早集合事實全文完全相同的資料。這不包含近似文字去重。

## 刑期狀態

刑期輸出設計尚在討論，沒有新增正式訓練 config。預設維持原 Taiwan 的 11 類：≤6、>6–12、>12–18、>18–24、>24–30、>30–36、>36–60、>60–<120、≥120 月，加上無期徒刑與死刑。只保留 `term_task=true` 且刑種合格的資料，拘役與混合刑預設排除。

可用 `--mtl-bins` 重現舊 MTL：先 `int(months)` 再分組，保留 ≥120 月為末組。可用 `--include-detention` 將安全萃取的單被告拘役獨立加入第 12 類；須有正的有限日數、`term_reason=detention_not_prison`、`defendant_status=單被告`。**尚未實作將拘役日數換算成月後合併分組，也尚未實作新的短刑期細分或交叉驗證。** 兩個選項供刑期前處理使用。

## 輸出與版控

每次訓練的獨立 output 目錄會保存 `best.pt`、tokenizer、encoder config、實際 `run_config.json`、資料 manifest 與指標。已含 `best.pt` 的目錄會拒絕覆寫。預設只以 VALID 選模，TEST 評估須明確啟用。

本目錄 `.gitignore` 排除本機 `data/`、`runs/`、`reports/`、JSONL、CSV 與模型權重，避免將資料集及訓練產物加入版控。這份程式快照不會隨原 Taiwan 目錄自動同步。
