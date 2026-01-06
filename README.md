簡介
-
這個專案是一個簡單的音樂辨識系統範例。

檔案
-
- build_db.py: 從 `songs/` 目錄萃取每首歌的 MFCC 平均向量，輸出 `db_features.npy` 與 `db_meta.json`。
- recognize.py: 載入資料庫並對輸入音檔做比對，回傳 top-k 相似歌曲。

快速開始
-
1. 安裝 Python 套件（建議使用虛擬環境）：

```bash
pip install -r requirements.txt
# 若需要讀取 mp3，請在系統安裝 ffmpeg（Windows 可用 choco 或從官網下載）
```

2. 把音訊檔放到 `songs/` 資料夾（支援 .wav, .mp3, .flac, .m4a, .ogg）。

3. 建立資料庫：

```bash
python build_db.py --duration 60 --n-mfcc 20 --normalize
```

4. 辨識測試音檔：

```powershell
python recognize.py "C:\Users\linle\Downloads\test_clip2.mp3"
```

使用說明重點
-
- `build_db.py` 會從每首歌取前 N 秒（預設 60 秒）計算 MFCC 的時間平均向量並儲存。
- `--normalize` 選項會將每個向量做 L2 正規化，若辨識時使用 cosine similarity 可提高效果。
- `recognize.py` 支援 L2 距離或 cosine 相似度（程式內可切換），並會打印 top-k 結果。

故障排除
-
- 若出現 `FileNotFoundError`，請確認路徑是否正確並在命令列用引號包住含空格的路徑。
- 若 librosa 讀取 mp3 失敗，請安裝系統層級的 `ffmpeg`，或確認 `audioread` 是否正常。

下一步建議
-
- 若資料量大（數千首以上），可加入 FAISS 做向量索引加速檢索。
- 可改用短時窗比對（分段比對），以支援只提供片段的情境。
