"""
build_db.py（高精準版）
- MFCC + Chroma 特徵
- 多段取樣平均
- 輸出特徵資料庫供 recognize.py 使用
"""

import os
import json
import numpy as np
import librosa

SONGS_DIR = "songs"
OUT_FEATURES = "db_features.npy"
OUT_META = "db_meta.json"

SR = 22050
N_MFCC = 20
N_CHROMA = 12

SEGMENT_DURATION = 6    # 每段 6 秒
SEGMENT_COUNT = 10      # 每首歌取 10 段（約 1 分鐘）


# =========================
# 擷取單段特徵
# =========================
def extract_segment(path, offset, duration):
    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True,
        offset=offset,
        duration=duration
    )

    if y.size == 0:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR)

    feat = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1)
    ])

    return feat.astype("float32")


# =========================
# 擷取整首歌（多段平均）
# =========================
def extract_song_feature(path):
    feats = []

    for i in range(SEGMENT_COUNT):
        offset = i * SEGMENT_DURATION
        feat = extract_segment(path, offset, SEGMENT_DURATION)
        if feat is not None:
            feats.append(feat)

    if not feats:
        return None

    return np.mean(feats, axis=0)


# =========================
# 主程式
# =========================
def main():
    if not os.path.exists(SONGS_DIR):
        print("❌ 找不到 songs/ 資料夾")
        return

    files = [
        f for f in os.listdir(SONGS_DIR)
        if f.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg"))
    ]
    files.sort()

    if not files:
        print("❌ songs/ 內沒有音樂檔")
        return

    features = []
    meta = []

    print(f"🎵 找到 {len(files)} 首歌，開始建立高精準資料庫...\n")

    for idx, fname in enumerate(files):
        path = os.path.join(SONGS_DIR, fname)
        print(f"[{idx+1}/{len(files)}] 處理：{fname}")

        try:
            feat = extract_song_feature(path)
            if feat is None:
                print("   ⚠️ 擷取失敗，跳過")
                continue

            features.append(feat)
            meta.append({
                "idx": len(features) - 1,
                "filename": fname
            })

        except Exception as e:
            print("   ❌ 錯誤：", e)

    if not features:
        print("❌ 沒有成功建立任何特徵")
        return

    features = np.vstack(features).astype("float32")

    np.save(OUT_FEATURES, features)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ 建立完成")
    print("特徵檔：", OUT_FEATURES)
    print("資訊檔：", OUT_META)
    print("特徵維度：", features.shape)


if __name__ == "__main__":
    main()
