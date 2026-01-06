"""
recognize.py（高精準版）
- MFCC + Chroma 特徵
- 多段取樣 + 平均投票
- 信心差距判斷（避免亂猜）
"""

import sys
import os
import json
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DB_FEATURES = "db_features.npy"
DB_META = "db_meta.json"

SR = 22050
N_MFCC = 20
N_CHROMA = 12

SEGMENT_DURATION = 6     # 每段 6 秒
SEGMENT_COUNT = 8        # 共取 8 段（約 48 秒）

CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.03


# =========================
# 特徵擷取（MFCC + Chroma）
# =========================
def extract_feature(path, offset=0, duration=6):
    y, _ = librosa.load(path, sr=SR, mono=True,
                        offset=offset, duration=duration)

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
# 辨識主程式
# =========================
def recognize(test_path, top_k=3):

    if not os.path.exists(test_path):
        print("❌ 找不到測試音檔")
        return

    db = np.load(DB_FEATURES)
    with open(DB_META, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 標準化（很重要）
    scaler = StandardScaler()
    db_norm = scaler.fit_transform(db)

    scores_all = []

    # 多段取樣
    for i in range(SEGMENT_COUNT):
        offset = i * SEGMENT_DURATION
        feat = extract_feature(test_path, offset, SEGMENT_DURATION)
        if feat is None:
            continue

        feat_norm = scaler.transform(feat.reshape(1, -1))
        sims = cosine_similarity(feat_norm, db_norm).flatten()
        scores_all.append(sims)

    if not scores_all:
        print("❌ 無法擷取特徵")
        return

    # 平均投票
    score_avg = np.mean(scores_all, axis=0)

    idx_sorted = np.argsort(-score_avg)

    best = idx_sorted[0]
    second = idx_sorted[1]

    best_score = score_avg[best]
    second_score = score_avg[second]
    margin = best_score - second_score

    print("\n🎵 辨識結果：\n")

    if best_score >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
        print("✅【確定結果】")
        print(f"歌曲：{meta[best]['filename']}")
        print(f"信心度：{best_score:.4f}")
        print(f"與第二名差距：{margin:.4f}")
    else:
        print("⚠️【結果不夠確定】")
        print("可能候選歌曲：")
        for rank in range(min(top_k, len(idx_sorted))):
            idx = idx_sorted[rank]
            print(f"{rank+1}. {meta[idx]['filename']} "
                  f"(信心度: {score_avg[idx]:.4f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python recognize.py test_clip.wav")
    else:
        recognize(sys.argv[1])
