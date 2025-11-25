# scripts/align_embeddings.py
import numpy as np
from scipy.linalg import orthogonal_procrustes
import fasttext

print("Loading models...")
m_en = fasttext.load_model("output/eng_all.bin")
m_hi = fasttext.load_model("output/hin_all.bin")


print("Loading seed dictionary...")
pairs = []
with open("output/seed.txt", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            en, hi = line.split()
        except ValueError:
            continue
        try:
            v_en = m_en.get_word_vector(en)
            v_hi = m_hi.get_word_vector(hi)
            pairs.append((v_en, v_hi))
        except Exception:
            # skip words that are OOV, etc.
            continue

if not pairs:
    raise SystemExit("No valid seed pairs found!")

X = np.array([p[0] for p in pairs])  # English
Y = np.array([p[1] for p in pairs])  # Hindi

print("Running Orthogonal Procrustes on", X.shape[0], "seed pairs...")
R, _ = orthogonal_procrustes(X, Y)

np.save("output/R.npy", R)
print("Saved alignment matrix to output/R.npy")
