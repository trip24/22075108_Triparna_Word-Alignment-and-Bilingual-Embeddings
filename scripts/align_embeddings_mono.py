import numpy as np
from scipy.linalg import orthogonal_procrustes
import fasttext

print("Loading high-capacity monolingual models...")
m_en = fasttext.load_model("output/eng_mono.bin")
m_hi = fasttext.load_model("output/hin_mono.bin")

print("Loading seed dictionary from parallel corpus...")
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
            # skip OOV or problematic tokens
            pass

print("Valid seed pairs used for monolingual alignment:", len(pairs))
if not pairs:
    raise SystemExit("No valid seed pairs – check seed.txt or vocab.")

X = np.array([p[0] for p in pairs])  # English vectors
Y = np.array([p[1] for p in pairs])  # Hindi vectors

print("Running Orthogonal Procrustes for monolingual embeddings...")
R, _ = orthogonal_procrustes(X, Y)

np.save("output/R_mono.npy", R)
print("Saved monolingual alignment matrix → output/R_mono.npy")
