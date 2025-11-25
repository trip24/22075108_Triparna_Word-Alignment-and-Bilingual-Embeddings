import numpy as np
from scipy.linalg import orthogonal_procrustes
import fasttext

print("Loading embeddings...")
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
            v_en = m_en.get_word_vector(en)
            v_hi = m_hi.get_word_vector(hi)
            pairs.append((v_en, v_hi))
        except:
            pass

print("Seed pairs used:", len(pairs))

X = np.array([p[0] for p in pairs])
Y = np.array([p[1] for p in pairs])

print("Running Procrustes mapping...")
R, _ = orthogonal_procrustes(X, Y)

np.save("output/R_all.npy", R)
print("Saved → output/R_all.npy")
