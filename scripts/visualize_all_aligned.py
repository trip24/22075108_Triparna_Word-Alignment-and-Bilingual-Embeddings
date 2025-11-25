import fasttext
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

m_en = fasttext.load_model("output/eng_all.bin")
m_hi = fasttext.load_model("output/hin_all.bin")
R = np.load("output/R_all.npy")

vocab_en = m_en.get_words()[:200]
vocab_hi = m_hi.get_words()[:200]

E = np.array([m_en.get_word_vector(w) for w in vocab_en]).dot(R)
H = np.array([m_hi.get_word_vector(w) for w in vocab_hi])

X = np.vstack([E, H])
colors = ["blue"]*len(E) + ["red"]*len(H)

tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X2 = tsne.fit_transform(X)

plt.figure(figsize=(10,10))
for (x, y, c) in zip(X2[:,0], X2[:,1], colors):
    plt.scatter(x, y, c=c, s=10)

plt.title("Aligned EN-HI Embeddings (Parallel + Comparable)")
plt.savefig("output/tsne_all_aligned.png", dpi=300)
