# scripts/visualize_aligned.py
import fasttext
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("Loading models and alignment...")
m_en = fasttext.load_model("output/eng.bin")
m_hi = fasttext.load_model("output/hin.bin")
R = np.load("output/R.npy")

# Sample some words
vocab_en = m_en.get_words()[:150]
vocab_hi = m_hi.get_words()[:150]

E = np.array([m_en.get_word_vector(w) for w in vocab_en]).dot(R)  # aligned EN
H = np.array([m_hi.get_word_vector(w) for w in vocab_hi])         # HI

X = np.vstack([E, H])
colors = ["blue"] * len(E) + ["red"] * len(H)

print("Running t-SNE on aligned embeddings...")
tsne = TSNE(n_components=2, random_state=0, perplexity=30)
X2 = tsne.fit_transform(X)

plt.figure(figsize=(10,10))
for (x, y, c) in zip(X2[:,0], X2[:,1], colors):
    plt.scatter(x, y, c=c, s=10)

plt.title("Aligned EN (blue) + HI (red) embeddings")
plt.savefig("output/tsne_aligned.png", dpi=300)
print("Saved output/tsne_aligned.png")
