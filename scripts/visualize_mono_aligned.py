import fasttext
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("Loading monolingual models and alignment...")
m_en = fasttext.load_model("output/eng_mono.bin")
m_hi = fasttext.load_model("output/hin_mono.bin")
R = np.load("output/R_mono.npy")

# sample some words from each vocab
vocab_en = m_en.get_words()[:200]
vocab_hi = m_hi.get_words()[:200]

E = np.array([m_en.get_word_vector(w) for w in vocab_en]).dot(R)
H = np.array([m_hi.get_word_vector(w) for w in vocab_hi])

X = np.vstack([E, H])
colors = ["blue"] * len(E) + ["red"] * len(H)

print("Running t-SNE on monolingual-aligned embeddings...")
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X2 = tsne.fit_transform(X)

plt.figure(figsize=(10, 10))
for (x, y, c) in zip(X2[:, 0], X2[:, 1], colors):
    plt.scatter(x, y, c=c, s=10)

plt.title("Monolingual-trained EN (blue) + HI (red) in shared space")
plt.savefig("output/tsne_mono_aligned.png", dpi=300)
print("Saved → output/tsne_mono_aligned.png")
