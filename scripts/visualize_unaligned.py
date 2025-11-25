import fasttext
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load models
m_en = fasttext.load_model('output/eng.bin')
m_hi = fasttext.load_model('output/hin.bin')

# pick some frequent words from each vocab
vocab_en = m_en.get_words()[:200]   # adjust if needed
vocab_hi = m_hi.get_words()[:200]

# get vectors
E = np.array([m_en.get_word_vector(w) for w in vocab_en])
H = np.array([m_hi.get_word_vector(w) for w in vocab_hi])

# stack them
X = np.vstack([E, H])
labels = vocab_en + vocab_hi
colors = ['blue'] * len(vocab_en) + ['red'] * len(vocab_hi)

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=0)
X2 = tsne.fit_transform(X)

plt.figure(figsize=(10,10))
for (x, y, c) in zip(X2[:,0], X2[:,1], colors):
    plt.scatter(x, y, c=c, s=10)
plt.title("Unaligned EN (blue) vs HI (red) embeddings")
plt.savefig('output/tsne_unaligned.png', dpi=300)
print("Saved output/tsne_unaligned.png")
