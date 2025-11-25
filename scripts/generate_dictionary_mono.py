import fasttext
import numpy as np
from sklearn.preprocessing import normalize
import json

print("Loading monolingual models and alignment...")
m_en = fasttext.load_model("output/eng_mono.bin")
m_hi = fasttext.load_model("output/hin_mono.bin")
R = np.load("output/R_mono.npy")

vocab_en = m_en.get_words()[:8000]   # more, since you have more compute
vocab_hi = m_hi.get_words()[:12000]

print("Building embedding matrices...")
E = np.array([m_en.get_word_vector(w) for w in vocab_en]).dot(R)  # aligned EN
H = np.array([m_hi.get_word_vector(w) for w in vocab_hi])         # HI

# Normalize for cosine similarity
E = normalize(E)
H = normalize(H)

dictionary = {}
print("Generating bilingual dictionary from monolingual embeddings...")
for i, en_word in enumerate(vocab_en):
    sims = H.dot(E[i])
    best = np.argmax(sims)
    dictionary[en_word] = vocab_hi[best]

with open("output/dictionary_mono.json", "w", encoding="utf8") as f:
    json.dump(dictionary, f, ensure_ascii=False, indent=2)

print("Saved → output/dictionary_mono.json (monolingual-based bilingual dictionary)")
