import fasttext
import numpy as np
from sklearn.preprocessing import normalize
import json

print("Loading models and alignment (parallel + comparable)...")
m_en = fasttext.load_model("output/eng_all.bin")
m_hi = fasttext.load_model("output/hin_all.bin")
R = np.load("output/R_all.npy")

# choose vocab sizes; you can change these
vocab_en = m_en.get_words()[:5000]
vocab_hi = m_hi.get_words()[:8000]

print("Building embedding matrices...")
E = np.array([m_en.get_word_vector(w) for w in vocab_en]).dot(R)  # aligned English
H = np.array([m_hi.get_word_vector(w) for w in vocab_hi])         # Hindi

# normalize for cosine similarity
E = normalize(E)
H = normalize(H)

dictionary = {}
print("Generating bilingual dictionary (parallel + comparable)...")
for i, en_word in enumerate(vocab_en):
    sims = H.dot(E[i])              # cosine similarities to all Hindi words
    best = np.argmax(sims)
    dictionary[en_word] = vocab_hi[best]

with open("output/dictionary_all.json", "w", encoding="utf8") as f:
    json.dump(dictionary, f, ensure_ascii=False, indent=2)

print("Saved → output/dictionary_all.json")
