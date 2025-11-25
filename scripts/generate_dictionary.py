# scripts/generate_dictionary.py
import fasttext
import numpy as np
from sklearn.preprocessing import normalize
import json

print("Loading models and alignment...")
m_en = fasttext.load_model("output/eng.bin")
m_hi = fasttext.load_model("output/hin.bin")
R = np.load("output/R.npy")

# select vocabularies (you can adjust sizes)
vocab_en = m_en.get_words()[:5000]
vocab_hi = m_hi.get_words()[:8000]

print("Building embedding matrices...")
E = np.array([m_en.get_word_vector(w) for w in vocab_en]).dot(R)
H = np.array([m_hi.get_word_vector(w) for w in vocab_hi])

# normalize for cosine similarity
E = normalize(E)
H = normalize(H)

dictionary = {}

print("Generating dictionary (this may take a bit)...")
for i, en_word in enumerate(vocab_en):
    sims = np.dot(H, E[i])         # similarity to all Hindi words
    best = np.argmax(sims)         # index of best match
    dictionary[en_word] = vocab_hi[best]

with open("output/dictionary.json", "w", encoding="utf8") as f:
    json.dump(dictionary, f, ensure_ascii=False, indent=2)

print("Dictionary saved → output/dictionary.json with", len(dictionary), "entries")
