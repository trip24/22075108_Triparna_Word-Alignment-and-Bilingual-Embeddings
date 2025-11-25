import fasttext
import numpy as np
from sklearn.preprocessing import normalize
import json

def build_mono_dict(model, vocab_size=8000, top_k=5):
    words = model.get_words()[:vocab_size]
    M = np.array([model.get_word_vector(w) for w in words])
    M = normalize(M)
    sim = M.dot(M.T)   # cosine similarities
    mono_dict = {}
    for i, w in enumerate(words):
        # top_k+1 because first is the word itself
        idx = np.argpartition(-sim[i], top_k+1)[:top_k+1]
        idx = idx[idx != i]  # drop self
        neighbors = [words[j] for j in idx[:top_k]]
        mono_dict[w] = neighbors
    return mono_dict

print("Loading monolingual embeddings...")
m_en = fasttext.load_model("output/eng_mono.bin")
m_hi = fasttext.load_model("output/hin_mono.bin")

print("Building English monolingual dictionary (synonyms)...")
en_mono = build_mono_dict(m_en)
with open("output/en_monodict.json", "w", encoding="utf8") as f:
    json.dump(en_mono, f, ensure_ascii=False, indent=2)

print("Building Hindi monolingual dictionary (synonyms)...")
hi_mono = build_mono_dict(m_hi)
with open("output/hi_monodict.json", "w", encoding="utf8") as f:
    json.dump(hi_mono, f, ensure_ascii=False, indent=2)

print("Saved → en_monodict.json and hi_monodict.json")
