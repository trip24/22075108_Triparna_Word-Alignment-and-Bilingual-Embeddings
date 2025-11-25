import fasttext

print("Training English monolingual embeddings (high capacity)...")
m_en = fasttext.train_unsupervised(
    'data/en_mono.txt',
    model='skipgram',
    dim=300,      # higher dimension
    epoch=10,     # more training
    minn=2,       # subword n-grams
    maxn=6
)
m_en.save_model('output/eng_mono.bin')
print("Saved output/eng_mono.bin")

print("Training Hindi monolingual embeddings (high capacity)...")
m_hi = fasttext.train_unsupervised(
    'data/hi_mono.txt',
    model='skipgram',
    dim=300,
    epoch=10,
    minn=2,
    maxn=6
)
m_hi.save_model('output/hin_mono.bin')
print("Saved output/hin_mono.bin")

print("Done training monolingual embeddings.")
