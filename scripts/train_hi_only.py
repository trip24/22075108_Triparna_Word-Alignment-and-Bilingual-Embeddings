import fasttext

print("Training Hindi ONLY for debugging...")
model_hi = fasttext.train_unsupervised(
    'data/hi_train_all.txt',
    model='skipgram',
    dim=100,     # smaller, faster, smaller file
    epoch=5,
    minn=0,      # turn off subword for now (simpler, smaller model)
    maxn=0
)
model_hi.save_model('output/hin_all.bin')
print("Saved output/hin_all.bin")
