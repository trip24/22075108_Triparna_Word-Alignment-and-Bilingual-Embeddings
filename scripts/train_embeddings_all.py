import fasttext

print("Training English (parallel + comparable)...")
model_en = fasttext.train_unsupervised(
    "data/en_train_all.txt",
    model="skipgram",
    dim=300,
    epoch=10,
    minn=2,
    maxn=6
)
model_en.save_model("output/eng_all.bin")
print("Saved → output/eng_all.bin")

print("Training Hindi (parallel + comparable)...")
model_hi = fasttext.train_unsupervised(
    "data/hi_train_all.txt",
    model="skipgram",
    dim=300,
    epoch=10,
    minn=2,
    maxn=6
)
model_hi.save_model("output/hin_all.bin")
print("Saved → output/hin_all.bin")
