import fasttext

# Train English embeddings
print("Training English FastText model...")
model_en = fasttext.train_unsupervised('data/en.txt',
                                       model='skipgram',
                                       dim=300)
model_en.save_model('output/eng.bin')
print("Saved output/eng.bin")

# Train Hindi embeddings
print("Training Hindi FastText model...")
model_hi = fasttext.train_unsupervised('data/hi.txt',
                                       model='skipgram',
                                       dim=300)
model_hi.save_model('output/hin.bin')
print("Saved output/hin.bin")

print("Done training both models.")
