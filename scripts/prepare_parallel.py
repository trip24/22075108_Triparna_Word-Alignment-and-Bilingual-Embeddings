import pandas as pd

en = open('data/en.txt', encoding='utf8').read().strip().splitlines()
hi = open('data/hi.txt', encoding='utf8').read().strip().splitlines()

if len(en) != len(hi):
    raise SystemExit(f"Line mismatch: {len(en)} EN vs {len(hi)} HI")

df = pd.DataFrame({'en': en, 'hi': hi})
df.to_csv('data/parallel_corpus.txt', sep='\t', index=False, header=False)

print("Saved parallel_corpus.txt with", len(df), "pairs")
