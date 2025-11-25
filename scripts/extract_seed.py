from simalign import SentenceAligner
from collections import Counter

# Don't specify matching_methods; use library defaults
aligner = SentenceAligner(
    model="xlm-roberta-base",
    token_type="bpe"
)

pairs = Counter()

with open("data/parallel_corpus.txt", "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            en_sent, hi_sent = line.split("\t")
        except ValueError:
            continue  # skip malformed lines

        en_sent = en_sent.strip()
        hi_sent = hi_sent.strip()
        if not en_sent or not hi_sent:
            continue

        out = aligner.get_word_aligns(en_sent, hi_sent)
        # out is a dict: {method_name: list_of_alignments, ...}
        if not out:
            continue

        # pick the first available method (whatever it is in this version)
        method_name = next(iter(out.keys()))
        aligns = out[method_name]

        en_words = en_sent.split()
        hi_words = hi_sent.split()

        for i, j in aligns:
            if i < len(en_words) and j < len(hi_words):
                pairs[(en_words[i].lower(), hi_words[j])] += 1

# Save top 5000 seed pairs
with open("output/seed.txt", "w", encoding="utf8") as out_f:
    for (en, hi), freq in pairs.most_common(5000):
        out_f.write(f"{en} {hi}\n")

print("Saved output/seed.txt with", len(pairs), "distinct pairs")
