import random

# Read parallel corpora
with open("data/en.txt", encoding="utf8") as f:
    en_lines = [l.strip() for l in f if l.strip()]

with open("data/hi.txt", encoding="utf8") as f:
    hi_lines = [l.strip() for l in f if l.strip()]

print(f"Parallel EN lines: {len(en_lines)}")
print(f"Parallel HI lines: {len(hi_lines)}")

# Make copies and shuffle independently
en_comp = en_lines.copy()
hi_comp = hi_lines.copy()

random.shuffle(en_comp)
random.shuffle(hi_comp)

# Save as comparable corpora
with open("data/comp_en.txt", "w", encoding="utf8") as f:
    for line in en_comp:
        f.write(line + "\n")

with open("data/comp_hi.txt", "w", encoding="utf8") as f:
    for line in hi_comp:
        f.write(line + "\n")

print(f"Created comparable corpora: data/comp_en.txt, data/comp_hi.txt")
