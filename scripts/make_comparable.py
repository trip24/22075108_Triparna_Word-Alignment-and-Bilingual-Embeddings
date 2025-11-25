import random

print("Loading parallel corpora...")

en = open("data/en.txt", encoding="utf8").read().strip().splitlines()
hi = open("data/hi.txt", encoding="utf8").read().strip().splitlines()

print("English lines:", len(en))
print("Hindi lines:", len(hi))

# Shuffle independently
print("Shuffling to create comparable corpora...")
random.shuffle(en)
random.shuffle(hi)

# Save as comparable corpora
open("data/comp_en.txt", "w", encoding="utf8").write("\n".join(en))
open("data/comp_hi.txt", "w", encoding="utf8").write("\n".join(hi))

print("Saved:")
print("  data/comp_en.txt")
print("  data/comp_hi.txt")
print("Comparable corpora created successfully!")
