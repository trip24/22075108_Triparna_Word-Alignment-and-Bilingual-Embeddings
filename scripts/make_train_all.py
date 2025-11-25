print("Combining parallel and comparable corpora...")

en_par = open("data/en.txt", encoding="utf8").read().strip().splitlines()
hi_par = open("data/hi.txt", encoding="utf8").read().strip().splitlines()

en_comp = open("data/comp_en.txt", encoding="utf8").read().strip().splitlines()
hi_comp = open("data/comp_hi.txt", encoding="utf8").read().strip().splitlines()

# Combine
en_all = en_par + en_comp
hi_all = hi_par + hi_comp

open("data/en_train_all.txt", "w", encoding="utf8").write("\n".join(en_all))
open("data/hi_train_all.txt", "w", encoding="utf8").write("\n".join(hi_all))

print("Created:")
print("  data/en_train_all.txt  (parallel + comparable)")
print("  data/hi_train_all.txt  (parallel + comparable)")
