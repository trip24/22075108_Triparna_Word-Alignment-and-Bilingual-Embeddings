import re
from pathlib import Path

DATA_DIR = Path("data")

def clean_wiki_text(text: str) -> str:
    # Remove numeric references like [39], [123], [1][2]
    text = re.sub(r"\[\d+\]", "", text)
    # Remove [citation needed]-style tags
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def simple_sentence_split(text: str):
    # Very rough sentence splitter: split on . ? !
    text = text.replace("\n", " ")
    parts = re.split(r"[.?!]+", text)
    sents = [p.strip() for p in parts if p.strip()]
    return sents

def remove_english_from_hindi(text: str) -> str:
    # Remove sequences of Latin letters from Hindi text
    # Example: "भारत [39] India" -> "भारत  "
    text = re.sub(r"[A-Za-z]+", "", text)
    # Clean extra spaces again
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    # ---------- ENGLISH COMPARABLE ----------
    comp_en_raw_path = DATA_DIR / "comp_en_raw.txt"
    comp_hi_raw_path = DATA_DIR / "comp_hi_raw.txt"

    if comp_en_raw_path.exists():
        en_raw = comp_en_raw_path.read_text(encoding="utf8")
        en_clean = clean_wiki_text(en_raw)
        en_sents = simple_sentence_split(en_clean)
        (DATA_DIR / "comp_en.txt").write_text(
            "\n".join(en_sents),
            encoding="utf8"
        )
        print(f"English comparable sentences: {len(en_sents)}")
    else:
        en_sents = []
        print("WARNING: data/comp_en_raw.txt not found, skipping EN comparable.")

    # ---------- HINDI COMPARABLE ----------
    if comp_hi_raw_path.exists():
        hi_raw = comp_hi_raw_path.read_text(encoding="utf8")
        hi_clean = clean_wiki_text(hi_raw)
        hi_clean = remove_english_from_hindi(hi_clean)
        hi_sents = simple_sentence_split(hi_clean)
        (DATA_DIR / "comp_hi.txt").write_text(
            "\n".join(hi_sents),
            encoding="utf8"
        )
        print(f"Hindi comparable sentences (after removing English & citations): {len(hi_sents)}")
    else:
        hi_sents = []
        print("WARNING: data/comp_hi_raw.txt not found, skipping HI comparable.")

    # ---------- MERGE WITH PARALLEL CORPORA ----------
    en_parallel = (DATA_DIR / "en.txt").read_text(encoding="utf8").strip().splitlines()
    hi_parallel = (DATA_DIR / "hi.txt").read_text(encoding="utf8").strip().splitlines()

    # Train-all = parallel + comparable
    en_train_all = en_parallel + en_sents
    hi_train_all = hi_parallel + hi_sents

    (DATA_DIR / "en_train_all.txt").write_text(
        "\n".join(en_train_all),
        encoding="utf8"
    )
    (DATA_DIR / "hi_train_all.txt").write_text(
        "\n".join(hi_train_all),
        encoding="utf8"
    )

    print(f"en_train_all.txt lines: {len(en_train_all)}")
    print(f"hi_train_all.txt lines: {len(hi_train_all)}")
    print("Done: cleaned comparable corpora and merged with parallel data.")

if __name__ == "__main__":
    main()
