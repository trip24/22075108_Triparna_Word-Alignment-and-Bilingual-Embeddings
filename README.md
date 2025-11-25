# 🧑‍🎓 **BTP Project — Word Alignment and Bilingual Word Embeddings**

**Author:** *Triparna Samanta*

**Roll Number:** *22075108*

**Institution:** *IIT(BHU),Varanasi*

**BTP Title:** **Encoder–Decoder Word Alignment**

**Professor:** *Anil Singh*


# 📘 **Bilingual Embedding Pipeline for Hindi–English**

*A complete pipeline using Parallel → Comparable → Monolingual corpora, Word Alignment, Embedding Alignment, Visualization, and Dictionary Generation.*

---

## 📄 **Table of Contents**

1. [Project Overview](#project-overview)
2. [Datasets Used](#datasets-used)
3. [System Architecture](#system-architecture)
4. [Methodology](#methodology)
5. [Pipeline Stages](#pipeline-stages)

   * Parallel Corpora Stage
   * Comparable Corpora Stage
   * Monolingual Corpora Stage
6. [Word Alignment](#word-alignment)
7. [Embedding Alignment](#embedding-alignment)
8. [Dictionary Generation](#dictionary-generation)
9. [Visualization](#visualization)
10. [File Structure](#file-structure)
11. [How to Run the Project](#how-to-run-the-project)
12. [Final Outputs](#final-outputs)
13. [Future Work](#future-work)
14. [Project Report & Presentation](#project-report--presentation)

---

# 🧠 **Project Overview**

This project builds **high-quality bilingual embeddings** for the Hindi–English language pair using:

* Parallel corpora
* Comparable corpora
* Monolingual corpora

The pipeline includes:

* Word alignment using **Encoder-Decoder Model**
* Bilingual embedding alignment using **Orthogonal Procrustes**
* Training FastText embeddings
* Visualization using t-SNE
* Automatically generated bilingual & monolingual dictionaries

This follows the complete instruction given by the supervisor:

> “First use parallel corpora → then comparable corpora → then monolingual corpora,
> use word alignment, align embeddings, visualize them and finally create bilingual and monolingual dictionaries.”

---

# 📦 **Datasets Used**

### **1️⃣ Parallel Corpus – AI4Bharat Samanantar**

Source:
👉 [https://samanantar.com/](https://samanantar.com/) (AI4Bharat)

* Contains Hindi–English sentence-aligned data
* Used to train baseline bilingual embeddings
* Used for Encoder Decoder word alignment

Your extracted files:

* `data/en.txt`
* `data/hi.txt`

---

### **2️⃣ Comparable Corpus – Wikipedia Articles (Manually Collected)**

Since comparable corpora need to be *topically similar but not sentence-aligned*,
you manually collected Wikipedia pages in:

* Sports
* Technology
* Education
* Miscellaneous topics

You copy/pasted these into:

* `data/comp_en.txt`
* `data/comp_hi.txt`

These are **independently shuffled** to simulate non-parallel comparable corpora.

---

### **3️⃣ Monolingual Corpus – IIT Bombay English & Hindi Monolingual Dataset**

Source: IITB Monolingual Corpora
👉 [https://www.cfilt.iitb.ac.in/iitb_parallel/](https://www.cfilt.iitb.ac.in/iitb_parallel/)

Used to improve:

* Language modeling quality
* Richness of embedding space
* Quality of bilingual dictionary after projection

Stored in:

* `data/en_mono.txt`
* `data/hi_mono.txt`

---

# 🧬 **System Architecture**

```
          Parallel Corpus (Samanantar)
                     │
             Sentence Alignment
                     │
                Encoder Decoder Model
             (Word Alignments)
                     │
       ┌─────────────┴────────────────┐
       │                               │
Parallel Embeddings            Comparable Corpora
(FastText)                     (Shuffled Wikipedia)
       │                               │
       └─────────────┬────────────────┘
                     │
  Train Parallel + Comparable Embeddings
                     │
             Procrustes Alignment
                     │
            Comparable-Aligned Space
                     │
       ┌─────────────┴────────────────┐
       │                               │
  Monolingual EN Corpus       Monolingual HI Corpus
  (IIT Bombay)                (IIT Bombay)
       │                               │
        Train High-Capacity Monolingual FastText
                     │
             Procrustes Mapping
                     │
      Monolingual-Aligned Bilingual Space
                     │
       ┌─────────────┴─────────────────┐
       │                                │
 Bilingual Dictionary             Monolingual Dictionary
 (EN→HI / HI→EN)                 (Synonym Lists)
```

---

# 🔬 **Methodology**

This project follows a **3-stage pipeline**:

1. **Parallel Corpus Stage**
2. **Comparable Corpus Stage**
3. **Monolingual Corpus Stage**

Every stage trains its own embeddings, aligns them, visualizes them, and generates dictionaries.

### Core technologies used:

* **FastText Skip-gram embeddings**
* **Encoder Decoder based word alignment**
* **Orthogonal Procrustes mapping**
* **t-SNE visualization**
* **Cosine similarity search for dictionary generation**

---

# 🚦 **Pipeline Stages**

---

## 1️⃣ **Parallel Corpus Stage (Base Model)**

Using **Samanantar parallel corpus**:

✔ Train FastText EN/HI embeddings
✔ Extract word alignments using Encoder Decoder
✔ Build seed dictionary (`seed.txt`)
✔ Learn mapping matrix using Procrustes
✔ Visualize aligned vs unaligned spaces
✔ Generate first bilingual dictionary

---

## 2️⃣ **Comparable Corpus Stage (Rich Semantic Space)**

Using **Wikipedia comparable corpora**:

✔ Shuffle English & Hindi independently
✔ Combine parallel + comparable corpora
✔ Train improved FastText embeddings
✔ Align with the same seed dictionary
✔ Visualize comparable-enhanced embedding space
✔ Generate comparable-based bilingual dictionary

---

## 3️⃣ **Monolingual Corpus Stage (Best Model)**

Using **IIT Bombay monolingual corpora**:

✔ Train 300-dim high-capacity FastText embeddings
✔ Align them using parallel seed dictionary
✔ Visualize monolingual-aligned bilingual space
✔ Generate best-quality bilingual dictionary
✔ Generate monolingual synonym dictionaries

---

# 🔗 **Word Alignment**

We use **Encoder Decoder based Word Alignment**:

* Model: Encoder Decoder Architecture
* Matching method: `mwmf` (Many-to-Many Maximum F-score Word Matching)
* Produces high-quality bilingual word pairs

Output file:

```
output/seed.txt
```

This seed is essential for:

* Comparable embedding alignment
* Monolingual embedding alignment

---

# 🎯 **Embedding Alignment**

We align EN/HI embeddings using **Orthogonal Procrustes**:

[
R = \arg\min_R |XR - Y| \quad \text{s.t. } R^\top R = I
]

This produces alignment matrices:

```
output/R.npy         (parallel)
output/R_all.npy     (parallel + comparable)
output/R_mono.npy    (monolingual)
```

---

# 📘 **Dictionary Generation**

Three bilingual dictionaries:

* `dictionary.json` (parallel-only)
* `dictionary_all.json` (parallel + comparable)
* `dictionary_mono.json` (monolingual-enhanced)

Two monolingual dictionaries:

* `en_monodict.json`
* `hi_monodict.json`

Generated using cosine similarity of aligned embeddings.

---

# 🎨 **Visualization**

t-SNE plots generated:

```
output/tsne_unaligned.png
output/tsne_aligned.png
output/tsne_all_aligned.png
output/tsne_mono_aligned.png
```

These show the quality of alignment improving through:

parallel → comparable → monolingual.

---

# 📁 **File Structure**

```
BilingualProject/
│
├── data/
│   ├── en.txt                    # Samanantar parallel
│   ├── hi.txt
│   ├── comp_en.txt              # Comparable corpora (Wikipedia)
│   ├── comp_hi.txt
│   ├── en_train_all.txt         # Parallel + comparable
│   ├── hi_train_all.txt
│   ├── en_mono.txt              # IIT Bombay monolingual
│   ├── hi_mono.txt
├── model/
│   ├── model.py  
|
|         
├── output/
│   ├── eng.bin                  # Parallel embeddings
│   ├── hin.bin
│   ├── eng_all.bin              # Parallel + comparable embeddings
│   ├── hin_all.bin
│   ├── eng_mono.bin             # Monolingual embeddings
│   ├── hin_mono.bin
│   ├── seed.txt                 # Word alignments
│   ├── R.npy                    # Alignment matrices
│   ├── R_all.npy
│   ├── R_mono.npy
│   ├── dictionary.json
│   ├── dictionary_all.json
│   ├── dictionary_mono.json
│   ├── en_monodict.json
│   ├── hi_monodict.json
│   ├── tsne_unaligned.png
│   ├── tsne_aligned.png
│   ├── tsne_all_aligned.png
│   ├── tsne_mono_aligned.png
│
├── scripts/
│   ├── prepare_parallel.py
│   ├── make_comparable.py
│   ├── make_train_all.py
│   ├── train_embeddings.py
│   ├── train_embeddings_all.py
│   ├── train_embeddings_mono.py
│   ├── extract_seed.py
│   ├── align_embeddings.py
│   ├── align_embeddings_all.py
│   ├── align_embeddings_mono.py
│   ├── generate_dictionary.py
│   ├── generate_dictionary_all.py
│   ├── generate_dictionary_mono.py
│   ├── generate_monolingual_dicts.py
│   ├── visualize_unaligned.py
│   ├── visualize_aligned.py
│   ├── visualize_all_aligned.py
│   ├── visualize_mono_aligned.py
│
└── README.md
```

---

# ▶️ **How to Run the Project**

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install fasttext simalign numpy scipy sklearn matplotlib
```

---

### 2. Prepare parallel corpus

```bash
python scripts/prepare_parallel.py
```

---

### 3. Extract word alignments

```bash
python scripts/extract_seed.py
```

---

### 4. Train parallel embeddings

```bash
python scripts/train_embeddings.py
```

---

### 5. Align & visualize parallel embeddings

```bash
python scripts/align_embeddings.py
python scripts/visualize_aligned.py
```

---

### 6. Create comparable corpora

```bash
python scripts/make_comparable.py
python scripts/make_train_all.py
```

---

### 7. Train comparable embeddings

```bash
python scripts/train_embeddings_all.py
python scripts/align_embeddings_all.py
python scripts/visualize_all_aligned.py
```

---

### 8. Train monolingual embeddings

```bash
python scripts/train_embeddings_mono.py
python scripts/align_embeddings_mono.py
python scripts/visualize_mono_aligned.py
```

---

### 9. Generate dictionaries

```bash
python scripts/generate_dictionary.py
python scripts/generate_dictionary_all.py
python scripts/generate_dictionary_mono.py
python scripts/generate_monolingual_dicts.py
```

---

# 🏁 **Final Results**

### 🔤 Embeddings

* Parallel
* Comparable
* Monolingual

### 🧭 Alignments

* Mapping matrices
* Word alignment seeds

### 📘 Dictionaries

* Parallel dictionary
* Comparable-enhanced dictionary
* Monolingual-enhanced dictionary
* Monolingual synonym dictionaries

### 🎨 Visualizations

* Parallel (before & after alignment)
* Comparable-enhanced
* Monolingual-enhanced

---

# 🚀 **Future Work**

* Use contextual embeddings (mBERT, XLM-R, LaBSE)
* Train full MUSE unsupervised bilingual mapping
* Use CSLS instead of cosine similarity
* Build sentence-level dictionaries
* Integrate with Bhaashik annotation tool
* Add WordNet-style semantic graph construction

---

# 🎓 **Project Report & Presentation**

### 📌 **Presentation (Google Drive link)**

👉 https://drive.google.com/drive/folders/1Hd0S3gwncXq4ADlcNMk2uMsBL4byHJXl?usp=drive_link

### 📌 **Full PDF Report**

👉 https://drive.google.com/drive/folders/1Hd0S3gwncXq4ADlcNMk2uMsBL4byHJXl?usp=drive_link

### 📌 **Final Paper (Optional)**

👉 https://drive.google.com/drive/folders/1Hd0S3gwncXq4ADlcNMk2uMsBL4byHJXl?usp=drive_link

---

# 🙌 **End of README**


"# 22075108_Triparna_Word-Alignment-and-Bilingual-Embeddings" 
