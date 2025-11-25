import os
import math
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# At the top with other imports
from transformers.models.bart.modeling_bart import shift_tokens_right

# ─── Hyperparameters ───────────────────────────────────────────
train_src    = "/content/drive/MyDrive/datasets/eng-train.txt"
train_tgt    = "/content/drive/MyDrive/datasets/hi-train.txt"
train_align  = "/content/drive/MyDrive/datasets/alignments.txt"
test_src     = "/content/drive/MyDrive/datasets/eng-test.txt"
test_tgt     = "/content/drive/MyDrive/datasets/hi-test.txt"

batch_size   = 6
epochs       = 5  # Increased for better convergence
lr           = 1e-5
max_length   = 500
save_dir     = "./checkpoints"
model_name   = "facebook/mbart-large-50-many-to-many-mmt"

# ─── Device Setup ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_dir, exist_ok=True)
print(f"Using device: {device}")

# ─── Tokenizer with Special Handling ───────────────────────────
tokenizer = MBart50TokenizerFast.from_pretrained(
    model_name,
    src_lang="en_XX",
    tgt_lang="hi_IN"
)

# ─── Dataset with Robust Alignment Handling ────────────────────
class ParallelDataset(Dataset):
    def __init__(self, src_path, tgt_path, align_path, tokenizer, max_length=128):
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f]
        with open(align_path, "r", encoding="utf-8") as f:
            self.align_lines = [line.strip() for line in f]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_ids = set(tokenizer.all_special_ids)

        assert len(self.src_lines) == len(self.tgt_lines) == len(self.align_lines)

    def __len__(self):
        return len(self.src_lines)

    def _get_word_mapping(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True
        )
        word_map = defaultdict(list)
        for sub_idx, (word_id, (start, end)) in enumerate(zip(enc.word_ids(), enc["offset_mapping"])):
            if word_id is not None and start < end and enc.input_ids[sub_idx] not in self.special_ids:
                word_map[word_id].append(sub_idx)
        return word_map

    def __getitem__(self, idx):
    # Tokenize and get word mappings
      src_map = self._get_word_mapping(self.src_lines[idx])
      tgt_map = self._get_word_mapping(self.tgt_lines[idx])

      # Convert alignments
      sub_aligns = []
      for pair in self.align_lines[idx].split():
          try:
              src_word, tgt_word = map(int, pair.split("-"))
              for s_sub in src_map.get(src_word, []):
                  for t_sub in tgt_map.get(tgt_word, []):
                      if s_sub < self.max_length and t_sub < self.max_length:
                          sub_aligns.append((s_sub, t_sub))
          except (ValueError, IndexError):
              continue

      # Tokenize source normally
      src_enc = self.tokenizer(
          self.src_lines[idx],
          truncation=True,
          max_length=self.max_length,
          return_tensors="pt"
      )

      # Tokenize target with special handling (CRITICAL FIX)
      with self.tokenizer.as_target_tokenizer():  # <-- ADD THIS
          tgt_enc = self.tokenizer(
              self.tgt_lines[idx],
              truncation=True,
              max_length=self.max_length,
              return_tensors="pt"
          )

      return {
          "input_ids": src_enc["input_ids"][0],
          "attention_mask": src_enc["attention_mask"][0],
          "decoder_input_ids": tgt_enc["input_ids"][0],
          "decoder_attention_mask": tgt_enc["attention_mask"][0],
          "alignments": sub_aligns
      }

def collate_fn(batch):
    src_batch = tokenizer.pad(
        {"input_ids": [x["input_ids"] for x in batch],
         "attention_mask": [x["attention_mask"] for x in batch]},
        padding=True,
        return_tensors="pt"
    )
    tgt_batch = tokenizer.pad(
        {"input_ids": [x["decoder_input_ids"] for x in batch],
         "attention_mask": [x["decoder_attention_mask"] for x in batch]},
        padding=True,
        return_tensors="pt"
    )
    return {
        "input_ids": src_batch["input_ids"].to(device),
        "attention_mask": src_batch["attention_mask"].to(device),
        "decoder_input_ids": tgt_batch["input_ids"].to(device),
        "decoder_attention_mask": tgt_batch["attention_mask"].to(device),
        "alignments": [x["alignments"] for x in batch]
    }

# ─── Model with Stabilized Training ────────────────────────────
class AlignmentModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_name)

        # Freeze base model
        for param in self.mbart.parameters():
            param.requires_grad = False

        # Alignment components
        self.d_model = self.mbart.config.d_model
        self.self_attn = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.mbart.config.encoder_attention_heads,
            batch_first=True
        )
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)

        # Initialize alignment components
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, alignments=None):
        # Encoder
        encoder_out = self.mbart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # Decoder
        decoder_out = self.mbart.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=attention_mask
        ).last_hidden_state

        # Additional self-attention
        key_padding_mask = None
        attn_out, _ = self.self_attn(
            decoder_out,
            decoder_out,
            decoder_out,
            key_padding_mask=key_padding_mask
        )

        # Alignment projections
        Q = self.query(attn_out)
        K = self.key(encoder_out)

        # Alignment matrix
        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_model)
        A = F.softmax(logits, dim=-1)

        if alignments is None:
            return A, None

        # Guided alignment loss
        loss = 0.0
        valid_batches = 0
        special_ids = {self.mbart.config.pad_token_id,
                       self.mbart.config.bos_token_id,
                       self.mbart.config.eos_token_id}

        for batch_idx in range(A.size(0)):
            valid_tokens = []
            for t, token_id in enumerate(decoder_input_ids[batch_idx]):
                if token_id not in special_ids:
                    valid_tokens.append(t)

            if not valid_tokens:
                continue

            batch_loss = 0.0
            valid_count = 0

            for t in valid_tokens:
                src_indices = [s for s, t_align in alignments[batch_idx] if t_align == t]
                if not src_indices:
                    continue

                # Handle out-of-bounds indices
                src_indices = [s for s in src_indices if s < A.size(-1)]
                if not src_indices:
                    continue

                prob = A[batch_idx, t, src_indices].sum()
                batch_loss -= torch.log(prob + 1e-10)
                valid_count += 1

            if valid_count > 0:
                loss += batch_loss / valid_count
                valid_batches += 1

        if valid_batches > 0:
            loss = loss / valid_batches
        else:
            loss = torch.tensor(0.0, device=device)

        return A, loss

# ─── Training Infrastructure ───────────────────────────────────
def train(model, train_loader, epochs, optimizer):
    best_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        valid_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            decoder_input_ids = shift_tokens_right(
            batch['decoder_input_ids'],
            tokenizer.pad_token_id,
            tokenizer.lang_code_to_id["hi_IN"]  # Hindi language token
            )
            batch['decoder_input_ids'] = decoder_input_ids

            _, loss = model(**{k:v for k,v in batch.items() if k != 'alignments'},
                           alignments=batch['alignments'])

            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                valid_batches += 1

        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        scheduler.step(avg_loss)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"align_epoch_{epoch}.pt"))

    return model

# ─── Alignment Extraction ──────────────────────────────────────


def extract_alignments(model, src_texts, tgt_texts, output_file):
    model.eval()
    special_ids = set(tokenizer.all_special_ids)

    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
            if idx % 100 == 0:
                print(f"Processing {idx+1}/{len(src_texts)}")

            # Tokenize source and target
            src_enc = tokenizer(
                src,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_offsets_mapping=True
            ).to(device)

            with tokenizer.as_target_tokenizer():
                tgt_enc = tokenizer(
                    tgt,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_offsets_mapping=True
                ).to(device)

            # Fix 1: Proper decoder input shifting
            decoder_input_ids = shift_tokens_right(
                tgt_enc.input_ids,
                tokenizer.pad_token_id,
                tokenizer.lang_code_to_id["hi_IN"]
            )

            inputs = {
                "input_ids": src_enc.input_ids,
                "attention_mask": src_enc.attention_mask,
                "decoder_input_ids": decoder_input_ids,  # Use shifted IDs
                "decoder_attention_mask": tgt_enc.attention_mask
            }

            # Fix 2: Correct batch handling
            with torch.no_grad():
                A, _ = model(**inputs)  # A shape: [batch_size, tgt_len, src_len]

            # Word-to-subword mappings
            src_word_map = defaultdict(list)
            for sub_idx, (word_id, (start, end)) in enumerate(zip(src_enc.word_ids(), src_enc["offset_mapping"][0])):
                if word_id is not None and start < end and src_enc.input_ids[0, sub_idx] not in special_ids:
                    src_word_map[word_id].append(sub_idx)

            tgt_word_map = defaultdict(list)
            for sub_idx, (word_id, (start, end)) in enumerate(zip(tgt_enc.word_ids(), tgt_enc["offset_mapping"][0])):
                if word_id is not None and start < end and tgt_enc.input_ids[0, sub_idx] not in special_ids:
                    tgt_word_map[word_id].append(sub_idx)

            # Fix 3: Proper alignment extraction with batch indexing
            word_pairs = set()
            batch_idx = 0  # Since we process one sentence at a time
            for t_sub in range(A.size(1)):  # Iterate over target tokens
                # Skip special tokens
                if tgt_enc.input_ids[0, t_sub].item() in special_ids:
                    continue

                # Get best source alignment (shape: [batch, tgt, src])
                s_sub = A[batch_idx, t_sub].argmax().item()

                # Validate source index
                if s_sub >= src_enc.input_ids.size(1) or src_enc.input_ids[0, s_sub].item() in special_ids:
                    continue

                # Map subwords to original words
                src_word = next((k for k, v in src_word_map.items() if s_sub in v), None)
                tgt_word = next((k for k, v in tgt_word_map.items() if t_sub in v), None)

                if src_word is not None and tgt_word is not None:
                    word_pairs.add((src_word, tgt_word))

            # Write alignments in Pharaoh format
            sorted_pairs = sorted(word_pairs, key=lambda x: (x[0], x[1]))
            fout.write(" ".join(f"{s}-{t}" for s, t in sorted_pairs) + "\n")

    print(f"Alignments saved to {output_file}")

# ─── Main Execution ────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize data and model
    train_dataset = ParallelDataset(train_src, train_tgt, train_align, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    model = AlignmentModel(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Train
    trained_model = train(model, train_loader, epochs, optimizer)

    drive_output_path = "/content/drive/MyDrive/datasets/final_alignments.txt"

    # Load test data
    with open(test_src, "r", encoding="utf-8") as f:
        test_srcs = [line.strip() for line in f]
    with open(test_tgt, "r", encoding="utf-8") as f:
        test_tgts = [line.strip() for line in f]


    # Extract and save alignments
    extract_alignments(trained_model, test_srcs, test_tgts, drive_output_path)

    # Preview
    print("\nSample Alignments:")
    with open(drive_output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"Sentence {i+1}: {line.strip()}")