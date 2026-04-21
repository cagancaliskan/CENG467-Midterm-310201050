#!/usr/bin/env python3
"""
CENG 467 – Natural Language Understanding and Generation
Take-Home Midterm – Question 5: Language Modeling
Student ID : 310201050
Seed       : 3102
Device     : Apple M2 (MPS)

Model 1 : Trigram with add-k smoothing (statistical baseline)
Model 2 : 2-layer LSTM language model (neural)
Dataset : WikiText-2
"""

import os, json, math, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from datasets import load_dataset

# --- 0. reproducibility & device ---
SEED = 3102
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)

print(f"\n{'='*60}")
print(f"  CENG 467 – Question 5: Language Modeling")
print(f"  Student ID : 310201050  |  Seed : {SEED}")
print(f"  Device     : {DEVICE}")
print(f"{'='*60}\n")

OUT_DIR = "q5_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. data loading - wikitext-2 (v1, preprocessed) ---
print("[STEP 1] Loading WikiText-2 …")

ds = None
for name, cfg in [("wikitext", "wikitext-2-v1"),
                  ("Salesforce/wikitext", "wikitext-2-v1")]:
    try:
        ds = load_dataset(name, cfg, trust_remote_code=True)
        print(f"  Loaded via: {name} / {cfg}")
        break
    except Exception:
        continue
if ds is None:
    raise RuntimeError("Could not load WikiText-2 from any source.")

def stream_tokens(split):
    """Concatenate all lines of a split and tokenise on whitespace."""
    text = " ".join(ex["text"] for ex in ds[split])
    return text.split()

train_tokens = stream_tokens("train")
val_tokens   = stream_tokens("validation")
test_tokens  = stream_tokens("test")
print(f"  Token counts → Train: {len(train_tokens):,}  Val: {len(val_tokens):,}  Test: {len(test_tokens):,}\n")

# --- 2. vocabulary  (single shared vocabulary across both models) ---
counter   = Counter(train_tokens)
vocab     = sorted(counter.keys())          # use full training-set vocabulary
if "<unk>" not in vocab:
    vocab = ["<unk>"] + vocab
w2i       = {w: i for i, w in enumerate(vocab)}
i2w       = {i: w for w, i in w2i.items()}
UNK_ID    = w2i["<unk>"]
V         = len(vocab)

def encode(tokens):
    return [w2i.get(t, UNK_ID) for t in tokens]

train_ids = encode(train_tokens)
val_ids   = encode(val_tokens)
test_ids  = encode(test_tokens)

print(f"[STEP 2] Vocabulary size: {V:,}  (UNK id = {UNK_ID})\n")

# --- 3. model 1 - trigram with add-k smoothing ---
print("[MODEL 1] Trigram with add-k smoothing")

K_SMOOTH = 0.01

# Count bigram contexts and trigram next-words from training data
trigram = defaultdict(lambda: defaultdict(int))   # (w1,w2) -> {w3: count}
bigram  = defaultdict(int)                         # (w1,w2) -> total count

t0 = time.time()
for i in range(len(train_ids) - 2):
    ctx = (train_ids[i], train_ids[i+1])
    trigram[ctx][train_ids[i+2]] += 1
    bigram [ctx] += 1
print(f"  Counted {len(bigram):,} unique bigram contexts in {time.time()-t0:.1f}s.")

def trigram_log_prob(w1, w2, w3):
    """log P(w3 | w1, w2)  with add-k smoothing."""
    ctx_total = bigram.get((w1, w2), 0)
    num       = trigram[(w1, w2)].get(w3, 0) + K_SMOOTH
    den       = ctx_total + K_SMOOTH * V
    return math.log(num / den)

def trigram_perplexity(token_ids):
    n = 0
    log_sum = 0.0
    for i in range(2, len(token_ids)):
        log_sum += trigram_log_prob(token_ids[i-2], token_ids[i-1], token_ids[i])
        n += 1
    return math.exp(-log_sum / n)

t0 = time.time()
trigram_test_ppl = trigram_perplexity(test_ids)
trigram_val_ppl  = trigram_perplexity(val_ids)
print(f"  Validation PPL : {trigram_val_ppl:.2f}")
print(f"  Test PPL       : {trigram_test_ppl:.2f}    ({time.time()-t0:.1f}s)\n")

# --- 4. model 2 - lstm language model ---
print("[MODEL 2] 2-layer LSTM language model")

# Hyperparameters (intentionally non-textbook values)
EMBED_DIM   = 192
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.45
