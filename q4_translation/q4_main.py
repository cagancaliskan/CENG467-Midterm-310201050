#!/usr/bin/env python3
"""
CENG 467 – Natural Language Understanding and Generation
Take-Home Midterm – Question 4: Machine Translation (de → en)
Student ID : 310201050
Seed       : 3102
Device     : Apple M2 (MPS)

Model A : Seq2Seq with Bahdanau attention (trained from scratch on Multi30k)
Model B : Helsinki-NLP/opus-mt-de-en  (pretrained MarianMT Transformer)
"""

import os, json, random, re, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data       import Dataset, DataLoader
from torch.nn.utils.rnn     import pad_sequence, pack_padded_sequence, pad_packed_sequence
from collections            import Counter

from datasets   import load_dataset
import sacrebleu

import nltk
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_fn

from transformers import MarianMTModel, MarianTokenizer

# --- 0.  reproducibility & device ---
SEED = 3102
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)

print(f"\n{'='*60}")
print(f"  CENG 467 – Question 4: Machine Translation (de→en)")
print(f"  Student ID : 310201050  |  Seed : {SEED}")
print(f"  Device     : {DEVICE}")
print(f"{'='*60}\n")

OUT_DIR = "q4_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1.  data - multi30k (de↔en) ---
print("[STEP 1] Loading Multi30k …")
ds = load_dataset("bentrevett/multi30k")

de_train = [ex["de"] for ex in ds["train"]]
en_train = [ex["en"] for ex in ds["train"]]
de_val   = [ex["de"] for ex in ds["validation"]]
en_val   = [ex["en"] for ex in ds["validation"]]
de_test  = [ex["de"] for ex in ds["test"]]
en_test  = [ex["en"] for ex in ds["test"]]

print(f"  Train: {len(de_train)}  |  Val: {len(de_val)}  |  Test: {len(de_test)}\n")

# --- 2.  tokenisation & vocabulary  (used by seq2seq only; marianmt uses its own) ---
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

def tokenize(text: str) -> list:
    text = text.lower().strip()
    text = re.sub(r"([.!?,;:'\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def build_vocab(sents, max_size=8_000, min_freq=2):
    cnt = Counter()
    for s in sents: cnt.update(tokenize(s))
    keep = [w for w, c in cnt.most_common(max_size - len(SPECIALS)) if c >= min_freq]
    vocab = SPECIALS + keep
    return {w: i for i, w in enumerate(vocab)}

src_vocab = build_vocab(de_train, max_size=8_000)
tgt_vocab = build_vocab(en_train, max_size=8_000)
src_itos  = {i: w for w, i in src_vocab.items()}
tgt_itos  = {i: w for w, i in tgt_vocab.items()}
print(f"  Source vocab (de): {len(src_vocab)}   Target vocab (en): {len(tgt_vocab)}\n")

MAX_LEN = 50
def encode_src(text):
    ids = [src_vocab.get(t, UNK_IDX) for t in tokenize(text)[:MAX_LEN-1]]
    ids.append(EOS_IDX)
    return ids
def encode_tgt(text):
    body = [tgt_vocab.get(t, UNK_IDX) for t in tokenize(text)[:MAX_LEN-2]]
    return [SOS_IDX] + body + [EOS_IDX]

# --- 3.  dataset / dataloader ---
class M30kDataset(Dataset):
    def __init__(self, src_sents, tgt_sents):
        self.X = [torch.tensor(encode_src(s), dtype=torch.long) for s in src_sents]
        self.Y = [torch.tensor(encode_tgt(t), dtype=torch.long) for t in tgt_sents]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

def collate(batch):
    xs, ys  = zip(*batch)
    x_lens  = torch.tensor([len(x) for x in xs])
    x_pad   = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    y_pad   = pad_sequence(ys, batch_first=True, padding_value=PAD_IDX)
    return x_pad, y_pad, x_lens

BATCH = 32
dl_tr = DataLoader(M30kDataset(de_train, en_train), batch_size=BATCH, shuffle=True,  collate_fn=collate)
dl_vl = DataLoader(M30kDataset(de_val,   en_val),   batch_size=BATCH, shuffle=False, collate_fn=collate)
dl_te = DataLoader(M30kDataset(de_test,  en_test),  batch_size=BATCH, shuffle=False, collate_fn=collate)

# --- 4.  model a - seq2seq with bahdanau attention ---
EMBED_DIM = 256
HIDDEN    = 256
DROPOUT   = 0.30
EPOCHS    = 8
LR        = 1e-3
TF_RATIO  = 0.5
