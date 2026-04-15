#!/usr/bin/env python3
"""
CENG 467 – Natural Language Understanding and Generation
Take-Home Midterm – Question 2: Named Entity Recognition
Student ID : 310201050
Seed       : 3102
Device     : Apple M2 (MPS)

DATA LOADING NOTE: HuggingFace datasets ≥ 3.0 removed `trust_remote_code`,
so all script-based CoNLL-2003 mirrors fail.  We download the canonical
CoNLL-2003 BIO files from a stable static mirror and normalise to BIO
tagging (handling either IOB1 or BIO source format).
"""

import os, json, random, time, warnings, urllib.request, zipfile
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, defaultdict

from seqeval.metrics import (
    precision_score, recall_score, f1_score, classification_report
)
from torchcrf import CRF

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    get_linear_schedule_with_warmup,
)

# --- 0.  reproducibility & device ---
SEED = 3102
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)

print(f"\n{'='*60}")
print(f"  CENG 467 – Question 2: Named Entity Recognition")
print(f"  Student ID : 310201050  |  Seed : {SEED}")
print(f"  Device     : {DEVICE}")
print(f"{'='*60}\n")

OUT_DIR = "q2_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1.  data loading - conll-2003 (multi-source download with bio normalisation) ---
print("[STEP 1] Loading CoNLL-2003 …")

CACHE_DIR = "./conll2003_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

LABELS     = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
              "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
NUM_LABELS = len(LABELS)
id2label   = {i: l for i, l in enumerate(LABELS)}
label2id   = {l: i for i, l in enumerate(LABELS)}
TAG_KEY    = "ner_tags"

def to_bio(tag_seq):
    """Normalise an IOB1 tag sequence to BIO (IOB2). Idempotent on BIO input."""
    out, prev = [], "O"
    for tag in tag_seq:
        if tag.startswith("I-"):
            etype = tag[2:]
            # If previous tag is O or a different type, this I- is actually a B-
            if prev == "O" or prev[2:] != etype:
                out.append("B-" + etype)
            else:
                out.append(tag)
        else:
            out.append(tag)
        prev = out[-1]
    return out

def parse_conll_file(path):
    """Parse CoNLL-2003 BIO/IOB1 file → list of {tokens, ner_tags} dicts."""
    sentences = []
    toks, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        # read line by line to parse the conll format
        for line in f:
            line = line.rstrip("\n")
            # docstart means a new document is starting, we can ignore it
                        # docstart means a new document is starting, skip it
if not line or line.startswith("-DOCSTART-"):
                if toks:
                    sentences.append({"tokens": toks, "ner_tags": to_bio(tags)})
                    toks, tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 4:
                toks.append(parts[0])
                tags.append(parts[-1])
        if toks:
            sentences.append({"tokens": toks, "ner_tags": to_bio(tags)})
    return sentences

# ── Source 1: deepai.org zip (preferred — contains train.txt/valid.txt/test.txt)
def try_deepai():
    target = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}
    if all(os.path.exists(os.path.join(CACHE_DIR, fn)) for fn in target.values()):
        return {split: os.path.join(CACHE_DIR, fn) for split, fn in target.items()}
    print("  Trying deepai.org mirror …")
    zip_path = os.path.join(CACHE_DIR, "conll2003.zip")
    urllib.request.urlretrieve("https://data.deepai.org/conll2003.zip", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CACHE_DIR)
    paths = {split: os.path.join(CACHE_DIR, fn) for split, fn in target.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        raise FileNotFoundError("deepai.org zip did not contain expected files")
    return paths

# ── Source 2: synalp/NER on GitHub (fallback — IOB1, normalised on parse)
SYNALP_URLS = {
    "train"      : "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train",
    "validation" : "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa",
    "test"       : "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb",
}
SYNALP_FILES = {"train": "synalp_train.txt", "validation": "synalp_valid.txt", "test": "synalp_test.txt"}

def try_synalp():
    paths = {}
    print("  Trying synalp/NER mirror …")
    for split, url in SYNALP_URLS.items():
        fp = os.path.join(CACHE_DIR, SYNALP_FILES[split])
        if not os.path.exists(fp):
            print(f"    downloading {split} …")
            urllib.request.urlretrieve(url, fp)
        paths[split] = fp
    return paths

paths = None
for fn in (try_deepai, try_synalp):
    try:
        paths = fn()
        break
    except Exception as e:
        print(f"    failed: {e}")
        continue

if paths is None:
    raise RuntimeError("Could not download CoNLL-2003 from any mirror.")

raw = {split: parse_conll_file(p) for split, p in paths.items()}

print(f"  Labels ({NUM_LABELS}): {LABELS}")
print(f"  Train: {len(raw['train'])}  |  Val: {len(raw['validation'])}  |  Test: {len(raw['test'])}\n")

# Sanity: warn on any unknown label after normalisation
seen_labels = {t for split in raw.values() for s in split for t in s["ner_tags"]}
unknown = seen_labels - set(LABELS)
if unknown:
    print(f"  [WARN] unknown labels in data: {unknown} (will be mapped to O)")

# --- 2.  build vocabulary (for bilstm-crf) ---
print("[STEP 2] Building word vocabulary for BiLSTM-CRF …")

word_counter = Counter()
for ex in raw["train"]:
    for tok in ex["tokens"]:
        word_counter[tok.lower()] += 1

VOCAB_WORDS = ["<PAD>", "<UNK>"] + [w for w, c in word_counter.most_common() if c >= 2]
w2i         = {w: i for i, w in enumerate(VOCAB_WORDS)}
print(f"  Vocabulary size: {len(VOCAB_WORDS)}\n")

def encode_tokens(toks):
    return [w2i.get(t.lower(), 1) for t in toks]

def encode_tags(tags):
    return [label2id.get(t, label2id["O"]) for t in tags]

# --- 3.  model 1 - bilstm-crf ---
print("[MODEL 1] BiLSTM-CRF")

EMBED_DIM    = 96
HIDDEN_DIM   = 192
DROPOUT_LSTM = 0.40
BATCH_LSTM   = 32
EPOCHS_LSTM  = 6
LR_LSTM      = 8e-4
GRAD_CLIP    = 1.0

class CoNLLSeqDataset(Dataset):
    def __init__(self, sentences):
        self.X = [encode_tokens(s["tokens"])  for s in sentences]
        self.Y = [encode_tags(s["ner_tags"])  for s in sentences]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.Y[i], dtype=torch.long),
        )

def collate_seq(batch):
    xs, ys  = zip(*batch)
