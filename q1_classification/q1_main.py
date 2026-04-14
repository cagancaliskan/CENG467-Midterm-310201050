#!/usr/bin/env python3
"""
CENG 467 – Natural Language Understanding and Generation
Take-Home Midterm – Question 1: Representation Learning in Text Classification
Student ID : 310201050
Seed       : 3102
Device     : Apple M2 (MPS)
"""

import os, json, re, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus  import stopwords

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# --- 0. reproducibility & device ---
SEED = 3102
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cuda")  if torch.cuda.is_available()         else
    torch.device("cpu")
)
print(f"\n{'='*60}")
print(f"  CENG 467 – Question 1: Text Classification")
print(f"  Student ID : 310201050   |   Seed : {SEED}")
print(f"  Device     : {DEVICE}")
print(f"{'='*60}\n")

OUT_DIR = "q1_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. data loading  - imdb ---
print("[STEP 1] Loading IMDb dataset …")
raw = load_dataset("imdb")  # huggingface is a lifesaver here

rng = random.Random(SEED)

train_all = list(zip(raw["train"]["text"], raw["train"]["label"]))
test_all  = list(zip(raw["test"]["text"],  raw["test"]["label"]))
rng.shuffle(train_all)
rng.shuffle(test_all)

# Fixed, reproducible splits (used by ALL models)
# limiting dataset size so it trains faster on my mac
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 6000, 1000, 1000
train_data = train_all[:TRAIN_SIZE]
val_data   = train_all[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
test_data  = test_all[:TEST_SIZE]          # ← used ONCE at final evaluation

texts_train,  labels_train  = zip(*train_data)
texts_val,    labels_val    = zip(*val_data)
texts_test,   labels_test   = zip(*test_data)
texts_train, labels_train   = list(texts_train),  list(labels_train)
texts_val,   labels_val     = list(texts_val),    list(labels_val)
texts_test,  labels_test    = list(texts_test),   list(labels_test)

print(f"  Train: {len(train_data)}  |  Val: {len(val_data)}  |  Test: {len(test_data)}\n")

# --- 2. preprocessing  -  two tokenization strategies ---
STOPWORDS = set(stopwords.words("english"))

def clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)

# basic preprocessing steps
def normalize(text: str) -> str:
    text = clean_html(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Strategy A – whitespace split + stopword removal  (used by TF-IDF / LR)
def tokenize_whitespace(text: str, remove_sw: bool = True) -> list:
    tokens = normalize(text).split()
    return [t for t in tokens if t not in STOPWORDS] if remove_sw else tokens

# Strategy B – NLTK word_tokenize  (used by BiLSTM; preserves more morphology)
def tokenize_nltk(text: str, remove_sw: bool = False) -> list:
    tokens = word_tokenize(normalize(text))
    return [t for t in tokens if t not in STOPWORDS] if remove_sw else tokens

print("[STEP 2] Preprocessing complete (two tokenization strategies defined)\n")

# --- 3. model 1 - tf-idf + logistic regression ---
print("[MODEL 1] TF-IDF (unigram + bigram) + Logistic Regression")
t0 = time.time()

# Apply Strategy A
sw_train = [" ".join(tokenize_whitespace(t)) for t in texts_train]
sw_val   = [" ".join(tokenize_whitespace(t)) for t in texts_val]
sw_test  = [" ".join(tokenize_whitespace(t)) for t in texts_test]

vectorizer = TfidfVectorizer(
    max_features  = 30_000,
    ngram_range   = (1, 2),
    sublinear_tf  = True,
    min_df        = 2,
)
X_train = vectorizer.fit_transform(sw_train)
X_val   = vectorizer.transform(sw_val)
X_test  = vectorizer.transform(sw_test)

lr = LogisticRegression(C=0.7, max_iter=1000, random_state=SEED, solver="lbfgs")
lr.fit(X_train, labels_train)

val_preds_lr  = lr.predict(X_val)
val_acc_lr    = accuracy_score(labels_val,  val_preds_lr)
val_f1_lr     = f1_score(labels_val,  val_preds_lr,  average="macro")
test_preds_lr = lr.predict(X_test)
test_acc_lr   = accuracy_score(labels_test, test_preds_lr)
test_f1_lr    = f1_score(labels_test, test_preds_lr, average="macro")

print(f"  Val  → Acc: {val_acc_lr:.4f}  Macro-F1: {val_f1_lr:.4f}")
print(f"  Test → Acc: {test_acc_lr:.4f}  Macro-F1: {test_f1_lr:.4f}  ({time.time()-t0:.1f}s)\n")

# --- 4. model 2 - bilstm ---
print("[MODEL 2] BiLSTM  (2-layer, bidirectional, dropout=0.35)")

# Hyper-parameters
MAX_VOCAB       = 20_000
MAX_LEN_LSTM    = 256
EMBED_DIM       = 128
HIDDEN_DIM      = 128
LSTM_LAYERS     = 2
DROPOUT_LSTM    = 0.35
BATCH_LSTM      = 32
EPOCHS_LSTM     = 5
LR_LSTM         = 1e-3

# Build vocabulary from training tokens (Strategy B)
train_tokens = [tokenize_nltk(t) for t in texts_train]
counter      = Counter(tok for toks in train_tokens for tok in toks)
vocab_words  = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common(MAX_VOCAB - 2)]
w2i          = {w: i for i, w in enumerate(vocab_words)}

    # helper to convert text tokens into vocab ids with padding
def encode_seq(tokens, max_len=MAX_LEN_LSTM):
    ids = [w2i.get(tok, 1) for tok in tokens][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

class IMDbSeqDataset(Dataset):
    def __init__(self, texts, labels):
        self.X = [torch.tensor(encode_seq(tokenize_nltk(t)), dtype=torch.long) for t in texts]
        self.y = [torch.tensor(l, dtype=torch.long) for l in labels]
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

ds_tr = IMDbSeqDataset(texts_train, labels_train)
ds_vl = IMDbSeqDataset(texts_val,   labels_val)
ds_te = IMDbSeqDataset(texts_test,  labels_test)
dl_tr = DataLoader(ds_tr, batch_size=BATCH_LSTM, shuffle=True,  worker_init_fn=lambda _: np.random.seed(SEED))
dl_vl = DataLoader(ds_vl, batch_size=BATCH_LSTM, shuffle=False)
dl_te = DataLoader(ds_te, batch_size=BATCH_LSTM, shuffle=False)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, n_layers, dropout, n_classes=2):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm    = nn.LSTM(embed_dim, hidden, num_layers=n_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)
        self.drop    = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # embed tokens and apply dropout to prevent overfitting
        emb = self.drop(self.embed(x))
        # pass through bidirectional lstm
        _, (h, _) = self.lstm(emb)
        # Concatenate last forward and backward hidden states
        out = torch.cat([h[-2], h[-1]], dim=1)
        # classify the sequence based on final hidden states
                # classify the concatenated forward/backward representations
return self.fc(self.drop(out))

lstm_clf  = BiLSTMClassifier(len(vocab_words), EMBED_DIM, HIDDEN_DIM, LSTM_LAYERS, DROPOUT_LSTM).to(DEVICE)
opt_lstm  = torch.optim.Adam(lstm_clf.parameters(), lr=LR_LSTM)
criterion = nn.CrossEntropyLoss()

def eval_seq(model, dl):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            p  = model(xb).argmax(1).cpu().tolist()
            preds  += p
            truths += yb.tolist()
    return accuracy_score(truths, preds), f1_score(truths, preds, average="macro"), preds

t0 = time.time()
for ep in range(EPOCHS_LSTM):
    lstm_clf.train(); total_loss = 0
    for xb, yb in dl_tr:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_lstm.zero_grad()
        # forward pass and calculate cross entropy loss
        loss = criterion(lstm_clf(xb), yb)
        # backpropagate errors
                # backpropagate errors and step the optimizer
loss.backward()
        nn.utils.clip_grad_norm_(lstm_clf.parameters(), 1.0)
        opt_lstm.step()
        total_loss += loss.item()
    va, vf, _ = eval_seq(lstm_clf, dl_vl)
    print(f"  Epoch {ep+1}/{EPOCHS_LSTM} | Loss: {total_loss/len(dl_tr):.4f}  Val Acc: {va:.4f}  Val F1: {vf:.4f}")

# evaluate the final bilstm model on the unseen test set
test_acc_lstm, test_f1_lstm, test_preds_lstm = eval_seq(lstm_clf, dl_te)
print(f"  Test → Acc: {test_acc_lstm:.4f}  Macro-F1: {test_f1_lstm:.4f}  ({time.time()-t0:.1f}s)\n")
