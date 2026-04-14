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

# --- 5. model 3 - distilbert ---
print("[MODEL 3] DistilBERT (distilbert-base-uncased, fine-tuned)")

BERT_MAX_LEN = 256
BERT_BATCH   = 16
BERT_EPOCHS  = 3
BERT_LR      = 2e-5
WARMUP_RATIO = 0.1

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class BertIMDbDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc    = tokenizer(list(texts), truncation=True, padding="max_length",
                                max_length=BERT_MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

bd_tr = DataLoader(BertIMDbDataset(texts_train, labels_train), batch_size=BERT_BATCH, shuffle=True)
bd_vl = DataLoader(BertIMDbDataset(texts_val,   labels_val),   batch_size=BERT_BATCH)
bd_te = DataLoader(BertIMDbDataset(texts_test,  labels_test),  batch_size=BERT_BATCH)

bert = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2).to(DEVICE)

opt_bert   = torch.optim.AdamW(bert.parameters(), lr=BERT_LR, weight_decay=0.01)
tot_steps  = len(bd_tr) * BERT_EPOCHS
scheduler  = get_linear_schedule_with_warmup(
    opt_bert,
    num_warmup_steps  = int(tot_steps * WARMUP_RATIO),
    num_training_steps= tot_steps,
)

def eval_bert(model, dl):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch, labels in dl:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            p      = logits.argmax(1).cpu().tolist()
            preds  += p
            truths += labels.tolist()
    return accuracy_score(truths, preds), f1_score(truths, preds, average="macro"), preds

t0 = time.time()
for ep in range(BERT_EPOCHS):
    bert.train(); total_loss = 0
    for batch, labels in bd_tr:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        opt_bert.zero_grad()
        loss = bert(**batch, labels=labels).loss
                # backpropagate errors and step the optimizer
loss.backward()
        nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
        opt_bert.step(); scheduler.step()
        total_loss += loss.item()
    va, vf, _ = eval_bert(bert, bd_vl)
    print(f"  Epoch {ep+1}/{BERT_EPOCHS} | Loss: {total_loss/len(bd_tr):.4f}  Val Acc: {va:.4f}  Val F1: {vf:.4f}")

test_acc_bert, test_f1_bert, test_preds_bert = eval_bert(bert, bd_te)
print(f"  Test → Acc: {test_acc_bert:.4f}  Macro-F1: {test_f1_bert:.4f}  ({time.time()-t0:.1f}s)\n")

# --- 6. misclassification analysis  (distilbert, 5 samples) ---
print("[STEP 6] Misclassification Analysis – DistilBERT")
label_name = {0: "negative", 1: "positive"}
misclassified = []
for i, (pred, true) in enumerate(zip(test_preds_bert, labels_test)):
    if pred != true:
        misclassified.append({
            "index"     : i,
            "true_label": label_name[true],
            "pred_label": label_name[pred],
            "text_snippet": texts_test[i][:400],
        })
for m in misclassified[:5]:
    print(f"\n  [{m['index']}] TRUE={m['true_label']}  PRED={m['pred_label']}")
    print(f"  \"{m['text_snippet'][:200]}...\"")


# --- 7. save all results to json ---
results = {
    "meta": {
        "student_id" : "310201050",
        "seed"       : SEED,
        "device"     : str(DEVICE),
        "dataset"    : "IMDb",
        "splits"     : {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE},
    },
    "tokenization_strategies": {
        "strategy_A": "Whitespace split + lowercase + HTML removal + stopword removal  → used by TF-IDF",
        "strategy_B": "NLTK word_tokenize + lowercase + HTML removal (no stopword removal) → used by BiLSTM",
        "strategy_C": "WordPiece subword tokenization via DistilBertTokenizerFast → used by DistilBERT",
    },
    "models": {
        "tfidf_lr": {
            "val_accuracy" : round(val_acc_lr,   4),
            "val_macro_f1" : round(val_f1_lr,    4),
            "test_accuracy": round(test_acc_lr,  4),
            "test_macro_f1": round(test_f1_lr,   4),
            "hyperparams"  : {
                "max_features": 30000, "ngram_range": "(1,2)",
                "sublinear_tf": True,  "C": 0.7,
            },
        },
        "bilstm": {
            "test_accuracy": round(test_acc_lstm, 4),
            "test_macro_f1": round(test_f1_lstm,  4),
            "hyperparams"  : {
                "vocab_size": MAX_VOCAB, "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM, "num_layers": LSTM_LAYERS,
                "dropout": DROPOUT_LSTM, "max_len": MAX_LEN_LSTM,
                "batch_size": BATCH_LSTM, "epochs": EPOCHS_LSTM, "lr": LR_LSTM,
            },
        },
        "distilbert": {
            "test_accuracy": round(test_acc_bert, 4),
            "test_macro_f1": round(test_f1_bert,  4),
            "hyperparams"  : {
                "model"      : "distilbert-base-uncased",
                "max_len"    : BERT_MAX_LEN,
                "batch_size" : BERT_BATCH,
                "epochs"     : BERT_EPOCHS,
                "lr"         : BERT_LR,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": 0.01,
            },
        },
    },
    "misclassified_examples": misclassified[:5],
}

out_path = os.path.join(OUT_DIR, "q1_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# --- 8. final summary table ---
print(f"\n{'='*55}")
print(f"  FINAL RESULTS  (Test Set, n={TEST_SIZE})")
print(f"{'='*55}")
print(f"  {'Model':<22} {'Accuracy':>10}  {'Macro-F1':>10}")
print(f"  {'-'*50}")
print(f"  {'TF-IDF + LR':<22} {test_acc_lr:>10.4f}  {test_f1_lr:>10.4f}")
print(f"  {'BiLSTM':<22} {test_acc_lstm:>10.4f}  {test_f1_lstm:>10.4f}")
print(f"  {'DistilBERT':<22} {test_acc_bert:>10.4f}  {test_f1_bert:>10.4f}")
print(f"{'='*55}")
print(f"\n  Results saved → {out_path}")