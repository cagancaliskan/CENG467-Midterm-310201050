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
    lengths = torch.tensor([len(x) for x in xs])
    x_pad   = pad_sequence(xs, batch_first=True, padding_value=0)
    y_pad   = pad_sequence(ys, batch_first=True, padding_value=label2id["O"])
    max_len = x_pad.size(1)
    mask    = (torch.arange(max_len)[None, :] < lengths[:, None])
    return x_pad, y_pad, mask

ds_tr = CoNLLSeqDataset(raw["train"])
ds_vl = CoNLLSeqDataset(raw["validation"])
ds_te = CoNLLSeqDataset(raw["test"])

dl_tr = DataLoader(ds_tr, batch_size=BATCH_LSTM, shuffle=True,  collate_fn=collate_seq)
dl_vl = DataLoader(ds_vl, batch_size=BATCH_LSTM, shuffle=False, collate_fn=collate_seq)
dl_te = DataLoader(ds_te, batch_size=BATCH_LSTM, shuffle=False, collate_fn=collate_seq)

# using torchcrf for the crf layer to make life easier
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, num_tags, dropout=0.4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop  = nn.Dropout(dropout)
        self.lstm  = nn.LSTM(embed_dim, hidden, batch_first=True,
                             bidirectional=True, num_layers=1)
        self.fc    = nn.Linear(hidden * 2, num_tags)
        self.crf   = CRF(num_tags, batch_first=True)

    def emissions(self, x):
        e = self.drop(self.embed(x))
        h, _ = self.lstm(e)
        return self.fc(self.drop(h))

    def forward(self, x, mask, tags=None):
        emi = self.emissions(x)
        if tags is not None:
            # return the negative log likelihood for training
            return -self.crf(emi, tags, mask=mask, reduction="mean")
        # during eval, use viterbi decoding to get best path
        return self.crf.decode(emi, mask=mask)

model_lstm = BiLSTM_CRF(len(VOCAB_WORDS), EMBED_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT_LSTM).to(DEVICE)
opt_lstm   = torch.optim.Adam(model_lstm.parameters(), lr=LR_LSTM, weight_decay=1e-5)

def eval_bilstm_crf(model, dl):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb, mask in dl:
            xb   = xb.to(DEVICE)
            mb   = mask.to(DEVICE)
            preds = model(xb, mask=mb)
            for i, p in enumerate(preds):
                length = mask[i].sum().item()
                y_pred_all.append([id2label[t] for t in p[:length]])
                y_true_all.append([id2label[t] for t in yb[i, :length].tolist()])
    return y_true_all, y_pred_all

t0 = time.time()
for ep in range(EPOCHS_LSTM):
    model_lstm.train(); total_loss = 0
    for xb, yb, mask in dl_tr:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mask.to(DEVICE)
        opt_lstm.zero_grad()
        loss = model_lstm(xb, mask=mb, tags=yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), GRAD_CLIP)
        opt_lstm.step()
        total_loss += loss.item()
    yt, yp = eval_bilstm_crf(model_lstm, dl_vl)
    f1_dev = f1_score(yt, yp)
    print(f"  Epoch {ep+1}/{EPOCHS_LSTM} | Loss: {total_loss/len(dl_tr):.4f}  Val-F1: {f1_dev:.4f}")

yt_lstm, yp_lstm = eval_bilstm_crf(model_lstm, dl_te)
prec_lstm  = precision_score(yt_lstm, yp_lstm)
rec_lstm   = recall_score(yt_lstm, yp_lstm)
f1_lstm    = f1_score(yt_lstm, yp_lstm)
print(f"  Test → P: {prec_lstm:.4f}  R: {rec_lstm:.4f}  F1: {f1_lstm:.4f}  ({time.time()-t0:.1f}s)\n")
report_lstm = classification_report(yt_lstm, yp_lstm, digits=4)

# --- 4.  model 2 - distilbert for token classification ---
print("[MODEL 2] DistilBERT for Token Classification")

BERT_MAX_LEN = 128
BERT_BATCH   = 16
BERT_EPOCHS  = 3
BERT_LR      = 3e-5
WARMUP_RATIO = 0.1

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

def tokenise_and_align(token_lists, tag_lists):
    enc = tokenizer(
        token_lists,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=BERT_MAX_LEN,
        return_tensors="pt",
    )
    aligned = []
    for i, tag_seq in enumerate(tag_lists):
        wids   = enc.word_ids(batch_index=i)
        prev   = None
        labs   = []
        for wid in wids:
            if wid is None:
                labs.append(-100)
            elif wid != prev:
                labs.append(label2id.get(tag_seq[wid], label2id["O"]))
            else:
                labs.append(-100)
            prev = wid
        aligned.append(labs)
    enc["labels"] = torch.tensor(aligned, dtype=torch.long)
    return enc

class BertNERDataset(Dataset):
    def __init__(self, sentences):
        token_lists = [s["tokens"]   for s in sentences]
        tag_lists   = [s["ner_tags"] for s in sentences]
        self.enc    = tokenise_and_align(token_lists, tag_lists)
    def __len__(self): return self.enc["input_ids"].size(0)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}

bd_tr = DataLoader(BertNERDataset(raw["train"]),      batch_size=BERT_BATCH, shuffle=True)
bd_vl = DataLoader(BertNERDataset(raw["validation"]), batch_size=BERT_BATCH)
bd_te = DataLoader(BertNERDataset(raw["test"]),       batch_size=BERT_BATCH)

bert = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
).to(DEVICE)

opt_bert  = torch.optim.AdamW(bert.parameters(), lr=BERT_LR, weight_decay=0.01)
total_st  = len(bd_tr) * BERT_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    opt_bert,
    num_warmup_steps  = int(total_st * WARMUP_RATIO),
    num_training_steps= total_st,
)

def eval_bert_ner(model, dl):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dl:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]
            logits = model(**inputs).logits.cpu()
            preds  = logits.argmax(-1)
            for p_seq, l_seq in zip(preds, labels):
                t_lst, p_lst = [], []
                for p_t, l_t in zip(p_seq.tolist(), l_seq.tolist()):
                    if l_t == -100: continue
                    t_lst.append(id2label[l_t])
                    p_lst.append(id2label[p_t])
                y_true.append(t_lst)
                y_pred.append(p_lst)
    return y_true, y_pred

t0 = time.time()
for ep in range(BERT_EPOCHS):
    bert.train(); total_loss = 0
    for batch in bd_tr:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        opt_bert.zero_grad()
        out = bert(**batch)
        out.loss.backward()
        nn.utils.clip_grad_norm_(bert.parameters(), GRAD_CLIP)
        opt_bert.step(); scheduler.step()
        total_loss += out.loss.item()
    yt, yp = eval_bert_ner(bert, bd_vl)
    f1_dev = f1_score(yt, yp)
    print(f"  Epoch {ep+1}/{BERT_EPOCHS} | Loss: {total_loss/len(bd_tr):.4f}  Val-F1: {f1_dev:.4f}")

yt_bert, yp_bert = eval_bert_ner(bert, bd_te)
prec_bert = precision_score(yt_bert, yp_bert)
rec_bert  = recall_score(yt_bert,    yp_bert)
f1_bert   = f1_score(yt_bert,        yp_bert)
print(f"  Test → P: {prec_bert:.4f}  R: {rec_bert:.4f}  F1: {f1_bert:.4f}  ({time.time()-t0:.1f}s)\n")
report_bert = classification_report(yt_bert, yp_bert, digits=4)

# --- 5.  error analysis  -  boundary errors, type confusion, fp, fn ---
print("[STEP 5] Error Analysis (DistilBERT predictions on test set)")

def extract_entities(tags):
    spans, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            start = i; i += 1
            while i < len(tags) and tags[i] == f"I-{etype}":
                i += 1
            spans.append((etype, start, i - 1))
        else:
            i += 1
    return spans

err_counts = defaultdict(int)
boundary_examples, type_examples, fp_examples, fn_examples = [], [], [], []

for sent_idx, (gold, pred) in enumerate(zip(yt_bert, yp_bert)):
    gold_spans = set(extract_entities(gold))
    pred_spans = set(extract_entities(pred))

    matched_gold = set()
    matched_pred = set()
    for g in gold_spans:
        for p in pred_spans:
            if g == p:
                err_counts["true_positive"] += 1
                matched_gold.add(g); matched_pred.add(p)
                break

    for g in gold_spans - matched_gold:
        boundary_hit = False
        for p in pred_spans - matched_pred:
            if g[0] == p[0] and not (g[2] < p[1] or p[2] < g[1]):
                err_counts["boundary"] += 1
                if len(boundary_examples) < 3:
                    boundary_examples.append({"sent_idx": sent_idx, "gold": g, "pred": p})
                matched_pred.add(p); boundary_hit = True; break
        if boundary_hit: continue
        type_hit = False
        for p in pred_spans - matched_pred:
            if g[1] == p[1] and g[2] == p[2] and g[0] != p[0]:
                err_counts["type_confusion"] += 1
                if len(type_examples) < 3:
                    type_examples.append({"sent_idx": sent_idx, "gold": g, "pred": p})
                matched_pred.add(p); type_hit = True; break
        if type_hit: continue
        err_counts["false_negative"] += 1
        if len(fn_examples) < 3:
            fn_examples.append({"sent_idx": sent_idx, "gold": g})

    for p in pred_spans - matched_pred:
        err_counts["false_positive"] += 1
        if len(fp_examples) < 3:
            fp_examples.append({"sent_idx": sent_idx, "pred": p})

print(f"  True Positives  : {err_counts['true_positive']}")
print(f"  Boundary errors : {err_counts['boundary']}")
print(f"  Type confusion  : {err_counts['type_confusion']}")
print(f"  False positives : {err_counts['false_positive']}")
print(f"  False negatives : {err_counts['false_negative']}\n")

# --- 6.  save results ---
results = {
    "meta": {
        "student_id": "310201050", "seed": SEED, "device": str(DEVICE),
        "dataset": "CoNLL-2003 (manual download, BIO-normalised)",
        "num_labels": NUM_LABELS, "labels": LABELS,
    },
    "models": {
        "bilstm_crf": {
            "test_precision": round(prec_lstm, 4),
            "test_recall"   : round(rec_lstm,  4),
            "test_f1"       : round(f1_lstm,   4),
            "report"        : report_lstm,
            "hyperparams"   : {
                "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT_LSTM, "batch": BATCH_LSTM,
                "epochs": EPOCHS_LSTM, "lr": LR_LSTM,
                "vocab_size": len(VOCAB_WORDS),
            },
        },
        "distilbert": {
            "test_precision": round(prec_bert, 4),
            "test_recall"   : round(rec_bert,  4),
            "test_f1"       : round(f1_bert,   4),
            "report"        : report_bert,
            "hyperparams"   : {
                "model"        : "distilbert-base-cased",
                "max_len"      : BERT_MAX_LEN,
                "batch"        : BERT_BATCH,
                "epochs"       : BERT_EPOCHS,
                "lr"           : BERT_LR,
                "warmup_ratio" : WARMUP_RATIO,
                "weight_decay" : 0.01,
            },
        },
    },
    "error_analysis": {
        "counts" : dict(err_counts),
        "examples": {
            "boundary"      : boundary_examples,
            "type_confusion": type_examples,
            "false_positive": fp_examples,
            "false_negative": fn_examples,
        },
    },
}

with open(os.path.join(OUT_DIR, "q2_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# --- 7.  final summary table ---
print(f"\n{'='*60}")
print(f"  FINAL RESULTS – CoNLL-2003 Test Set")
print(f"{'='*60}")
print(f"  {'Model':<22} {'Precision':>11} {'Recall':>9} {'F1':>9}")
print(f"  {'-'*55}")
print(f"  {'BiLSTM-CRF':<22} {prec_lstm:>11.4f} {rec_lstm:>9.4f} {f1_lstm:>9.4f}")
print(f"  {'DistilBERT (cased)':<22} {prec_bert:>11.4f} {rec_bert:>9.4f} {f1_bert:>9.4f}")
print(f"{'='*60}")
print(f"\n  Results saved → {OUT_DIR}/q2_results.json")