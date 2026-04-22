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
BPTT_LEN    = 35
BATCH       = 20
EPOCHS      = 6
LR          = 1e-3
GRAD_CLIP   = 0.25

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, n_layers, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim)
        self.lstm    = nn.LSTM(embed_dim, hidden, num_layers=n_layers,
                               dropout=dropout, batch_first=True)
        self.fc      = nn.Linear(hidden, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden  = hidden
        self.layers  = n_layers

    def forward(self, x, hid=None):
        e        = self.dropout(self.embed(x))
        out, hid = self.lstm(e, hid)
        out      = self.dropout(out)
        return self.fc(out), hid

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.layers, batch_size, self.hidden, device=device),
                torch.zeros(self.layers, batch_size, self.hidden, device=device))

# ── batchify into BPTT layout ────────────────────────────────────────────────
def batchify(ids, bsz):
    nb   = len(ids) // bsz
    ids  = ids[: nb * bsz]
    return torch.tensor(ids, dtype=torch.long).view(bsz, -1)

train_data = batchify(train_ids, BATCH)
val_data   = batchify(val_ids,   BATCH)
test_data  = batchify(test_ids,  BATCH)
print(f"  BPTT layout → train: {tuple(train_data.shape)}  val: {tuple(val_data.shape)}  test: {tuple(test_data.shape)}")

def get_batch(source, i):
    sl = min(BPTT_LEN, source.size(1) - 1 - i)
    x  = source[:, i : i + sl]
    y  = source[:, i + 1 : i + 1 + sl]
    return x, y

model_lstm = LSTMLanguageModel(V, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
n_lstm_params = sum(p.numel() for p in model_lstm.parameters())
print(f"  Parameters: {n_lstm_params/1e6:.2f}M")

opt_lstm   = torch.optim.Adam(model_lstm.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.StepLR(opt_lstm, step_size=2, gamma=0.5)
criterion  = nn.CrossEntropyLoss()

def evaluate(model, data):
    model.eval()
    total_loss, n_tok = 0.0, 0
    hid = model.init_hidden(data.size(0), DEVICE)
    with torch.no_grad():
        for i in range(0, data.size(1) - 1, BPTT_LEN):
            x, y = get_batch(data, i)
            x, y = x.to(DEVICE), y.to(DEVICE)
            hid  = tuple(h.detach() for h in hid)
            logits, hid = model(x, hid)
            loss = criterion(logits.reshape(-1, V), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            n_tok      += y.numel()
    return total_loss / n_tok, math.exp(total_loss / n_tok)

t0 = time.time()
best_val = float("inf")
for ep in range(EPOCHS):
    model_lstm.train()
    total_loss, n_tok = 0.0, 0
    hid = model_lstm.init_hidden(BATCH, DEVICE)
    for i in range(0, train_data.size(1) - 1, BPTT_LEN):
        x, y = get_batch(train_data, i)
        x, y = x.to(DEVICE), y.to(DEVICE)
        hid  = tuple(h.detach() for h in hid)
        opt_lstm.zero_grad()
        logits, hid = model_lstm(x, hid)
        loss = criterion(logits.reshape(-1, V), y.reshape(-1))
                    # propagate loss through time (BPTT)
loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), GRAD_CLIP)
        opt_lstm.step()
        total_loss += loss.item() * y.numel()
        n_tok      += y.numel()

    train_ppl   = math.exp(total_loss / n_tok)
    val_loss, val_ppl = evaluate(model_lstm, val_data)
    print(f"  Epoch {ep+1}/{EPOCHS} | TrainPPL: {train_ppl:7.2f}  ValPPL: {val_ppl:7.2f}")
    scheduler.step()
    best_val = min(best_val, val_ppl)

test_loss, lstm_test_ppl = evaluate(model_lstm, test_data)
print(f"  Test PPL: {lstm_test_ppl:.2f}    ({time.time()-t0:.1f}s)\n")

# --- 5. text generation  (both models, three prompts) ---
print("[STEP 5] Generating text samples …")

PROMPTS = ["the city of", "in the year", "she said that"]
GEN_LEN = 25
TEMPERATURE = 0.9

def trigram_generate(prompt, max_len=GEN_LEN):
    out_ids = encode(prompt.split())
    if len(out_ids) < 2:
        out_ids = [UNK_ID, UNK_ID] + out_ids
    for _ in range(max_len):
        ctx = (out_ids[-2], out_ids[-1])
        nxt = trigram[ctx]
        if nxt:
            ids, cs = zip(*nxt.items())
            probs   = np.array(cs, dtype=float) / sum(cs)
            choice  = int(np.random.choice(ids, p=probs))
        else:
            choice = random.randint(0, V - 1)
        out_ids.append(choice)
    return " ".join(i2w[i] for i in out_ids)

def lstm_generate(prompt, max_len=GEN_LEN, temp=TEMPERATURE):
    model_lstm.eval()
    out_ids = encode(prompt.split())
    hid     = model_lstm.init_hidden(1, DEVICE)
    with torch.no_grad():
        # Prime the hidden state with the prompt
        for tok in out_ids[:-1]:
            x   = torch.tensor([[tok]], device=DEVICE)
            _, hid = model_lstm(x, hid)
        x = torch.tensor([[out_ids[-1]]], device=DEVICE)
        for _ in range(max_len):
            logits, hid = model_lstm(x, hid)
            probs = torch.softmax(logits[0, -1] / temp, dim=-1)
            nxt   = int(torch.multinomial(probs, 1).item())
            out_ids.append(nxt)
            x = torch.tensor([[nxt]], device=DEVICE)
    return " ".join(i2w[i] for i in out_ids)

samples = []
for p in PROMPTS:
    tri_out  = trigram_generate(p)
    lstm_out = lstm_generate(p)
    samples.append({"prompt": p, "trigram": tri_out, "lstm": lstm_out})
    print(f"\n  PROMPT     : {p}")
    print(f"  Trigram    : {tri_out}")
    print(f"  LSTM       : {lstm_out}")

# --- 6. save results ---
results = {
    "meta": {
        "student_id": "310201050", "seed": SEED, "device": str(DEVICE),
        "dataset": "WikiText-2 (v1)",
        "vocab_size": V,
        "token_counts": {
            "train": len(train_tokens),
            "val"  : len(val_tokens),
            "test" : len(test_tokens),
        },
    },
    "models": {
        "trigram_addk": {
            "test_perplexity"     : round(trigram_test_ppl, 2),
            "validation_perplexity": round(trigram_val_ppl, 2),
            "config": {
                "n"          : 3,
                "smoothing"  : "add-k",
                "k"          : K_SMOOTH,
                "vocab_size" : V,
            },
        },
        "lstm": {
            "test_perplexity"     : round(lstm_test_ppl, 2),
            "best_validation_ppl" : round(best_val, 2),
            "config": {
                "embed_dim"  : EMBED_DIM,
                "hidden_dim" : HIDDEN_DIM,
                "num_layers" : NUM_LAYERS,
                "dropout"    : DROPOUT,
                "bptt_len"   : BPTT_LEN,
                "batch"      : BATCH,
                "epochs"     : EPOCHS,
                "lr"         : LR,
                "grad_clip"  : GRAD_CLIP,
                "n_parameters_M": round(n_lstm_params / 1e6, 2),
            },
        },
    },
    "text_generation": {
        "config"  : {"max_len": GEN_LEN, "temperature": TEMPERATURE},
        "samples" : samples,
    },
}

out_path = os.path.join(OUT_DIR, "q5_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# --- 7. summary ---
print(f"\n{'='*55}")
print(f"  FINAL RESULTS – WikiText-2 Test Perplexity")
print(f"{'='*55}")
print(f"  {'Model':<28} {'Val PPL':>10} {'Test PPL':>10}")
print(f"  {'-'*50}")
print(f"  {'Trigram (add-k=0.01)':<28} {trigram_val_ppl:>10.2f} {trigram_test_ppl:>10.2f}")
print(f"  {'2-layer LSTM':<28} {best_val:>10.2f} {lstm_test_ppl:>10.2f}")
print(f"{'='*55}")
print(f"\n  Results saved → {out_path}")