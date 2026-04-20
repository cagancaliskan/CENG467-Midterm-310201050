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

# bi-directional gru to encode the german source text
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm    = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc_h    = nn.Linear(hidden * 2, hidden)
        self.fc_c    = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lens):
        emb = self.dropout(self.embed(x))
        packed = pack_padded_sequence(emb, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        out_p, (h, c) = self.lstm(packed)
        out, _ = pad_packed_sequence(out_p, batch_first=True, padding_value=0.0)
        # Combine forward + backward final states
        h_cat = torch.cat([h[0], h[1]], dim=1)
        c_cat = torch.cat([c[0], c[1]], dim=1)
        h_init = torch.tanh(self.fc_h(h_cat)).unsqueeze(0)
        c_init = torch.tanh(self.fc_c(c_cat)).unsqueeze(0)
        return out, (h_init, c_init)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W = nn.Linear(enc_hidden + dec_hidden, dec_hidden)
        self.v = nn.Linear(dec_hidden, 1, bias=False)
    def forward(self, dec_h, enc_outs, mask):
        # dec_h: (B, dec_hidden)   enc_outs: (B, T, enc_hidden)
        T = enc_outs.size(1)
        h_exp  = dec_h.unsqueeze(1).expand(-1, T, -1)
        energy = torch.tanh(self.W(torch.cat([h_exp, enc_outs], dim=2)))
        scores = self.v(energy).squeeze(2).masked_fill(~mask, -1e9)
        return F.softmax(scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, enc_hidden, dropout):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(enc_hidden, hidden)
        self.lstm      = nn.LSTM(embed_dim + enc_hidden, hidden, batch_first=True)
        self.fc_out    = nn.Linear(hidden + enc_hidden + embed_dim, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def step(self, x, hidden, enc_outs, mask):
        emb     = self.dropout(self.embed(x).unsqueeze(1))
        h_state = hidden[0].squeeze(0)
        attn    = self.attention(h_state, enc_outs, mask)
        ctx     = torch.bmm(attn.unsqueeze(1), enc_outs)
        out, hidden = self.lstm(torch.cat([emb, ctx], dim=2), hidden)
        combined    = torch.cat([out.squeeze(1), ctx.squeeze(1), emb.squeeze(1)], dim=1)
        return self.fc_out(combined), hidden, attn

# finally got the seq2seq wiring right
class Seq2Seq(nn.Module):
    def __init__(self, src_vsize, tgt_vsize, embed_dim, hidden, dropout):
        super().__init__()
        self.encoder    = Encoder(src_vsize, embed_dim, hidden, dropout)
        self.decoder    = Decoder(tgt_vsize, embed_dim, hidden, hidden * 2, dropout)
        self.tgt_vsize  = tgt_vsize

    def _src_mask(self, src, enc_len):
        mask = (src != PAD_IDX)
        if mask.size(1) > enc_len:
            mask = mask[:, :enc_len]
        elif mask.size(1) < enc_len:
            mask = F.pad(mask, (0, enc_len - mask.size(1)), value=False)
        return mask

    def forward(self, src, src_lens, tgt, teacher_forcing=0.5):
        B, T = tgt.shape
        enc_outs, hidden = self.encoder(src, src_lens)
        mask = self._src_mask(src, enc_outs.size(1))
        outs = torch.zeros(B, T, self.tgt_vsize, device=src.device)
        x = tgt[:, 0]
        for t in range(1, T):
            logits, hidden, _ = self.decoder.step(x, hidden, enc_outs, mask)
            outs[:, t] = logits
            x = tgt[:, t] if random.random() < teacher_forcing else logits.argmax(1)
        return outs

    @torch.no_grad()
    def greedy_generate(self, src, src_lens, max_len=50):
        self.eval()
        B = src.size(0)
        enc_outs, hidden = self.encoder(src, src_lens)
        mask = self._src_mask(src, enc_outs.size(1))
        x = torch.full((B,), SOS_IDX, dtype=torch.long, device=src.device)
        outs = []
        for _ in range(max_len):
            logits, hidden, _ = self.decoder.step(x, hidden, enc_outs, mask)
            x = logits.argmax(1)
            outs.append(x.cpu())
        return torch.stack(outs, dim=1)

# Build & summarise
seq2seq = Seq2Seq(len(src_vocab), len(tgt_vocab), EMBED_DIM, HIDDEN, DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in seq2seq.parameters())
print(f"[MODEL A] Seq2Seq + attention   ({n_params/1e6:.2f}M parameters)")

opt        = torch.optim.Adam(seq2seq.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
criterion  = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

t0 = time.time()
for ep in range(EPOCHS):
    seq2seq.train(); tr_loss = 0
    for src, tgt, src_lens in dl_tr:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        opt.zero_grad()
        logits = seq2seq(src, src_lens, tgt, TF_RATIO)
        loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)),
                         tgt   [:, 1:].reshape(-1))
                # calculate gradients and update weights
loss.backward()
        nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item()

    seq2seq.eval(); vl_loss = 0
    with torch.no_grad():
        for src, tgt, src_lens in dl_vl:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = seq2seq(src, src_lens, tgt, 0.0)
            vl_loss += criterion(
                logits[:, 1:].reshape(-1, logits.size(-1)),
                tgt   [:, 1:].reshape(-1)
            ).item()

    tr_avg, vl_avg = tr_loss/len(dl_tr), vl_loss/len(dl_vl)
    print(f"  Epoch {ep+1}/{EPOCHS}  Train: {tr_avg:.4f}  Val: {vl_avg:.4f}  ValPPL: {np.exp(vl_avg):.2f}")
    scheduler.step()
print(f"  Training time: {time.time()-t0:.1f}s\n")

# Generate test predictions
def ids_to_text(ids):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_IDX: break
        if i in (PAD_IDX, SOS_IDX): continue
        out.append(tgt_itos.get(i, UNK))
    return " ".join(out)

print("[STEP] Decoding test set with Seq2Seq (greedy) …")
seq2seq_preds = []
for src, _, src_lens in dl_te:
    src = src.to(DEVICE)
    pred_ids = seq2seq.greedy_generate(src, src_lens, max_len=MAX_LEN)
    seq2seq_preds.extend(ids_to_text(row) for row in pred_ids)
print(f"  Generated {len(seq2seq_preds)} translations.\n")

# --- 5.  model b - marianmt (helsinki-nlp/opus-mt-de-en) ---
print("[MODEL B] MarianMT  (Helsinki-NLP/opus-mt-de-en)")
MARIAN_NAME = "Helsinki-NLP/opus-mt-de-en"
m_tok   = MarianTokenizer.from_pretrained(MARIAN_NAME)
m_model = MarianMTModel.from_pretrained(MARIAN_NAME).to(DEVICE)
m_model.eval()

t0 = time.time()
marian_preds = []
M_BATCH, M_BEAMS = 16, 4
for i in range(0, len(de_test), M_BATCH):
    batch = de_test[i : i + M_BATCH]
    enc   = m_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = m_model.generate(**enc, num_beams=M_BEAMS, max_length=128, early_stopping=True)
    marian_preds.extend(m_tok.decode(o, skip_special_tokens=True) for o in out)
    if (i // M_BATCH + 1) % 10 == 0:
        print(f"  {min(i+M_BATCH, len(de_test))}/{len(de_test)} done")
print(f"  Marian inference: {time.time()-t0:.1f}s\n")

# --- 6.  metrics  (bleu, chrf, meteor, bertscore) ---
print("[STEP 6] Computing metrics …")

def safe_text(s): return s if s.strip() else "<empty>"

def compute_metrics(hyps, refs, label):
    hyps = [safe_text(h) for h in hyps]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    meteor = float(np.mean([
        meteor_score([nltk.word_tokenize(r.lower())], nltk.word_tokenize(h.lower()))
        for h, r in zip(hyps, refs)
    ]))
    P, R, F = bert_score_fn(
        hyps, refs,
        lang        = "en",
        model_type  = "distilbert-base-uncased",
        verbose     = False,
        batch_size  = 16,
        device      = str(DEVICE),
    )
    bs = float(F.mean().item())
    print(f"  {label}: BLEU={bleu:.2f}  ChrF={chrf:.2f}  METEOR={meteor:.4f}  BERTScore={bs:.4f}")
    return {"bleu": round(bleu, 2),  "chrf":   round(chrf, 2),
            "meteor": round(meteor, 4), "bertscore_f1": round(bs, 4)}

seq2seq_metrics = compute_metrics(seq2seq_preds, en_test, "Seq2Seq+Attn")
marian_metrics  = compute_metrics(marian_preds,  en_test, "MarianMT    ")

# --- 7.  qualitative examples ---
print("\n[STEP 7] Qualitative examples")
qualitative = []
for idx in [0, 100, 500]:
    qualitative.append({
        "index"          : idx,
        "source_de"      : de_test[idx],
        "reference_en"   : en_test[idx],
        "seq2seq_output" : seq2seq_preds[idx],
        "marian_output"  : marian_preds[idx],
    })
    print(f"\n  [{idx}] DE: {de_test[idx]}")
    print(f"       REF   : {en_test[idx]}")
    print(f"       S2S   : {seq2seq_preds[idx]}")
    print(f"       Marian: {marian_preds[idx]}")

# --- 8.  save ---
results = {
    "meta": {
        "student_id": "310201050", "seed": SEED, "device": str(DEVICE),
        "dataset": "Multi30k (de→en)",
        "splits": {"train": len(de_train), "val": len(de_val), "test": len(de_test)},
    },
    "models": {
        "seq2seq_attention": {
            **seq2seq_metrics,
            "config": {
                "embed_dim": EMBED_DIM, "hidden": HIDDEN, "dropout": DROPOUT,
                "epochs": EPOCHS, "lr": LR, "batch_size": BATCH,
                "teacher_forcing": TF_RATIO,
                "src_vocab_size": len(src_vocab), "tgt_vocab_size": len(tgt_vocab),
                "n_parameters_M": round(n_params / 1e6, 2),
                "decoding": "greedy",
            },
        },
        "marian_pretrained": {
            **marian_metrics,
            "config": {"model": MARIAN_NAME, "num_beams": M_BEAMS, "max_length": 128},
        },
    },
    "qualitative_examples": qualitative,
}

with open(os.path.join(OUT_DIR, "q4_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# --- 9.  summary ---
print(f"\n{'='*55}")
print(f"  FINAL RESULTS – Multi30k Test (n={len(de_test)})")
print(f"{'='*55}")
print(f"  {'Metric':<16} {'Seq2Seq+Attn':>16} {'MarianMT':>14}")
print(f"  {'-'*48}")
print(f"  {'BLEU':<16} {seq2seq_metrics['bleu']:>16.2f} {marian_metrics['bleu']:>14.2f}")
print(f"  {'ChrF':<16} {seq2seq_metrics['chrf']:>16.2f} {marian_metrics['chrf']:>14.2f}")
print(f"  {'METEOR':<16} {seq2seq_metrics['meteor']:>16.4f} {marian_metrics['meteor']:>14.4f}")
print(f"  {'BERTScore F1':<16} {seq2seq_metrics['bertscore_f1']:>16.4f} {marian_metrics['bertscore_f1']:>14.4f}")
print(f"{'='*55}")
print(f"\n  Results saved → {OUT_DIR}/q4_results.json")