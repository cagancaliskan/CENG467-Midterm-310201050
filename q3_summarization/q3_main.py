#!/usr/bin/env python3
"""
CENG 467 – Natural Language Understanding and Generation
Take-Home Midterm – Question 3: Text Summarization
Student ID : 310201050
Seed       : 3102
Device     : Apple M2 (MPS)

Extractive  : LexRank (sumy)
Abstractive : DistilBART fine-tuned on CNN/DailyMail
"""

import os, json, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from datasets import load_dataset

# ── extractive (sumy) ────────────────────────────────────────────────────────
from sumy.parsers.plaintext  import PlaintextParser
from sumy.nlp.tokenizers     import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers       import Stemmer
from sumy.utils              import get_stop_words

# ── abstractive (transformers) ────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── metrics ───────────────────────────────────────────────────────────────────
from rouge_score import rouge_scorer  # these metrics are so slow to calculate
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)
from nltk.translate.bleu_score   import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_fn

# --- 0. reproducibility & device ---
SEED = 3102
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)

print(f"\n{'='*60}")
print(f"  CENG 467 – Question 3: Text Summarization")
print(f"  Student ID : 310201050  |  Seed : {SEED}")
print(f"  Device     : {DEVICE}")
print(f"{'='*60}\n")

OUT_DIR = "q3_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. data loading - cnn/dailymail v3.0.0 ---
N_SAMPLES = 200   # reduce to 100 if memory pressure is observed

print(f"[STEP 1] Loading CNN/DailyMail (test split, {N_SAMPLES} examples) …")
ds = load_dataset(
    "cnn_dailymail", "3.0.0",
    split=f"test[:{N_SAMPLES}]",
    trust_remote_code=True,
)

articles    = [ex["article"]    for ex in ds]
references  = [ex["highlights"] for ex in ds]
print(f"  Loaded {len(articles)} article–summary pairs.")
avg_art_len = int(np.mean([len(a.split()) for a in articles]))
avg_ref_len = int(np.mean([len(r.split()) for r in references]))
print(f"  Avg. article length: {avg_art_len} words   |   Avg. reference: {avg_ref_len} words\n")

# --- 2. extractive - lexrank ---
print("[MODEL 1] LexRank (graph-based extractive)")
NUM_SENTENCES_LEXRANK = 3   # CNN/DailyMail references avg. ~3 sentences

stemmer    = Stemmer("english")
summarizer = LexRankSummarizer(stemmer)
summarizer.stop_words = get_stop_words("english")

def lexrank_summary(text: str, k: int = NUM_SENTENCES_LEXRANK) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    return " ".join(str(s) for s in summarizer(parser.document, k))

t0 = time.time()
extractive_summaries = []
for i, art in enumerate(articles):
    extractive_summaries.append(lexrank_summary(art))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_SAMPLES} processed")
print(f"  Completed in {time.time()-t0:.1f}s\n")

# --- 3. abstractive - distilbart (cnn/dailymail-fine-tuned) ---
print("[MODEL 2] DistilBART (sshleifer/distilbart-cnn-12-6)")
BART_NAME      = "sshleifer/distilbart-cnn-12-6"
MAX_INPUT_LEN  = 1024
MAX_GEN_LEN    = 130
MIN_GEN_LEN    = 30
NUM_BEAMS      = 4
LEN_PENALTY    = 2.0
NO_REPEAT_NG   = 3

bart_tok   = AutoTokenizer.from_pretrained(BART_NAME)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_NAME).to(DEVICE)
bart_model.eval()

def bart_summary(text: str) -> str:
    inputs = bart_tok(
        text, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        ids = bart_model.generate(
            **inputs,
            max_length          = MAX_GEN_LEN,
            min_length          = MIN_GEN_LEN,
            num_beams           = NUM_BEAMS,
            length_penalty      = LEN_PENALTY,
            no_repeat_ngram_size= NO_REPEAT_NG,
            early_stopping      = True,
        )
    return bart_tok.decode(ids[0], skip_special_tokens=True)

t0 = time.time()
abstractive_summaries = []
for i, art in enumerate(articles):
    abstractive_summaries.append(bart_summary(art))
    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (N_SAMPLES - i - 1)
        print(f"  {i+1}/{N_SAMPLES} processed  (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
print(f"  Completed in {time.time()-t0:.1f}s\n")

# --- 4. metrics ---
print("[STEP 4] Computing metrics …")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def rouge_avg(refs, hyps):
    sums = {"rouge1": [], "rouge2": [], "rougeL": []}
    for r, h in zip(refs, hyps):
