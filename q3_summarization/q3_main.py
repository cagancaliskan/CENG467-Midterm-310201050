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
        s = scorer.score(r, h)
        for k in sums:
            sums[k].append(s[k].fmeasure)
    return {k: float(np.mean(v)) for k, v in sums.items()}

def bleu_avg(refs, hyps):
    smooth = SmoothingFunction().method4
    out = []
    for r, h in zip(refs, hyps):
        r_tok = nltk.word_tokenize(r.lower())
        h_tok = nltk.word_tokenize(h.lower())
        if not h_tok:
            out.append(0.0); continue
        out.append(sentence_bleu([r_tok], h_tok, smoothing_function=smooth))
    return float(np.mean(out))

def meteor_avg(refs, hyps):
    out = []
    for r, h in zip(refs, hyps):
        r_tok = nltk.word_tokenize(r.lower())
        h_tok = nltk.word_tokenize(h.lower())
        out.append(meteor_score([r_tok], h_tok))
    return float(np.mean(out))

def bertscore_avg(refs, hyps):
    # Use DistilBERT scorer to keep memory footprint low on the M2
    P, R, F = bert_score_fn(
        hyps, refs,
        lang        = "en",
        model_type  = "distilbert-base-uncased",
        verbose     = False,
        batch_size  = 8,
        device      = str(DEVICE),
    )
    return float(F.mean().item())

print("  → LexRank metrics …")
rouge_lex   = rouge_avg(references, extractive_summaries)
bleu_lex    = bleu_avg(references, extractive_summaries)
meteor_lex  = meteor_avg(references, extractive_summaries)
bert_lex    = bertscore_avg(references, extractive_summaries)

print("  → DistilBART metrics …")
rouge_bart  = rouge_avg(references, abstractive_summaries)
bleu_bart   = bleu_avg(references, abstractive_summaries)
meteor_bart = meteor_avg(references, abstractive_summaries)
bert_bart   = bertscore_avg(references, abstractive_summaries)

# --- 5. qualitative examples ---
sample_idx = [0, 50, 100]
qualitative = []
for i in sample_idx:
    qualitative.append({
        "index"            : i,
        "article_preview"  : articles[i][:600] + " …",
        "reference"        : references[i],
        "lexrank_output"   : extractive_summaries[i],
        "distilbart_output": abstractive_summaries[i],
    })

# --- 6. save results ---
results = {
    "meta": {
        "student_id" : "310201050",
        "seed"       : SEED,
        "device"     : str(DEVICE),
        "dataset"    : "CNN/DailyMail v3.0.0",
        "n_samples"  : N_SAMPLES,
        "avg_article_words"  : avg_art_len,
        "avg_reference_words": avg_ref_len,
    },
    "models": {
        "lexrank": {
            "rouge1"      : round(rouge_lex["rouge1"], 4),
            "rouge2"      : round(rouge_lex["rouge2"], 4),
            "rougeL"      : round(rouge_lex["rougeL"], 4),
            "bleu"        : round(bleu_lex,  4),
            "meteor"      : round(meteor_lex, 4),
            "bertscore_f1": round(bert_lex,  4),
            "config"      : {"num_sentences": NUM_SENTENCES_LEXRANK},
        },
        "distilbart": {
            "rouge1"      : round(rouge_bart["rouge1"], 4),
            "rouge2"      : round(rouge_bart["rouge2"], 4),
            "rougeL"      : round(rouge_bart["rougeL"], 4),
            "bleu"        : round(bleu_bart,  4),
            "meteor"      : round(meteor_bart, 4),
            "bertscore_f1": round(bert_bart,  4),
            "config": {
                "model"            : BART_NAME,
                "max_input_len"    : MAX_INPUT_LEN,
                "max_gen_len"      : MAX_GEN_LEN,
                "min_gen_len"      : MIN_GEN_LEN,
                "num_beams"        : NUM_BEAMS,
                "length_penalty"   : LEN_PENALTY,
                "no_repeat_ngram"  : NO_REPEAT_NG,
            },
        },
    },
    "qualitative_examples": qualitative,
}

out_path = os.path.join(OUT_DIR, "q3_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# --- 7. summary table ---
print(f"\n{'='*55}")
print(f"  FINAL RESULTS – CNN/DailyMail (n={N_SAMPLES})")
print(f"{'='*55}")
print(f"  {'Metric':<14} {'LexRank':>14} {'DistilBART':>14}")
print(f"  {'-'*48}")
print(f"  {'ROUGE-1':<14} {rouge_lex['rouge1']:>14.4f} {rouge_bart['rouge1']:>14.4f}")
print(f"  {'ROUGE-2':<14} {rouge_lex['rouge2']:>14.4f} {rouge_bart['rouge2']:>14.4f}")
print(f"  {'ROUGE-L':<14} {rouge_lex['rougeL']:>14.4f} {rouge_bart['rougeL']:>14.4f}")
print(f"  {'BLEU':<14} {bleu_lex:>14.4f} {bleu_bart:>14.4f}")
print(f"  {'METEOR':<14} {meteor_lex:>14.4f} {meteor_bart:>14.4f}")
print(f"  {'BERTScore F1':<14} {bert_lex:>14.4f} {bert_bart:>14.4f}")
print(f"{'='*55}")
print(f"\n  Results saved → {out_path}")