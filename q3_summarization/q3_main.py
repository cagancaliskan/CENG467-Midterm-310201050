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

