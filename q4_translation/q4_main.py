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

