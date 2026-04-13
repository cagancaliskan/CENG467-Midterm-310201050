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

