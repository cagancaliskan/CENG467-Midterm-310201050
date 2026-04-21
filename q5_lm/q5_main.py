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
