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
