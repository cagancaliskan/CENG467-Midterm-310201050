# CENG 467 — Take-Home Midterm Examination

**Course:** Natural Language Understanding and Generation
**Instructor:** Prof. Dr. Aytuğ Onan
**Student ID:** 310201050
**Date:** April 2026

---

## Overview

This repository contains the complete code, results, and report for the
CENG 467 take-home midterm examination, covering five core NLP tasks:

| #  | Task                          | Models compared                          | Dataset            |
|----|-------------------------------|------------------------------------------|--------------------|
| Q1 | Text Classification           | TF-IDF + LR, BiLSTM, DistilBERT          | IMDb               |
| Q2 | Named Entity Recognition      | BiLSTM-CRF, DistilBERT (cased)           | CoNLL-2003         |
| Q3 | Text Summarization            | LexRank (extractive), DistilBART (abstractive) | CNN/DailyMail v3.0.0 |
| Q4 | Machine Translation (de→en)   | Seq2Seq + Bahdanau attention, MarianMT   | Multi30k           |
| Q5 | Language Modeling             | Trigram (add-k), 2-layer LSTM            | WikiText-2         |

All experiments are reproducible from a single random seed (`3102`,
derived from the student ID).

---

## Repository Structure

```
CENG467_Midterm_310201050/
├── README.md                     # this file
├── requirements.txt              # pinned Python dependencies
├── run_all.sh                    # convenience runner
│
├── q1_classification/
│   ├── q1_main.py
│   ├── q1_section.tex
│   └── q1_results/q1_results.json     ← generated after running
│
├── q2_ner/
│   ├── q2_main.py
│   ├── q2_section.tex
│   └── q2_results/q2_results.json     ← generated after running
│
├── q3_summarization/
│   ├── q3_main.py
│   ├── q3_section.tex
│   └── q3_results/q3_results.json     ← generated after running
│
├── q4_translation/
│   ├── q4_main.py
│   ├── q4_section.tex
│   └── q4_results/q4_results.json     ← generated after running
│
├── q5_lm/
│   ├── q5_main.py
│   ├── q5_section.tex
│   └── q5_results/q5_results.json     ← generated after running
│
└── report/
    ├── main.tex                  # top-level LaTeX document
    ├── references.bib            # BibTeX bibliography
    └── (compiled output → main.pdf)
```

---

## Reproducing the Results

### 1. Environment

The experiments were run on an **Apple MacBook Air (M2, 8 GB RAM)** with
Python 3.10+. PyTorch is configured to use the Metal Performance
Shaders (MPS) backend for hardware acceleration; the code automatically
falls back to CPU on unsupported devices.

Set up the environment:

```bash
python3 -m venv ceng467_env
source ceng467_env/bin/activate
pip install -r requirements.txt
```

### 2. Running each question

Each question is fully self-contained:

```bash
# Question 1 — Classification
cd q1_classification && python q1_main.py && cd ..

# Question 2 — NER
cd q2_ner && python q2_main.py && cd ..

# Question 3 — Summarization (~25–35 min)
cd q3_summarization && python q3_main.py && cd ..

# Question 4 — Machine Translation (~15–20 min)
cd q4_translation && python q4_main.py && cd ..

# Question 5 — Language Modeling (~10–15 min)
cd q5_lm && python q5_main.py && cd ..
```

Or run everything sequentially:

```bash
bash run_all.sh
```

### 3. Compiling the report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This produces `main.pdf`. Before compilation, populate the placeholder
dashes (`--`) in each section's results table from the corresponding
`*_results.json` file.

---

## Reproducibility Notes

- **Random seed:** `3102` (set for `random`, `numpy`, and `torch`).
- **Device:** Apple MPS auto-selected; falls back to CPU if MPS is
  unavailable.
- **Test sets** are evaluated **exactly once** per question.
- **Hyperparameters** are listed in each question's `*_section.tex`
  and embedded in the corresponding `*_results.json`.

Re-running a script on the same machine with the same seed should
reproduce the reported numbers up to non-deterministic GPU/MPS
operations (typically below the third decimal place).

---

## Notes on Hardware Constraints

Several model choices reflect the 8 GB memory budget of the M2:

- DistilBERT instead of BERT-base (Q1, Q2)
- DistilBART instead of BART-large (Q3)
- DistilBERT backbone for BERTScore instead of the default
  RoBERTa-large (Q3, Q4)
- Pretrained MarianMT instead of training a Transformer from scratch
  on Multi30k (Q4)

These substitutions are noted and justified in the corresponding
sections of the report.

---

## License & Academic Integrity

All work is original and prepared individually for the CENG 467
take-home midterm.  External resources are cited in the report's
bibliography (`report/references.bib`).