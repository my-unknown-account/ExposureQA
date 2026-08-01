<div align="center">

# *Analyzing Factual Recall in Large Language Models through Relation-Aware Pretraining Support*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-2EA44F)]()
[![EMNLP](https://img.shields.io/badge/EMNLP-2026-C0392B)]()

**A benchmark and analysis framework for studying how semantic pretraining evidence shapes factual recall, confidence, calibration, and hallucinations in LLMs.**

`Recall` · `Confidence` · `Calibration` · `Hallucination Analysis`

</div>

## ✨ Overview

Large Language Models (LLMs) often appear to “know” facts — but *why* do they reliably recall some facts while hallucinating others?

ExposureQA introduces **Relation-Aware Support**, a semantic measure of whether pretraining passages actually support a factual relation, moving beyond:
- lexical overlap
- entity co-occurrence
- popularity statistics

Unlike prior work, ExposureQA jointly analyzes:
- 📚 factual recall
- 🎯 confidence
- 📉 calibration
- ⚠️ hallucination behavior

across multiple **fully open LLM ecosystems** and their corresponding pretraining corpora.


## 🔥 Key Findings

- ✅ Relation-Aware Support predicts factual recall substantially better than lexical baselines
- ✅ Confidence strongly depends on semantic supporting evidence
- ✅ Sparse-support regimes are substantially more hallucination-prone and overconfident
- ✅ Calibration consistently improves when stronger semantic evidence exists in pretraining data


## 🧠 What is Relation-Aware Support?

Traditional exposure proxies often fail to capture whether a retrieved passage actually expresses a target fact.

| Method | Limitation |
|---|---|
| Lexical S-O overlap | Subject and object may co-occur without expressing the relation |
| Lexical S-R-O overlap | Relation words may appear without semantic support |
| Popularity statistics | Popular entities do not necessarily imply factual support |

ExposureQA instead estimates whether retrieved passages **semantically support** the target relation.

Example:

> *(Barack Obama, place_of_birth, Hawaii)*

```text
"Hawaii native Barack Obama returned to Honolulu..."
```

Although the passage never explicitly states *"Obama was born in Hawaii"*, it still semantically supports the relation.


# ⚙️ Installation

## Requirements

- Python 3.10+
- CUDA is recommended for RAG experiments

## Install Dependencies

```bash
pip install -r requirements.txt
```

For gated Hugging Face models:

```bash
export HF_TOKEN=YOUR_TOKEN
```


# 🤗 Dataset

## Download

Run from `dataset/`:

```bash
bash download.sh 1   # simple version
bash download.sh 2   # full version
```

### Simple Version

Contains:
- `exposureQA.json`

### Full Version

Contains:
- `exposureQA.json`
- `runs/`
- `passages/`

The full version is required for most experiments.

### Storage Requirements

| Version | Size |
|---|---|
| Simple | ~40 MB |
| Full (compressed) | ~140 GB |
| Full (extracted) | ~400 GB |

### Which Version Should You Use?

Use the **simple version** if you only need:
- Relation-Aware Support
- Lexical-SRO
- Lexical-SO
- Entity Popularity

Use the **full version** if you need:
- retrieved passages
- supporting evidence
- run-level outputs


# 🚀 Experiments


## 📈 Accuracy Analysis

```bash
cd experiments/accuracy
python compute.py
```

### Computes

- factual recall trends
- support-bin accuracy curves
- aggregated statistics across 20 runs

### Outputs

- `results.json`
- PDF plots


## 🎯 Confidence Analysis

```bash
cd experiments/confidence
python compute.py
```

### Confidence Methods

- Self-Consistency
- Token Log Probabilities
- Verbalized Confidence
- P(True)

### Outputs

- `results_all_indexes.json`
- PDF figures


## 📉 Calibration Analysis

```bash
cd experiments/calibration
python compute.py
```

### Computes

- Expected Calibration Error (ECE)
- support-aware calibration trends

### Outputs

- `results.json`
- PDF plots


## ⚠️ Failure Case Extraction

```bash
cd experiments/failure_examples
python extract_examples.py
```

### Requirements

Run calibration first.

### Extracts

- hallucinations
- overconfident failures
- sparse-support failures
- qualitative examples


## 👥 Human Verification

Human verification results are provided in:

```text
excel_filled/
├── Jamshid/
├── Kurosh/
└── Zahra/
```

To recreate the annotation pipeline:

```bash
cd experiments/human_evaluation

python sampling.py
python make_excels.py
```

After manually filling the generated Excel files, place completed annotations into `excel_filled/` and run:

```bash
python analyze.py
```


## 🔍 RAG Experiments

```bash
cd experiments/rag
```

### Step 1 — Build Question Sets

```bash
python questions.py
```

### Step 2 — Run RAG

```bash
python rag.py --model amber --question_type simple
python rag.py --model redpajama --question_type simple
python rag.py --model olmo --question_type simple
```

Supported question types:
- `simple`
- `complex`
- `template_based`

### Step 3 — LLM-Based Evaluation

```bash
python gpt_eval.py --model amber
python gpt_eval.py --model redpajama
python gpt_eval.py --model olmo
```

### Step 4 — Aggregate Results

```bash
python analyze.py
```

Expected outputs:

```text
results_rag/
├── amber_simple.json
├── amber_complex.json
└── ...
```


## 🧪 Sensitivity Analysis

```bash
cd experiments/sensitivity

python compute.py
python analysis.py
```

Analyzes:
- binning robustness
- aggregation stability


# ⚡ Quick Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
cd dataset
bash download.sh 2

# 3. Accuracy
cd ../experiments/accuracy
python compute.py

# 4. Confidence
cd ../confidence
python compute.py

# 5. Calibration
cd ../calibration
python compute.py

# 6. Failure examples
cd ../failure_examples
python extract_examples.py

# 7. RAG pipeline
cd ../rag
python questions.py
python rag.py --model amber --question_type simple
python gpt_eval.py --model amber
python analyze.py
```


# 📊 Main Contributions

- First large-scale study analyzing factual recall using **semantic pretraining evidence**
- Introduces **Relation-Aware Support**
- Benchmarks multiple fully open LLM ecosystems
- Studies recall, confidence, calibration, and hallucination jointly
- Releases the ExposureQA benchmark and experimental framework

---

# 📝 Citation

```bibtex
@article{exposureqa2026,
  title={Analyzing Factual Recall in Large Language Models through Relation-Aware Pretraining Support},
  author={Anonymous ARR submission},
  year={2026}
}
```
