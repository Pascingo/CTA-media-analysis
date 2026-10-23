# CTA Media Analysis (Track 1: AI Governance)

## Stage 1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pascingo/CTA-media-analysis/blob/main/01_Stage1_CLT.ipynb)

Stage 1 of our analysis of the [AI Media Dataset](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset) (Sep 2024 – Aug 2026): cleaning, AI-governance filter, spaCy preprocessing, EDA, sentiment, entity/relation extraction (spaCy NER + LangExtract) and a topic-specific knowledge graph.

| File | Purpose |
|---|---|
| `01_Stage1_CLT.ipynb` | Main notebook (all sections 1–10) |
| `langextract_ollama_local.py` | Runs the LangExtract step locally with Ollama (does not work reliably in Colab) |
| `requirements.txt` | Python packages incl. spaCy model `en_core_web_sm` |

## Data and caches – no slow steps needed

All input data and caches are public, so you can skip the slow steps:

- **Google Drive** ([public folder](https://drive.google.com/drive/folders/14cAxvtxuTfwENtSDCmGcRpl6qhl-z62z)): the setup cell downloads it into `data/` with `gdown`.
- **GitHub**: the same files can be taken from this repo's `data/` folder (e.g. after `git clone`).

| File in `data/` | Content | Saves |
|---|---|---|
| `ai_media_dataset_20260911.csv` | Fixed dataset version (reproducibility) | Kaggle download |
| `spacy_processed_text.csv` | Lemmatized texts | spaCy preprocessing (~40 M words) |
| `entities_spacy.csv`, `ner_processed_docs.csv` | spaCy NER results + progress | spaCy NER |
| `langextract_ollama_25pm.jsonl` | LangExtract results (gemma3:4b, 25 articles/month) | Local LLM run (hours) |

Every cache is only used if it matches the current corpus; otherwise the step re-runs and overwrites it.

## Run the notebook

**Google Colab (recommended):** click the badge above and run all cells top to bottom. Requirements and data are installed/downloaded automatically. The last export cell mounts *your* Google Drive and writes the graphs to `MyDrive/CTA-media-analysis/kg_export`.

**Local:**

```bash
git clone https://github.com/Pascingo/CTA-media-analysis.git
cd CTA-media-analysis
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook 01_Stage1_CLT.ipynb
```

Remove the Colab-only lines (`from google.colab import ...`, `drive.mount(...)`) and skip the Drive export cell. Python ≥ 3.10.

## Optional: re-run LangExtract locally (Ollama)

Only needed if you want new extraction results – the notebook just reads `data/langextract_ollama_25pm.jsonl`. The script rebuilds the same governance corpus (same `doc_id`s) and writes the same JSONL format. Tested on Apple Silicon (M4 Pro).

```bash
brew install ollama            # or app from ollama.com
pip install langextract pandas tqdm requests
ollama pull gemma3:4b

# separate terminal (quit the Ollama menu bar app first)
OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve

python langextract_ollama_local.py --limit 20                     # benchmark + time estimate
caffeinate -i python langextract_ollama_local.py --per-month 25 \
    --output data/langextract_ollama_25pm.jsonl                    # result used in the notebook
```

Runs are resumable (Ctrl+C, then start again). More options: `python langextract_ollama_local.py --help` (`--model`, `--workers`, `--max-chars`, `--sample-only`, …).

## Knowledge graph schema

- **Entities:** ORGANIZATION, PERSON, TECHNOLOGY, AI_MODEL, PRODUCT, REGULATION, STANDARD, GOVERNMENT_BODY
- **Relations:** REGULATES, DEVELOPS, INTRODUCES, PUBLISHES, INVESTIGATES, COMPLIES_WITH, APPLIES_TO, SUPPORTS, OPPOSES, PARTNERS_WITH

Main outputs: `data/ai_governance_cleaned.csv`, co-occurrence matrix, knowledge graphs (`.graphml`, node/edge CSVs) and interactive PyVis HTML views.

---

# Stage 2 – Embedding Models (Track 1: AI Governance)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pascingo/CTA-media-analysis/blob/main/Stage_2.ipynb)

Stage 2 trains and compares text representations on the same dataset version: a classical Word2Vec model, compact open LLMs fine-tuned with QLoRA (Qwen3-1.7B, Qwen3-4B), and TF-IDF / zero-shot baselines. The extrinsic task is the classification of AI-governance subtopics; the best model is then used to analyse the governance discourse over the whole corpus.

| File | Purpose |
|---|---|
| `Stage_2.ipynb` | Main notebook (sections 1–11) |

## Pipeline

1. **Cleaning** – two text versions: `base_text` (original casing, for the LLMs) and `w2v_text` (lowercase, for Word2Vec), plus a language filter
2. **Split** – 80/10/10 train/val/test on article level, stratified by month; used throughout the stage
3. **Word2Vec** – skip-gram (200 dim.) with multi-word terms (`eu_ai_act`, `prompt_injection`, …), learned on the training split only
4. **Labels** – 2,600 articles labelled by Phi-4 (silver labels) into `REGULATION_POLICY`, `SECURITY_SAFETY`, `BIAS_FAIRNESS_ETHICS`, `ACCOUNTABILITY_TRANSPARENCY`, `NOT_GOVERNANCE`
5. **Fine-tuning** – QLoRA (4-bit base model, LoRA r = 16) on Qwen3-1.7B and Qwen3-4B
6. **Evaluation** – word similarity (WordSim-353, SimLex-999, governance triplets), clustering quality, classification (macro-F1 with confidence intervals) and costs (time, GPU memory)
7. **Insights** – governance share over time, term trajectories, semantic shift between year 1 and year 2, themes and peak months

## Caches

Slow steps are stored in the team's Google Drive (`MyDrive/CTA-media-analysis/stage2`). With `USE_CACHE = True` (setup cell) existing results are loaded instead of recomputed; set it to `False` to rebuild everything.

| File | Content | Saves |
|---|---|---|
| `stage2_cleaned.parquet`, `splits.csv` | Cleaned corpus and fixed split | Cleaning (several minutes) |
| `bigram.phraser`, `w2v_train.model` | Phrase model and Word2Vec | Word2Vec training |
| `finetuning/llm_labels.jsonl` | Phi-4 labels | Labelling (10–60 min) |
| `finetuning/adapter_<model>` | LoRA adapters | Fine-tuning |
| `finetuning/corpus_predictions_<model>.csv` | Subtopic for every article | Corpus classification (30–45 min) |

## Run the notebook

Colab with a **GPU runtime is required from section 9 on** (Runtime → Change runtime type → L4 or A100). Sections 1–8 also run on CPU. The dataset is downloaded automatically from the same Drive folder as in Stage 1.