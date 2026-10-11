# CTA Media Analysis – Stage 1 (Track 1: AI Governance)

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

Main outputs: `data/ai_governance_cleaned.csv` (input for Stage 2), co-occurrence matrix, knowledge graphs (`.graphml`, node/edge CSVs) and interactive PyVis HTML views.
