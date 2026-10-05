# AI Media Analysis – Stage 1: AI Governance (Track 1)

<a href="https://colab.research.google.com/github/Pascingo/CTA-media-analysis/blob/main/01_Stage1_CLT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This repository contains Stage 1 of our analysis of the **AI Media Dataset** for **Track 1: AI Governance**.
The notebook `01_Stage1_CLT.ipynb` covers data collection and cleaning, text preprocessing, exploratory
data analysis, entity and relation extraction, and the construction of a topic-specific knowledge graph
for the observation period **September 2024 – August 2026**.

## Contents of the notebook

| Section | What we do |
|---|---|
| 1. Setup | Imports and central settings (text column, time frame, governance keywords, file paths) |
| 2. Loading and cleaning | Load the fixed dataset version, remove missing texts and duplicates, filter the time frame, normalize text (lowercase, HTML tags, symbols) |
| 3. AI Governance filter | Keyword filter with word boundaries; the number of hits per article (`gov_keyword_hits`) is kept for stricter filters later |
| 4. Corpus statistics | Words per article (min, median, max) |
| 5. Text preprocessing | Unification of acronyms and law names (e.g. `eu ai act` → `eu_ai_act`, `general data protection regulation` → `gdpr`), tokenization, lemmatization and stop-word removal with spaCy |
| 6. Most frequent words | Top 20 lemmas with additional domain stop words |
| 7. Temporal analysis | Articles per month and average sentiment (TextBlob polarity) per month |
| 8. Saving | Cleaned AI Governance dataset as CSV for the next stages |
| 9. Entity & relation extraction | spaCy NER on the full corpus, cleaning and alias normalization, structured relation extraction with LangExtract and a local LLM, document-level co-occurrence matrix |
| 10. Knowledge graph | Topic-specific schema (entity and relation types), graph construction with NetworkX, interactive visualization with PyVis |

### Knowledge graph schema

**Entity types:** `ORGANIZATION`, `PERSON`, `TECHNOLOGY`, `AI_MODEL`, `PRODUCT`, `REGULATION`, `STANDARD`, `GOVERNMENT_BODY`

**Relation types:** `REGULATES`, `DEVELOPS`, `INTRODUCES`, `PUBLISHES`, `INVESTIGATES`, `COMPLIES_WITH`, `APPLIES_TO`, `SUPPORTS`, `OPPOSES`, `PARTNERS_WITH`

The allowed combinations (which relation may connect which entity types) are defined in the `schema`
dictionary in section 10.

## Data

- **Source:** [AI Media Dataset on Kaggle](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset)
- **Fixed version:** To make all results reproducible, we use the version from 2026-09-11
  (`ai_media_dataset_20260911.csv`). This file is stored in a public Google Drive folder together with
  the spaCy cache and is downloaded automatically with `gdown` at the start of the notebook.
- The data files are **not** part of this repository because of their size.

## How to run

### Option A: Google Colab (recommended)

1. Click the "Open in Colab" badge above.
2. Run the cells from top to bottom. The dataset and the spaCy cache are downloaded automatically.
3. For the LangExtract section, select a **GPU runtime** (Runtime → Change runtime type) and start
   Ollama as described below.

The notebook uses Google Drive (`drive.mount`) to store caches between runtime resets. To write to our
shared folder, add a shortcut of the shared folder to your "My Drive" and name it `NLP_Projekt`.

### Option B: Local

```bash
git clone https://github.com/Pascingo/CTA-media-analysis.git
cd CTA-media-analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook 01_Stage1_CLT.ipynb
```

The notebook was developed in Colab. When running locally, remove or skip the Colab-specific lines
(`from google.colab import ...`, `drive.mount(...)`) and set `DRIVE_FOLDER` to a local folder, e.g. `data`.

## Local LLM for LangExtract (Ollama)

We do not use a commercial API. LangExtract runs with a local 8B model via [Ollama](https://ollama.com).
Ollama must be started with several parallel slots, otherwise the parallel requests in the notebook are
processed one after another:

```bash
# Colab: prefix every line with "!"
curl -fsSL https://ollama.com/install.sh | sh
OLLAMA_NUM_PARALLEL=8 nohup ollama serve > ollama.log 2>&1 &
ollama pull llama3.1:8b
```

Important settings at the top of the LangExtract cell:

| Setting | Meaning |
|---|---|
| `OLLAMA_MODEL` | Name of the local model (must be pulled in Ollama) |
| `N_PARALLEL` | Number of articles processed in parallel, must match `OLLAMA_NUM_PARALLEL` (use 4 if GPU memory is full) |
| `NUM_CTX` | Context window in tokens (LangExtract's Ollama default is only 2048) |
| `MAX_CHAR_BUFFER` | Chunk size in characters |
| `MAX_ARTICLE_CHARS` | Only the first part of each article is processed (`None` = full article) |
| `RUN_FULL_CORPUS` | `False` = stratified sample of 480 articles, `True` = all articles with `gov_keyword_hits >= MIN_GOV_HITS` |

### Speeding up the extraction

A first run took about 20 seconds per article, which would mean more than a week for the full corpus.
We reduced the runtime with these changes:

1. **Parallel requests:** LangExtract's Ollama provider sends its requests sequentially, so we process
   several articles at the same time with a thread pool and run Ollama with `OLLAMA_NUM_PARALLEL`.
2. **No waiting time:** The `time.sleep(5)` between articles was only needed for the rate limit of the
   Gemini free tier.
3. **Relations only:** Entities for the full corpus already come from spaCy. The LLM only extracts
   relationships; the types of subject and object are attributes of each relationship. This makes the
   model output, which is the slowest part, much shorter.
4. **Correct context size:** With the default of 2048 tokens, prompt and chunk did not fit into the
   context window. We use `num_ctx = 4096` with smaller chunks.
5. **Less input:** Only the first part of each article and, for the full run, only articles with at
   least three governance keyword hits are processed.

All results are cached in a JSONL file after every article, so an interrupted run continues where it
stopped.

## Files created by the notebook

| File | Content |
|---|---|
| `data/ai_governance_cleaned.csv` | Cleaned AI Governance dataset (input for Stage 2) |
| `spacy_processed_text.csv` | Lemmatized texts (cache, spaCy only runs once) |
| `entities_spacy.csv` | spaCy entity mentions with character positions |
| `ner_processed_docs.csv` | Progress file for the NER run |
| `langextract_sample_selection.csv` | Stratified sample (20 articles per month) |
| `langextract_sample_results.jsonl` | LangExtract results of the first run with Gemini (480 articles) |
| `langextract_local_results.jsonl` | LangExtract results with the local LLM |
| `schema.html` | Interactive visualization (PyVis) |

Cache files are stored in the Google Drive folder `NLP_Projekt`.

## Requirements

- Python 3.10 or newer
- Packages: see [`requirements.txt`](requirements.txt) (includes the spaCy model `en_core_web_sm`)
- For the LangExtract section: Ollama and a GPU (e.g. Colab T4) are strongly recommended

## Runtime notes

- **spaCy preprocessing** (about 40 million words) is the slowest CPU step. The result is cached and
  downloaded from Google Drive, so it normally does not have to run again.
- **spaCy NER** is processed in batches of 250 articles with checkpoints.
- **Sentiment** with TextBlob takes a few minutes for the full corpus.
- **LangExtract:** first run the 480-article sample (`RUN_FULL_CORPUS = False`). The cell prints the
  seconds per article, so the time for the full corpus can be estimated.
