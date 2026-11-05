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

**Google Colab (recommended):** click the badge above and select a **GPU runtime** (Runtime → Change runtime type → L4 or A100). The dataset is downloaded automatically from the same Drive folder as in Stage 1, and missing packages are installed by the notebook.

**Local:** install the packages with `pip install -r requirements.txt` (see Stage 1 for the full setup). Sections 1–8 (cleaning, Word2Vec) run on any machine. Sections 9–11 (QLoRA fine-tuning, evaluation, insights) need an **NVIDIA GPU**, because `bitsandbytes` does not support Apple Silicon or CPU-only machines – on a Mac, run these sections in Colab.

---

# Stage 3 – RAG System for Question Answering (Track 1: AI Governance)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pascingo/CTA-media-analysis/blob/main/Stage_3.ipynb)

Stage 3 builds and evaluates a retrieval-augmented generation (RAG) system for questions about AI governance (regulation, security, accountability, bias and fairness). It combines passage retrieval with a revised knowledge graph. Before building the RAG system, the notebook addresses the Stage 1 feedback (topic filter too broad, inconsistent relation predicates, co-occurrence graph treated as evidence), because a RAG system can only be as good as its corpus and its graph.

| File | Purpose |
|---|---|
| `Stage_3.ipynb` | Main notebook (parts 0–8) |

## Pipeline

0. **Setup** – same settings as Stage 1 (time frame, keywords); the Stage 1 corpus is reconstructed with its original `doc_id`s
1. **Stage 1 feedback**
   - tightened topic filter: lexical candidate filter (high recall) + semantic margin against governance / off-topic prototypes (`bge-small-en-v1.5`) → filter `gov_v2`
   - sentence-based passages (≤ 180 words, one sentence overlap) and exclusion of non-news sources (*builtin*, *skyquestt*, *zbrain*, *huggingface*)
   - reproducible relation extraction with `qwen3:14b` (Ollama) and an enforced JSON schema: 5 entity types, 9 predicates, automatic checks, quality filters and entity normalisation
   - manual validation of 50 edges (30 most important + 20 random, Wilson CIs)
   - knowledge graph v2 with a **factual layer** (162 edges, 211 entities) and a separate, non-factual **co-occurrence layer** (887 edges, passage-level, PMI)
2. **Q&A dataset** – 295 question–answer pairs (factual, temporal, relational, analytical, comparative) generated with `gemma3:12b`, automatic checks and repairs, manual review in Colab
3. **Text-only RAG baseline** – built on the course toolbox `anatoolbox`; sparse (BM25), dense (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`) and hybrid retrieval, cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), answers with `qwen3:8b`; retrieval metrics hit@k, recall@k, MRR
4. **Graph-enhanced RAG** – entity linking with surface forms, cross-encoder selection of relevant facts, graph passages added to retrieval, graph facts cited as `[G1]…[Gn]` in the prompt
5. **System evaluation** – text-only vs. graph-enhanced on all 295 questions, LLM judge (`gemma3:12b`) on faithfulness, relevance, context grounding, temporal grounding and correctness; error analysis and robustness check (reviewed vs. unreviewed pairs)
6. **RAG enhancement** – time-aware retrieval: dates in the question are classified (publication vs. event date) and a publication date is used as a retrieval filter
7. **Insights** – lessons for trend monitoring in AI governance with RAG and knowledge graphs
8. **Contributions and AI tools disclosure**

## Knowledge graph v2 schema

- **Entities:** LEGAL_INSTRUMENT, PUBLIC_AUTHORITY, ORGANIZATION, AI_SYSTEM, PERSON
- **Relations (with fixed type signatures):** PROPOSES, ADOPTS, ENFORCES, INVESTIGATES_OR_SANCTIONS, SUES, SUPPORTS, OPPOSES, COMPLIES_WITH, APPLIES_TO
- **Co-occurrence layer:** `CO_MENTIONED_WITH` (`factual = False`), only used to expand retrieval, never as evidence in an answer

## Main results

| | hit@5 | MRR |
|---|---:|---:|
| Text-only (hybrid · bge-small → rerank) | 0.881 | 0.780 |
| + graph facts in retrieval | 0.888 | 0.785 |
| + time-aware publication filter | **0.895** | **0.793** |

- Hybrid retrieval with reranking is the best text configuration; BM25 alone beats dense retrieval.
- The graph facts make no measurable difference to the answer quality overall (judge scores 4.6–5.0 for both systems).
- The publication-date filter is the clearest improvement: on the 20 affected questions, hit@5 rises from 0.75 to 0.95 and no unsupported claims remain.

## Caches and data

The Stage 1 input files are downloaded automatically from the same public Google Drive folder as in Stage 1. Results and caches of Stage 3 are stored in the team's Google Drive (`MyDrive/NLP_Projekt/stage3`), so slow steps survive a runtime reset. Each later part starts with a loading cell, so parts 2–7 can run without re-running Part 1.

| File / folder | Content | Saves |
|---|---|---|
| `export_articles.parquet`, `export_manifest.json` | Article table with all filter columns (`gov_v2`, `in_rag_corpus`, …) | Topic filter (Part 1) |
| `rag/rag_passages.parquet` | Search corpus (passages with `id`, `source_id`, `date`, metadata) | Passage splitting |
| `extraction_cache_ollama.jsonl` | Raw `qwen3:14b` answers of the relation extraction | LLM extraction (GPU) |
| `relations_normalized.parquet`, `entities_kg.parquet`, `kg_schema_stage3.json` | Filtered and normalised relations, entities, schema | Extraction + filters |
| `validation/edge_review.csv` | Manual verdicts on the 50 sampled edges | Manual review |
| `kg_v2/` | `kg_v2_facts.csv`, `kg_v2_cooccurrence.csv`, `kg_v2_entities.csv`, `.graphml` of both layers | Knowledge graph build |
| `qa/qa_testset.json`, `qa/qa_review.csv` | Final Q&A test set and review decisions | Q&A generation + review |
| `rag/` embeddings | Passage embeddings (file name contains model and corpus fingerprint) | Embedding |
| `eval/` | Answers, judgments and results (`rag_answers.jsonl`, `rag_judgments.jsonl`, `time_aware_*`) | Answer generation + judge |

## Run the notebook

**Google Colab (recommended):** click the badge above and select a **GPU runtime** (Runtime → Change runtime type → L4 or T4). The notebook installs the requirements, `anatoolbox` and Ollama itself and pulls the models it needs (`qwen3:14b`, `gemma3:12b`, `qwen3:8b`). It mounts your Google Drive and writes results to `MyDrive/NLP_Projekt/stage3`. A GPU is needed for the LLM steps (relation extraction, Q&A generation, answer generation, judge). Sections 1.5–1.6 and the manual Q&A review (2.5) run on a CPU runtime once the cached files exist.

**Local:** install the packages with `pip install -r requirements.txt` and `pip install "anatoolbox[embeddings] @ git+https://github.com/umagira/anatoolbox"`, install [Ollama](https://ollama.com) and pull the three models above. Remove the Colab-only lines (`from google.colab import ...`, `drive.mount(...)`) and set `STAGE3_FOLDER` to a local folder.
