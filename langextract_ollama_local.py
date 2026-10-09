"""
LangExtract with a local Ollama model (optimised for Apple Silicon, e.g. MacBook Pro M4 Pro)
==========================================================================================

Runs the same entity/relation extraction as the notebook (01_Stage1_CLT.ipynb), but with
a local Ollama model instead of the Gemini API, over ALL articles of df_gov
(>= MIN_KEYWORD_HITS governance keyword hits) or only over the existing sample.

The output has exactly the same format as the notebook cache
(one JSON line per article: doc_id, title, date, url, extractions), so the evaluation
and knowledge graph cells in the notebook can read it directly.

------------------------------------------------------------------------------------------
1. One-time setup (Terminal)
------------------------------------------------------------------------------------------
    brew install ollama                 # or the Ollama app from ollama.com
    pip install langextract pandas tqdm requests

    ollama pull gemma3:4b               # fast default model
    # optional, better quality but slower:  ollama pull qwen3:8b   /   ollama pull gemma3:12b

2. Start Ollama with parallel slots (important for speed!)
   Quit the Ollama menu bar app first, then in a separate Terminal window:

    OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve

   (If you prefer the Ollama app: set these variables with
    `launchctl setenv OLLAMA_NUM_PARALLEL 4` etc. and restart the app.)

3. Put this file in the same folder as the notebook. The folder "data" must contain
   ai_media_dataset_20260911.csv (the notebook downloads it there).

------------------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------------------
    # Benchmark first: 20 random articles, prints speed and estimated total time
    python langextract_ollama_local.py --limit 20

    # Full run over all df_gov articles (can be stopped with Ctrl+C and resumed any time)
    caffeinate -i python langextract_ollama_local.py

    # Stratified sample: 50 random articles per month (24 months -> 1'200 articles)
    python langextract_ollama_local.py --per-month 50 --output data/langextract_ollama_50pm.jsonl

    # Only the articles of the existing LangExtract sample (comparison with Gemini)
    python langextract_ollama_local.py --sample-only

    # Faster: only the first 8000 characters per article (= exactly one chunk)
    python langextract_ollama_local.py --max-chars 8000 --limit 20

    # Other model / more parallel requests
    python langextract_ollama_local.py --model qwen3:8b --workers 4 --limit 20

"caffeinate -i" prevents the Mac from going to sleep during long runs. Keep the
MacBook plugged in: on battery, macOS throttles the GPU.
"""

import argparse
import json
import os
import random
import sys
import textwrap
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

import langextract as lx


# ----------------------------------------------------------------------------------------
# Settings: identical to the notebook, so that doc_id = index of df_gov in the notebook
# ----------------------------------------------------------------------------------------
TEXT_COLUMN = "content"
START_DATE = "2024-09-01"
END_DATE = "2026-08-31"
GOVERNANCE_KEYWORDS = [
    "security", "accountability", "bias", "fairness",
    "regulatory", "regulation", "governance", "policy",
    "ethics", "compliance", "law"
]
MIN_KEYWORD_HITS = 5

DATA_FOLDER = "data"
DATASET_FILE = os.path.join(DATA_FOLDER, "ai_media_dataset_20260911.csv")
SAMPLE_FILE = os.path.join(DATA_FOLDER, "langextract_sample_selection.csv")
DEFAULT_OUTPUT = os.path.join(DATA_FOLDER, "langextract_ollama_results.jsonl")

OLLAMA_URL = "http://localhost:11434"


# ----------------------------------------------------------------------------------------
# Prompt and few-shot examples: identical to the notebook
# ----------------------------------------------------------------------------------------
PROMPT = textwrap.dedent("""
Extract information relevant to AI Governance.

Extract:

1. Entities with exactly one of these entity types:
   ORGANIZATION
   PERSON
   TECHNOLOGY
   AI_MODEL
   PRODUCT
   REGULATION
   STANDARD
   GOVERNMENT_BODY

2. Explicit relationships between entities.

For relationships, use exactly one of these predicates:
   REGULATES
   DEVELOPS
   INTRODUCES
   PUBLISHES
   INVESTIGATES
   COMPLIES_WITH
   APPLIES_TO
   SUPPORTS
   OPPOSES
   PARTNERS_WITH

Use exact text from the source.

Only extract information explicitly stated in the text.
Do not use outside knowledge.
Do not infer relationships that are not supported by the article.
""")

EXAMPLES = [
    lx.data.ExampleData(
        text="OpenAI is an artificial intelligence research organization.",
        extractions=[
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="OpenAI",
                attributes={"entity_type": "ORGANIZATION"},
            )
        ],
    ),
    lx.data.ExampleData(
        text="The European Commission introduced the AI Act.",
        extractions=[
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="introduced",
                attributes={
                    "subject": "European Commission",
                    "predicate": "INTRODUCES",
                    "object": "AI Act",
                },
            )
        ],
    ),
]


# ----------------------------------------------------------------------------------------
# 1. Rebuild df_gov exactly like the notebook (same cleaning -> same index = doc_id)
# ----------------------------------------------------------------------------------------
def build_df_gov():
    if not os.path.exists(DATASET_FILE):
        sys.exit(f"{DATASET_FILE} not found. Run the download cell of the notebook first.")

    print("Loading dataset and rebuilding df_gov (same steps as in the notebook)...")
    df = pd.read_csv(DATASET_FILE, usecols=["title", "date", "content", "domain", "url", "tags"])

    df = df.dropna(subset=[TEXT_COLUMN])
    df = df.drop_duplicates(subset=[TEXT_COLUMN])

    df["published_at"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["published_at"] >= START_DATE) & (df["published_at"] <= END_DATE)].copy()

    df["cleaned_text"] = df[TEXT_COLUMN].apply(
        lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    )
    df["cleaned_text"] = (
        df["cleaned_text"]
        .str.lower()
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"[-/]", " ", regex=True)
        .str.replace(r"[^a-z0-9\s.,!?]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    pattern = r"\b(?:" + "|".join(GOVERNANCE_KEYWORDS) + r")\b"
    df["gov_keyword_hits"] = df["cleaned_text"].str.count(pattern)
    df_gov = df[df["gov_keyword_hits"] >= MIN_KEYWORD_HITS].copy()

    # Only keep what we need (saves memory)
    df_gov = df_gov[["title", "date", "url", "content"]]
    print(f"df_gov: {len(df_gov):,} articles (>= {MIN_KEYWORD_HITS} keyword hits)")
    return df_gov


# ----------------------------------------------------------------------------------------
# 2. Ollama checks
# ----------------------------------------------------------------------------------------
def check_ollama(model):
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.exceptions.RequestException:
        sys.exit("Ollama is not running. Start it with:\n"
                 "  OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 "
                 "OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve")

    installed = {m["name"] for m in tags.get("models", [])}
    if model not in installed and f"{model}:latest" not in installed:
        sys.exit(f"Model '{model}' is not installed. Run: ollama pull {model}\n"
                 f"Installed models: {', '.join(sorted(installed)) or '-'}")

    # Load the model once (warm-up), so the first article is not slowed down
    print(f"Loading model '{model}' into memory...")
    requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": "ok", "stream": False, "keep_alive": "30m",
              "options": {"num_predict": 1}},
        timeout=300,
    )


# ----------------------------------------------------------------------------------------
# 3. Extraction of one article
# ----------------------------------------------------------------------------------------
def extract_article(doc_id, article, args):
    """Returns the record in the same format as the notebook cache."""
    record = {
        "doc_id": int(doc_id),
        "title": article["title"],
        "date": str(article["date"]),
        "url": article["url"],
        "extractions": [],
    }

    text = str(article["content"])
    if args.max_chars:
        # Only the beginning of the article (char positions stay valid, because it is a prefix)
        text = text[:args.max_chars]
    if not text.strip():
        return record

    last_error = None
    for attempt in range(args.retries):
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=PROMPT,
                examples=EXAMPLES,
                model_id=args.model,
                model_url=OLLAMA_URL,
                # Recommended settings for Ollama (see LangExtract README)
                fence_output=False,
                use_schema_constraints=False,
                temperature=0.0,                 # reproducible results
                max_char_buffer=args.max_char_buffer,
                extraction_passes=1,
                batch_length=10,
                max_workers=1,                   # parallelism is done across articles
                language_model_params={
                    # LangExtract's Ollama default is only 2048 tokens -> text would be cut off
                    "num_ctx": args.num_ctx,
                    "timeout": args.timeout,
                    "keep_alive": "30m",         # keep the model in memory
                },
                show_progress=False,
            )

            for extraction in result.extractions:
                if extraction.char_interval is None:   # keep only grounded results
                    continue
                record["extractions"].append({
                    "class": extraction.extraction_class,
                    "text": extraction.extraction_text,
                    "attributes": extraction.attributes,
                    "start_char": extraction.char_interval.start_pos,
                    "end_char": extraction.char_interval.end_pos,
                })
            return record

        except Exception as e:   # noqa: BLE001  (model output can fail in many ways)
            last_error = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


# ----------------------------------------------------------------------------------------
# 4. Main loop: parallel, resumable, writes every result immediately
# ----------------------------------------------------------------------------------------
def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["doc_id"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    return done


def main():
    parser = argparse.ArgumentParser(description="LangExtract with a local Ollama model")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model (default: gemma3:4b)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Articles processed in parallel. Should match OLLAMA_NUM_PARALLEL.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process N random articles (benchmark).")
    parser.add_argument("--per-month", type=int, default=None,
                        help="Random sample of N articles per month (reproducible, seed 42)")
    parser.add_argument("--sample-only", action="store_true",
                        help="Only the articles of langextract_sample_selection.csv")
    parser.add_argument("--max-char-buffer", type=int, default=8000,
                        help="Characters per chunk (same as notebook: 8000)")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Only use the first N characters of each article (e.g. 8000 = 1 chunk)")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context window in tokens")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per request in seconds")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", default=None, help=f"Output file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    output_file = args.output or DEFAULT_OUTPUT
    error_file = output_file.replace(".jsonl", "_errors.jsonl")

    df_gov = build_df_gov()
    doc_ids = list(df_gov.index)

    if args.sample_only:
        if not os.path.exists(SAMPLE_FILE):
            sys.exit(f"{SAMPLE_FILE} not found.")
        sample_ids = set(pd.read_csv(SAMPLE_FILE)["doc_id"].astype(int))
        doc_ids = [i for i in doc_ids if i in sample_ids]
        print(f"Sample only: {len(doc_ids):,} articles of the sample are in df_gov")

    if args.per_month:
        # Simple random sample per month: same weight for every month, reproducible
        months = pd.to_datetime(df_gov.loc[doc_ids, "date"]).dt.to_period("M")
        sampled = (
            pd.Series(doc_ids, index=doc_ids)
            .groupby(months.values)
            .apply(lambda s: s.sample(n=min(args.per_month, len(s)), random_state=42))
        )
        doc_ids = sorted(sampled.astype(int).tolist())
        print(f"Per month: {args.per_month} articles -> {len(doc_ids):,} articles in total")

    done = load_done_ids(output_file)
    todo = [i for i in doc_ids if i not in done]

    # Random order: the ETA is then representative (articles are sorted by date in the data)
    random.Random(42).shuffle(todo)
    total_remaining = len(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"Already done: {len(done & set(doc_ids)):,} | remaining: {total_remaining:,} "
          f"| this run: {len(todo):,}")
    if not todo:
        print("Nothing to do.")
        return

    check_ollama(args.model)

    # Save the settings of this run (separate file, so the result format stays identical)
    with open(output_file.replace(".jsonl", "_runinfo.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "max_char_buffer": args.max_char_buffer,
                   "max_chars": args.max_chars, "per_month": args.per_month,
                   "num_ctx": args.num_ctx, "min_keyword_hits": MIN_KEYWORD_HITS,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    write_lock = threading.Lock()
    n_ok = n_err = n_extr = 0
    start = time.time()

    print(f"\nModel: {args.model} | parallel: {args.workers} | chunk: {args.max_char_buffer} chars"
          f" | num_ctx: {args.num_ctx}\nOutput: {output_file}\n(Stop with Ctrl+C, resume later)\n")

    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        with tqdm(total=len(todo), desc="LangExtract (Ollama)", unit="art") as bar:

            futures = {pool.submit(extract_article, i, df_gov.loc[i], args): i for i in todo}

            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    record = future.result()
                    with write_lock, open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_ok += 1
                    n_extr += len(record["extractions"])
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    with write_lock, open(error_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"doc_id": int(doc_id), "error": str(e)}) + "\n")
                bar.update(1)
                bar.set_postfix(ok=n_ok, errors=n_err, extractions=n_extr)

    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nStopped. All finished articles are saved; just run the script again to resume.")
        os._exit(1)   # do not wait for the running requests
    pool.shutdown()

    # Summary and estimate for the full run
    elapsed = time.time() - start
    per_article = elapsed / max(n_ok + n_err, 1)
    still_open = total_remaining - n_ok
    print(f"\nFinished {n_ok:,} articles ({n_err} errors) in {elapsed / 60:.1f} min")
    print(f"Average: {per_article:.1f} s per article, {n_extr / max(n_ok, 1):.1f} extractions per article")
    if still_open > 0:
        print(f"Estimated time for the remaining {still_open:,} articles: "
              f"{still_open * per_article / 3600:.1f} hours")
    if n_err:
        print(f"Failed articles are listed in {error_file}; they are retried in the next run.")


if __name__ == "__main__":
    main()
