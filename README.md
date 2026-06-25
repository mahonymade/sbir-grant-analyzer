<!-- The YAML block below is required by Hugging Face Spaces. It is ignored by GitHub. -->
---
title: SBIR Grant Analyzer
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# SBIR Grant Analyzer

A Streamlit web app for exploring and analyzing SBIR/STTR grant data. Built to help researchers identify relevant prior awards and benchmark Phase I → Phase II conversion rates.

## Features

### Project Similarity Search
Find grants similar to your research project using three search modes:

| Mode | Description | Cost |
|---|---|---|
| **Keyword** | Boolean search over titles + abstracts | Free, instant |
| **Embeddings** | Semantic similarity via `all-MiniLM-L6-v2` against **precomputed** corpus embeddings | Free, sub-second |
| **LLM Scoring** | Groq (`llama-3.1-8b-instant`) rates each abstract 0–10 for relevance | API cost, admin-only |

### Phase Conversion Analysis
Estimates the rate at which Phase I awardees went on to receive a Phase II award for the same project. Matching uses company name normalization + fuzzy title matching (via rapidfuzz). Shows:
- Overall conversion rate and per-agency / per-year breakdowns
- Side-by-side comparison with similarity-filtered grants (if a search is active)
- Downloadable matched pairs table

## Quick Start

### 1. Data

The app loads slim, precomputed artifacts (a zstd Parquet table + float16 embeddings,
~228 MB total) hosted on a **Hugging Face Hub dataset repo** — downloaded once onto the
server at startup, never by the browser visitor. There is no large file for you to
manage. See [`data/README.md`](data/README.md) for the artifact spec and how to build
them from the raw CSV.

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` and `torch` are only required for Embeddings mode. `groq` is only required for LLM Scoring mode. The app runs fine without them if you only use Keyword mode.

### 3. (Optional) Configure secrets

Copy the example secrets file and fill in your values to enable LLM Scoring mode:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:
```toml
ADMIN_PASSWORD = "your-password-here"
GROQ_API_KEY = "gsk_..."
# Optional — override the default HF Hub dataset repo for the hosted artifacts:
# SBIR_DATA_REPO = "your-user/sbir-grant-analyzer-data"
```

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Deployment (Hugging Face Spaces)

The app is sized for a low-footprint hosted deploy:

1. **Build + upload artifacts** to a HF Hub dataset repo:
   ```bash
   .venv/bin/python scripts/build_artifacts.py --upload --repo <user>/sbir-grant-analyzer-data
   ```
2. **Create a Streamlit Space** and point it at this repo. Add a `README.md` header with
   `sdk: streamlit` and `app_file: app.py` in the Space (HF requires this frontmatter).
3. **Set secrets / variables** in the Space settings: `SBIR_DATA_REPO` (your dataset repo),
   plus `ADMIN_PASSWORD` and `GROQ_API_KEY` for LLM mode.

`requirements.txt` pins the **CPU-only PyTorch wheel** (`--extra-index-url .../whl/cpu`),
so the deploy image is ~200 MB of torch instead of ~2 GB of unused CUDA. HF Spaces' free
tier (16 GB RAM) comfortably fits the model + mmap'd embeddings + Parquet table.

## Tech Stack

| Component | Library |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Data | [pandas](https://pandas.pydata.org) |
| Fuzzy matching | [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) |
| Semantic embeddings | [sentence-transformers](https://www.sbert.net) |
| LLM scoring | [Groq](https://groq.com) (`llama-3.1-8b-instant`) |

## Project Structure

```
├── app.py                          # Main Streamlit app (three tabs)
├── scripts/
│   └── build_artifacts.py          # Build/upload slim parquet + embeddings artifacts
├── src/
│   ├── data_loader.py              # Parquet/CSV loaders, HF Hub artifact fetch, embeddings loader
│   ├── similarity.py               # Keyword / embeddings / LLM search engines
│   └── conversion.py               # Phase I → II matching logic
├── requirements.txt
├── .streamlit/
│   ├── config.toml                 # Blue theme
│   └── secrets.toml.example        # Template — copy to secrets.toml
└── data/
    └── README.md                   # Data file setup instructions
```
