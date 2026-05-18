# SBIR Grant Analyzer

A Streamlit web app for exploring and analyzing SBIR/STTR grant data. Built to help researchers identify relevant prior awards and benchmark Phase I → Phase II conversion rates.

## Features

### Project Similarity Search
Find grants similar to your research project using three search modes:

| Mode | Description | Cost |
|---|---|---|
| **Keyword** | Boolean search over titles + abstracts | Free, instant |
| **Embeddings** | Semantic similarity via `all-MiniLM-L6-v2` (local, ~80 MB, downloaded once) | Free, ~30s first run |
| **LLM Scoring** | Claude API rates each abstract 0–10 for relevance | API cost, admin-only |

### Phase Conversion Analysis
Estimates the rate at which Phase I awardees went on to receive a Phase II award for the same project. Matching uses company name normalization + fuzzy title matching (via rapidfuzz). Shows:
- Overall conversion rate and per-agency / per-year breakdowns
- Side-by-side comparison with similarity-filtered grants (if a search is active)
- Downloadable matched pairs table

## Quick Start

### 1. Get the data

The SBIR award dataset is not included in this repository (too large for GitHub). See [`data/README.md`](data/README.md) for instructions on downloading it from SBIR.gov.

Place the file at `award_data.csv` in the project root (same folder as `app.py`).

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` and `torch` are only required for Embeddings mode. `anthropic` is only required for LLM Scoring mode. The app runs fine without them if you only use Keyword mode.

### 3. (Optional) Configure secrets

Copy the example secrets file and fill in your values to enable LLM Scoring mode:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:
```toml
ADMIN_PASSWORD = "your-password-here"
ANTHROPIC_API_KEY = "sk-ant-..."
```

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Tech Stack

| Component | Library |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Data | [pandas](https://pandas.pydata.org) |
| Fuzzy matching | [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) |
| Semantic embeddings | [sentence-transformers](https://www.sbert.net) |
| LLM scoring | [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) (Claude) |

## Project Structure

```
├── app.py                          # Main Streamlit app (two tabs)
├── src/
│   ├── data_loader.py              # CSV loader + column normalization
│   ├── similarity.py               # Keyword / embeddings / LLM search engines
│   └── conversion.py               # Phase I → II matching logic
├── requirements.txt
├── .streamlit/
│   ├── config.toml                 # Blue theme
│   └── secrets.toml.example        # Template — copy to secrets.toml
└── data/
    └── README.md                   # Data file setup instructions
```
