# Data Files

The app runs on two slim, precomputed **artifacts** rather than the raw 351 MB CSV:

| Artifact | Size | Contents |
|---|---|---|
| `grants.parquet` | ~81 MB | slim grant table (8 columns + `_row_id`), zstd-compressed |
| `embeddings.npy` | ~147 MB | float16 `all-MiniLM-L6-v2` embeddings, row-aligned to the parquet via `_row_id` |
| `meta.json` | <1 KB | model name, row count, dims — runtime sanity check |

These are **hosted on a Hugging Face Hub dataset repo** (default `mahonymade/sbir-grant-analyzer-data`) and downloaded **once onto the server** at startup — the browser visitor never downloads them. None of these files are committed to git.

## Running the deployed / hosted-dataset path (default)

Just run the app. With the "Hosted dataset" data source selected (the default), it
downloads and caches the artifacts automatically. Override the repo with the
`SBIR_DATA_REPO` env var or a `SBIR_DATA_REPO` entry in `.streamlit/secrets.toml`.

## Building the artifacts from raw CSV

1. Download the full SBIR award dataset from [SBIR.gov Data Resources](https://www.sbir.gov/data-resources) and save it as `award_data.csv` in the repo root.
2. Build the artifacts (encodes all abstracts — uses Apple MPS / CUDA if available):
   ```
   .venv/bin/python scripts/build_artifacts.py
   ```
   This writes `grants.parquet`, `embeddings.npy`, and `meta.json` into `data/`.
   The app prefers these local files when present (no download needed).
3. Upload to the Hugging Face Hub (requires `huggingface-cli login` or `HF_TOKEN`):
   ```
   .venv/bin/python scripts/build_artifacts.py --upload --repo <user>/<dataset-repo>
   ```

## Local raw-CSV path (dev/fallback)

Select "Local CSV" in the sidebar and point it at `award_data.csv`. In this mode the
embeddings search has no precomputed artifact, so it encodes the filtered corpus on the
fly (slower) — fine for development.

## Expected columns

The file should have the following columns (the app normalizes names automatically):

| Column | Description |
|---|---|
| Company | Recipient company name |
| Award Title | Project title |
| Agency | Awarding agency (e.g., DoD, NSF) |
| Branch | Sub-agency branch |
| Phase | Phase I or Phase II |
| Program | SBIR or STTR |
| Award Year | Fiscal year of award |
| Award Amount | Dollar amount |
| Abstract | Project abstract text |
| ... | (37 additional columns) |

## Future API integration

When the SBIR API is restored, the `src/data_loader.py` → `load_from_api()` function contains a documented stub ready for implementation.
