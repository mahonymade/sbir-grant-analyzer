"""
Build slim, web-ready artifacts from the raw SBIR CSV.

Produces three files in ``data/``:
    grants.parquet   — slim, compressed grant table (~30–60 MB vs 351 MB CSV)
    embeddings.npy   — float16 sentence embeddings, aligned row-for-row to the parquet
    meta.json        — model name, row count, dims, dtype, build date (for runtime validation)

These artifacts are what the deployed app loads. They are hosted on a Hugging Face
Hub *dataset* repo and downloaded once onto the server at startup — the end user
(browser visitor) never downloads them.

Usage
-----
Build only (writes to data/):
    .venv/bin/python scripts/build_artifacts.py

Build and upload to Hugging Face Hub (requires `huggingface-cli login` or HF_TOKEN):
    .venv/bin/python scripts/build_artifacts.py --upload --repo mahonymade/sbir-grant-analyzer-data

Row alignment
-------------
The parquet carries an explicit ``_row_id`` (0..N-1). Row i of ``embeddings.npy``
corresponds to ``_row_id == i``. The runtime uses ``_row_id`` to slice embeddings for
any filtered subset, so alignment survives every filter/sort/index-reset in the app.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

# Make `src` importable when run as a script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_from_csv  # noqa: E402

# Slim column set — everything the app's tabs actually read. Derived text columns
# (combined_text_lc etc.) are rebuilt cheaply at load time, so they're not stored.
SLIM_COLS = [
    "company",
    "award_title",
    "phase",
    "agency",
    "program",
    "award_year",
    "award_amount",
    "abstract",
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = PROJECT_ROOT / "data"


def build(csv_path: str, device: str | None = None) -> dict:
    print(f"Loading raw CSV from {csv_path} …")
    df = load_from_csv(csv_path)
    print(f"  loaded {len(df):,} rows")

    # Drop rows with neither a title nor an abstract — nothing to search or embed.
    has_text = (
        df["award_title"].fillna("").str.strip().ne("")
        | df["abstract"].fillna("").str.strip().ne("")
    )
    df = df[has_text].reset_index(drop=True)
    print(f"  {len(df):,} rows after dropping empty title+abstract")

    # combined_text_lc is built by load_from_csv; use it verbatim so runtime
    # embeddings (query side) and build-time embeddings (corpus side) match.
    texts = df["combined_text_lc"].tolist()

    # --- Slim parquet ---
    slim = df[[c for c in SLIM_COLS if c in df.columns]].copy()
    slim.insert(0, "_row_id", np.arange(len(slim), dtype=np.int64))
    DATA_DIR.mkdir(exist_ok=True)
    parquet_path = DATA_DIR / "grants.parquet"
    # zstd compresses the long abstract text far better than snappy (~47% smaller
    # here) for a negligible read-time cost.
    slim.to_parquet(parquet_path, compression="zstd", compression_level=10, index=False)
    print(f"  wrote {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # --- Embeddings ---
    if device is None:
        try:
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"Encoding {len(texts):,} abstracts on device={device} …")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine becomes a plain dot product at runtime
        show_progress_bar=True,
    ).astype(np.float16)
    print(f"  encoded in {time.time() - t0:.1f}s, shape={emb.shape}, dtype={emb.dtype}")

    emb_path = DATA_DIR / "embeddings.npy"
    np.save(emb_path, emb)
    print(f"  wrote {emb_path} ({emb_path.stat().st_size / 1e6:.1f} MB)")

    # --- Metadata ---
    meta = {
        "model": EMBEDDING_MODEL,
        "n_rows": int(len(slim)),
        "dim": int(emb.shape[1]),
        "dtype": "float16",
        "normalized": True,
        "built": date.today().isoformat(),
    }
    meta_path = DATA_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path}: {meta}")
    return meta


def upload(repo_id: str) -> None:
    """Upload the three artifacts to a Hugging Face Hub dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    for name in ("grants.parquet", "embeddings.npy", "meta.json"):
        path = DATA_DIR / name
        print(f"Uploading {name} → {repo_id} …")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"Done. Set SBIR_DATA_REPO={repo_id} in the deploy environment / secrets.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slim SBIR artifacts.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "award_data.csv"))
    parser.add_argument("--device", default=None, help="mps | cpu | cuda (auto-detected)")
    parser.add_argument("--upload", action="store_true", help="upload to Hugging Face Hub")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="skip the build; upload the existing artifacts in data/ (no re-encode)",
    )
    parser.add_argument("--repo", default="mahonymade/sbir-grant-analyzer-data")
    args = parser.parse_args()

    if args.upload_only:
        missing = [
            n for n in ("grants.parquet", "embeddings.npy", "meta.json")
            if not (DATA_DIR / n).exists()
        ]
        if missing:
            parser.error(f"--upload-only but missing artifacts in {DATA_DIR}: {missing}")
        upload(args.repo)
        return

    build(args.csv, device=args.device)
    if args.upload:
        upload(args.repo)


if __name__ == "__main__":
    main()
