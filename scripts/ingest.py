"""
CLI script: ingest all documents in a folder (incremental — skips unchanged files).
Usage:
    python scripts/ingest.py --path data/raw/
    python scripts/ingest.py --path path/to/my/docs
"""

import argparse
import sys
from pathlib import Path

# Make sure 'app' is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import sync_folder


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the knowledge base")
    parser.add_argument(
        "--path",
        type=str,
        default="data/raw",
        help="Folder containing documents to ingest (default: data/raw)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only ingest files at the top level of the folder",
    )
    args = parser.parse_args()

    print("\n=== Knowledge Agent — Document Ingestion ===")
    result = sync_folder(args.path, recursive=not args.no_recursive)
    print("\n=== Done ===")
    print(f"  New files indexed   : {result['new']}")
    print(f"  Unchanged (skipped) : {result['skipped']}")
    print(f"  Chunks in store     : {result['total_chunks']}")
    print("\nReady to query. Run: streamlit run ui.py   (or: uvicorn main:app --reload)")


if __name__ == "__main__":
    main()
