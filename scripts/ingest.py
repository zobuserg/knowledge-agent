"""
CLI script: ingest all documents in a folder.
Usage:
    python scripts/ingest.py --path data/raw/
    python scripts/ingest.py --path path/to/my/docs
"""

import argparse
import sys
from pathlib import Path

# Make sure 'app' is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into knowledge base")
    parser.add_argument(
        "--path",
        type=str,
        default="data/raw",
        help="Folder containing documents to ingest (default: data/raw)",
    )
    args = parser.parse_args()

    print(f"\n=== Knowledge Agent — Document Ingestion ===")
    result = ingest_directory(args.path)
    print(f"\n=== Done ===")
    print(f"  Documents loaded : {result['docs_loaded']}")
    print(f"  Chunks in store  : {result['chunks_created']}")
    print(f"  Collection       : {result['collection']}")
    print(f"\nReady to query. Run: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
