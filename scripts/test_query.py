"""Quick test: ask a question directly without starting the API server."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.query.agent import query

questions = [
    "What is an AI agent?",
    "What is ChromaDB used for?",
    "What are embeddings?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")
    result = query(q)
    print(f"A: {result['answer']}")
    if result['sources']:
        print(f"\nSources:")
        for s in result['sources']:
            print(f"  - {s['file']} (score: {s['score']})")
