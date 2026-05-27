"""Test queries against the UPC curriculum."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.query.agent import query

questions = [
    "What courses are in the first semester?",
    "What mathematics courses are required?",
    "How many total credits does the program have?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")
    result = query(q, session_id="upc-test")
    print(f"A: {result['answer']}")
    if result['sources']:
        print(f"\nSources: {[s['file'] for s in result['sources']]}")
