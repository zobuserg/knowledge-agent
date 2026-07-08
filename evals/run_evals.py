"""
Retrieval & end-to-end evals
----------------------------
Measures whether the knowledge base actually retrieves the right documents —
so a change (embedding model, chunk size, top_k) can be judged by a score
instead of a feeling.

Dataset: evals/dataset.json (personal, gitignored). If missing, falls back to
evals/dataset.example.json so the script always runs. Each case:

    {
      "question":        "what to ask",
      "expect_file":     "substring that must appear in a retrieved file name",
      "must_contain":    ["optional substrings the final ANSWER must include (e2e only)"]
    }

Usage:
    python evals/run_evals.py                # retrieval eval (cheap: embeddings only)
    python evals/run_evals.py --e2e          # + full agent answers (costs LLM calls)
    python evals/run_evals.py --top-k 6
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import get_vector_store, get_embed_model  # noqa: E402

EVALS_DIR = Path(__file__).parent


def load_dataset() -> list[dict]:
    personal = EVALS_DIR / "dataset.json"
    example = EVALS_DIR / "dataset.example.json"
    if personal.exists():
        return json.loads(personal.read_text(encoding="utf-8"))
    print("NOTE: evals/dataset.json not found — using dataset.example.json.")
    print("      Copy it to dataset.json and edit the cases to match YOUR documents.\n")
    return json.loads(example.read_text(encoding="utf-8"))


def eval_retrieval(cases: list[dict], top_k: int) -> int:
    """Hit@k: does the expected file appear among the top-k retrieved chunks?"""
    from llama_index.core import VectorStoreIndex

    _, vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())
    retriever = index.as_retriever(similarity_top_k=top_k)

    hits = 0
    print(f"-- Retrieval eval (hit@{top_k}) " + "-" * 40)
    for case in cases:
        nodes = retriever.retrieve(case["question"])
        files = [n.metadata.get("file_name", "") for n in nodes]
        hit = any(case["expect_file"].lower() in f.lower() for f in files)
        hits += hit
        mark = "PASS" if hit else "FAIL"
        print(f"  [{mark}] {case['question'][:60]}")
        if not hit:
            print(f"         expected file ~ '{case['expect_file']}', got: {files[:3]}")
    print(f"\n  Retrieval score: {hits}/{len(cases)} ({100 * hits // len(cases)}%)\n")
    return hits


def eval_e2e(cases: list[dict]) -> int:
    """Full pipeline: agent answer must contain the expected substrings."""
    from app.query.agent import query, clear_history

    passed = 0
    print("-- End-to-end eval (agent answers) " + "-" * 35)
    for i, case in enumerate(cases):
        must = case.get("must_contain") or []
        if not must:
            continue
        session = f"_eval_{i}"
        clear_history(session)
        result = query(case["question"], session_id=session)
        answer = result["answer"].lower()
        ok = all(m.lower() in answer for m in must)
        cited = bool(result["sources"])
        passed += ok and cited
        mark = "PASS" if (ok and cited) else "FAIL"
        print(f"  [{mark}] {case['question'][:60]}")
        if not ok:
            missing = [m for m in must if m.lower() not in answer]
            print(f"         answer missing: {missing}")
        if not cited:
            print("         no sources cited")
        clear_history(session)
    total = sum(1 for c in cases if c.get("must_contain"))
    print(f"\n  E2E score: {passed}/{total}\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Run knowledge-base evals")
    parser.add_argument("--e2e", action="store_true", help="also run full agent answers (LLM cost)")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    cases = load_dataset()
    eval_retrieval(cases, args.top_k)
    if args.e2e:
        eval_e2e(cases)


if __name__ == "__main__":
    main()
