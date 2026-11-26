"""Teste determinístico do índice RAG local."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rag_local import LocalRAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    rag = LocalRAG(args.index, args.minilm)
    for query in args.query:
        print(json.dumps({"query": query, "results": rag.search(query, args.top_k)},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
