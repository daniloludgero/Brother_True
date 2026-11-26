"""Prepara uma amostra determinística do grafo para experimento local."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4096)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.records.open(encoding="utf-8") as source, args.out.open("w", encoding="utf-8") as target:
        for line in source:
            if count >= args.limit:
                break
            item = json.loads(line)
            query = str(item.get("query", "")).strip()
            response = str(item.get("response", "")).strip()
            if not query or not response:
                continue
            target.write(json.dumps({
                "prompt": query,
                "response": response,
                "source": item.get("source"),
                "row": item.get("row"),
            }, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"records": count, "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
