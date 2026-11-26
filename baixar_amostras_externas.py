"""Baixa amostras locais dos corpora Hugging Face sem ocupar o disco inteiro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def stream(repo: str, config: str | None, limit: int, out: Path, cache_dir: Path):
    kwargs = {"path": repo, "split": "train", "streaming": True,
              "cache_dir": str(cache_dir)}
    if config:
        kwargs["name"] = config
    dataset = load_dataset(**kwargs)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            handle.write(json.dumps({"text": text, "source": repo,
                                     "config": config}, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--english", type=int, default=100_000)
    parser.add_argument("--code", type=int, default=50_000)
    parser.add_argument("--chats", type=int, default=100_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("Polygl0t/tokenizers", "english", args.english, "tokenizers_english_sample.jsonl"),
        ("Polygl0t/tokenizers", "code", args.code, "tokenizers_code_sample.jsonl"),
        ("asdf98/human-chats-worldwide", None, args.chats, "human_chats_sample.jsonl"),
    ]
    manifest = {"note": "Amostras locais; datasets integrais excedem o disco disponível.", "files": {}}
    for repo, config, limit, filename in jobs:
        path = args.out / filename
        count = stream(repo, config, limit, path, args.cache_dir)
        manifest["files"][filename] = {"records": count, "repo": repo,
                                         "config": config, "limit": limit}
        print(json.dumps({"file": filename, "records": count}, ensure_ascii=False), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
