"""Constrói um GraphRAG local com MiniLM congelado.

O índice semântico fornece as sementes; SQLite preserva fontes, tópicos,
perguntas, respostas e caminhos de relação para expansão posterior.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .resource_governor import ResourceGovernor, ResourcePolicy


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]{4,}")
STOPWORDS = {
    "about", "after", "also", "com", "como", "from", "have", "into",
    "mais", "para", "que", "this", "that", "the", "they", "what",
    "when", "where", "with", "your", "user", "assistant", "please",
}


def retrieval_text(prompt: str) -> str:
    text = re.sub(r"</?(?:user|assistant)>", " ", prompt, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def topics(text: str) -> list[str]:
    values = []
    seen = set()
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        values.append(token)
        if len(values) >= 8:
            break
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    db_path = args.out / "graph.sqlite3"
    if db_path.exists():
        raise FileExistsError(f"Índice já existe; escolha outro --out: {db_path}")

    records = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            query = retrieval_text(item.get("prompt", ""))
            response = item.get("response", "").strip()
            if query and response:
                records.append({
                    "source": item.get("source"), "row": item.get("row"),
                    "query": query, "response": response,
                })

    torch.set_num_threads(2)
    model = SentenceTransformer(str(args.minilm), device="cpu",
                                local_files_only=True)
    model.eval()
    governor = ResourceGovernor(ResourcePolicy(cpu_limit_percent=75,
                                                ram_limit_percent=90,
                                                sleep_seconds=1.0))
    vectors = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        vectors.append(model.encode(
            [item["query"] for item in batch],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False))
        if (start // args.batch_size + 1) % 25 == 0 or start + args.batch_size >= len(records):
            print(json.dumps({"embedded": min(start + args.batch_size, len(records)),
                              "total": len(records), **governor.throttle()},
                             ensure_ascii=False), flush=True)
    matrix = np.concatenate(vectors, axis=0)
    np.save(args.out / "embeddings.npy", matrix)
    with (args.out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    connection = sqlite3.connect(db_path)
    connection.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT,
            text TEXT, source TEXT, row_number INTEGER, record_index INTEGER
        );
        CREATE TABLE edges (
            src TEXT NOT NULL, dst TEXT NOT NULL, relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0
        );
        CREATE INDEX edges_src_relation ON edges(src, relation);
        CREATE INDEX edges_dst_relation ON edges(dst, relation);
        CREATE INDEX nodes_record_index ON nodes(record_index);
    """)
    node_rows = []
    edge_rows = []
    sources = sorted({str(item["source"]) for item in records})
    for source in sources:
        node_rows.append((f"source:{source}", "source", source, source, source, None, None))
    topic_seen = set()
    for index, item in enumerate(records):
        item_id = f"item:{index}"
        answer_id = f"answer:{index}"
        source_id = f"source:{item['source']}"
        node_rows.extend([
            (item_id, "question", item["query"], item["query"], item["source"], item["row"], index),
            (answer_id, "answer", item["response"][:4000], item["response"], item["source"], item["row"], index),
        ])
        edge_rows.extend([
            (item_id, answer_id, "answers", 1.0),
            (item_id, source_id, "from_source", 1.0),
        ])
        for topic in topics(item["query"]):
            topic_id = f"topic:{topic}"
            if topic_id not in topic_seen:
                topic_seen.add(topic_id)
                node_rows.append((topic_id, "topic", topic, topic, None, None, None))
            edge_rows.append((item_id, topic_id, "has_topic", 1.0))
    connection.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", node_rows)
    connection.executemany("INSERT INTO edges VALUES (?, ?, ?, ?)", edge_rows)
    connection.commit()
    connection.close()
    manifest = {
        "type": "local_graph_rag", "model": "all-MiniLM-L6-v2",
        "dimension": int(matrix.shape[1]), "records": len(records),
        "nodes": len(node_rows), "edges": len(edge_rows),
        "reasoning_indexed": False, "response_policy": "grounded_retrieval_first",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
