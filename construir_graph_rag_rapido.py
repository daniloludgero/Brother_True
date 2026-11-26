"""Constrói o grafo completo sem pré-calcular todos os embeddings.

O MiniLM será usado on-line para reranquear apenas candidatos lexicais e
topicamente relacionados, mantendo latência e uso de RAM compatíveis.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]{4,}")
STOPWORDS = {
    "about", "after", "also", "com", "como", "from", "have", "into",
    "mais", "para", "que", "this", "that", "the", "they", "what",
    "when", "where", "with", "your", "user", "assistant", "please",
}


def clean(text: str) -> str:
    text = re.sub(r"</?(?:user|assistant)>", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def terms(text: str) -> list[str]:
    result = []
    seen = set()
    for token in TOKEN_RE.findall(text.lower()):
        if token not in STOPWORDS and token not in seen:
            seen.add(token)
            result.append(token)
        if len(result) >= 12:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    db_path = args.out / "graph.sqlite3"
    if db_path.exists():
        raise FileExistsError(f"Índice já existe; escolha outro --out: {db_path}")
    records = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            query = clean(item.get("prompt", ""))
            response = item.get("response", "").strip()
            if query and response:
                records.append({"source": item.get("source"), "row": item.get("row"),
                                "query": query, "response": response})
    with (args.out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    db = sqlite3.connect(db_path)
    db.executescript("""
      PRAGMA journal_mode=WAL;
      CREATE TABLE nodes(node_id TEXT PRIMARY KEY,node_type TEXT,label TEXT,text TEXT,source TEXT,row_number INTEGER,record_index INTEGER);
      CREATE TABLE edges(src TEXT NOT NULL,dst TEXT NOT NULL,relation TEXT NOT NULL,weight REAL NOT NULL DEFAULT 1.0);
      CREATE INDEX edges_src_relation ON edges(src,relation);
      CREATE INDEX edges_dst_relation ON edges(dst,relation);
      CREATE TABLE term_index(term TEXT NOT NULL,record_index INTEGER NOT NULL);
      CREATE INDEX term_index_term ON term_index(term);
    """)
    nodes, edges, term_rows = [], [], []
    seen_sources, seen_topics = set(), set()
    for index, item in enumerate(records):
        source = str(item["source"])
        source_id = f"source:{source}"
        if source_id not in seen_sources:
            seen_sources.add(source_id)
            nodes.append((source_id, "source", source, source, source, None, None))
        item_id, answer_id = f"item:{index}", f"answer:{index}"
        nodes.extend([
            (item_id, "question", item["query"], item["query"], source, item["row"], index),
            (answer_id, "answer", item["response"][:4000], item["response"], source, item["row"], index),
        ])
        edges.extend([(item_id, answer_id, "answers", 1.0),
                      (item_id, source_id, "from_source", 1.0)])
        for term in terms(item["query"]):
            topic_id = f"topic:{term}"
            if topic_id not in seen_topics:
                seen_topics.add(topic_id)
                nodes.append((topic_id, "topic", term, term, None, None, None))
            edges.append((item_id, topic_id, "has_topic", 1.0))
            term_rows.append((term, index))
    db.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?)", nodes)
    db.executemany("INSERT INTO edges VALUES (?,?,?,?)", edges)
    db.executemany("INSERT INTO term_index VALUES (?,?)", term_rows)
    db.commit(); db.close()
    manifest = {"type":"local_graph_rag_fast", "records":len(records),
                "nodes":len(nodes), "edges":len(edges), "terms":len(term_rows),
                "semantic_reranker":"all-MiniLM-L6-v2", "reasoning_indexed":False,
                "response_policy":"grounded_retrieval_first"}
    (args.out / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
