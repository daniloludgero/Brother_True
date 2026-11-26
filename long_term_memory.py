import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


class LongTermMemory:
    def __init__(self, db_path: str = "qualia_ltp.db", max_size: int = 5000):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.max_size = max_size
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ltp (
                    id INTEGER PRIMARY KEY,
                    label TEXT,
                    E REAL, L REAL, Q REAL,
                    neuro_json TEXT,
                    embedding_json TEXT,
                    timestamp TEXT,
                    consolidation_score REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 1,
                    replay_priority REAL DEFAULT 1.0
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ltp_label ON ltp(label)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ltp_time ON ltp(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ltp_priority ON ltp(replay_priority DESC)")

    def consolidate(self, short_term_entry: Dict, embedding_vec: Optional[np.ndarray] = None):
        neuro_str = json.dumps(short_term_entry.get("neuro", {}))
        emb_str = json.dumps(embedding_vec.tolist()) if embedding_vec is not None else "[]"
        label = short_term_entry.get("label", "irrelevante")
        score = short_term_entry.get("score", 0.5)

        with self.conn:
            self.conn.execute("""
                INSERT INTO ltp
                (label, E, L, Q, neuro_json, embedding_json, timestamp, consolidation_score, replay_priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (label, short_term_entry.get("E", 0), short_term_entry.get("L", 0),
                  short_term_entry.get("Q", 0), neuro_str, emb_str,
                  datetime.now().isoformat(), score, score * 1.5))

        self._prune_old_memories()

    def replay(self, num_replays: int = 3) -> List[Dict]:
        cursor = self.conn.cursor()
        # CORREÇÃO: o id agora é selecionado explicitamente, para que o
        # reforço de prioridade abaixo afete apenas a memória replayada
        # (antes, o UPDATE usava `WHERE label = ?`, afetando TODAS as
        # memórias que compartilhassem o mesmo rótulo).
        cursor.execute("""
            SELECT id, label, E, L, Q, neuro_json, replay_priority
            FROM ltp
            ORDER BY replay_priority DESC, access_count DESC
            LIMIT ?
        """, (num_replays * 2,))

        rows = cursor.fetchall()
        replays = []

        for row in rows[:num_replays]:
            mem_id, label, E, L, Q, neuro_json, priority = row
            neuro = json.loads(neuro_json)
            replays.append({
                "id": mem_id, "label": label, "E": E, "L": L, "Q": Q,
                "neuro": neuro, "priority": priority,
            })
            with self.conn:
                self.conn.execute(
                    "UPDATE ltp SET access_count = access_count + 1, replay_priority = replay_priority * 0.98 WHERE id = ?",
                    (mem_id,),
                )

        return replays

    def recall_long_term(self, query_neuro: Dict, top_k: int = 5) -> List[Dict]:
        d = query_neuro.get("dopamina", 0.5)
        s = query_neuro.get("serotonina", 0.5)
        n = query_neuro.get("noradrenalina", 0.3)

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT label, E, L, Q, neuro_json, consolidation_score, access_count
            FROM ltp
            ORDER BY consolidation_score DESC LIMIT 50
        """)
        rows = cursor.fetchall()

        candidates = []
        for row in rows:
            neuro_mem = json.loads(row[4])
            # CORREÇÃO: as chaves usadas aqui ('d','s','n') nunca existiam no
            # JSON armazenado (que usa os nomes completos vindos de
            # NeuroTransmissor.estado_atual()), então a similaridade sempre
            # caía no default e nunca discriminava de fato pela memória.
            sim = self._cosine_sim(
                [d, s, n],
                [neuro_mem.get("dopamina", 0.5), neuro_mem.get("serotonina", 0.5), neuro_mem.get("noradrenalina", 0.3)],
            )
            weight = row[5] * sim * (1 + row[6] * 0.1)
            candidates.append((weight, {"label": row[0], "E": row[1], "L": row[2], "Q": row[3]}))

        candidates.sort(reverse=True, key=lambda x: x[0])
        return [c[1] for c in candidates[:top_k]]

    def _cosine_sim(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def _prune_old_memories(self):
        with self.conn:
            self.conn.execute("""
                DELETE FROM ltp WHERE id NOT IN (
                    SELECT id FROM ltp ORDER BY replay_priority DESC LIMIT ?
                )
            """, (self.max_size,))

    def close(self):
        self.conn.close()
