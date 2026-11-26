import os
import sqlite3
from datetime import datetime
from typing import Dict, Optional

import numpy as np

try:
    import hnswlib
    _HNSWLIB_DISPONIVEL = True
except ImportError:
    _HNSWLIB_DISPONIVEL = False


class _NumpyIndex:
    """Fallback exato para bases pequenas quando hnswlib não está disponível."""

    def __init__(self, space: str, dim: int):
        self.dim = dim
        self.vectors = {}

    def init_index(self, **kwargs):
        return None

    def add_items(self, vectors, ids):
        for vector, item_id in zip(np.asarray(vectors), np.asarray(ids)):
            self.vectors[int(item_id)] = np.asarray(vector, dtype=np.float32)

    def get_current_count(self):
        return len(self.vectors)

    def knn_query(self, query, k=1):
        query_vec = np.asarray(query, dtype=np.float32).reshape(-1, self.dim)[0]
        query_norm = np.linalg.norm(query_vec)
        ranked = []
        for item_id, vector in self.vectors.items():
            similarity = float(
                np.dot(query_vec, vector) /
                (query_norm * np.linalg.norm(vector) + 1e-8)
            )
            ranked.append((1.0 - similarity, item_id))
        ranked.sort(key=lambda item: item[0])
        selected = ranked[:k]
        return (
            np.asarray([[item_id for _, item_id in selected]], dtype=np.int64),
            np.asarray([[distance for distance, _ in selected]], dtype=np.float32),
        )

    def save_index(self, path):
        # O SQLite continua sendo a fonte persistente; o índice é reconstruído
        # por _load_from_db na próxima inicialização.
        return None


# CORREÇÃO: antes era `{ ... }` — um placeholder nunca preenchido, que
# faria `classify_and_store` quebrar em qualquer execução real.
# Vetores em (dopamina, serotonina, noradrenalina), valores ilustrativos
# e claramente ajustáveis — não derivados de psicometria validada, apenas
# um ponto de partida funcional coerente com a intenção de cada rótulo.
EMOTION_MATRIX_PADRAO: Dict[str, list] = {
    "gostei":      [0.80, 0.70, 0.30],
    "magoei":      [0.20, 0.20, 0.60],
    "lembrarei":   [0.50, 0.50, 0.20],
    "pensarei":    [0.50, 0.40, 0.50],
    "irrelevante": [0.20, 0.50, 0.10],
    "irritou":     [0.20, 0.20, 0.80],
    "medo":        [0.10, 0.20, 0.90],
    "impulso":     [0.80, 0.30, 0.70],
    "interesse":   [0.70, 0.50, 0.50],
    "paixao":      [0.90, 0.60, 0.70],
}


class Memory:
    def __init__(self, db_path: str = "qualia_memory.db", dim: int = 3,
                 max_elements: int = 10000, emotion_matrix: Optional[Dict] = None):
        self.db_path = db_path
        self.dim = dim
        self.emotion_matrix = emotion_matrix or EMOTION_MATRIX_PADRAO

        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)

        index_class = hnswlib.Index if _HNSWLIB_DISPONIVEL else _NumpyIndex
        self.hnsw = index_class(space="cosine", dim=dim)
        self.hnsw.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.id_counter = 0
        self.memory_map = {}

        self.conn = sqlite3.connect(db_path)
        self._create_table_and_indexes()
        self._load_from_db()

    def _create_table_and_indexes(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    label TEXT,
                    E REAL, L REAL, Q REAL,
                    dopamina REAL, serotonina REAL, noradrenalina REAL,
                    timestamp TEXT, score REAL
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON memories(label)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_neuro ON memories(dopamina, serotonina, noradrenalina)")

    def _load_from_db(self):
        cursor = self.conn.cursor()
        # Ordem das colunas: id(0), dopamina(1), serotonina(2), noradrenalina(3),
        # label(4), E(5), L(6), Q(7), score(8), timestamp(9)
        cursor.execute("SELECT id, dopamina, serotonina, noradrenalina, label, E, L, Q, score, timestamp FROM memories")
        for row in cursor.fetchall():
            vec = np.array([row[1], row[2], row[3]], dtype=np.float32)
            self.hnsw.add_items(vec.reshape(1, -1), np.array([row[0]]))
            # CORREÇÃO: antes lia 'label': row[1] (que é a dopamina) e
            # 'E': row[4] (que é o label, uma string) — índices trocados.
            self.memory_map[row[0]] = {
                "label": row[4], "E": row[5], "L": row[6], "Q": row[7],
                "score": row[8], "timestamp": row[9],
            }
            self.id_counter = max(self.id_counter, row[0] + 1)

    def classify_and_store(self, neuro_state: Dict, E: float, L: float, Q: float):
        d = neuro_state.get("dopamina", 0.5)
        s = neuro_state.get("serotonina", 0.5)
        n = neuro_state.get("noradrenalina", 0.3)
        input_vec = np.array([d, s, n], dtype=np.float32)

        best_label = "irrelevante"
        best_score = -1.0
        for label, weights in self.emotion_matrix.items():
            w_vec = np.array(weights, dtype=np.float32)
            score = np.dot(input_vec, w_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(w_vec) + 1e-8)
            if score > best_score:
                best_score = score
                best_label = label

        mem_id = self.id_counter
        self.hnsw.add_items(input_vec.reshape(1, -1), np.array([mem_id]))
        self.memory_map[mem_id] = {"label": best_label, "E": E, "L": L, "Q": Q, "score": best_score}

        with self.conn:
            self.conn.execute("""
                INSERT INTO memories (id, label, E, L, Q, dopamina, serotonina, noradrenalina, timestamp, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (mem_id, best_label, E, L, Q, d, s, n, datetime.now().isoformat(), best_score))

        self.id_counter += 1
        return best_label, best_score

    def recall_ponderado(self, neuro_state: Dict, top_k: int = 5) -> Optional[Dict]:
        # CORREÇÃO: hnswlib lança erro se k > nº de elementos indexados, ou
        # se o índice estiver vazio — guardamos os dois casos.
        n_elementos = self.hnsw.get_current_count()
        if n_elementos == 0:
            return None
        k = min(top_k, n_elementos)

        d = neuro_state.get("dopamina", 0.5)
        s = neuro_state.get("serotonina", 0.5)
        n = neuro_state.get("noradrenalina", 0.3)
        query_vec = np.array([d, s, n], dtype=np.float32).reshape(1, -1)

        labels, distances = self.hnsw.knn_query(query_vec, k=k)

        candidates = []
        for i, lbl in enumerate(labels[0]):
            if lbl not in self.memory_map:
                continue
            meta = self.memory_map[lbl]
            sim = 1.0 - distances[0][i]
            if sim < 0.65:
                continue
            weight = meta["score"] * sim
            candidates.append((weight, meta))

        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]

    def close(self):
        self.conn.close()
        try:
            self.hnsw.save_index(self.db_path.replace(".db", "_hnsw.bin"))
        except Exception:
            pass
