"""Conferência estática (somente leitura) do pipeline canônico do BrotherTrue.

Não importa modelos nem altera nada: analisa via AST a ordem das chamadas em
`BrotherTrue.falar` e verifica herança/existência dos nós.

Uso: python tests/conferir_arquitetura.py
"""
import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# (nó, padrões de chamada que o evidenciam dentro de falar())
PIPELINE = [
    ("Encoder semântico", ["encoder_sem.codificar", "proj_sem"]),
    ("QualiaODE + Neuroquímica + Memória", ["ode.evolve", "neuro.atualizar", "memory.classify_and_store"]),
    ("Pensamento + Palavras", ["pensamento.process_and_learn", "palavras.forward"]),
    ("Registro biocognitivo estruturado", ["BiocognitiveReasoning"]),
    ("Evidence Gate do RAG", ["evidence_gate.filtrar"]),
    ("Renderer de <thought> simbólico", ["render_thought"]),
    ("Decoder próprio autoregressivo", ["voz.gerar", "voz.gerar_biocognitivo"]),
    ("Resposta verbal limpa", ["DeepSeekCoTParser.parse"]),
]

HERANCA = {  # arquivo: (classe, deve herdar nn.Module?)
    "semantica.py": [("EncoderSemantico", False), ("ProjetorSemantico", True)],
    "ode.py": [("QualiaODE", True)],
    "pensamento.py": [("Pensamento", True)],
    "palavras.py": [("Palavras", True)],
    "voz.py": [("Voz", True)],
    "memoria.py": [("Memory", False)],
    "long_term_memory.py": [("LongTermMemory", False)],
    "models/cot_parser.py": [("BiocognitiveReasoning", False), ("DeepSeekCoTParser", False)],
    "models/evidence_gate.py": [("EvidenceGate", False)],
}


def _nome(n):
    if isinstance(n, ast.Attribute):
        return f"{_nome(n.value)}.{n.attr}"
    if isinstance(n, ast.Name):
        return n.id
    return ""


def chamadas_em_ordem(func):
    out = [(c.lineno, _nome(c.func)) for c in ast.walk(func) if isinstance(c, ast.Call)]
    return sorted(out)


def conferir_pipeline():
    arvore = ast.parse((RAIZ / "brothertrue.py").read_text(encoding="utf-8"))
    falar = next(n for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef) and n.name == "falar")
    calls = chamadas_em_ordem(falar)
    linhas, ok = [], True
    ultima = 0
    for no, padroes in PIPELINE:
        hits = [ln for ln, nm in calls if any(nm.endswith(p) or p in nm for p in padroes)]
        if not hits:
            linhas.append(f"  [AUSENTE ] {no}")
            ok = False
            continue
        primeira = min(hits)
        ordem = "OK" if primeira >= ultima else "FORA DE ORDEM"
        if ordem != "OK":
            ok = False
        linhas.append(f"  [{ordem:^9}] {no} (linha {primeira})")
        ultima = max(ultima, primeira)
    return ok, linhas


def conferir_heranca():
    linhas, ok = [], True
    for arq, classes in HERANCA.items():
        p = RAIZ / arq
        if not p.exists():
            linhas.append(f"  [AUSENTE ] {arq}"); ok = False; continue
        defs = {n.name: n for n in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
                if isinstance(n, ast.ClassDef)}
        for nome, nn_mod in classes:
            c = defs.get(nome)
            if c is None:
                linhas.append(f"  [AUSENTE ] {arq}:{nome}"); ok = False; continue
            bases = [_nome(b) for b in c.bases]
            herda = any(b.endswith("Module") for b in bases)
            st = "OK" if (herda or not nn_mod) else "SEM nn.Module"
            ok &= st == "OK"
            linhas.append(f"  [{st:^9}] {arq}:{nome}({', '.join(bases)})")
    return ok, linhas


if __name__ == "__main__":
    ok1, l1 = conferir_pipeline()
    ok2, l2 = conferir_heranca()
    print("== Pipeline em BrotherTrue.falar ==", *l1, "== Herança dos nós ==", *l2, sep="\n")
    print("\nRESULTADO:", "CONFORME" if ok1 and ok2 else "NÃO CONFORME")
    sys.exit(0 if ok1 and ok2 else 1)
