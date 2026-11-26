import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from .ode import QualiaODE
    from .memoria import Memory
    from .long_term_memory import LongTermMemory
    from .pensamento import Pensamento
    from .palavras import Palavras
    from .ciclo_rem import CicloREM
    from .semantica import EncoderSemantico, ProjetorSemantico, _SENTENCE_TRANSFORMERS_DISPONIVEL
    from .utils_texto import SimpleTokenizer
    from .voz import Voz
except (ImportError, ValueError):
    from ode import QualiaODE
    from memoria import Memory
    from long_term_memory import LongTermMemory
    from pensamento import Pensamento
    from palavras import Palavras
    from ciclo_rem import CicloREM
    from semantica import EncoderSemantico, ProjetorSemantico, _SENTENCE_TRANSFORMERS_DISPONIVEL
    from utils_texto import SimpleTokenizer
    from voz import Voz
try:
    from .voz_distribuida import VozDistribuida
except (ImportError, ValueError):
    try:
        from voz_distribuida import VozDistribuida
    except ImportError:
        VozDistribuida = None

try:
    from .models.biocognitive_projector import BiocognitiveProjector, build_biocognitive_vector
    from .models.qwen_adapter import QwenBiocognitiveAdapter
    from .models.cot_parser import BiocognitiveReasoning, DeepSeekCoTParser
except (ImportError, ValueError):
    try:
        from models.biocognitive_projector import BiocognitiveProjector, build_biocognitive_vector
        from models.qwen_adapter import QwenBiocognitiveAdapter
        from models.cot_parser import BiocognitiveReasoning, DeepSeekCoTParser
    except ImportError:
        BiocognitiveProjector = None
        build_biocognitive_vector = None
        QwenBiocognitiveAdapter = None
        BiocognitiveReasoning = None
        DeepSeekCoTParser = None

try:
    from .models.evidence_gate import EvidenceGate
    from .models.thought_renderer import render_thought, render_thought_body
except (ImportError, ValueError):
    from models.evidence_gate import EvidenceGate
    from models.thought_renderer import render_thought, render_thought_body


class BrotherTrue:
    def __init__(self, db_memory: str = "qualia_memory.db", db_ltp: str = "qualia_ltp.db",
                 tokenizer_path: str = "tokenizer_voz.json", device: str = "cpu",
                 usar_voz_distribuida: bool = False, usar_qwen: bool = False,
                 modelo_voz: Optional[torch.nn.Module] = None,
                 qwen_model_path: Optional[str] = None, **ode_kwargs):
        self.device = device
        self.db_memory = db_memory
        self.usar_voz_distribuida = usar_voz_distribuida
        self.usar_qwen = usar_qwen
        self.modelo_voz = modelo_voz
        self.qwen_model_path = qwen_model_path
        self.db_ltp = db_ltp
        self.tokenizer_path = tokenizer_path
        self.ultimo_cot = None
        self.ultimo_raciocinio = None
        self.banco_vetorial = None
        self.banco_vetorial_search_fn = None
        # Evidence Gate do RAG e condicionamento do decoder pelo <thought>.
        self.evidence_gate = EvidenceGate()
        self.ultimo_gate = None
        self.thought_no_decoder = True

        self.ode = QualiaODE(**ode_kwargs).to(device)
        self.neuro = self.ode.neuro

        self.memory = Memory(db_path=db_memory, dim=3)
        self.ltp = LongTermMemory(db_path=db_ltp)

        self.pensamento = Pensamento(ode_model=self.ode, seq_len=10, d_model=8, nhead=2, lr=0.005, device=device)
        self.palavras = Palavras(dim_state=3, latent_dim=2, threshold_kl=0.5, device=device)

        self.rem = None
        self.encoder_sem = None
        self.proj_sem = None
        if _SENTENCE_TRANSFORMERS_DISPONIVEL:
            self.encoder_sem = EncoderSemantico(device=device)
            self.proj_sem = ProjetorSemantico()
            self.rem = CicloREM(neuro=self.neuro, long_term=self.ltp, memory=self.memory, embedding=self.encoder_sem)
        else:
            print("[AVISO] sentence-transformers não instalado. Encoder semântico desabilitado.")

        self.voz = None
        self.tokenizer_voz = None
        self._inicializar_voz()

        self.state_history = {"L": [], "Q": [], "F": [], "E": []}
        self.age = 0.35
        self.contexto_sem = None
        self.contexto_humor = None

        print("[OK] BrotherTrue inicializado.")

    def _inicializar_voz(self):
        try:
            if self.modelo_voz is not None:
                self.voz = self.modelo_voz.to(self.device)
                print("[OK] Modelo de voz customizado ativado.")
                return

            if self.usar_qwen and QwenBiocognitiveAdapter is not None:
                self.voz = QwenBiocognitiveAdapter(
                    model_name_or_path=self.qwen_model_path or "Qwen/Qwen2.5-3B-Instruct",
                    device=self.device
                ).to(self.device)
                print("[OK] QwenBiocognitiveAdapter ativado.")
                return

            if os.path.exists(self.tokenizer_path):
                from tokenizers import Tokenizer
                self.tokenizer_voz = Tokenizer.from_file(self.tokenizer_path)
                vocab_size = self.tokenizer_voz.get_vocab_size()
            else:
                self.tokenizer_voz = SimpleTokenizer()
                vocab_size = 8192

            if self.usar_voz_distribuida and VozDistribuida is not None:
                self.voz = VozDistribuida(
                    vocab_size=vocab_size,
                    d_model=768,
                    nhead=12,
                    num_layers=14,
                    dim_feedforward=2048,
                    max_len=512
                ).to(self.device)
                print("[OK] VozDistribuida ativada (14 camadas, 768D, SwiGLU, Qualia+RAG).")
            else:
                self.voz = Voz(vocab_size=vocab_size, d_model=256, nhead=4, num_layers=3,
                               dim_feedforward=512, max_len=128).to(self.device)
        except Exception as e:
            print(f"[AVISO] Voz não pôde ser inicializada: {e}")
            self.voz = None

    # ============================================================
    # API PÚBLICA
    # ============================================================

    def falar(self, texto: str, max_tokens: int = 64, temperatura: float = 0.9) -> str:
        if self.encoder_sem is None:
            return "[Encoder semântico não disponível. Instale sentence-transformers.]"
        if self.voz is None:
            return "[Voz não inicializada. Treine o decoder primeiro.]"

        vec_sem = self.encoder_sem.codificar(texto).clone()
        H_t, eta_base, contexto_sem = self.proj_sem(vec_sem)
        self.contexto_sem = contexto_sem.detach()

        X0 = self._obter_estado_inicial()
        t_span = torch.linspace(0, 4.0, 15)
        solucao = self.ode.evolve(X0, t_span, lambda t: H_t.item(), self.age, eta_base)

        final_X = solucao[-1]
        L, Q, F = final_X[0], final_X[1], final_X[2]
        E = 0.5 * L + 0.5 * Q

        E_crit = self.ode.E_crit(self.age)
        if E > E_crit:
            novo_X, colapsou = self.ode.apply_collapse(final_X, E, E_crit)
            L, Q, F = novo_X[0], novo_X[1], novo_X[2]
            E = 0.5 * L + 0.5 * Q
            self.neuro.atualizar(E.item(), L.item(), Q.item(), collapsed=True)
        else:
            self.neuro.atualizar(E.item(), L.item(), Q.item(), collapsed=False)

        estado_atual = torch.tensor([L.item(), Q.item(), F.item()])
        predito, loss_p = self.pensamento.process_and_learn(estado_atual)

        neuro_estado = self.neuro.estado_atual()
        resultado_palavras = self.palavras.forward(estado_atual, neuro_estado)

        # Registro do CoT endógeno e calibração epistêmica para o organismo
        self.ultimo_cot = {
            "predito": predito.detach() if hasattr(predito, "detach") else predito,
            "loss_p": float(loss_p),
            "kl": float(resultado_palavras.get("kl", 0.0)),
            "surprise": bool(resultado_palavras.get("surprise", False)),
            "estado_atual": estado_atual.detach(),
        }

        # CORREÇÃO: uma única chamada de classify_and_store/consolidate por
        # turno (antes, Palavras também escrevia na LTP, duplicando entradas
        # com rótulos e scores divergentes). O sinal de "surpresa" de
        # Palavras agora apenas modula o score desta única escrita.
        label, score = self.memory.classify_and_store(neuro_estado, E.item(), L.item(), Q.item())
        score_final = max(score, resultado_palavras["score_sugerido"]) if resultado_palavras["surprise"] else score
        self.ltp.consolidate({
            "label": label, "E": E.item(), "L": L.item(), "Q": Q.item(),
            "neuro": neuro_estado, "score": score_final,
        })

        humor = self._construir_vetor_humor(L, Q, F, neuro_estado)
        bio_state = {"L": L.item(), "Q": Q.item(), "F": F.item(), "E": E.item(), "neuro": neuro_estado}

        # ── Nó: Registro biocognitivo estruturado (ANTES do decoder) ──────────
        # thought_text e clean_response são preenchidos pelos nós seguintes.
        registro = None
        if BiocognitiveReasoning is not None:
            registro = BiocognitiveReasoning(
                thought_text="",
                qualia={"L": L.item(), "Q": Q.item(), "F": F.item(), "E": E.item()},
                neurochemistry=neuro_estado,
                introspective_prediction={"predito": predito, "loss_p": float(loss_p)},
                epistemic_surprise={
                    "kl": float(resultado_palavras.get("kl", 0.0)),
                    "surprise": bool(resultado_palavras.get("surprise", False)),
                },
                clean_response="",
            )

        # ── Nó: Evidence Gate do RAG ───────────────────────────────────────────
        # Consulta opcional ao Banco Vetorial desacoplado
        candidatos_vetoriais = []
        if self.banco_vetorial is not None:
            try:
                if self.banco_vetorial_search_fn is not None:
                    candidatos_vetoriais = self.banco_vetorial_search_fn(self.banco_vetorial, texto, top_k=3)
                elif hasattr(self.banco_vetorial, "buscar"):
                    candidatos_vetoriais = self.banco_vetorial.buscar(texto, top_k=3)
                elif hasattr(self.banco_vetorial, "search"):
                    res = self.banco_vetorial.search(texto, k=3)
                    candidatos_vetoriais = res if isinstance(res, list) else [res]
            except Exception as e:
                print(f"[Aviso Banco Vetorial] Consulta ao banco vetorial falhou: {e}")

        candidatos_ltp = []
        try:
            top_mems = self.ltp.recall_long_term(neuro_estado, top_k=3)
            candidatos_ltp = [f"{m.get('label', '')} (E={m.get('E', 0):.2f})" for m in (top_mems or [])]
        except Exception:
            candidatos_ltp = []

        gate_vetorial = self.evidence_gate.filtrar(candidatos_vetoriais, origem="banco_vetorial")
        gate_ltp = self.evidence_gate.filtrar(candidatos_ltp, origem="ltp")
        textos_vetoriais = gate_vetorial.textos
        evidencias = gate_vetorial.aprovadas + gate_ltp.aprovadas
        self.ultimo_gate = {
            "banco_vetorial": self.evidence_gate.to_dict(gate_vetorial),
            "ltp": self.evidence_gate.to_dict(gate_ltp),
        }

        rag_context_tensor = None
        if gate_ltp.aprovadas and self.encoder_sem is not None:
            try:
                mem_embeddings = [self.encoder_sem.codificar(t).clone().unsqueeze(0) for t in gate_ltp.textos]
                rag_context_tensor = torch.cat(mem_embeddings, dim=0).unsqueeze(0).to(self.device)
            except Exception:
                rag_context_tensor = None

        # ── Nó: Renderer de <thought> simbólico ────────────────────────────────
        formatter = getattr(self.voz, "format_endogenous_thought", None)
        thought_text = render_thought_body(bio_state, self.ultimo_cot, evidencias, formatter=formatter)
        if DeepSeekCoTParser is not None:
            thought_text = DeepSeekCoTParser.sanitize_thought(thought_text)
        thought_block = render_thought(thought_text)
        if registro is not None:
            registro.thought_text = thought_text
        self.ultimo_cot["thought_text"] = thought_text

        # ── Nó: Decoder próprio autoregressivo (condicionado pelo <thought>) ──
        thought_ids = self._tokenizar(thought_block) if self.thought_no_decoder else []

        if hasattr(self.voz, "gerar_biocognitivo"):
            import inspect
            bio_kwargs = dict(
                texto=texto,
                humor=humor,
                predito=predito,
                loss_p=loss_p,
                kl=resultado_palavras.get("kl", 0.0),
                surprise=resultado_palavras.get("surprise", False),
                rag_context=rag_context_tensor,
                rag_textos=textos_vetoriais,
                max_tokens=max_tokens,
                temperatura=temperatura
            )
            if "thought_text" in inspect.signature(self.voz.gerar_biocognitivo).parameters:
                bio_kwargs["thought_text"] = thought_text
            resposta = self.voz.gerar_biocognitivo(**bio_kwargs)
        elif hasattr(self.voz, "construir_memoria_distribuida"):
            prompt_ids = [getattr(self.voz, "bos_token_id", 1)]
            if self.tokenizer_voz and hasattr(self.tokenizer_voz, "encode"):
                try:
                    enc = self.tokenizer_voz.encode(texto, add_special_tokens=False)
                    ids = enc.ids if hasattr(enc, "ids") else list(enc)
                    prompt_ids = [getattr(self.voz, "bos_token_id", 1)] + ids[:16]
                except Exception:
                    prompt_ids = [getattr(self.voz, "bos_token_id", 1)]

            import inspect
            sig = inspect.signature(self.voz.gerar)
            gerar_kwargs = {
                "prompt_ids": prompt_ids,
                "humor": humor,
                "rag_context": rag_context_tensor,
                "max_new_tokens": max_tokens,
                "temperatura": temperatura,
            }
            if "cot_signals" in sig.parameters:
                gerar_kwargs["cot_signals"] = self.ultimo_cot
            if "bio_state" in sig.parameters:
                gerar_kwargs["bio_state"] = bio_state
            if "thought_text" in sig.parameters:
                gerar_kwargs["thought_text"] = thought_text

            ids_resposta = self.voz.gerar(**gerar_kwargs)
            resposta = self._decodificar_tokens(ids_resposta)
        else:
            import inspect
            gerar_kwargs = {"max_novos_tokens": max_tokens, "temperatura": temperatura}
            if thought_ids and "prompt_ids" in inspect.signature(self.voz.gerar).parameters:
                gerar_kwargs["prompt_ids"] = thought_ids
            ids_resposta = self.voz.gerar(humor, **gerar_kwargs)
            resposta = self._decodificar_tokens(ids_resposta.squeeze().tolist())

        # ── Nó: Resposta verbal limpa (Zero-Leakage estilo DeepSeek) ──────────
        if DeepSeekCoTParser is not None:
            thought_from_resp, clean_resp = DeepSeekCoTParser.parse(resposta)
            resposta = clean_resp

        # Registro do raciocínio estruturado no canal de primeira classe.
        # Adaptadores que mantêm o próprio registro (ex.: Qwen) têm precedência.
        if hasattr(self.voz, "ultimo_raciocinio") and self.voz.ultimo_raciocinio is not None:
            self.ultimo_raciocinio = self.voz.ultimo_raciocinio
        elif registro is not None:
            registro.clean_response = resposta
            self.ultimo_raciocinio = registro

        # Atualização do dicionário retrocompatível ultimo_cot com os campos de CoT
        if self.ultimo_cot is not None:
            self.ultimo_cot["thought_text"] = self.ultimo_raciocinio.thought_text if self.ultimo_raciocinio else thought_text
            self.ultimo_cot["clean_response"] = resposta

        self._atualizar_historico(L, Q, F, E)
        self.contexto_humor = humor.detach()

        return resposta

    def obter_ultimo_raciocinio(self) -> Optional[Any]:
        """Retorna o raciocínio estruturado (Chain-of-Thought) de primeira classe do último turno."""
        return self.ultimo_raciocinio

    def conectar_banco_vetorial(self, cliente_vetorial: Any, search_fn: Optional[Any] = None) -> None:
        """Conecta um banco vetorial local (FAISS, LanceDB, sqlite-vec, etc.) para RAG contextual.

        Args:
            cliente_vetorial: Instância ou cliente do banco vetorial.
            search_fn: Função opcional com assinatura (db, query_str, top_k) -> List[str].
        """
        self.banco_vetorial = cliente_vetorial
        self.banco_vetorial_search_fn = search_fn
        nome = getattr(cliente_vetorial, "__class__", type(cliente_vetorial)).__name__
        print(f"[OK] Banco Vetorial conectado com sucesso ao BrotherTrue ({nome}).")

    def ouvir_e_processar(self, texto: str) -> Dict:
        if self.encoder_sem is None:
            return {"erro": "Encoder não disponível"}

        vec_sem = self.encoder_sem.codificar(texto).clone()
        H_t, eta_base, contexto_sem = self.proj_sem(vec_sem)

        X0 = self._obter_estado_inicial()
        t_span = torch.linspace(0, 4.0, 15)
        solucao = self.ode.evolve(X0, t_span, lambda t: H_t.item(), self.age, eta_base)

        final_X = solucao[-1]
        L, Q, F = final_X[0], final_X[1], final_X[2]
        E = 0.5 * L + 0.5 * Q

        E_crit = self.ode.E_crit(self.age)
        if E > E_crit:
            novo_X, colapsou = self.ode.apply_collapse(final_X, E, E_crit)
            L, Q, F = novo_X[0], novo_X[1], novo_X[2]
            E = 0.5 * L + 0.5 * Q
            self.neuro.atualizar(E.item(), L.item(), Q.item(), collapsed=True)
        else:
            self.neuro.atualizar(E.item(), L.item(), Q.item(), collapsed=False)
        neuro_estado = self.neuro.estado_atual()

        estado_atual = torch.tensor([L.item(), Q.item(), F.item()])
        predito, loss_p = self.pensamento.process_and_learn(estado_atual)
        resultado_palavras = self.palavras.forward(estado_atual, neuro_estado)

        humor = self._construir_vetor_humor(L, Q, F, neuro_estado)
        self._atualizar_historico(L, Q, F, E)

        return {
            "L": L.item(), "Q": Q.item(), "F": F.item(), "E": E.item(),
            "neuro": neuro_estado, "humor": humor.squeeze().tolist(),
            "kl": resultado_palavras["kl"], "surprise": resultado_palavras["surprise"],
            "loss_pensamento": loss_p, "contexto_sem": contexto_sem.tolist(),
        }

    def sonhar(self, duracao_rem: int = 20) -> Dict:
        if self.rem is None:
            return {"erro": "CicloREM não inicializado (encoder ausente)."}
        return self.rem.processar_rem(duracao_rem)

    # ============================================================
    # PERSISTÊNCIA
    # ============================================================

    def salvar_checkpoint(self, diretorio: str):
        os.makedirs(diretorio, exist_ok=True)

        torch.save(self.ode.state_dict(), f"{diretorio}/ode.pt")
        torch.save(self.pensamento.state_dict(), f"{diretorio}/pensamento.pt")
        torch.save(self.palavras.state_dict(), f"{diretorio}/palavras.pt")
        if self.voz:
            torch.save(self.voz.state_dict(), f"{diretorio}/voz.pt")

        torch.save(self.state_history, f"{diretorio}/state_history.pt")

        # Fecha conexões para garantir que os arquivos .db no disco estejam
        # consistentes antes de copiar.
        self.memory.close()
        self.ltp.close()

        if os.path.exists(self.db_memory):
            shutil.copy(self.db_memory, f"{diretorio}/qualia_memory.db")
        if os.path.exists(self.db_ltp):
            shutil.copy(self.db_ltp, f"{diretorio}/qualia_ltp.db")

        if self.tokenizer_voz and hasattr(self.tokenizer_voz, "save"):
            self.tokenizer_voz.save(f"{diretorio}/tokenizer_voz.json")

        meta = {"timestamp": datetime.now().isoformat(), "age": self.age}
        with open(f"{diretorio}/meta.json", "w") as f:
            json.dump(meta, f)

        # CORREÇÃO: antes, o objeto ficava com conexões fechadas após salvar
        # (qualquer chamada subsequente a falar()/ouvir_e_processar() falhava
        # até que carregar_checkpoint() fosse chamado manualmente). Agora as
        # conexões são reabertas imediatamente, e o objeto continua usável.
        self.memory = Memory(db_path=self.db_memory, dim=3)
        self.ltp = LongTermMemory(db_path=self.db_ltp)

        print(f"[OK] Checkpoint salvo em: {diretorio}")

    def carregar_checkpoint(self, diretorio: str):
        if not os.path.exists(diretorio):
            print(f"[ERRO] Diretório {diretorio} não encontrado.")
            return

        self.ode.load_state_dict(torch.load(f"{diretorio}/ode.pt", map_location=self.device))
        self.pensamento.load_state_dict(torch.load(f"{diretorio}/pensamento.pt", map_location=self.device))
        self.palavras.load_state_dict(torch.load(f"{diretorio}/palavras.pt", map_location=self.device))
        if self.voz and os.path.exists(f"{diretorio}/voz.pt"):
            self.voz.load_state_dict(torch.load(f"{diretorio}/voz.pt", map_location=self.device))

        self.state_history = torch.load(f"{diretorio}/state_history.pt")

        self.memory.close()
        self.ltp.close()
        if os.path.exists(f"{diretorio}/qualia_memory.db"):
            shutil.copy(f"{diretorio}/qualia_memory.db", self.db_memory)
        if os.path.exists(f"{diretorio}/qualia_ltp.db"):
            shutil.copy(f"{diretorio}/qualia_ltp.db", self.db_ltp)
        self.memory = Memory(db_path=self.db_memory, dim=3)
        self.ltp = LongTermMemory(db_path=self.db_ltp)

        if os.path.exists(f"{diretorio}/tokenizer_voz.json"):
            from tokenizers import Tokenizer
            self.tokenizer_voz = Tokenizer.from_file(f"{diretorio}/tokenizer_voz.json")

        if os.path.exists(f"{diretorio}/meta.json"):
            with open(f"{diretorio}/meta.json", "r") as f:
                meta = json.load(f)
            self.age = meta.get("age", 0.35)

        print(f"[OK] Checkpoint carregado de: {diretorio}")

    # ============================================================
    # AUXILIARES
    # ============================================================

    def _obter_estado_inicial(self) -> torch.Tensor:
        if self.state_history["L"]:
            return torch.tensor([
                self.state_history["L"][-1], self.state_history["Q"][-1], self.state_history["F"][-1],
            ], dtype=torch.float32)
        return torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

    def _construir_vetor_humor(self, L, Q, F, neuro: Dict) -> torch.Tensor:
        humor = torch.tensor([
            L.item(), Q.item(), F.item(),
            neuro["dopamina"], neuro["serotonina"], neuro["noradrenalina"],
            neuro["adrenalina"], neuro["gaba"], neuro["ocitocina"],
            neuro["acetilcolina"], neuro["glutamato"],
        ], dtype=torch.float32).unsqueeze(0)
        return humor.to(self.device)

    def _atualizar_historico(self, L, Q, F, E):
        self.state_history["L"].append(L.item())
        self.state_history["Q"].append(Q.item())
        self.state_history["F"].append(F.item())
        self.state_history["E"].append(E.item())

    def _tokenizar(self, texto: str) -> List[int]:
        """Tokeniza o <thought> para condicionar o decoder. Ignora o
        SimpleTokenizer de fallback, cujo encode() muta o vocabulário e
        alteraria o mapeamento de decode() dos tokens gerados."""
        if self.tokenizer_voz is None or isinstance(self.tokenizer_voz, SimpleTokenizer):
            return []
        try:
            enc = self.tokenizer_voz.encode(texto)
            return list(enc.ids) if hasattr(enc, "ids") else [int(t) for t in enc]
        except Exception:
            return []

    def _decodificar_tokens(self, ids: List[int]) -> str:
        if self.tokenizer_voz is None:
            return "[tokenizer não carregado]"
        try:
            if hasattr(self.tokenizer_voz, "decode"):
                if isinstance(self.tokenizer_voz, SimpleTokenizer):
                    return self.tokenizer_voz.decode(np.array(ids))
                return self.tokenizer_voz.decode(ids)
            return "[tokenizer sem método decode]"
        except Exception as e:
            return f"[erro na decodificação: {e}]"

    # ============================================================
    # CONSULTAS
    # ============================================================

    def estado_emocional(self) -> Dict:
        neuro = self.neuro.estado_atual()
        return {
            "L": self.state_history["L"][-1] if self.state_history["L"] else 1.0,
            "Q": self.state_history["Q"][-1] if self.state_history["Q"] else 1.0,
            "F": self.state_history["F"][-1] if self.state_history["F"] else 0.0,
            "E": self.state_history["E"][-1] if self.state_history["E"] else 1.0,
            "neuro": neuro,
        }

    def historico_memorias(self, top_k: int = 5) -> List[Dict]:
        return self.ltp.recall_long_term(self.neuro.estado_atual(), top_k=top_k)

    def resumir_sessao(self) -> str:
        estado = self.estado_emocional()
        neuro = estado["neuro"]
        emocao_max = max(neuro, key=neuro.get)
        return (f"Estado: L={estado['L']:.2f}, Q={estado['Q']:.2f}, E={estado['E']:.2f}, "
                f"Emoção dominante: {emocao_max} ({neuro[emocao_max]:.2f})")

    def __repr__(self):
        return f"<BrotherTrue: age={self.age}, neuro={self.neuro.estado_atual()}>"
