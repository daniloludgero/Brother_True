import torch
import torch.nn as nn

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_DISPONIVEL = True
except ImportError:
    _SENTENCE_TRANSFORMERS_DISPONIVEL = False


class EncoderSemantico:
    """Encoder de linguagem leve (Apache 2.0). Converte texto em vetor denso (384-d)."""

    def __init__(self, modelo_nome: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        if not _SENTENCE_TRANSFORMERS_DISPONIVEL:
            raise ImportError(
                "sentence-transformers não está instalado. "
                "Rode: pip install sentence-transformers --break-system-packages"
            )
        self.modelo = SentenceTransformer(modelo_nome, device=device)
        self.dim_saida = self.modelo.get_sentence_embedding_dimension()

    def codificar(self, texto: str) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.modelo.encode(texto, convert_to_tensor=True)
        return embedding


class ProjetorSemantico(nn.Module):
    """Projeta o vetor semântico para H_t (intensidade), eta_t (ruído base) e contexto latente."""

    def __init__(self, dim_entrada: int = 384):
        super().__init__()
        self.proj_h = nn.Linear(dim_entrada, 1)
        self.proj_eta = nn.Linear(dim_entrada, 1)
        self.proj_contexto = nn.Linear(dim_entrada, 64)

    def forward(self, vec_sem: torch.Tensor):
        h = torch.sigmoid(self.proj_h(vec_sem)) * 5.0
        eta_base = torch.tanh(self.proj_eta(vec_sem)) * 2.0
        contexto = self.proj_contexto(vec_sem)
        return h.squeeze(), eta_base.squeeze(), contexto
