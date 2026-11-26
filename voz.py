from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Voz(nn.Module):
    """
    Decoder Transformer autoral. Recebe o humor (11 dims) e gera texto
    token a token, condicionado via cross-attention a um "token de humor".
    """

    def __init__(self, vocab_size: int = 8192, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 512, max_len: int = 128,
                 dropout: float = 0.1, pad_token_id: int = 0, bos_token_id: int = 1,
                 eos_token_id: int = 2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.humor_proj = nn.Linear(11, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # CORREÇÃO: antes, xavier_uniform_ era aplicado indiscriminadamente a
        # todo parâmetro com dim() > 1, incluindo embeddings — que se
        # beneficiam mais de inicialização normal com desvio pequeno
        # (padrão GPT-2/BERT), não de Xavier (pensada para camadas lineares
        # com fan-in/fan-out simétrico).
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if "token_embedding" in name or "pos_embedding" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def _prepare_memory(self, humor: torch.Tensor) -> torch.Tensor:
        return self.humor_proj(humor).unsqueeze(1)

    def forward(self, input_ids: torch.Tensor, humor: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # CORREÇÃO: proteção explícita contra overflow do índice de posição
        # (antes, exceder max_len quebrava silenciosamente dentro do
        # embedding de posição, com um erro de índice difícil de rastrear).
        if seq_len > self.max_len:
            raise ValueError(f"Sequência de tamanho {seq_len} excede max_len={self.max_len}.")

        pos = torch.arange(0, seq_len, device=device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos)
        tgt = tok_emb + pos_emb

        memory = self._prepare_memory(humor)

        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device)

        output = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.lm_head(output)

    @torch.no_grad()
    def gerar(self, humor: torch.Tensor, max_novos_tokens: int = 64,
              temperatura: float = 1.0, top_k: int = 0, top_p: float = 0.9,
              prompt_ids: Optional[list] = None) -> torch.Tensor:
        """`prompt_ids` (opcional): prefixo de condicionamento (ex.: <thought>
        renderizado). O prefixo é consumido pelo decoder mas NÃO é devolvido:
        a saída contém apenas BOS + tokens novos. Com `prompt_ids=None` o
        comportamento é idêntico ao original."""
        self.eval()
        device = next(self.parameters()).device

        # CORREÇÃO: limita max_novos_tokens para nunca ultrapassar max_len
        # (em vez de deixar o forward() levantar erro no meio da geração).
        max_novos_tokens = min(max_novos_tokens, self.max_len - 1)

        prefixo = [self.bos_token_id]
        if prompt_ids:
            # Orçamento de contexto: BOS + prefixo + novos tokens <= max_len.
            # Truncamento à esquerda preserva a parte mais recente do prefixo.
            orcamento = max(self.max_len - 1 - max_novos_tokens, 0)
            corpo = [int(t) for t in prompt_ids if int(t) != self.bos_token_id]
            prefixo += corpo[-orcamento:] if orcamento > 0 else []
        input_ids = torch.tensor([prefixo], device=device)
        gerados = [self.bos_token_id]

        for _ in range(max_novos_tokens):
            seq_len = input_ids.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device)

            logits = self.forward(input_ids, humor, tgt_mask=tgt_mask)
            logits_ultimo = logits[0, -1, :] / temperatura

            if top_k > 0:
                indices_to_remove = logits_ultimo < torch.topk(logits_ultimo, top_k)[0][..., -1, None]
                logits_ultimo[indices_to_remove] = -1e10

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_ultimo, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits_ultimo[indices_to_remove] = -1e10

            probs = F.softmax(logits_ultimo, dim=-1)
            proximo_id = torch.multinomial(probs, num_samples=1)
            gerados.append(proximo_id.item())

            if proximo_id.item() == self.eos_token_id:
                break

            input_ids = torch.cat([input_ids, proximo_id.unsqueeze(0)], dim=1)

        return torch.tensor(gerados, device=device).unsqueeze(0)
