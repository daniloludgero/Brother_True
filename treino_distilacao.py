import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm


def treinar_voz_com_destilacao(
    brother_true,
    caminho_dataset_logits: str,
    caminho_saida: str = "checkpoints_voz_destil",
    batch_size: int = 8,
    epochs: int = 5,
    lr: float = 5e-4,
    acumular_grad: int = 1,
    alpha_ce: float = 0.5,
    alpha_kl: float = 0.5,
    temperature: float = 2.0,
    device: str = "cpu",
):
    """
    Treina a Voz por destilação (cross-entropy + KL contra logits de um
    professor), com temperatura, na forma padrão de Hinton et al.

    ATENÇÃO — lacuna ainda aberta (não corrigida aqui por falta da etapa
    anterior do pipeline): se `dados_prof[i]` não contiver uma chave
    'humor' com o vetor de 11 dimensões correspondente àquele exemplo,
    o treino cai no fallback de humor zerado abaixo, e a Voz aprende a
    IGNORAR o condicionamento por humor — o canal que distingue este
    sistema de um chatbot comum. Antes de rodar isto em produção, ajuste
    a etapa de geração do dataset de logits do professor para também
    persistir `ouvir_e_processar(texto)['humor']` por exemplo.
    """
    dados_prof = torch.load(caminho_dataset_logits)
    print(f"[OK] Dataset de logits carregado: {len(dados_prof)} exemplos")

    tokenizer = brother_true.tokenizer_voz
    pad_id = tokenizer.token_to_id("<pad>")
    modelo = brother_true.voz.to(device)
    modelo.train()

    optimizer = AdamW(modelo.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1)

    indices = list(range(len(dados_prof)))
    random.shuffle(indices)

    perdas_epoca = []
    os.makedirs(caminho_saida, exist_ok=True)

    for epoca in range(1, epochs + 1):
        modelo.train()
        perda_total = 0.0
        n_batches = 0
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Época {epoca}")

        for i in pbar:
            batch_idx = indices[i:i + batch_size]
            batch = [dados_prof[j] for j in batch_idx]

            input_ids_list = [torch.tensor(item["input_ids"][:-1], dtype=torch.long) for item in batch]
            target_ids_list = [torch.tensor(item["input_ids"][1:], dtype=torch.long) for item in batch]
            logits_prof_list = [torch.tensor(item["logits"][:-1]) for item in batch]

            input_padded = nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            target_padded = nn.utils.rnn.pad_sequence(target_ids_list, batch_first=True, padding_value=pad_id)

            max_len = input_padded.size(1)
            vocab_size = logits_prof_list[0].size(-1)
            logits_prof_padded = torch.zeros(len(batch), max_len, vocab_size)
            for k, lp in enumerate(logits_prof_list):
                seq_len = lp.size(0)
                logits_prof_padded[k, :seq_len] = lp

            # Humor real, se disponível no dataset; caso contrário, fallback
            # zerado (ver aviso na docstring acima).
            humor_batch = torch.stack([
                torch.tensor(item["humor"], dtype=torch.float32) if "humor" in item
                else torch.zeros(11, dtype=torch.float32)
                for item in batch
            ])

            input_padded = input_padded.to(device)
            target_padded = target_padded.to(device)
            logits_prof_padded = logits_prof_padded.to(device)
            humor_batch = humor_batch.to(device)

            # CORREÇÃO: máscara de padding agora é construída e passada ao
            # decoder. Antes, posições de padding eram atendidas pela
            # self-attention como se fossem conteúdo válido; apenas a saída
            # era neutralizada via ignore_index na cross-entropy.
            padding_mask = (input_padded == pad_id)

            logits_aluno = modelo(input_padded, humor_batch, tgt_key_padding_mask=padding_mask)

            ce_loss = F.cross_entropy(
                logits_aluno.reshape(-1, vocab_size), target_padded.reshape(-1), ignore_index=pad_id
            )

            log_probs_aluno = F.log_softmax(logits_aluno / temperature, dim=-1)
            probs_prof = F.softmax(logits_prof_padded / temperature, dim=-1)
            kl_loss = F.kl_div(
                log_probs_aluno, probs_prof, reduction="batchmean", log_target=False
            ) * (temperature ** 2)

            loss = alpha_ce * ce_loss + alpha_kl * kl_loss
            loss = loss / acumular_grad
            loss.backward()

            if (i + 1) % acumular_grad == 0:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            perda_total += loss.item() * acumular_grad
            n_batches += 1
            pbar.set_postfix({"ce": ce_loss.item(), "kl": kl_loss.item()})

        perda_media = perda_total / max(n_batches, 1)
        perdas_epoca.append(perda_media)
        print(f"[OK] Época {epoca}: perda média = {perda_media:.4f}")

        torch.save(modelo.state_dict(), f"{caminho_saida}/voz_ep{epoca}.pt")

    torch.save(modelo.state_dict(), f"{caminho_saida}/voz_destilada_final.pt")
    return perdas_epoca
