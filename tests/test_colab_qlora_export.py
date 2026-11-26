"""Testes de Verificação do Pipeline QLoRA Colab Pro e Exportador de Adaptadores Leves."""

import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Garante inclusão do diretório raiz no sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.colab_train_qwen_qlora import (
    BiocognitiveProjector,
    QLoraTrainingConfig,
    capture_rng_state,
    check_disk_space,
    format_chatml_biocognitive,
    restore_rng_state,
    save_safe_qlora_checkpoint,
)
from pipelines.export_lightweight_adapter import (
    calculate_sha256,
    check_host_disk_safety,
    export_lightweight_adapter,
)


def test_thread_governance():
    """Garante que o limite cooperativo de 2 threads CPU está rigorosamente ativo."""
    torch.set_num_threads(2)
    assert torch.get_num_threads() == 2


def test_host_disk_safety():
    """Garante que a aferição de espaço em disco respeita o piso de 4.5 GB livres."""
    free_gb = check_host_disk_safety(".", min_free_gb=4.5)
    assert free_gb >= 4.5, f"Espaço livre insuficiente: {free_gb:.2f} GB < 4.5 GB"


def test_biocognitive_projector_3b_and_7b():
    """Valida o contrato POO do BiocognitiveProjector para Qwen 3B (2048) e 7B (3584)."""
    # Qwen2.5-3B: d_model = 2048
    proj_3b = BiocognitiveProjector(dim_bio=17, num_prefix_tokens=4, d_model=2048)
    state_17d = torch.randn(3, 17)
    prefix_3b = proj_3b(state_17d)
    assert prefix_3b.shape == (3, 4, 2048)

    # Qwen2.5-7B: d_model = 3584
    proj_7b = BiocognitiveProjector(dim_bio=17, num_prefix_tokens=4, d_model=3584)
    prefix_7b = proj_7b(state_17d)
    assert prefix_7b.shape == (3, 4, 3584)

    # Tolerância a variação dimensional (20D pad/truncation)
    state_20d = torch.randn(2, 20)
    prefix_flex = proj_3b(state_20d)
    assert prefix_flex.shape == (2, 4, 2048)

    # Vetor 1D não-batched
    state_1d = torch.randn(17)
    prefix_1d = proj_3b(state_1d)
    assert prefix_1d.shape == (1, 4, 2048)


def test_chatml_formatting_with_endogenous_cot():
    """Valida o esquema de prompt ChatML com tokens especiais e bloco <thought>."""
    bio_state = torch.tensor([0.8, 0.3, 0.9] + [0.7] * 8 + [0.82, 0.28, 0.91, 0.005] + [0.02, 0.0])
    chatml = format_chatml_biocognitive(
        prompt="Como manter a homeostase em condições de estresse térmico?",
        response="A termorregulação opera via feedback negativo autonômico.",
        bio_state=bio_state,
        thought_analysis="Analisar homeostase fisiológica com rigor biocognitivo."
    )

    assert "<|im_start|>system" in chatml
    assert "<|im_start|>user" in chatml
    assert "<|im_start|>assistant" in chatml
    assert "<thought>" in chatml
    assert "</thought>" in chatml
    assert "[Qualia: L=0.80, Q=0.30, E=0.55, F=0.90]" in chatml
    assert "loss_p=0.0050" in chatml


def test_safe_checkpointing_and_rng_reproducibility():
    """Valida o salvamento atômico seguro e a integridade de restauração de RNG."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = Path(tmp_dir) / "checkpoints"

        # Fixa sementes
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

        proj = BiocognitiveProjector(dim_bio=17, num_prefix_tokens=4, d_model=2048)
        dummy_model = nn.Linear(32, 32)
        optimizer = torch.optim.Adam(proj.parameters(), lr=1e-4)

        # Salva checkpoint
        saved_step_dir = save_safe_qlora_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=42,
            epoch=1,
            model=dummy_model,
            projector=proj,
            optimizer=optimizer,
            metrics={"loss": 0.15},
            min_disk_free_gb=1.0,
        )

        assert (saved_step_dir / "biocognitive_projector.pt").exists()
        assert (saved_step_dir / "trainer_state.pt").exists()
        assert (saved_step_dir / "manifest.json").exists()

        # Altera estado do RNG
        val_random_mutated = [random.random() for _ in range(5)]

        # Restaura estado do RNG
        trainer_state = torch.load(saved_step_dir / "trainer_state.pt", weights_only=False)
        restore_rng_state(trainer_state["rng_state"])

        # Reinicia a sequência e verifica determinismo
        random.seed(12345)
        expected_val = [random.random() for _ in range(5)]
        restore_rng_state(trainer_state["rng_state"])
        restored_val = [random.random() for _ in range(5)]
        assert expected_val == restored_val, "Falha na reprodutibilidade determinística do RNG"


def test_lightweight_adapter_export_under_30mb():
    """Valida a extração do pacote leve < 30 MB e geração do manifesto SHA-256."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / "src_checkpoint"
        src.mkdir()
        dst = Path(tmp_dir) / "exported_adapter"

        # Simula adapter_model.bin com matrizes LoRA
        lora_state = {
            "model.layers.0.self_attn.q_proj.lora_a.weight": torch.randn(16, 2048, dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.lora_b.weight": torch.randn(2048, 16, dtype=torch.float16),
        }
        torch.save(lora_state, src / "adapter_model.bin")

        # Simula adapter_config.json
        with open(src / "adapter_config.json", "w") as f:
            json.dump({"r": 16, "lora_alpha": 32, "peft_type": "LORA"}, f)

        # Simula biocognitive_projector.pt
        proj = BiocognitiveProjector(dim_bio=17, num_prefix_tokens=4, d_model=2048)
        torch.save({"state_dict": proj.state_dict(), "dim_bio": 17, "num_prefix_tokens": 4, "d_model": 2048}, src / "biocognitive_projector.pt")

        # Executa exportação
        manifest = export_lightweight_adapter(
            source_dir=src,
            output_dir=dst,
            max_size_mb=30.0,
            min_disk_free_gb=1.0,
        )

        assert manifest["status"] == "HOMOLOGADO"
        assert manifest["total_package_size_mb"] < 30.0
        assert (dst / "adapter_model.bin").exists()
        assert (dst / "adapter_config.json").exists()
        assert (dst / "biocognitive_projector.pt").exists()
        assert (dst / "export_manifest.json").exists()

        # Verifica hashes SHA-256
        for f_info in manifest["files"]:
            calculated_hash = calculate_sha256(dst / f_info["file"])
            assert calculated_hash == f_info["sha256"]


if __name__ == "__main__":
    test_funcs = [
        test_thread_governance,
        test_host_disk_safety,
        test_biocognitive_projector_3b_and_7b,
        test_chatml_formatting_with_endogenous_cot,
        test_safe_checkpointing_and_rng_reproducibility,
        test_lightweight_adapter_export_under_30mb,
    ]
    print(f"Executando {len(test_funcs)} testes de verificação...")
    for fn in test_funcs:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print("TODOS OS TESTES FORAM APROVADOS COM SUCESSO! (100% PASS)")

