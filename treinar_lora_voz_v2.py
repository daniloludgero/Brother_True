"""Lançador compatível do piloto LoRA para MultiheadAttention do PyTorch."""

from . import treinar_lora_voz


treinar_lora_voz.LoRALinear.weight = property(
    lambda self: self.base.weight
)
treinar_lora_voz.LoRALinear.bias = property(
    lambda self: self.base.bias
)


if __name__ == "__main__":
    treinar_lora_voz.main()
