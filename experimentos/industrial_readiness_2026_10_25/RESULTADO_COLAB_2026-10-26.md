# Resultado dos testes no Colab — 26/10/2026

O notebook experimental verificou o hash do pacote e os 34 arquivos de fonte
antes da execução. O resultado persistido no Google Drive foi lido novamente:
https://drive.google.com/file/d/19f8GNyC4N8GIqogZ10fJEgVcIJCdx1ep/view?usp=drivesdk

- Contratos de estado, evidência e canais: 16 testes aprovados.
- Contratos do runner: 4 testes aprovados.
- Decoder próprio: 11 testes aprovados, incluindo máscaras causais, padding,
  gradientes, condicionamento biocognitivo e por evidência, geração e checkpoint.
- Total: 31 aprovados; códigos de saída 0 nas três suítes.
- Runtime: PyTorch `2.11.0+cpu`; GPU CUDA indisponível nesta sessão.
- Um teste emitiu aviso de conversão de tensor com gradiente para escalar; ele
  não falhou e o aviso permanece registrado no relatório completo.
- SHA-256 do pacote: `a50832d7ee0ded27e03c8db24d5f945c53fdec793b9e81cfb3e4078ccda7df1b`.

Estas verificações utilizam tensores e um modelo pequeno de teste. Não houve
treinamento linguístico ou avaliação semântica, e os resultados não demonstram
qualidade comercial nem equivalência funcional ao organismo completo.
