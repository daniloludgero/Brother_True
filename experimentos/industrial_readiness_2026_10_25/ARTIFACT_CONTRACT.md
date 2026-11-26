# Contrato de artefato v1 — aluno próprio

Um experimento só pode ser reproduzido se estes itens viajarem juntos:

| Item | Identidade e verificação exigida |
|---|---|
| Fontes | Manifesto por arquivo SHA-256 do núcleo lido e da implementação experimental |
| Tokenizer | JSON exato, hash, vocabulário e IDs distintos BOS/EOS/PAD |
| Modelo | `architecture_version`, configuração, pesos completos e SHA-256 |
| Estado | Layout20 versionado e traço do núcleo vinculado à mensagem, colapso verificado, limiar e hash |
| Evidência | Fontes, citações, veredicto por subquestão e proveniência do adjudicador |
| Dados | Split congelado, conteúdo identificado por hash, licença/proveniência e revisão docente do registro completo |
| Treino | Semente, hiperparâmetros, cursor, modelo/otimizador/scheduler/scaler, RNG Python/NumPy/Torch CPU e CUDA |
| Avaliação | Mesmo checkpoint/tokenizer/entradas nas ablações, saídas brutas identificadas, métricas e revisão semântica |

Hashes vinculam os bytes observados. Não autenticam a autoria do professor nem
provam que uma citação sustenta uma conclusão. A revisão real deve preservar
modelo/versão, entrada, saída e critérios; esses registros precisam de auditoria.

Os parâmetros arquiteturais de um checkpoint não podem ser reconstruídos a
partir do nome da pasta. O carregamento de pesos é estrito; incompatibilidade
de tokenizer, dimensão, layout, causalidade ou configuração aborta a execução.

No runner piloto, o arquivo de checkpoint é salvo temporariamente e promovido
por `os.replace`; seu manifesto é escrito depois, também atomicamente. É preciso
que ambos existam e seus hashes confiram. Uma falha entre as duas gravações deixa
um arquivo sem promoção válida. Isso não é uma transação atômica entre arquivos,
nem prova de sincronização durável do servidor Google Drive. A promoção requer
readback no runtime e, em uma etapa posterior, após remontar o Drive.

O runner atual salva estado necessário para uma futura retomada, mas não
implementa `resume`. Apenas armazenamento dos RNGs não demonstra retomada
bit-a-bit. O próximo teste é comparar execução contínua com salvar/carregar e
continuar, mantendo hardware, versões, ordem de dados e determinismo.

## Critérios de promoção

1. Passar regressões causais, padding, inicialização, sensibilidade/gradientes e
   checkpoint do decoder no Colab. Passar formas de tensor não basta.
2. Executar um ensaio pequeno de aprendizado e comparar contra inicialização
   aleatória. Relatar custo real; nenhuma conclusão industrial nessa etapa.
3. Pré-treinar linguagem em corpus amplo aprovado e licenciado; depois realizar
   SFT com estados/evidências variados e alvos linguisticamente válidos.
4. Comparar estado real, ausência dos tokens contínuos, troca contrafactual e
   remoção do resumo simbólico. Documentar exatamente qual caminho foi removido.
5. Avaliar pertinência, correção, memória contextual, calibração, grounding,
   conversa casual e robustez em casos reservados. Diversidade lexical e EOS
   são auxiliares; uma diferença de logits não demonstra melhoria de qualidade.

Nenhum portão nesta versão concede automaticamente um selo comercial.
