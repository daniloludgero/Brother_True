# Direção de Arte e Identidade Visual — BrotherTrue ("Irmão de Verdade")

**Padrão de Engenharia Visual:** Portfólio AAA Internacional + W3C DTCG (3 Camadas) + WCAG 2.1/2.2 Nível AAA  
**Arquivos Gerados:**
- Banner de Identidade e Brasão: [brothertrue-brand-hero.svg](file:///mnt/c/Users/Samsung/Downloads/brothertrue/docs/assets/brothertrue-brand-hero.svg)
- Arquitetura Canônica de 8 Nós: [brothertrue-pipeline-8-nos.svg](file:///mnt/c/Users/Samsung/Downloads/brothertrue/docs/assets/brothertrue-pipeline-8-nos.svg)
- Dinâmica de Contenção e Halo Homeostático: [energy-collapse-dynamics.svg](file:///mnt/c/Users/Samsung/Downloads/brothertrue/docs/assets/energy-collapse-dynamics.svg)
- Dicionário de Design Tokens (DAG 3 Camadas): [tokens.dtcg.json](file:///mnt/c/Users/Samsung/Downloads/brothertrue/docs/design/tokens.dtcg.json)

---

## 1. Conceito Semiótico e Narrativa de Marca ("O Irmão / O Outro")

O nome **BrotherTrue** opera simultaneamente em duas camadas estratégicas extraídas diretamente das 4 referências visuais:

1. **Dualidade Complementar (*Brother* — Polo Quente $\times$ *True* — Polo Frio):**
   - Nas artes de referência, dois arquétipos opostos e equivalentes coexistem em equilíbrio: um movido a **energia cinética solar (Vermelhão/Âmbar)** e outro regido por **disciplina tática e precisão (Azul Cobalto/Cerúleo)**.
   - No produto, o **Organismo Endógeno** (`QualiaODE`, `Neuroquímica`, `Memória`, `Pensamento`) é o irmão orgânico que sente e projeta trajetórias contínuas; o **Rigor Epistêmico** (`EvidenceGate`, `<thought>`, `DeepSeekCoTParser`) é o parceiro racional que filtra alucinações e entrega a fala limpa ao usuário.
2. **Convergência Simétrica (*A Fusão*):**
   - A 3ª referência (os dois arcos inclinados tocando exatamente o ponto central) dá origem ao **Brasão Oficial do BrotherTrue** em geometria escalonada *pixel-stepped* (`shape-rendering="crispEdges"`): o ponto exato de toque é o **Nó 04 → Nó 05 (Registro Biocognitivo $\to$ Evidence Gate)**, onde a intuição contínua se funde à evidência verificada.
3. **O Halo Dourado (*Homeostase e Limiar Crítico $E_{\text{crit}}$*):**
   - A 4ª referência (as duas silhuetas lado a lado no vazio profundo `#060B1B` sob anéis dourados `#E5B031`) traduz visualmente o invariante matemático central do repositório: a coroa homeostática $E_{\text{crit}}(a) = E_0 + \eta_E a$ que governa e protege o organismo contra explosão de estados.

---

## 2. Paleta Cromática Extraída e Auditada (WCAG 2.2 AA/AAA)

Todas as cores provêm exclusivamente da quantização cromática das 4 referências enviadas, estruturadas em isolamento tripartite (`brand.*` $\to$ `semantic.*` $\to$ `component.*`):

| Papel Semiótico | Token DTCG | Hex sRGB | Origem na Referência | Contraste sobre `#0C1326` |
| :--- | :--- | :---: | :--- | :---: |
| **Void Cosmos (Canvas)** | `brand.void.cosmos` | `#060B1B` | Fundo espacial profundo (Ref. 1 & 4) | — |
| **Tactical Indigo (Cards)** | `brand.void.indigo` | `#0C1326` | Sombra noturna de armadura (Ref. 3 & 4) | — |
| **Solar Ki Amber (*Brother*)** | `brand.brother_warm.solar_amber` | `#FBA049` | Energia luminosa quente (Ref. 4) | **9.00 : 1** (AAA) |
| **Vermilion Drive (Alerta ODE)** | `brand.brother_warm.vermilion_ki` | `#D83800` | Túnica vermelhão intensa (Ref. 1) | **4.62 : 1** (AA) |
| **Cerulean Horizon (*True*)** | `brand.true_cool.cerulean_horizon` | `#31B1E0` | Céu cerúleo nítido (Ref. 2) | **7.47 : 1** (AAA) |
| **Royal Cobalt (Estrutura)** | `brand.true_cool.royal_cobalt` | `#1B2C7E` | Azul cobalto profundo (Ref. 1) | Contorno UI |
| **Halo Gold (Homeostase)** | `brand.synchronicity.halo_gold` | `#E5B031` | Anéis dourados superiores (Ref. 1 & 4) | **9.31 : 1** (AAA) |
| **Fusion Cream (Fagulha)** | `brand.synchronicity.fusion_cream` | `#FAEFC1` | Brilho central de convergência (Ref. 4) | **16.97 : 1** (AAA) |
| **Canyon Sand (Subtexto)** | `brand.synchronicity.desert_ochre` | `#CBB694` | Rocha ocre do cenário (Ref. 2) | **9.38 : 1** (AAA) |

---

## 3. Tipografia, Grid 16-Bit e Regras de Interface

1. **Grid Modular de 16px (`patternUnits` 16×16):** Homenagem limpa e adulta à engenharia de *sprites* 16-bit/32-bit, sem cair em estética infantil ou caricata. Os cantos do Hero Banner utilizam colchetes escalonados (*pixel-stepped brackets*) com `crispEdges`, mantendo o restante dos diagramas em precisão vetorial cirúrgica.
2. **Tipografia Quantitativa:** Uso estrito de pilha monoespaçada técnica (`ui-monospace, SFMono-Regular, Menlo, Consolas`) com `font-variant-numeric: tabular-nums` em todas as leituras de $L, Q, F, E$, `loss_p` e divergência `KL`, impedindo oscilação horizontal (*reflow jank*) no Web Lab.
3. **Zero Violação de Propriedade Intelectual:** Nenhum *sprite* comercial ou personagem de terceiros foi copiado ou embutido no repositório; 100% da procedência artística foi destilada em geometria autoral abstrata, proporção cromática e narrativa conceitual própria do **BrotherTrue**.
