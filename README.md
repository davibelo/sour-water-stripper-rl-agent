# sour-water-stripper-rl-agent

Projeto de pesquisa (Mestrado EPQB) sobre um **agente de Reinforcement Learning supervisório** para otimização do ponto operacional de uma **coluna stripper de águas ácidas (H₂S/NH₃)**.

O agente ajusta o setpoint de carga térmica do refervedor (`Qreb`) para conduzir a coluna ao seu **joelho operacional** — máxima eficiência térmica de stripping com mínimo consumo energético, **sem medir diretamente H₂S/NH₃**. O treinamento é feito contra **modelos surrogates DNN** (que substituem o simulador rigoroso para acelerar os episódios), e a **validação** é feita de forma **estática no Aspen**: aplicam-se os `Qreb` finais determinados pelo agente, coletam-se as concentrações de H₂S/NH₃ resultantes e o número de passos que o agente fez para chegar no último estado estacionário.

> **Estado atual (V3 — implementado):** os experimentos principais são PPO e DDPG,
> ambos com hiperparâmetros e pesos da recompensa otimizados via **Optuna**, validados
> no **Aspen estático**. A validação **dinâmica** no Aspen Dynamics foi **concebida e
> planejada** (ver [`06-agentRL-PPO/README.md`](06-agentRL-PPO/README.md)), mas, pela
> complexidade, **não foi realizada**: em seu lugar fez-se **validação estática em
> passos** — avaliando quantos passos estáticos o agente precisa para levar a planta de
> um estado estacionário a outro, otimizado (pastas `07`–`17`). Ver
> [`00-anotacoes-discussoes-chat/projetoV3.md`](00-anotacoes-discussoes-chat/projetoV3.md)
> e a apresentação de resultados em
> [`00-anotacoes-documentos/Apresentação Agente RL rev0.pptx`](00-anotacoes-documentos/Apresentação%20Agente%20RL%20rev0.pptx).

---

## Visão geral do pipeline

```
01 Análise de        02 Amostragem        03 Simulação          04 Análise
   sensibilidade  →     LHS            →     (Aspen)         →     de dados
   (quais vars)         (sim_cases)          (sim_results)         (limpeza/EDA)
                                                                      │
                                                                      ▼
   Validação estática    ←     Agentes RL (PPO/DDPG)   ←    05 Surrogate DNN
   no Aspen (cH₂S,             + Optuna (HPO)                (Tbottom, Ttop)
    cNH₃)                      10/12/...                     05b Detecção do joelho
```

1. **Identificar** quais variáveis afetam o processo (análise de sensibilidade).
2. **Amostrar** o espaço de 7 entradas via Latin Hypercube Sampling (LHS).
3. **Simular** os casos amostrados no Aspen até regime estacionário e consolidar os resultados.
4. **Analisar/limpar** os dados e preparar os conjuntos de treino.
5. **Treinar surrogates DNN** que preveem `Tbottom` e `Ttop` a partir das 7 entradas; detectar o joelho.
6. **Treinar agentes RL** (PPO/DDPG) usando os surrogates como ambiente Gymnasium, com **Optuna** otimizando hiperparâmetros e os pesos da recompensa.
7. **Validar** estaticamente no Aspen: simular episódios → `Qreb` final → rodar o Aspen e coletar `cH2S`/`cNH3`.

---

## O processo (coluna stripper)

| Variável | Papel | Faixa |
|---|---|---|
| `Qreb` | **Ação do agente** (setpoint do refervedor) | 2,4–6,0 Gcal/h |
| `Qfeed` | Vazão da alimentação (observada) | 47 500–52 500 kg/h |
| `Tfeed` | Temperatura da alimentação (observada) | 118,95–125,05 °C |
| `Pcolumn` | Pressão da coluna (observada) | 5,7–6,3 kgf/cm² |
| `Tbottom` | Temperatura no fundo (saída do surrogate) | ~160–168 °C |
| `Ttop` | Temperatura no topo (saída do surrogate) | ~60–166 °C |
| `cH2S` | Concentração de H₂S na carga (**latente**) | 875–3 500 ppm |
| `cNH3` | Concentração de NH₃ na carga (**latente**) | 750–3 000 ppm |
| `Ecolumn` | Eficiência de Murphree (**latente**) | 66,5–73,5 % |

- **Estado interno do ambiente** (7 vars): `[Qfeed, Tfeed, cH2S, cNH3, Qreb, Pcolumn, Ecolumn]`.
- **Observação do agente** (6 vars): `[Qfeed, Tfeed, Qreb, Pcolumn, Tbottom, Ttop]`. Formulação **POMDP**: `cH2S`, `cNH3` e `Ecolumn` influenciam o surrogate mas **não** são observadas; são randomizadas por episódio para forçar robustez. `Pcolumn` é observada, mas também randomizada entre episódios (≈ constante dentro de cada um).
- **Ação incremental**: contínua em `[-1, 1]` → `ΔQreb = a × 0.3` Gcal/h; `Qreb(t+1) = clip(Qreb(t) + ΔQreb, 2.4, 6.0)`. Evita saltos bruscos e permite percorrer a faixa em ~12 passos (episódio = 30 passos).

### Recompensa

Apenas a **Recompensa A** foi usada no treinamento:

```
r = w1·(Tbottom − Ttop)_norm − w2·Qreb_norm − w3·var(Tbottom)
```

Maximiza o gradiente térmico útil (proxy de eficiência de stripping), penaliza consumo energético e instabilidade de `Tbottom`. Violação de limites de temperatura → penalidade `−50` e terminação do episódio. Os pesos `w1/w2/w3` foram **otimizados via Optuna**; os valores reportados são `w1 = 0,105`, `w2 = 0,9`, `w3 = 0,6`.

> A **Recompensa B** (`r = (Tbottom − Ttop) / (Qreb + ε)`) está definida em `config.py` mas **não foi utilizada** no treinamento. Tipo ativo e pesos ficam em `REWARD_TYPES` e `REWARD_WEIGHTS_A`.

---

## Estrutura do repositório

As pastas são numeradas na ordem do pipeline. Pastas `0X-agentRL-*` treinam agentes; a pasta seguinte (`0X+1-...-aspen-test`) homologa o agente correspondente no Aspen.

| Diretório | Conteúdo |
|---|---|
| `00-*` | Anotações, documentos (artigo/dissertação), desenhos e arquivos de simulação |
| `01-analise_sensibilidade` | Análise de sensibilidade do processo |
| `02-amostragem_dnn` | Geração de amostras (LHS) para treino da DNN |
| `03-simulation_cases` | Scripts de simulação em lote no Aspen e consolidação dos resultados |
| `04-analise_dados` | Análise exploratória e limpeza dos dados |
| `05-surrogate` | Treino dos surrogates DNN (Keras) + normalização min-max |
| `05b-detectar_joelho` | Detecção do ponto de joelho |
| `06-agentRL-PPO` | Agente **PPO** (base) |
| `07-agentRL-PPO-aspen-test` | Homologação do agente PPO no Aspen |
| `08-agentRL-DDPG` / `09-...-aspen-test` | Agente **DDPG** (base) e homologação |
| `10-agentRL-PPO-Opt` / `11-...-aspen-test` | PPO com **otimização de hiperparâmetros (Optuna)** e homologação |
| `12-agentRL-DDPG-Opt` / `13-...-aspen-test` | DDPG com Optuna e homologação |
| `14-agentRL-PPO-Opt2` / `15-...-aspen-test` | PPO Optuna (2ª rodada) e homologação |
| `16-agentRL-DDPG-Opt2` / `17-...-aspen-test` | DDPG Optuna (2ª rodada) e homologação |

Cada pasta de agente segue o mesmo padrão interno:

```
0X-agentRL-.../
├── scripts/
│   ├── config.py          # Hiperparâmetros, faixas operacionais, caminhos
│   ├── surrogate_env.py   # Ambiente Gymnasium com surrogates Keras
│   ├── train_*.py         # Treinamento (PPO/DDPG; variantes Opt usam Optuna)
│   ├── evaluate.py        # Avaliação e gráficos (prévia no surrogate)
│   └── tensorboard_launch.py
├── models/                # Modelos treinados (gerado pelo treino)
│   └── PPO_rewardX_TAG_TIMESTAMP/
│       ├── best_model.zip         # Melhor checkpoint (EvalCallback)
│       ├── final_model.zip        # Modelo ao final do treino
│       ├── vec_normalize.pkl      # Estatísticas de normalização do VecEnv
│       └── training_meta.json     # Metadados do treino
└── results/
    ├── figures/
    └── logs/              # Logs de treino / TensorBoard
```

> Detalhes de treino e avaliação em [`06-agentRL-PPO/README.md`](06-agentRL-PPO/README.md).
> Observação: aquele README descreve o **conceito/plano** de integração com o
> simulador **dinâmico** (não executado); a validação efetiva foi estática (ver abaixo).

### Otimização de hiperparâmetros (Optuna) e matriz de experimentos

As variantes `*-Opt` e `*-Opt2` usam **Optuna** em um protocolo de duas fases:

1. **Busca**: ~50 trials × 150 000 timesteps, explorando hiperparâmetros do agente **e** os pesos `w1/w2/w3` da recompensa.
2. **Retreinamento**: melhor configuração treinada por 500 000 timesteps.

A comparação-alvo do estudo é **PPO-Opt (`10`) vs DDPG-Opt (`12`)**, ambos com Recompensa A e pesos otimizados:

| Diretório | Algoritmo | Observação |
|---|---|---|
| `10-agentRL-PPO-Opt` | PPO + Optuna | experimento principal |
| `12-agentRL-DDPG-Opt` | DDPG + Optuna (ruído Ornstein-Uhlenbeck) | experimento principal |
| `06` / `08` | PPO / DDPG (base) | hiperparâmetros fixos (sem Optuna) |
| `14` / `16` | PPO / DDPG Optuna (2ª rodada) | variações `Opt2` |

> Modelos de *trials* intermediários do Optuna não são versionados (ver `.gitignore`); apenas o melhor é mantido.

### Dependências entre diretórios

```
05-surrogate/output_files/02b_Tbottom_best.keras      ← usado por surrogate_env.py
05-surrogate/output_files/02b_Ttop_best.keras         ← usado por surrogate_env.py
05-surrogate/sim_results_normalization-{DATA_ID}.json ← normalização min-max
0X-agentRL-.../models/PPO_rewardX_{TAG}_{TS}/         ← usado na homologação Aspen
```

(Os sufixos `MODEL_ID`/`DATA_ID`, ex.: `02b`/`04`, são definidos em cada `config.py`.)

---

## Ambiente e execução

- **Gerenciador de pacotes**: [`uv`](https://docs.astral.sh/uv/) (não usar `pip` diretamente).
- **Python**: 3.11 (ver `.python-version`).
- **Todos os scripts devem ser executados a partir da raiz do repositório.**

### Setup

```bash
uv sync
```

Isso cria o `.venv` e instala as dependências travadas em `uv.lock`
(stable-baselines3, gymnasium, tensorflow, optuna, scikit-learn, matplotlib,
seaborn, pandas, pywin32 para a interface com o Aspen, etc.).

### Treinar um agente (ex.: PPO base)

```bash
.venv\Scripts\python.exe 06-agentRL-PPO/scripts/train_ppo.py
```

As configurações de cada run (tipo de recompensa, `TOTAL_TIMESTEPS`, `SEED`,
`RUN_TAG`, hiperparâmetros) são editadas diretamente no `config.py` daquela pasta.

### Avaliar (prévia no surrogate)

```bash
.venv\Scripts\python.exe 06-agentRL-PPO/scripts/evaluate.py
```

### Acompanhar o treino (TensorBoard)

```bash
.venv\Scripts\python.exe 06-agentRL-PPO/scripts/tensorboard_launch.py
# abrir http://localhost:6006
```

> **Nota (Windows + uv):** use `tensorboard_launch.py`; o `tensorboard.exe`
> gerado pelo `uv` falha ("Failed to canonicalize script path"). O TensorBoard
> 2.20 também depende de `pkg_resources` — por isso `setuptools<81` está fixado
> nas dependências.

### Validar no Aspen (estático)

A validação é **estática** (sem Aspen Dynamics): avalia-se quantos **passos estáticos**
o agente precisa para levar a planta de um estado estacionário a outro, otimizado. Tem
dois passos, cobertos pelos scripts em `0X+1-...-aspen-test/`:

```bash
# 1. Simula episódios com o agente (surrogate como ambiente) → Qreb final
#    Episódios em 3 bandas de Qreb inicial: low, mid, high
.venv\Scripts\python.exe 07-agentRL-PPO-aspen-test/01_simulate_agent.py

# 2. Roda o Aspen (COM/Win32, instâncias paralelas) com o Qreb final de cada
#    episódio até convergir, coletando cH2S/cNH3 de saída, Tbottom, Ttop
.venv\Scripts\python.exe 07-agentRL-PPO-aspen-test/02_simulate_aspen.py

# 3. Gráficos/histogramas dos resultados
.venv\Scripts\python.exe 07-agentRL-PPO-aspen-test/03_plot_histograms.py
```

---

## Convenções de nomenclatura

- Modelos do agente: `PPO_reward{REWARD}_{RUN_TAG}_{TIMESTAMP}/`
- Resultados de simulação do agente: `episodes_results_{RUN}_{REWARD}.csv`
- Pontos para avaliação no Aspen: `sim_points_{RUN}_{REWARD}_{band}.csv`
  (bandas: `low`, `mid`, `high` — faixas inicial de `Qreb`)
- IDs de run: `RUN_TAG` em `config.py`; `RUN` nos scripts de simulação.

---

## Normalização

Os surrogates foram treinados com normalização **min-max para `[0, 1]`**. O
`surrogate_env.py` normaliza as entradas antes de chamar os modelos Keras e
desnormaliza as saídas para valores de engenharia. Adicionalmente, o
`VecNormalize` do stable-baselines3 aplica normalização adaptativa sobre as
observações — por isso o `vec_normalize.pkl` é **obrigatório** na homologação.

---

## Resultados (resumo)

Resumo dos resultados consolidados na apresentação
[`Apresentação Agente RL rev0.pptx`](00-anotacoes-documentos/Apresentação%20Agente%20RL%20rev0.pptx).

**Dataset e surrogate:**
- Amostragem LHS com **1 000 pontos** sobre as 7 entradas, simulados no Aspen Plus V14 (bloco `RadFrac`).
- Um modelo DNN por saída (`Tbottom`, `Ttop`); divisão treino/validação/teste de 75 %/15 %/15 % (conforme apresentação); hiperparâmetros da DNN otimizados com **Optuna** (nº de camadas 1–5, neurônios 16–256, ativação ReLU/Swish/Tanh/SELU, dropout 0–0,3, L2 1e-6–1e-2, batchnorm on/off).

**Protocolo de teste (estático):** 3 faixas de `Qreb` inicial (`low`, `mid`, `high`) × **10 episódios** = **30 episódios**; tolerância de convergência de `Qreb` = 0,01; os `Qreb` finais são simulados no Aspen para obter recuperação de H₂S e perda de NH₃.

**PPO vs DDPG:**

| Métrica | PPO | DDPG |
|---|---|---|
| Recuperação de H₂S | ~100 % em **25/30** episódios | menor que o PPO |
| Perda máxima de NH₃ | **< 0,7 %** | ~**7 %** |
| Passos até convergir | no máx. **11** (nos 30 episódios) | até **30** iterações |

**Conclusão:** o **PPO teve o melhor desempenho** — maior recuperação de H₂S, menor perda de NH₃ e convergência em menos passos que o DDPG.

---

## Situação atual e próximos passos

**Implementado:**
- [x] Geração de dataset por LHS + simulação Aspen
- [x] Treino e validação do surrogate DNN (`Tbottom`, `Ttop`)
- [x] Treino RL com Optuna (PPO e DDPG)
- [x] Simulação de episódios com os agentes treinados
- [x] Validação estática no Aspen (`cH2S`, `cNH3`)
- [x] Comparação PPO-Opt vs DDPG-Opt

**Etapas futuras:**
- [ ] Validação dinâmica (Aspen Dynamics) para regime transiente
- [ ] Fine-tuning do agente com dados do Aspen (active learning)
- [ ] Piloto em processo real

---

## Notas

- Fluxo de trabalho Git do projeto: trabalhar direto na branch `main`; não criar
  branches nem fazer commit automaticamente (ver `CLAUDE.md`).
- Arquivos de simulação do Aspen e modelos de *trials* do Optuna são, em parte,
  ignorados pelo Git (ver `.gitignore`); apenas os melhores modelos são versionados.
```#   s o u r - w a t e r - s t r i p p e r - r l - a g e n t  
 