# CLAUDE.md — sour-water-stripper-rl-agent

Projeto de pesquisa: agente RL (PPO) supervisório para otimização do joelho operacional de uma coluna stripper H₂S/NH₃, usando surrogates DNN como ambiente de treinamento e Aspen Dynamics para homologação.

---

## Fluxo de trabalho com Git

- **Não criar branches** — implementar modificações diretamente na branch atual (`main`)
- **Não estagiar nem fazer commit** — após implementar, parar; o código será verificado manualmente e o commit criado pelo usuário

---

## Ambiente e execução

- **Gerenciador de pacotes**: `uv` (não usar `pip` diretamente)
- **Python do ambiente virtual** (Windows): `.venv\Scripts\python.exe`
- **Todos os scripts devem ser executados a partir da raiz** (`sour-water-stripper-rl-agent/`)

```bash
.venv\Scripts\python.exe 06-agentRL-PPO/scripts/train_ppo.py
.venv\Scripts\python.exe 07-agentRL-PPO-aspen-static/01_simulate_agent.py
```


## Estrutura do projeto

| Diretório | Conteúdo |
|---|---|
| `00-*` | Anotações, desenhos, arquivos de simulação |
| `01-analise_sensibilidade` | Análise de sensibilidade do processo |
| `02-amostragem_dnn` | Geração de amostras para treino DNN |
| `03-simulation_cases` | Casos de simulação para treinamento da DNN |
| `04-analise_dados` | Análise e exploração de dados para treinamento DNN|
| `05-surrogate` | Modelos surrogate DNN (Keras) |
| `05b-detectar_joelho` | Detecção do ponto de joelho |
| `06-agentRL-PPO` | Treinamento e prévia de avaliação do agente PPO |
| `07-agentRL-PPO-aspen-static` | Avaliação dos resultados do agente no Aspen utilizando o modelo estático|

### Arquivos-chave em `06-agentRL-PPO/scripts/`

- `config.py` — hiperparâmetros, faixas operacionais, caminhos de arquivos
- `surrogate_env.py` — ambiente Gymnasium com surrogates Keras
- `train_ppo.py` — treinamento PPO
- `evaluate.py` — avaliação e gráficos

### Dependências entre diretórios

```
05-surrogate/output_files/02_Tbottom_best.keras  ← usado por surrogate_env.py
05-surrogate/output_files/02_Ttop_best.keras     ← usado por surrogate_env.py
05-surrogate/sim_results_normalization-{DATA_ID}.json ← normalização min-max
06-agentRL-PPO/models/PPO_reward{X}_{TAG}_{TS}/ ← modelos usados pelo 07
```

---

## Variáveis do processo (stripper)

| Variável | Papel | Faixa aproximada |
|---|---|---|
| `Qreb` | Ação do agente (setpoint do refervedor) | 2.4–6.0 Gcal/h |
| `Tbottom` | Temperatura no fundo da coluna | ~160–168 °C |
| `Ttop` | Temperatura no topo da coluna | ~60–166 °C |
| `Qfeed`, `Tfeed` | Vazão e temperatura da alimentação da coluna | Perturbações observadas | variável |
| `cH2S`, `cNH3`, `Ecolumn` | Concentrações de contaminantes na alimentação e Eficiência da coluna | Variáveis latentes (não observadas pelo agente) | — |

**Ação**: contínua em `[-1, 1]` → `ΔQreb ∈ [-0.3, +0.3]` Gcal/h

---

## Recompensas implementadas

- **A** (robusta): `r = w1·(Tbottom−Ttop)_norm − w2·Qreb_norm − w3·var(Tbottom)`
- **B** (sensível): `r = (Tbottom−Ttop) / (Qreb + ε)`

Pesos em `config.py` → `REWARD_WEIGHTS_A` e `REWARD_TYPES`.

---

## Convenções de nomenclatura

- Arquivos de modelo do agente RL: `PPO_reward{REWARD}_{RUN_TAG}_{TIMESTAMP}/`
- Resultados de simulação do agente RL: `episodes_results_{RUN}_{REWARD}.csv`
- Pontos para avaliação do agente no Aspen: `sim_points_{RUN}_{REWARD}_{band}.csv` (bands: low, mid, high)
- IDs de run: `RUN_TAG` em `config.py`, `RUN` nos scripts de simulação
