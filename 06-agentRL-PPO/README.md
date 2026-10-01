# 06 - Agente RL (PPO) — Otimização do Joelho Operacional do Stripper

Agente de Reinforcement Learning (PPO) supervisório que ajusta o setpoint de carga térmica do refervedor (`Q̇_SP`) para conduzir a coluna stripper H₂S/NH₃ ao ponto ótimo de operação ("joelho"), usando modelos surrogates DNN como ambiente de treinamento.

---

## Estrutura de Pastas

```
06-agentRL-PPO/
├── scripts/
│   ├── config.py           # Hiperparâmetros, faixas operacionais, caminhos
│   ├── surrogate_env.py    # Ambiente Gymnasium com surrogates Keras
│   ├── train_ppo.py        # Script de treinamento PPO
│   └── evaluate.py         # Avaliação, gráficos e comparação A vs B
├── models/                 # Modelos RL treinados (gerado pelo treino)
│   └── PPO_rewardX_TAG_TIMESTAMP/
│       ├── best_model.zip        # Melhor modelo (EvalCallback)
│       ├── final_model.zip       # Modelo ao final do treino
│       ├── vec_normalize.pkl     # Estatísticas de normalização do VecEnv
│       └── training_meta.json    # Metadados do treino
├── results/
│   ├── figures/            # Gráficos de avaliação
│   └── logs/               # Logs de treino, TensorBoard, evaluations.npz
└── README.md
```

---

## Pré-requisitos

```bash
uv pip install stable-baselines3 gymnasium tensorflow
```

Os modelos surrogates devem existir em:
- `05-surrogate/output_files/02_Tbottom_best.keras`
- `05-surrogate/output_files/02_Ttop_best.keras`

E o arquivo de normalização em:
- `05-surrogate/sim_results_normalization.json`

---

## Fase 1 — Treinamento

Todos os comandos devem ser executados a partir da raiz do workspace (`sour-water-stripper-rl-agent/`), utilizando o Python do ambiente virtual para garantir que os pacotes instalados via `uv` sejam usados.

As configurações de treinamento (tipos de recompensa, timesteps, seed, hiperparâmetros, etc.) são definidas diretamente no arquivo `config.py`.

### Executar treinamento

```bash
.venv\Scripts\python.exe 06-agentRL-PPO/scripts/train_ppo.py
```

### Principais configurações em `config.py`

| Variável          | Padrão       | Descrição                              |
|-------------------|--------------|----------------------------------------|
| `REWARD_TYPES`    | `["A", "B"]` | Tipos de recompensa a treinar          |
| `TOTAL_TIMESTEPS` | `500_000`    | Total de timesteps de treinamento      |
| `SEED`            | `42`         | Seed para reprodutibilidade            |
| `RUN_TAG`         | `""`         | Tag opcional para identificar a run    |

### Monitoramento com TensorBoard

O TensorBoard é um painel web para acompanhar as métricas do treino (reward, loss, entropia, etc.). Neste projeto, os logs são gravados pelo `train_ppo.py` em `results/logs/tensorboard/` dentro de `06-agentRL-PPO/`. Cada run fica em um subdiretório com o nome `PPO_rewardX_TAG_TIMESTAMP`.

Passos:

1. Inicie o treino:

```bash
.venv\Scripts\python.exe 06-agentRL-PPO/scripts/train_ppo.py
```

2. Em outro terminal, a partir da raiz do workspace (`sour-water-stripper-rl-agent/`), rode o script de lançamento:

```bash
.venv\Scripts\python.exe 06-agentRL-PPO/scripts/tensorboard_launch.py
```

3. Abra o navegador em `http://localhost:6006` e selecione a run desejada.

> **Nota (Windows + uv):** O executável `tensorboard.exe` gerado pelo `uv` apresenta o erro
> *"Failed to canonicalize script path"*, e o pacote não possui `__main__.py` (impedindo `python -m tensorboard`).
> O script `tensorboard_launch.py` contorna ambos os problemas invocando `tensorboard.main.run_main()` diretamente via Python.
> Além disso, o `tensorboard 2.20` depende de `pkg_resources`, que foi removido no `setuptools>=82`.
> Caso encontre `ModuleNotFoundError: No module named 'pkg_resources'`, instale uma versão compatível:
>
> ```bash
> uv pip install "setuptools<81"
> ```

### Descrição das Recompensas

**Recompensa A** — Trade-off linear (robusta):
```
r_t = w1·(Tbottom − Ttop)_norm − w2·Qreb_norm − w3·var(Tbottom)
```
- `w1=1.0`, `w2=0.5`, `w3=0.2` (ajustáveis em `config.py`)

**Recompensa B** — Eficiência normalizada (sensível):
```
r_t = (Tbottom − Ttop) / (Qreb + ε)
```

---

## Fase 2 — Avaliação

### Avaliar o último modelo treinado de cada tipo

```bash
python evaluate.py
```

### Avaliar uma run específica

```bash
python evaluate.py --run_dir ../models/PPO_rewardA_20260224_120000 --reward A
```

### Comparar Recompensa A vs B

```bash
python evaluate.py --compare --n_episodes 50
```

Gráficos gerados em `results/figures/`:
- Trajetórias de episódio (Qreb, temperaturas, ΔT, reward por step)
- Distribuição do Qreb final convergido
- Curva de aprendizado (reward média vs timesteps)
- Boxplot comparativo A vs B (retorno, Qreb final, ΔT final)

---

## Fase 3 — Integração com Simulador Dinâmico (Homologação)

Após o treinamento no surrogate, o agente deve ser validado no simulador de alta fidelidade (Aspen Dynamics ou equivalente). O procedimento de integração utiliza os artefatos salvos na pasta `models/`.

### Artefatos necessários para integração

Para cada run treinada, a pasta `models/PPO_rewardX_...` contém:

| Arquivo                | Descrição                                                  |
|------------------------|------------------------------------------------------------|
| `best_model.zip`       | Pesos da política PPO (melhor checkpoint)                  |
| `vec_normalize.pkl`    | Estatísticas de normalização (média/std das observações)   |
| `training_meta.json`   | Metadados: reward type, hiperparâmetros, seed              |

### Como carregar e usar o agente em produção/homologação

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

# --- 1. Carregar modelo e normalização ---
MODEL_PATH = "models/PPO_rewardA_.../best_model.zip"
VECNORM_PATH = "models/PPO_rewardA_.../vec_normalize.pkl"

# Ambiente dummy (necessário apenas para shape; não será usado para stepping)
from surrogate_env import StripperSurrogateEnv
from stable_baselines3.common.monitor import Monitor
dummy_env = DummyVecEnv([lambda: Monitor(StripperSurrogateEnv(reward_type="A"))])
env = VecNormalize.load(VECNORM_PATH, dummy_env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL_PATH, env=env)

# --- 2. Montar observação a partir de leitura do simulador/planta ---
# Observação crua (valores de engenharia):
#   [Qfeed (kg/h), Tfeed (°C), Qreb (Gcal/h), Pcolumn (kgf/cm²), Tbottom (°C), Ttop (°C)]
obs_raw = np.array([50000.0, 122.0, 3.5, 6.0, 164.0, 110.0])

# Normalizar para [0, 1] usando as faixas do config.py:
obs_mins = np.array([47500.0, 118.95, 2.4, 5.7, 160.094, 60.009])
obs_maxs = np.array([52500.0, 125.05, 6.0, 6.3, 168.108, 165.655])
obs_norm = np.clip((obs_raw - obs_mins) / (obs_maxs - obs_mins), 0.0, 1.0).astype(np.float32)

# VecNormalize espera shape (1, 6)
obs_vec = env.normalize_obs(obs_norm.reshape(1, -1))

# --- 3. Obter ação (determinística, sem exploração) ---
action, _ = model.predict(obs_vec, deterministic=True)

# Ação em [-1, 1] → converter para ΔQreb real:
DELTA_QREB_MAX = 0.3  # Gcal/h (de config.py)
delta_qreb = float(action[0]) * DELTA_QREB_MAX

# --- 4. Aplicar no simulador ---
QREB_MIN, QREB_MAX = 2.4, 6.0
qreb_atual = 3.5
qreb_novo = np.clip(qreb_atual + delta_qreb, QREB_MIN, QREB_MAX)
# Enviar qreb_novo como novo setpoint Q̇_SP ao controlador PID do simulador
```

### Protocolo de integração com loop de simulação dinâmica

```
┌─────────────────────────────────────────────────────┐
│              LOOP DE HOMOLOGAÇÃO                    │
│                                                     │
│  1. Simulador roda até estado estacionário          │
│  2. Ler variáveis: Qfeed, Tfeed, Qreb, P, Tb, Tt  │
│  3. Montar obs_raw e normalizar                     │
│  4. model.predict(obs, deterministic=True)          │
│  5. Converter ação → ΔQreb                          │
│  6. Qreb_novo = clip(Qreb + ΔQreb)                 │
│  7. Enviar Qreb_novo como setpoint ao PID           │
│  8. Aguardar estabilização (detector de SS)         │
│  9. Voltar ao passo 2                               │
└─────────────────────────────────────────────────────┘
```

### Detector de estado estacionário (critérios sugeridos)

Antes de cada ação do agente, verificar:
- `|dTbottom/dt| < 0.05 °C/min`
- `|dTtop/dt| < 0.05 °C/min`
- Variância de Tbottom e Ttop nos últimos N pontos abaixo de limiar
- Vazão de carga aproximadamente constante

Somente quando todas as condições forem atendidas, o agente aplica nova ação.

### Comportamento esperado do agente treinado

```
Q̇ baixo  → ΔQ̇_SP > 0 → agente aumenta carga térmica
Q̇ ótimo  → ΔQ̇_SP ≈ 0 → agente mantém (joelho)
Q̇ alto   → ΔQ̇_SP < 0 → agente reduz carga térmica
```

### Segurança em produção

- Limites rígidos de `Qreb`: [2.4, 6.0] Gcal/h (clipping obrigatório)
- Limites de temperatura: Tbottom ∈ [155, 175] °C, Ttop ∈ [55, 170] °C
- Se qualquer limite for violado → fallback total para PID convencional
- Pesos do agente congelados (sem exploração, `deterministic=True`)

---

## Descrição dos Scripts

### `config.py`
Centraliza todas as constantes: caminhos dos surrogates, faixas de normalização das 7 entradas e 2 saídas, limites operacionais, hiperparâmetros do PPO, pesos das recompensas e limites de segurança.

### `surrogate_env.py`
Ambiente Gymnasium (`StripperSurrogateEnv`) que:
- Carrega os modelos Keras `02_Tbottom_best.keras` e `02_Ttop_best.keras`
- Implementa formulação POMDP: agente observa `[Qfeed, Tfeed, Qreb, Pcolumn, Tbottom, Ttop]`; variáveis latentes `cH2S`, `cNH3`, `Ecolumn` são randomizadas por episódio mas não observadas
- Ação contínua em `[-1, 1]` mapeada para `ΔQreb ∈ [-0.3, +0.3]` Gcal/h
- Inicialização diversificada: Qreb inicial em faixa baixa, média ou alta
- Suporta ambos os tipos de recompensa (`reward_type="A"` ou `"B"`)
- Terminação por violação de segurança ou truncamento em 30 steps

### `train_ppo.py`
Script de treinamento com:
- `VecNormalize` para normalização adaptativa das observações
- `EvalCallback` para salvar melhor modelo durante o treino
- Logging de métricas a cada 5000 steps
- Suporte a TensorBoard
- Salva `best_model.zip`, `final_model.zip`, `vec_normalize.pkl` e `training_meta.json`

### `evaluate.py`
Script de avaliação que:
- Carrega modelo treinado e executa episódios determinísticos
- Gera gráficos de trajetória (Qreb, temperaturas, ΔT, reward)
- Gera distribuição do Qreb final convergido
- Plota curva de aprendizado a partir dos logs de avaliação
- Compara runs de Recompensa A vs B com boxplots

---

## Observações sobre Normalização

Os surrogates foram treinados com dados normalizados min-max para `[0, 1]`:
- **Entradas**: `x_norm = (x - x_min) / (x_max - x_min)`
- **Saídas**: `y_norm = (y - y_min) / (y_max - y_min)`

O ambiente `surrogate_env.py` já cuida de normalizar as entradas antes de chamar os surrogates e desnormalizar as saídas para valores de engenharia. A observação entregue ao agente também é normalizada para `[0, 1]`.

Adicionalmente, o `VecNormalize` do stable-baselines3 aplica uma normalização adaptativa (running mean/std) sobre as observações — por isso o arquivo `vec_normalize.pkl` é obrigatório na integração.
