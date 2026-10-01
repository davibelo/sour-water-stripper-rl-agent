### Projeto – Agente de Reinforcement Learning Supervisório (V3 — Implementado)

#### Otimização do Joelho Operacional do Stripper H₂S / NH₃ com Surrogate DNN (LHS)

---

### 1. Objetivo

Desenvolver e comparar agentes de **Reinforcement Learning (RL)** capazes de ajustar o **setpoint da carga térmica do refervedor (Q̇reb)** para conduzir a coluna stripper ao **ponto ótimo de operação ("joelho")**, caracterizado por máxima eficiência térmica de stripping com mínimo consumo energético, **sem medições diretas de H₂S/NH₃**.

O treinamento é realizado em um **surrogate estacionário DNN** (treinado com amostras por Latin Hypercube Sampling – LHS + simulações Aspen), seguido de **validação estática no Aspen** utilizando os setpoints de Q̇reb determinados pelo agente para obter as concentrações finais de H₂S e NH₃.

---

### 2. Arquitetura Geral

**Em produção (planta — conceitual):**

- O agente RL atua como supervisório sobre o setpoint de Q̇reb
- O PID executa a manipulação física (vapor)
- O agente age somente quando o detector de estabilidade permite

**No desenvolvimento/treinamento (implementado):**

- Treinamento intensivo no **surrogate DNN estacionário** (ambiente Gymnasium rápido)
- Otimização de hiperparâmetros com **Optuna** (50 trials, 150k timesteps cada)
- Retreinamento com os melhores hiperparâmetros (500k timesteps)
- Validação no **Aspen estático** (simulador de alta fidelidade) para obter H₂S e NH₃

```text
                     (Treinamento)
          ┌───────────────────────────────────┐
          │  Agente RL (PPO ou DDPG)          │
          │  Ação: ΔQ̇reb (incremental)        │
          └───────────────┬───────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Surrogate DNN        │
              │   (Tbottom, Ttop)      │
              └───────────────────────┘

                     (Validação)
          ┌───────────────────────────────────┐
          │  Agente RL (pesos congelados)     │
          │  Simula episódios → Q̇reb final    │
          └───────────────┬───────────────────┘
                          │
              ┌───────────▼───────────┐
              │  Aspen (estático)     │
              │  Entrada: Q̇reb final  │
              │  Saída: cH₂S, cNH₃   │
              └───────────────────────┘
```

---

### 3. Formulação do Problema de RL (POMDP)

O problema é formulado como **POMDP** (parcialmente observável), pois pressão e eficiência interna influenciam o processo mas **não são observadas pelo agente**:

**Estado interno do ambiente:**

```text
x_t = [Qfeed, Tfeed, cH2S, cNH3, Qreb, Pcolumn, Ecolumn]
```

**Observação entregue ao agente (variáveis medidas — 6 entradas):**

```text
s_t = [Qfeed, Tfeed, Qreb, Pcolumn, Tbottom, Ttop]
```

onde `cH2S`, `cNH3` e `Ecolumn` influenciam o surrogate, mas **não compõem a observação** do agente — são randomizados por episódio para forçar robustez da política.

| Variável | Papel | Faixa |
|---|---|---|
| `Qfeed` | Vazão da alimentação | 47 500 – 52 500 kg/h |
| `Tfeed` | Temperatura da alimentação | 118,95 – 125,05 °C |
| `cH2S` | Concentração de H₂S (latente) | 875 – 3 500 ppm |
| `cNH3` | Concentração de NH₃ (latente) | 750 – 3 000 ppm |
| `Qreb` | Carga térmica no refervedor (ação) | 2,4 – 6,0 Gcal/h |
| `Pcolumn` | Pressão na coluna | 5,7 – 6,3 kgf/cm²m |
| `Ecolumn` | Eficiência de Murphree (latente) | 66,5% – 73,5% |
| `Tbottom` | Temperatura no fundo (saída surrogate) | ~160 – 168 °C |
| `Ttop` | Temperatura no topo (saída surrogate) | ~60 – 166 °C |

---

### 4. Espaço de Ações (incremental)

A ação é **incremental** sobre o setpoint de carga térmica:

```text
a_t ∈ [-1, +1]  →  ΔQreb = a_t × 0.3 Gcal/h
Qreb(t+1) = clip(Qreb(t) + ΔQreb, 2.4, 6.0)
```

Essa formulação garante:

- Tracking local a partir do estado atual
- Ausência de saltos bruscos
- Comportamento suave e interpretável
- Possibilidade de percorrer toda a faixa em ~12 passos (episódio = 30 passos)

---

### 5. Função de Recompensa

Apenas a **Recompensa A** foi utilizada no treinamento:

```text
r_t = w1 · (Tbottom − Ttop)_norm
    − w2 · Qreb_norm
    − w3 · var(Tbottom)
```

Interpretação:

- Maximiza gradiente térmico útil (proxy de eficiência de stripping)
- Penaliza consumo energético
- Penaliza instabilidade residual da temperatura de fundo

Penalidade de segurança: `−50.0` por violação de limites de temperatura, com terminação do episódio.

Os pesos `w1`, `w2`, `w3` foram otimizados pelo Optuna em conjunto com os hiperparâmetros do agente.

> **Nota:** A Recompensa B (`r = (Tbottom − Ttop) / (Qreb + ε)`) foi definida mas **não utilizada** no treinamento.

---

### 6. Surrogate Model (DNN)

**Localização:** `05-surrogate/`

Dois modelos Keras independentes (um por saída):

- `02b_Tbottom_best.keras` — prediz temperatura de fundo
- `02b_Ttop_best.keras` — prediz temperatura de topo

**Entradas (7 variáveis, normalizadas min-max):**
`Qfeed, Tfeed, cH2S, cNH3, Qreb, Pcolumn, Ecolumn`

**Saídas (2 variáveis):**
`Tbottom, Ttop`

O dataset foi gerado via **Latin Hypercube Sampling (LHS)** com critério space-filling sobre o espaço de 7 entradas e simulado no Aspen até regime estacionário.

---

### 7. Agentes RL Implementados

Foram implementados e comparados **dois algoritmos**, ambos com hiperparâmetros otimizados via Optuna:

#### 7.1 PPO (Proximal Policy Optimization) — on-policy

- Biblioteca: `stable-baselines3`
- Alta estabilidade de treinamento
- Menor sensibilidade à escala da recompensa

#### 7.2 DDPG (Deep Deterministic Policy Gradient) — off-policy

- Biblioteca: `stable-baselines3`
- Ruído de exploração Ornstein-Uhlenbeck (σ otimizado pelo Optuna)
- Alta sensibilidade ao gradiente da recompensa

---

### 8. Otimização de Hiperparâmetros com Optuna

Ambos os agentes foram submetidos à otimização de hiperparâmetros com **Optuna**:

**Protocolo (2 fases):**

1. **Busca (50 trials × 150 000 timesteps):** Optuna explora o espaço de hiperparâmetros e pesos da recompensa
2. **Retreinamento (500 000 timesteps):** Melhor configuração retrain completo

**Espaço de busca — PPO:**

| Parâmetro | Tipo | Faixa |
|---|---|---|
| `learning_rate` | float (log) | 1e-5 – 1e-3 |
| `n_steps` | categórico | 512, 1 024, 2 048, 4 096 |
| `batch_size` | categórico | 32, 64, 128, 256 |
| `n_epochs` | int | 4 – 20 |
| `gamma` | float | 0,95 – 0,999 |
| `gae_lambda` | float | 0,90 – 1,0 |
| `clip_range` | float | 0,10 – 0,40 |
| `ent_coef` | float | 0,0 – 0,10 |
| `w1` | float | 0,05 – 0,30 |
| `w2` | float | 0,50 – 1,50 |
| `w3` | float | 0,10 – 1,50 |

**Espaço de busca — DDPG:**

| Parâmetro | Tipo | Faixa |
|---|---|---|
| `learning_rate` | float (log) | 1e-4 – 1e-2 |
| `buffer_size` | categórico | 50 000, 100 000, 200 000 |
| `learning_starts` | int | 500 – 5 000 |
| `batch_size` | categórico | 64, 128, 256, 512 |
| `tau` | float | 0,001 – 0,05 |
| `gamma` | float | 0,90 – 0,999 |
| `train_freq` | categórico | 1, 2, 4 |
| `gradient_steps` | categórico | 1, 2, 4 |
| `action_noise_sigma` | float | 0,01 – 0,30 |
| `w1` | float | 0,05 – 0,30 |
| `w2` | float | 0,50 – 1,50 |
| `w3` | float | 0,10 – 1,50 |

---

### 9. Matriz de Experimentos

| Diretório | Algoritmo | Recompensa |
|---|---|---|
| `10-agentRL-PPO-Opt` | PPO + Optuna (50 trials) | A (pesos otimizados) |
| `12-agentRL-DDPG-Opt` | DDPG + Optuna (50 trials) | A (pesos otimizados) |

---

### 10. Estratégia de Treinamento

#### 10.1 Randomização por episódio

- **Observadas (randomizadas):** `Qfeed`, `Tfeed`
- **Latentes (randomizadas, não observadas):** `cH2S`, `cNH3`, `Ecolumn`
- **Pressão:** `Pcolumn` observada, mas randomizada entre episódios

`Pcolumn` e `Ecolumn` permanecem aproximadamente constantes dentro de cada episódio, refletindo o comportamento físico real.

#### 10.2 Episódios de treinamento

- Duração: **30 passos** por episódio
- Q̇reb inicial: randomizado em 3 bandas (baixo / médio / alto)
- Avaliação: a cada **10 000 timesteps**, em 20 episódios determinísticos

#### 10.3 Inicialização dos episódios

- `Q̇` inicial baixo, médio e alto
- Estados próximos e distantes do joelho

Objetivo: ensinar a política a **subir**, **descer** e **estacionar** no ótimo independentemente do ponto de partida.

---

### 11. Validação no Aspen (Estática)

**Não foi realizada simulação dinâmica.** A validação foi feita em dois passos:

**Passo 1 — Simulação de episódios com o agente:**

```text
Para cada episódio:
  ├─ Inicializa condições (Qfeed, Tfeed, cH2S, cNH3, Pcolumn, Ecolumn)
  ├─ Agente age incrementalmente por 30 passos (surrogate como ambiente)
  └─ Registra trajetória de Qreb, Tbottom, Ttop → Q̇reb final
```

Episódios organizados em 3 bandas de Q̇reb inicial: `low`, `mid`, `high` (10 episódios cada).

**Passo 2 — Simulação estática no Aspen:**

```text
Para cada episódio:
  ├─ Configura Aspen com as condições do episódio + Q̇reb final do agente
  ├─ Roda até regime estacionário (convergência)
  └─ Coleta: cH₂S saída, cNH₃ saída, Tbottom, Ttop, iterações
```

O Aspen é controlado via **COM (Win32)**, com instâncias paralelas para reduzir tempo total.

**Resultados coletados por episódio:**
- Concentração de H₂S na saída (recuperação)
- Concentração de NH₃ na saída (perda)
- `Tbottom`, `Ttop`, `Qreb` final
- Convergência (flag `valid`)

---

### 12. Fluxo Completo do Projeto

```text
02-amostragem_dnn/
 └─ Geração de amostras por LHS
      │
      ▼
03-simulation_cases/
 └─ Simulação no Aspen (LHS → dataset)
      │
      ▼
05-surrogate/
 └─ Treino surrogate DNN (Tbottom, Ttop)
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
10-agentRL-PPO-Opt/                 12-agentRL-DDPG-Opt/
 └─ Optuna HPO (50 trials)           └─ Optuna HPO (50 trials)
 └─ Retreinamento (500k steps)       └─ Retreinamento (500k steps)
      │                                      │
      ▼                                      ▼
11-agentRL-PPO-Opt-aspen-test/      13-agentRL-DDPG-Opt-aspen-test/
 └─ Simula episódios                  └─ Simula episódios
 └─ Valida Q̇reb no Aspen estático    └─ Valida Q̇reb no Aspen estático
 └─ Coleta cH₂S, cNH₃               └─ Coleta cH₂S, cNH₃
```

---

### 13. Métricas de Avaliação

**Durante treinamento:**
- Recompensa média por episódio (reward A)
- Q̇reb convergido e estabilidade

**Na validação Aspen:**
- Concentração de H₂S na saída (recuperação)
- Concentração de NH₃ na saída (perda)
- ΔT = Tbottom − Ttop
- Q̇reb final determinado pelo agente
- Taxa de convergência do Aspen (`valid`)

**Comparação entre agentes:**
- PPO-Opt vs. DDPG-Opt

---

### 14. Segurança e Governança

- Limites rígidos de `Q̇reb` ∈ [2,4 – 6,0] Gcal/h aplicados pelo `clip`
- Penalidade severa (−50) por violação de limites de temperatura
- Terminação de episódio em estados inseguros (durante treino)
- Ação incremental garante ausência de saltos bruscos

---

### 15. Situação Atual e Próximos Passos

**Implementado:**
- [x] Geração de dataset por LHS + Aspen
- [x] Treinamento e validação do surrogate DNN
- [x] Treinamento RL com Optuna (PPO e DDPG)
- [x] Simulação de episódios com os agentes treinados
- [x] Validação estática no Aspen (cH₂S e cNH₃)
- [x] Avaliação comparativa PPO-Opt vs. DDPG-Opt

**Não realizado / etapas futuras:**
- [ ] Simulação dinâmica (Aspen Dynamics) para validação transiente
- [ ] Fine-tuning do agente com dados do Aspen (active learning)
- [ ] Piloto em ambiente de processo real
