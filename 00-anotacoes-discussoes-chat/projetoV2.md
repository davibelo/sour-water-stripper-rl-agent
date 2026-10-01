### Projeto Conceitual – Agente de Reinforcement Learning Supervisório

#### Otimização do Joelho Operacional do Stripper H₂S / NH₃ com Surrogate DNN (LHS)

---

### 1. Objetivo

Desenvolver e comparar agentes de **Reinforcement Learning (RL)** capazes de ajustar o **setpoint da carga térmica do refervedor (Q̇_SP)** para conduzir a coluna stripper ao **ponto ótimo de operação (“joelho”)**, caracterizado por máxima eficiência térmica de stripping com mínimo consumo energético, **sem medições diretas de H₂S/NH₃**.

Evolução proposta: incluir uma etapa de **modelo surrogate estacionário em DNN** (treinado com **Latin Hypercube Sampling – LHS** + simulações Aspen) para reduzir drasticamente o custo computacional de interações necessárias no treinamento do agente RL, seguida de **validação e fine-tuning** no simulador de alta fidelidade.

Requisitos principais:

- Operação por *tracking* a partir do estado atual da torre
- Ação lenta (event-based), compatível com a dinâmica do processo
- Separação clara entre **otimização** (RL) e **controle** (PID de carga térmica)
- Arquitetura defensável para aplicação industrial
- Robustez a **distúrbios não medidos** (ex.: pressão e eficiência interna)

---

### 2. Arquitetura Geral (treinamento vs produção)

**Em produção (planta):**

- O RL atua como supervisório sobre o setpoint de Q̇_SP
- O PID executa a manipulação física (vapor)
- O RL age somente quando permitido pelo detector de estabilidade

**No desenvolvimento/treinamento (simulação):**

- Treinamento intensivo em **surrogate DNN estacionário** (ambiente rápido)
- Validação e ajuste em **simulador rigoroso** (Aspen) para reduzir *model bias*

Arquitetura conceitual:

```text
                     (Treinamento)
          ┌───────────────────────────────────┐
          │   Agente RL (PPO ou TD3)          │
          │   Ação: ΔQ̇_SP (event-based)       │
          └───────────────┬───────────────────┘
                          │
         ┌────────────────┴─────────────────┐
         │                                  │
┌────────▼─────────┐                ┌────────▼─────────┐
│  Surrogate DNN    │                │ Simulador Aspen   │
│  estacionário     │                │ (alta fidelidade) │
│  rápido           │                │ lento (referência)│
└────────┬─────────┘                └────────┬─────────┘
         │                                    │
   Tbottom, Ttop                       Tbottom, Ttop
         │                                    │
         └────────── validação / AL ──────────┘

                     (Produção)
┌─────────────────────────────────────────────┐
│      Agente RL (pesos congelados)           │
│      Ação: ΔQ̇_SP (event-based)             │
└───────────────────┬─────────────────────────┘
                    │
                 Setpoint Q̇_SP
                    │
┌───────────────────▼─────────────────────────┐
│     Malha de Carga Térmica (PID)            │
│     Manipula vazão de vapor                 │
└───────────────────┬─────────────────────────┘
                    │
                Torre Stripper
                    │
      Tbottom, Ttop, T_carga, vazão, Q̇
```

Princípios-chave:

- O RL decide *quanto de energia* aplicar (otimização)
- O PID garante execução física robusta (controle)
- O RL não fecha malha rápida nem reage a ruído instantâneo

---

### 3. Estratégia Temporal do Agente

O agente RL atua de forma **event-based**:

- Quando um **detector de estado estacionário** indica estabilidade
- A atualização de ΔQ̇_SP ocorre em janelas compatíveis com a dinâmica do processo (ex.: 15–30 min ou evento)

#### 3.1 Detector de Estado Estacionário (exemplo)

Critérios típicos:

- |dTbottom/dt| < ε₁
- |dTtop/dt| < ε₂
- Variância de Tbottom e Ttop abaixo de limiar
- Vazão de carga aproximadamente constante

Somente quando as condições são atendidas o RL aplica uma nova ação.

---

### 4. Formulação do Problema de RL (com distúrbios não medidos)

Pressão de operação e eficiência interna podem variar entre colunas (ou ao longo do tempo), e **não são necessariamente medidas** ou disponibilizadas ao agente. Assim, o problema é formulado como **POMDP** (parcialmente observável):

- **Estado interno do ambiente (com distúrbios latentes):**

```text
x_t = [Qfeed, Tfeed, cH2S, cNH3, Q̇reb, Pcolumn, Ecolumn]
```

- **Observação entregue ao agente (somente variáveis medidas):**

```text
s_t = [Qfeed, Tfeed, Q̇reb, Pcolumn, Tbottom, Ttop]
```

onde `cH2S`, `cNH3` e `Ecolumn` influenciam o processo (e o simulador/surrogate), mas **não compõem a observação**.

onde:

Qfeed = vazão da corrente de alimentação,
Tfeed = temperatura da corrente de alimentação,
cH2S = concentração de H₂S na corrente de alimentação,
cNH3 = concentração de NH₃ na corrente de alimentação,
Q̇reb = carga térmica no refervedor,
Pcolumn = pressão na coluna,
Ecolumn = eficiência de Murphree a coluna,
Tbottom = temperatura no fundo da coluna,
Ttop = temperatura no topo da coluna

### 5. Espaço de Ações

Ação incremental no setpoint de carga térmica:

```text
a_t = ΔQ̇_SP
```

Com restrições:

```text
ΔQ̇_SP ∈ [−ΔQ̇_max, +ΔQ̇_max]
```

Atualização do setpoint:

$$
Q̇_{SP}(t+1) = Q̇_{SP}(t) + a_t
$$

Essa formulação garante:

- Tracking local
- Ausência de saltos perigosos
- Comportamento suave e interpretável

---

### 6. Funções de Recompensa

Duas formulações serão avaliadas.

#### 6.1 Recompensa A – Trade-off Linear (robusta)

```text
r_t = w1 · (Tbottom − Ttop)
    − w2 · Q̇_reb
    − w3 · var(Tbottom)
```

Interpretação:

- Maximiza gradiente térmico útil
- Penaliza consumo energético
- Penaliza instabilidade residual

Propriedades:

- Alta robustez a ruído
- Melhor desempenho em ambiente real
- Fácil interpretação operacional

#### 6.2 Recompensa B – Eficiência Normalizada (sensível)

```text
r_t = (Tbottom − Ttop) / (Q̇_reb + ε)
```

Interpretação:

- Maximiza eficiência térmica marginal
- Pico claro no joelho

Propriedades:

- Convergência mais rápida
- Maior sensibilidade a ruído
- Mais adequada para simulador / surrogate

---

### 7. Frameworks de Actor–Critic

#### 7.1 PPO (Proximal Policy Optimization)

Características:

- On-policy
- Alta estabilidade
- Menor sensibilidade à escala da reward

Combinações testadas:

- PPO + Recompensa A (baseline industrial)
- PPO + Recompensa B (teste de robustez)

#### 7.2 TD3 (Twin Delayed DDPG)

Características:

- Off-policy
- Double critic (reduz overestimation)
- Alta sensibilidade ao gradiente da reward

Combinações testadas:

- TD3 + Recompensa A
- TD3 + Recompensa B

---

### 8. Matriz de Experimentos (RL)

| Experimento | Algoritmo | Recompensa  | Objetivo                |
| ----------- | --------- | ----------- | ----------------------- |
| E1          | PPO       | Linear      | Robustez e tracking     |
| E2          | PPO       | Normalizada | Estabilidade vs razão  |
| E3          | TD3       | Linear      | Sensibilidade do critic |
| E4          | TD3       | Normalizada | Precisão do joelho     |

---

### 9. Etapa de Surrogate Model (DNN) com Dataset via LHS

#### 9.1 Objetivo do surrogate (estacionário)

Construir um **surrogate estacionário** para aproximar o mapeamento do regime estacionário da coluna, reduzindo custo computacional de interação no treinamento do RL:

$$
\hat{f}:\; (Qfeed, Tfeed, cH2S, cNH3, Qreb, Pcolumn, Ecolumn) \rightarrow (Tbottom, Ttop)
$$

Uso no RL (POMDP):

- O ambiente usa `Pcolumn` e `Ecolumn` internamente
- O agente não observa `Pcolumn` e `Ecolumn`, mas deve operar corretamente sob suas variações

#### 9.2 Entradas e saídas do surrogate (com faixas)

**Entradas (variáveis do ambiente):**

- Vazão da carga | `Qfeed` | 47500 a 52500 kg/h
- Temperatura da carga | `Tfeed` | 118.95 a 125.05 °C
- Concentração de H₂S na alimentação | `cH2S` | 875 a 3500 ppm
- Concentração de NH₃ na alimentação | `cNH3` | 750 a 3000 ppm
- Carga térmica no refervedor | `Qreb` | 2.4 a 6.0 Gcal/h
- Pressão da coluna (latente) | `Pcolumn` | 5.7 a 6.3 kgf/cm²m
- Eficiência de Murphree global (latente) | `Ecolumn` | 66.5% a 73.5%

**Saídas (variáveis térmicas observáveis):**

- Temperatura de fundo | `Tbottom`
- Temperatura de topo | `Ttop`

**Metadados recomendados no dataset (não como saída principal de regressão):**

- `valid` (0/1): convergiu e respeitou restrições?
- flags de violação: limites de temperatura/hidráulica (se disponível)

Observação: `Ecolumn` é aplicado igualmente em todos os pratos no Aspen, sendo um parâmetro escalar global de eficiência interna.

#### 9.3 Amostragem do dataset por Latin Hypercube Sampling (LHS)

Gerar um conjunto de pontos com LHS no espaço:

```text
[Qfeed, Tfeed, cH2S, cNH3, Qreb, Pcolumn, Ecolumn]
```

Recomendações:

- Critério *space-filling* (ex.: maximin) para cobrir bem o espaço
- Checagem de inviabilidade:
  - Se não convergir, registrar como `valid=0` e não usar no treino de regressão (ou treinar um classificador separado de validade)

#### 9.4 Geração de dados no Aspen (alta fidelidade)

Para cada ponto LHS:

1) Configurar o caso no Aspen com as entradas
2) Rodar até regime estacionário / convergência
3) Registrar:
   - `Tbottom`, `Ttop`
   - `valid`
   - (opcional) tempo de convergência, iterações, flags de restrição

#### 9.5 Treinamento do surrogate DNN

- Modelo: MLP (DNN feedforward) com 2 saídas (`Tbottom`, `Ttop`)
- Pré-processamento:
  - normalização das entradas por faixa
  - `Ecolumn` como fração (0–1)
- Validação:
  - split treino/val/test
  - métricas por variável (MAE/RMSE)
- Critério de aceitação (conceitual):
  - erro térmico suficientemente baixo para não deslocar a localização do “joelho” na reward
  - estabilidade de predição nas bordas do domínio

#### 9.6 Incerteza e mitigação de *model bias* no RL

Para reduzir o risco do RL “explorar falhas” do surrogate:

- Usar **ensemble de DNNs** (vários modelos treinados com inicializações/dados) para estimar incerteza
- Penalizar ações/estados em regiões com alta incerteza (regularização de robustez)
- Pipeline iterativo (Dyna / *active learning*):
  1) LHS inicial + Aspen → dataset
  2) Treino surrogate
  3) Pré-treino RL no surrogate (muitas interações)
  4) Avaliar política no Aspen e coletar pontos adicionais nas regiões visitadas
  5) Re-treinar surrogate e ajustar RL

---

### 10. Estratégia de Treinamento do Agente (com surrogate)

#### 10.1 Randomização por episódio (robustez)

- Variáveis medidas (observadas) randomizadas:
  - `Qfeed`, `Tfeed`, `cH2S`, `cNH3`
- Variáveis latentes (não observadas) randomizadas por episódio:
  - `Pcolumn` (5.7–6.3 kgf/cm²m)
  - `Ecolumn` (0.665–0.735)

Durante um episódio, `Pcolumn` e `Ecolumn` permanecem aproximadamente constantes (variações lentas), refletindo comportamento físico.

#### 10.2 Fases do treinamento

1) **Pré-treinamento no surrogate DNN (rápido)**

   - muitas interações
   - exploração e coleta de experiências barata
2) **Validação e fine-tuning no Aspen (lento, referência)**

   - poucas interações (caro)
   - objetivo: reduzir diferença sim2surrogate e validar robustez

#### 10.3 Inicialização dos episódios

- `Q̇` inicial baixo, médio e alto
- Estados próximos e distantes do joelho

Objetivo:

- Ensinar a política a **subir**, **descer** e **estacionar** no ótimo

---

### 11. Execução em Produção

Durante a operação real:

- Pesos do agente congelados
- Política determinística (sem exploração)
- Ação somente quando evento permitido (detector de estacionário)
- Fallback total para PID em caso de anomalia

Comportamento esperado:

```text
Q̇ baixo  → ΔQ̇_SP > 0 → aumenta
Q̇ ótimo → ΔQ̇_SP ≈ 0 → mantém
Q̇ alto  → ΔQ̇_SP < 0 → reduz
```

---

### 12. Métricas de Avaliação

Além da reward média:

- `Q̇` convergido e estabilidade de `Q̇`
- ΔT = Tbottom − Ttop
- Tempo até convergência
- Robustez a perturbações de carga
- Generalização sob variações latentes:
  - avaliar desempenho em combinações de `Pcolumn`/`Ecolumn` fora do conjunto de treino (holdout)

Para o surrogate:

- MAE/RMSE de `Tbottom` e `Ttop` em test set
- taxa de falhas (`valid=0`) e tratamento
- análise de erro nas bordas do domínio (onde o RL pode operar)

---

### 13. Segurança e Governança

- Limites rígidos de `Q̇_SP`
- Limites de `Tbottom` e `Ttop`
- Penalidades severas por violação
- Terminação de episódio em estados inseguros (no treino)
- Fallback total para PID

---

### 14. Conclusão

O projeto propõe um **agente RL supervisório** fisicamente consistente e alinhado à prática industrial, capaz de localizar e manter o joelho operacional do stripper H₂S/NH₃ utilizando apenas variáveis térmicas e operacionais, com separação clara entre otimização e controle.

A inclusão do **surrogate estacionário em DNN**, treinado com **dataset gerado via LHS** no Aspen e com **randomização de pressão e eficiência de Murphree (latentes)**, viabiliza treinamento intensivo do RL com menor custo computacional e melhora a robustez da política a variações reais de planta/coluna.

O desenho está pronto para:

- Geração do dataset por LHS + Aspen
- Treinamento/validação do surrogate DNN
- Treinamento RL no surrogate + fine-tuning no Aspen
- Avaliação comparativa PPO vs TD3 e rewards
- Evolução para piloto e planta

```

```
