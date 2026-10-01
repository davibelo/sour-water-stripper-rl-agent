# Projeto Conceitual – Agente de Reinforcement Learning Supervisório

## Otimização do Joelho Operacional do Stripper H₂S / NH₃

---

## 1. Objetivo

Desenvolver e comparar agentes de **Reinforcement Learning (RL)** capazes de ajustar o **setpoint da carga térmica do refervedor (Q̇_SP)** para conduzir a coluna stripper ao  **ponto ótimo de operação ("joelho")** , caracterizado por máxima eficiência térmica de stripping com mínimo consumo energético,  **sem medições diretas de H₂S/NH₃** .

Requisitos principais:

* Operação por *tracking* a partir do estado atual da torre
* Ação lenta (event-based), compatível com a dinâmica do processo
* Separação clara entre **otimização** (RL) e **controle** (PID de carga térmica)
* Arquitetura defensável para aplicação industrial

---

## 2. Arquitetura Geral

```
┌───────────────────────────────────────────┐
│        Agente RL (PPO ou TD3)             │
│  Ação: ΔQ̇_SP (15–30 min ou evento)        |
└────────────────────── ────────────────────┘
                       │
                 Setpoint Q̇_SP
                       │
┌───────────────────────────────────────────┐
│     Malha de Carga Térmica (PID)          │
│     Manipula vazão de vapor               │
└────────────────────── ────────────────────┘
                       │
                 Vazão de vapor
                       │
                Torre Stripper
                       │
   Tbottom, Ttop, T_carga, vazão, Q̇
```

 **Princípios-chave** :

* O RL decide *quanto de energia* aplicar
* O PID garante execução física robusta
* O RL não fecha malha rápida nem reage a ruído instantâneo

---

## 3. Estratégia Temporal do Agente

O agente RL atua de forma  **event-based** :

* Quando um **detector de estado estacionário** indica estabilidade

### Detector de Estado Estacionário (exemplo)

Critérios típicos:

* |dTbottom/dt| < ε₁
* |dTtop/dt| < ε₂
* Variância de Tbottom e Ttop abaixo de limiar
* Vazão de carga aproximadamente constante

Somente quando as condições são atendidas o RL aplica uma nova ação.

---

## 4. Formulação do Problema de RL

### 4.1 Espaço de Estados (Observações)

```text
s_t = [
  vazão_de_carga,  
  T_carga,   
  Tbottom,
  Ttop,  
  Q̇_reb,
  dTbottom/dt  
]
```

Características:

* Variáveis filtradas (média móvel)
* Normalização por faixas operacionais
* Representação suficiente do regime térmico e energético

---

### 4.2 Espaço de Ações

Ação incremental no setpoint de carga térmica:

```text
a_t = ΔQ̇_SP
```

Com restrições:

```text
ΔQ̇_SP ∈ [−ΔQ̇_max, +ΔQ̇_max]
```

Atualização do setpoint:

```text
Q̇_SP(t+1) = Q̇_SP(t) + a_t
```

Essa formulação garante:

* Tracking local
* Ausência de saltos perigosos
* Comportamento suave e interpretável

---

## 5. Funções de Recompensa

Duas formulações serão avaliadas.

---

### 5.1 Recompensa A – Trade-off Linear (Robusta)

```text
r_t = w1 · (Tbottom − Ttop)
    − w2 · Q̇_reb
    − w3 · var(Tbottom)
```

 **Interpretação** :

* Maximiza gradiente térmico útil
* Penaliza consumo energético
* Penaliza instabilidade residual

 **Propriedades** :

* Alta robustez a ruído
* Melhor desempenho em ambiente real
* Fácil interpretação operacional

---

### 5.2 Recompensa B – Eficiência Normalizada (Sensível)

```text
r_t = (Tbottom − Ttop) / (Q̇_reb + ε)
```

 **Interpretação** :

* Maximiza eficiência térmica marginal
* Pico claro no joelho

 **Propriedades** :

* Convergência mais rápida
* Maior sensibilidade a ruído
* Mais adequada para simulador

---

## 6. Frameworks de Actor–Critic

### 6.1 PPO (Proximal Policy Optimization)

Características:

* On-policy
* Alta estabilidade
* Menor sensibilidade à escala da reward

Combinações testadas:

* PPO + Recompensa A (baseline industrial)
* PPO + Recompensa B (teste de robustez)

---

### 6.2 TD3 (Twin Delayed DDPG)

Características:

* Off-policy
* Double critic (reduz overestimation)
* Alta sensibilidade ao gradiente da reward

Combinações testadas:

* TD3 + Recompensa A
* TD3 + Recompensa B

---

## 7. Matriz de Experimentos

| Experimento | Algoritmo | Recompensa  | Objetivo                |
| ----------- | --------- | ----------- | ----------------------- |
| E1          | PPO       | Linear      | Robustez e tracking     |
| E2          | PPO       | Normalizada | Estabilidade vs razão  |
| E3          | TD3       | Linear      | Sensibilidade do critic |
| E4          | TD3       | Normalizada | Precisão do joelho     |

---

## 8. Estratégia de Treinamento

### Ambiente

* Simulador (Aspen Plus/Dynamics ou surrogate)
* *Domain randomization* :
  * Vazão de carga
  * Temperatura da carga
  * Eficiência interna
  * Pressão

### Inicialização dos Episódios

* Q̇ inicial baixo, médio e alto
* Estados próximos e distantes do joelho

Objetivo:

* Ensinar a política a  **subir** , **descer** e **estacionar** no ótimo

---

## 9. Execução em Produção

Durante a operação real:

* Pesos do agente congelados
* Política determinística
* Sem exploração
* Ação somente quando evento permitido

Comportamento esperado:

```text
Q̇ baixo  → ΔQ̇_SP > 0 → aumenta
Q̇ ótimo → ΔQ̇_SP ≈ 0 → mantém
Q̇ alto  → ΔQ̇_SP < 0 → reduz
```

---

## 10. Métricas de Avaliação

Além da reward média:

* Q̇ convergido
* ΔT = Tbottom − Ttop
* Variância de Q̇
* Tempo até convergência
* Robustez a perturbações de carga

---

## 11. Segurança e Governança

* Limites rígidos de Q̇_SP
* Limites de Tbottom e Ttop
* Penalidades severas por violação
* Fallback total para PID

---

## 12. Conclusão

O projeto propõe um  **agente RL supervisório** , fisicamente consistente e alinhado à prática industrial, capaz de localizar e manter o joelho operacional do stripper H₂S/NH₃ utilizando apenas variáveis térmicas e operacionais, com clara separação entre otimização e controle.

Este desenho está pronto para:

* Implementação em simulador
* Avaliação comparativa de algoritmos
* Evolução para piloto e planta
