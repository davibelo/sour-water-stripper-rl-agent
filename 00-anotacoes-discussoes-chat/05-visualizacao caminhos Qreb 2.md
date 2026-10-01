Perfeito. Vou estruturar **as 8 sugestões completas**, com:

* Objetivo analítico
* O que logar no treino
* Como plotar
* O que indica comportamento correto
* O que indica problema

Sem priorização.

---

# 1️⃣ Scatter + tendência: Q̇_final vs episódio

## 🎯 Objetivo

Ver **convergência inter-episódios** e redução de variância.

## 📦 O que logar

Para cada episódio `e`:

* `Qreb_final[e]`
* `Qreb_mean[e]`
* `reward_mean[e]`
* `ΔT_final[e]`

## 📊 Plot

* Eixo X: episódio
* Eixo Y: Q̇_final
* Overlay: média móvel (ex: janela 50 episódios)

## ✅ Comportamento saudável

* Convergência para uma banda estreita
* Redução da variância ao longo do treino
* Média móvel estabilizada

## ❌ Problemas típicos

* Drift contínuo → reward mal escalada
* Alta variância persistente → política instável
* Oscilação periódica → overfitting ao surrogate

---

# 2️⃣ Heatmap: Q̇_inicial → Q̇_final

## 🎯 Objetivo

Ver **independência do estado inicial**.

## 📦 Log necessário

Para cada episódio:

* `Qreb_init`
* `Qreb_final`

## 📊 Plot

* Eixo X: Q̇ inicial
* Eixo Y: Q̇ final
* Cor: densidade

## ✅ Esperado

* Convergência para uma **linha horizontal**
  (mesmo ótimo independentemente do início)

## ❌ Problemas

* Diagonal → agente apenas mantém estado
* Múltiplas bandas → múltiplos “óptimos falsos”
* Alta dispersão → reward mal definida

---

# 3️⃣ Trajetórias intra-episódio (Q̇ vs passo)

## 🎯 Objetivo

Analisar dinâmica de decisão.

## 📦 Log necessário

Por passo de decisão:

* `Qreb_t`
* `ΔQ_action`
* `reward_t`
* `ΔT_t`

## 📊 Plot

Selecionar:

* 5 episódios no início do treino
* 5 no final

Plotar:

* Q̇ vs passo
* ΔT vs passo
* Reward vs passo

## ✅ Esperado

* Convergência suave
* Poucos overshoots
* Aproximação amortecida

## ❌ Problemas

* Zig-zag → ΔQ_max grande
* Oscilação permanente → reward não penaliza variabilidade
* Convergência lenta → exploração excessiva

---

# 4️⃣ Sobreposição no mapa físico ΔT vs Q̇

## 🎯 Objetivo

Verificar se o agente aprendeu o **fenômeno físico real**.

## 📦 Log necessário

* Curva ΔT(Q̇) de referência (Aspen ou surrogate)
* Q̇_final de múltiplos episódios

## 📊 Plot

* Curva física ΔT(Q̇)
* Pontos marcando Q̇_final

## ✅ Esperado

* Pontos concentrados no joelho físico

## ❌ Problemas

* Convergência fora do joelho
* Concentração em região linear → reward mal balanceada
* Exploração em região instável → surrogate bias

Essa é a validação fenomenológica.

---

# 5️⃣ Histogramas finais

## 🎯 Objetivo

Avaliar multimodalidade e dispersão.

## 📦 Log necessário

* Q̇_final
* ΔT_final

## 📊 Plot

Dois histogramas:

* Q̇_final
* ΔT_final

## ✅ Esperado

* Distribuição estreita
* Unimodal

## ❌ Problemas

* Bimodal → dois ótimos artificiais
* Cauda longa → política instável

---

# 6️⃣ Erro absoluto em relação ao joelho

Se você conhece o ótimo físico ( Q̇^* ):

[
Erro_e = |Q̇_{final,e} - Q̇^*|
]

## 🎯 Objetivo

Transformar análise qualitativa em quantitativa.

## 📊 Plot

* Erro vs episódio
* Média móvel do erro

## ✅ Esperado

* Decaimento exponencial
* Estabilização baixa

## ❌ Problemas

* Platô alto → reward desalinhada
* Aumento tardio → overfitting

---

# 7️⃣ Dispersão Reward vs Q̇_final

## 🎯 Objetivo

Diagnosticar estrutura da função recompensa.

## 📦 Log necessário

* reward_mean por episódio
* Q̇_final

## 📊 Plot

Scatter:

* X: Q̇_final
* Y: reward_mean

## ✅ Esperado

* Pico claro próximo ao joelho
* Relação monotônica ao redor do ótimo

## ❌ Problemas

* Reward alta longe do joelho → mal formulada
* Reward plana → gradiente fraco

---

# 8️⃣ Evolução da variância intra-episódio

## 🎯 Objetivo

Ver estabilidade operacional.

Para cada episódio:

[
Var_e = Var(Q̇_t)
]

## 📊 Plot

* Var(Q̇) vs episódio

## ✅ Esperado

* Alta variância no início do treino
* Queda progressiva
* Estabilização baixa

## ❌ Problemas

* Variância persistente → política exploratória demais
* Variância zero precoce → exploração insuficiente

---

# Estrutura ideal de logging (resumo)

Durante treino, salve:

### Por episódio:

* Qreb_init
* Qreb_final
* Qreb_mean
* reward_mean
* ΔT_final
* var_Qreb
* erro_joelho (se conhecido)

### Por passo:

* Qreb_t
* ΔQ_action
* reward_t
* ΔT_t

---

# Observação importante para seu caso (surrogate + POMDP)

Eu acrescentaria uma análise extra opcional:

### Separar resultados por bins de Pcolumn e Ecolumn

Para verificar robustez a variáveis latentes.

Isso revela se o agente aprendeu política robusta ou apenas média do domínio.

---

Se quiser, posso agora:

* Sugerir qual dessas é mais sensível para detectar erro na Reward A
* Ou qual melhor revela instabilidade no TD3
* Ou estruturar um template de análise comparativa PPO vs TD3

Qual direção você quer seguir?
