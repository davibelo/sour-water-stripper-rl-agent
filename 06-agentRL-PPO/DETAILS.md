# Detalhamento Técnico do Agente RL PPO

Este documento descreve as escolhas de projeto feitas na construção do agente de Reinforcement Learning para otimização do joelho operacional do stripper H₂S/NH₃.

---

## 1. Por que PPO?

O PPO (Proximal Policy Optimization) foi escolhido como algoritmo principal por quatro razões:

1. **Estabilidade de treinamento** — O PPO usa uma função objetivo com clipping que impede atualizações muito grandes da política. Isso é fundamental em um problema de controle de processo onde a política deve convergir de forma suave, sem oscilações destrutivas entre iterações de treinamento.

2. **On-policy com boa eficiência amostral** — Embora on-policy seja tipicamente menos eficiente que off-policy (como TD3), o custo de interação com o surrogate DNN é desprezível (~ms por step). Isso elimina a principal desvantagem do on-policy e permite aproveitar a estabilidade superior.

3. **Robustez à escala da recompensa** — O PPO tolera melhor variações na magnitude da reward do que algoritmos baseados em Q-value (TD3/SAC). Como testamos duas funções de recompensa com escalas diferentes (Recompensa A em ~[-1, 1] e Recompensa B em ~[0, 30]), essa robustez é valiosa.

4. **Actor-Critic com GAE** — A arquitetura actor-critic com Generalized Advantage Estimation (GAE, λ=0.95) reduz a variância das estimativas de vantagem sem introduzir viés significativo, acelerando a convergência.

### Função objetivo do PPO

A atualização da política é dada por:

```
L_CLIP(θ) = E[ min( r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]
```

onde `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` é a razão de probabilidades e `ε = 0.2` é o parâmetro de clipping. Isso garante que a nova política não se afaste demais da anterior em cada atualização.

---

## 2. Formulação como POMDP

O problema é formulado como um Processo de Decisão de Markov Parcialmente Observável (POMDP). Essa é uma escolha deliberada que reflete a realidade da planta:

### Estado completo do ambiente (7 variáveis)

```
x_t = [Qfeed, Tfeed, cH2S, cNH3, Qreb, Pcolumn, Ecolumn]
```

Todas as 7 variáveis são usadas como entrada do surrogate para prever as temperaturas.

### Observação entregue ao agente (6 variáveis)

```
s_t = [Qfeed, Tfeed, Qreb, Pcolumn, Tbottom, Ttop]
```

### Variáveis ocultas (3 variáveis)

| Variável  | Faixa          | Por que é oculta                                              |
|-----------|----------------|---------------------------------------------------------------|
| `cH2S`    | 875–3500 ppm   | Concentração de H₂S na alimentação — não medida em tempo real |
| `cNH3`    | 750–3000 ppm   | Concentração de NH₃ na alimentação — não medida em tempo real |
| `Ecolumn` | 66.5%–73.5%    | Eficiência de Murphree — parâmetro interno, não mensurável    |

### Consequência para o agente

O agente nunca vê `cH2S`, `cNH3` nem `Ecolumn`, mas essas variáveis influenciam as temperaturas que ele observa. Ele precisa aprender uma política robusta que funcione bem independentemente dos valores dessas variáveis latentes. Isso é alcançado pela **randomização por episódio**: a cada reset, novos valores são sorteados uniformemente nas faixas definidas.

### Por que Pcolumn está na observação?

A pressão da coluna, embora varie entre colunas e ao longo do tempo, é mensurável por instrumentação padrão. Incluí-la na observação permite ao agente adaptar sua política à pressão atual, o que é fisicamente correto — o equilíbrio térmico depende fortemente da pressão.

---

## 3. Espaço de Ações — Ação Incremental

A ação do agente é **incremental** no setpoint de carga térmica:

```
a_t ∈ [-1, +1]  →  ΔQreb = a_t × 0.3 Gcal/h
Qreb(t+1) = clip(Qreb(t) + ΔQreb, 2.4, 6.0)
```

### Por que incremental e não absoluta?

1. **Tracking local** — O agente ajusta a partir do estado atual, não precisa "adivinhar" o valor ótimo absoluto de Qreb. Isso é mais natural para um controlador supervisório.

2. **Suavidade** — Ações incrementais pequenas (±0.3 Gcal/h por step) evitam saltos perigosos de carga térmica, compatível com a dinâmica lenta do processo real.

3. **Comportamento interpretável** — Em produção, o operador pode verificar: o agente está pedindo para subir, descer ou manter? A magnitude do incremento é razoável?

### Por que ΔQreb_max = 0.3 Gcal/h?

A faixa total de Qreb é [2.4, 6.0], ou seja, 3.6 Gcal/h de amplitude. Com ΔQreb_max = 0.3, o agente pode percorrer a faixa completa em ~12 steps (de 30 disponíveis por episódio). Isso dá margem suficiente para convergir ao ótimo sem permitir variações excessivas por step.

### Por que o action space é [-1, +1] e não [-0.3, +0.3]?

O stable-baselines3 funciona melhor com ações normalizadas em [-1, 1]. A conversão para engenharia (`× 0.3`) é feita dentro do `env.step()`. Isso melhora a estabilidade numérica do treinamento.

---

## 4. Normalização — Três Camadas

Existem três níveis de normalização no pipeline, cada um com propósito distinto:

### Camada 1: Normalização dos surrogates (min-max [0,1])

Os modelos Keras foram treinados com entradas e saídas normalizadas por min-max:

```
x_norm = (x - x_min) / (x_max - x_min)    →  [0, 1]
y_norm = (y - y_min) / (y_max - y_min)    →  [0, 1]
```

O `surrogate_env.py` normaliza antes de chamar o surrogate e desnormaliza a saída para valores de engenharia (°C). Os min/max vêm do `sim_results_normalization.json` gerado na etapa 05.

### Camada 2: Normalização da observação (min-max [0,1])

A observação entregue ao agente é normalizada com as faixas conhecidas de cada variável:

```
obs = [Qfeed_norm, Tfeed_norm, Qreb_norm, Pcolumn_norm, Tbottom_norm, Ttop_norm]
```

Cada componente fica em [0, 1]. Isso garante que o observation_space do Gymnasium seja respeitado e que todas as features tenham magnitude comparável.

### Camada 3: VecNormalize (running mean/std)

O `VecNormalize` do stable-baselines3 aplica uma normalização adaptativa com running mean e running standard deviation sobre as observações. Isso é calculado durante o treinamento e salvo em `vec_normalize.pkl`.

**Por que a camada 3 se já temos a camada 2?** A normalização min-max (camada 2) coloca tudo em [0,1], mas a distribuição real das observações durante o treinamento pode não ser uniforme nesse intervalo. O VecNormalize ajusta para média zero e variância unitária considerando a distribuição real, o que melhora o aprendizado das redes neurais do actor e critic.

**Consequência para integração**: Na hora de usar o agente em produção, é obrigatório carregar o `vec_normalize.pkl` e aplicar `env.normalize_obs()` sobre a observação antes de chamar `model.predict()`.

---

## 5. Funções de Recompensa — Design e Trade-offs

### Recompensa A — Trade-off Linear

```
r_t = w1 · ΔT_norm − w2 · Qreb_norm − w3 · var(Tbottom)
```

onde:
- `ΔT_norm = (Tbottom − Ttop) / (Tbottom_max − Ttop_min)` — gradiente térmico normalizado
- `Qreb_norm = (Qreb − Qreb_min) / (Qreb_max − Qreb_min)` — carga térmica normalizada
- `var(Tbottom)` — variância das últimas 5 leituras de Tbottom

**Escolha dos pesos** (`w1=1.0, w2=0.5, w3=0.2`):
- O gradiente térmico é o objetivo principal (peso 1.0)
- A penalização energética é metade do incentivo térmico (peso 0.5), criando um trade-off explícito
- A penalização de variância é leve (peso 0.2), incentivando estabilidade sem dominar o sinal

**Por que normalizar ΔT e Qreb?** Para que os pesos w1, w2, w3 sejam interpretáveis como trade-offs relativos. Sem normalização, as magnitudes absolutas dominariam o balanço.

**Por que usar variância de Tbottom?** Penaliza oscilações do agente — uma política que fica "indo e voltando" terá variância alta e será punida. O deque de 5 posições corresponde à janela de memória curta.

**Propriedades**: robusta a ruído, fácil de interpretar, incentiva convergência suave ao joelho.

### Recompensa B — Eficiência Normalizada

```
r_t = (Tbottom − Ttop) / (Qreb + ε)
```

onde `ε = 0.001` evita divisão por zero.

**Interpretação**: Esta é literalmente a eficiência térmica marginal — graus de gradiente por unidade de energia. A curva `ΔT / Qreb` tem um pico natural no joelho operacional, onde o retorno marginal de energia é máximo.

**Propriedades**: sinal mais nítido no joelho (convergência potencialmente mais rápida), mas mais sensível a ruído porque o denominador amplifica variações em regiões de Qreb baixo.

### Por que duas recompensas?

A escolha da função de recompensa é uma das decisões mais impactantes em RL. Testar duas formulações com propriedades complementares permite:
- Verificar se ambas convergem para o mesmo ponto operacional (validação cruzada)
- Escolher a mais adequada para o ambiente final (surrogate vs planta)
- A Recompensa A é mais conservadora para produção; a B é mais informativa para treinamento rápido

---

## 6. Estrutura do Episódio

### Inicialização diversificada

A cada reset, o ambiente:

1. Sorteia `Qfeed`, `Tfeed`, `cH2S`, `cNH3`, `Pcolumn`, `Ecolumn` uniformemente nas faixas
2. Sorteia `Qreb` inicial em uma de três faixas com igual probabilidade:
   - **Baixo**: [2.4, 3.59] Gcal/h — abaixo do joelho
   - **Médio**: [3.59, 4.77] Gcal/h — próximo ao joelho
   - **Alto**: [4.77, 6.0] Gcal/h — acima do joelho

**Por que três faixas?** Para que o agente aprenda três comportamentos:
- Partindo de baixo → precisa **subir** Qreb
- Partindo do meio → precisa fazer **ajuste fino**
- Partindo de alto → precisa **descer** Qreb

Se inicializássemos sempre no mesmo ponto, o agente aprenderia apenas uma direção.

### Duração do episódio: 30 steps

O episódio dura no máximo 30 steps (truncamento) ou termina antes por violação de segurança.

**Por que 30?** Com ΔQreb_max = 0.3 e faixa total de 3.6 Gcal/h, o agente consegue percorrer toda a faixa em 12 steps. Os 30 steps dão margem (~2.5x) para exploração, ajuste fino e estabilização. É longo o suficiente para convergir, curto o suficiente para episódios rápidos no surrogate.

### Distúrbios fixos por episódio

As variáveis latentes (`cH2S`, `cNH3`, `Ecolumn`) e as variáveis de contexto (`Qfeed`, `Tfeed`, `Pcolumn`) são sorteadas no reset e **permanecem constantes** durante o episódio.

**Justificativa física**: No processo real, essas variáveis mudam lentamente (escala de horas). Dentro de uma janela de atuação do agente (minutos), é razoável tratá-las como constantes. Entre episódios, a randomização cobre todo o espaço operacional.

---

## 7. Segurança — Limites e Terminação

### Limites rígidos de Qreb

```
Qreb ∈ [2.4, 6.0] Gcal/h
```

O `np.clip()` no `env.step()` impede fisicamente que Qreb saia dessa faixa, independentemente da ação do agente. Mesmo que a política ordene ΔQreb = +0.3 quando Qreb = 5.95, o resultado será 6.0.

### Limites de temperatura e penalidade

```
Tbottom ∈ [155, 175] °C
Ttop ∈ [55, 170] °C
```

Se qualquer temperatura sair desses limites após uma ação:
- O agente recebe **penalidade severa** (`reward = -50`)
- O episódio é **terminado imediatamente** (`terminated = True`)

**Por que terminação e não apenas penalidade?** A terminação ensina ao agente que estados inseguros são "becos sem saída" — não há como se recuperar. Isso cria um gradiente de política muito forte para evitar essas regiões. Uma penalidade sem terminação permitiria ao agente "passar pela região perigosa" e voltar, o que é inaceitável na operação real.

**Por que -50?** A reward típica por step fica em [-1, +1] para Recompensa A e [0, 30] para Recompensa B. Uma penalidade de -50 é suficientemente negativa para dominar o retorno do episódio inteiro, tornando qualquer trajetória insegura pior que qualquer trajetória segura.

---

## 8. Hiperparâmetros do PPO

| Parâmetro        | Valor  | Justificativa                                                     |
|------------------|--------|-------------------------------------------------------------------|
| `learning_rate`  | 3e-4   | Valor padrão robusto para PPO; convergência estável               |
| `n_steps`        | 2048   | Steps coletados antes de cada atualização; bom balanço bias/var   |
| `batch_size`     | 64     | Minibatch para SGD; 2048/64 = 32 minibatches por época            |
| `n_epochs`       | 10     | Épocas de SGD por atualização; reutiliza os dados coletados       |
| `gamma`          | 0.99   | Fator de desconto alto — o agente valoriza recompensas futuras    |
| `gae_lambda`     | 0.95   | GAE λ alto — menor viés, mais variância                           |
| `clip_range`     | 0.2    | Clipping do PPO — limita mudanças na política                     |
| `ent_coef`       | 0.01   | Bônus de entropia — incentiva exploração inicial                  |
| `vf_coef`        | 0.5    | Peso da loss do value function — balanço actor/critic             |
| `max_grad_norm`  | 0.5    | Gradient clipping — estabilidade numérica                         |

### Por que gamma = 0.99?

Com 30 steps por episódio, γ^30 ≈ 0.74. Isso significa que o agente dá ~74% do peso à recompensa 30 steps no futuro. Um gamma alto é adequado porque o objetivo é a convergência ao joelho (longo prazo), não a recompensa imediata.

### Por que ent_coef = 0.01?

O coeficiente de entropia incentiva a política a manter alguma aleatoriedade durante o treinamento (exploração). O valor 0.01 é baixo o suficiente para não atrapalhar a convergência, mas suficiente para evitar colapso prematuro da política em um único comportamento.

---

## 9. Arquitetura da Rede Neural (MlpPolicy)

O `MlpPolicy` padrão do stable-baselines3 cria:

- **Actor** (política): MLP com 2 camadas ocultas de 64 neurônios cada, ativação `tanh`, saída em distribuição Gaussiana com média e log-std
- **Critic** (value function): MLP separada, mesma arquitetura, saída escalar (estimativa de V(s))

```
Actor:  obs(6) → Dense(64, tanh) → Dense(64, tanh) → μ(1), log_σ(1)
Critic: obs(6) → Dense(64, tanh) → Dense(64, tanh) → V(1)
```

### Por que redes separadas para actor e critic?

O stable-baselines3 usa por padrão redes separadas (não compartilhadas). Isso evita que gradientes conflitantes entre a loss de política e a loss de value function interfiram um no outro, resultando em treinamento mais estável.

### Por que tanh e não ReLU?

A ativação `tanh` é preferida em PPO porque:
- Saídas limitadas em [-1, 1] mantêm ativações em faixa controlada
- Gradientes mais suaves nas bordas (vs ReLU com gradiente zero para negativos)
- O PPO original foi validado extensivamente com tanh

---

## 10. VecNormalize — Normalização Adaptativa

O wrapper `VecNormalize` mantém estimativas correntes de média e variância das observações:

```
obs_normalizado = (obs - running_mean) / sqrt(running_var + 1e-8)
```

### Na observação

- `norm_obs=True` — normaliza observações para o agente
- `clip_obs=10.0` — limita observações normalizadas a [-10, 10]

### Na recompensa

- **Treino**: `norm_reward=True` — normaliza a reward durante o treinamento, estabilizando a magnitude do sinal de aprendizado
- **Avaliação**: `norm_reward=False` — recompensas em escala original para métricas interpretáveis

### Persistência

As estatísticas são salvas em `vec_normalize.pkl`. Na integração com o simulador, é **obrigatório** carregar esse arquivo para que o agente receba observações na mesma escala em que foi treinado.

---

## 11. Callbacks de Treinamento

### EvalCallback

A cada 10.000 timesteps, executa 20 episódios determinísticos em um ambiente de avaliação separado. Se a reward média for a melhor até o momento, salva o modelo como `best_model.zip`.

**Por que um ambiente de avaliação separado?** O ambiente de treino tem VecNormalize com estatísticas sendo atualizadas (training=True). O de avaliação usa estatísticas congeladas (training=False), garantindo avaliação consistente.

### RewardLoggerCallback

Registra reward acumulada e comprimento de cada episódio, imprimindo estatísticas (média e std dos últimos 50 episódios) a cada 5.000 timesteps no log.

---

## 12. Reprodutibilidade

- `SEED = 42` é usado para o PPO, para o gerador de números aleatórios do ambiente e para o ambiente de avaliação (seed + 1000)
- O `numpy.random.default_rng(seed)` garante sequências reprodutíveis de condições iniciais
- Metadados do treino (hiperparâmetros, seed, reward type) são salvos em `training_meta.json`

---

## 13. Resumo das Escolhas de Projeto

| Aspecto                  | Escolha                      | Alternativa descartada       | Motivo                                     |
|--------------------------|------------------------------|------------------------------|--------------------------------------------|
| Algoritmo                | PPO                          | TD3, SAC                     | Estabilidade, robustez à escala de reward   |
| Formulação               | POMDP                        | MDP completo                 | Variáveis latentes não mensuráveis na planta|
| Ação                     | Incremental (ΔQreb)          | Absoluta (Qreb)              | Tracking, suavidade, segurança              |
| Recompensa               | Duas formulações (A e B)     | Uma única                    | Validação cruzada, robustez                 |
| Inicialização            | Três faixas de Qreb          | Uniforme                     | Ensina subir, descer e manter               |
| Segurança                | Terminação + penalidade      | Apenas penalidade            | Ensina que estados inseguros são terminais  |
| Normalização             | 3 camadas (surrogate, obs, VecNorm) | Apenas uma             | Cada camada tem propósito distinto          |
| Distúrbios               | Fixos por episódio           | Variáveis dentro do episódio | Coerência com dinâmica lenta do processo    |
