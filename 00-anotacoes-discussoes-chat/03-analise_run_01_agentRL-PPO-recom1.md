Ótimo, agora dá para ser  **cirúrgico** .
Li a definição completa das rewards — e **o comportamento “laranja” é 100% explicável pela forma matemática da Recompensa A** .

Vou estruturar assim:

1. **Por que a Reward A gera exatamente aqueles gráficos**
2. **Qual termo é o culpado principal**
3. **O que mudar na função (com sugestões concretas)**
4. **Versão recomendada da Reward A (industrial + PPO-friendly)**

---

## 1. Por que a Reward A gera alto KL, clipping e colapso de entropia

Reward A (laranja):

$$
r_t =
w_1 (T_{bottom} - T_{top})

* w_2 \dot Q_{reb}
* w_3 ,\mathrm{var}(T_{bottom})
$$

Essa reward tem  **três características perigosas para PPO** :

---

### (A) **Gradiente não saturado em ΔT**

O termo
$$
w_1 (T_{bottom} - T_{top})
$$

* cresce **linearmente**
* não tem platô
* não tem noção de “ótimo local”

➡️ Para o PPO,  **sempre vale a pena empurrar mais** .

Isso explica:

* `approx_kl` crescente
* policy tentando dar passos grandes
* PPO sendo obrigado a clippar

---

### (B) Penalidade de energia **linear e fraca no regime do joelho**

$$
* w_2 \dot Q_{reb}
$$

Perto do joelho:

* ΔT cresce rápido
* Q̇ cresce mais devagar

➡️ O gradiente líquido **favorece agressividade**
➡️ PPO vê vantagem em “forçar” a policy

---

### (C) Penalidade de variância = **gradiente instável**

$$
* w_3 ,\mathrm{var}(T_{bottom})
$$

Esse termo é o mais subestimado — e  **muito importante** :

* Variância:
  * não é suave
  * depende da janela
  * muda abruptamente entre steps

➡️ Isso injeta **ruído de alto ganho** no gradiente da policy
➡️ PPO reage tentando compensar com updates maiores
➡️ Clip fraction explode

---

## 2. Quem é o principal culpado?

### 🟥 O maior vilão é **a combinação** de:

> **ΔT linear não saturado + penalidade de variância direta**

Separadamente, cada um é administrável.
Juntos, eles criam exatamente o regime que você observou:

* policy agressiva
* entropia colapsando
* PPO “freando” o tempo todo

---

## 3. O que mudar — sugestões objetivas

### 🔧 Mudança 1 — saturar o ganho térmico (ESSENCIAL)

Troque:

```text
(Tbottom − Ttop)
```

por algo com  **platô físico** :

**Opção A (mais simples, recomendo):**

```text
tanh((Tbottom − Ttop − ΔT_ref)/σ)
```

**Opção B (quadrática local):**

```text
− (Tbottom − Ttop − ΔT_opt)^2
```

➡️ Isso cria um **joelho explícito na reward**
➡️ O gradiente **vai a zero no ótimo**

---

### 🔧 Mudança 2 — normalizar ou suavizar a penalidade de energia

Troque:

```text
− w2 · Q̇_reb
```

por:

```text
− w2 · (Q̇_reb − Q̇_ref)
```

ou:

```text
− w2 · (Q̇_reb / Q̇_nom)
```

➡️ Evita gradiente constante empurrando para extremos

---

### 🔧 Mudança 3 — reformular o termo de variância (MUITO importante)

❌ Evite:

```text
var(Tbottom)
```

✅ Prefira  **proxy suave de instabilidade** :

```text
|dTbottom/dt|
```

ou:

```
|Tbottom(t) − Tbottom(t−1)|
```

ou:

```text
EMA(|ΔTbottom|)
```

➡️ Mantém penalização de instabilidade
➡️ Remove ruído não diferenciável

---

## 4. Reward A — versão recomendada (PPO-safe + industrial)

Minha sugestão concreta:

```text
r_t =
+ w1 · tanh((Tbottom − Ttop − ΔT_ref)/σ)
− w2 · (Q̇_reb / Q̇_nom)
− w3 · |Tbottom(t) − Tbottom(t−1)|
```

### Propriedades dessa forma:

* ✔️ gradiente **limitado**
* ✔️ ótimo explícito
* ✔️ PPO não entra em regime de clipping crônico
* ✔️ interpretação operacional preservada
* ✔️ robusta para surrogate **e** Aspen

---

## 5. O que você deve observar nos gráficos após a mudança

Se estiver correto, você verá:

* `approx_kl` → cai para < 0.01
* `clip_fraction` → < 0.1
* `entropy_loss` → estabiliza
* reward converge **sem ruído crescente**

👉 Aí sim  **mais tempo de treino passa a fazer sentido** .

---

## Conclusão direta

> O caso laranja **não precisa de mais steps**
> Ele precisa de **saturação e suavização da reward**

A forma atual da Reward A é  **fisicamente correta** ,
mas  **numericamente agressiva demais para PPO** .

Se quiser, no próximo passo posso:

* te ajudar a **calibrar `ΔT_ref` e `σ` a partir dos dados Aspen**
* ou comparar essa reward reformulada com a Reward B (razão) de forma híbrida.
