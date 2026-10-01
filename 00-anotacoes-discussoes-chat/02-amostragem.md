### O que é Latin Hypercube Sampling (LHS)

**Latin Hypercube Sampling** é uma técnica de *Design of Experiments* para gerar amostras em um espaço de parâmetros de alta dimensão de forma **mais “uniforme”** do que amostragem aleatória simples (Monte Carlo), usando  **estratificação por dimensão** .

A ideia central:

* Para cada variável de entrada xj (com j=1,…,d), você divide o intervalo [0,1] em N estratos (faixas) de mesma “probabilidade”/largura.
* Você escolhe **exatamente 1 ponto em cada estrato** por dimensão.
* Depois você **permuta** as escolhas entre dimensões para formar N vetores d-dimensionais.

Isso garante que,  **em cada dimensão individualmente** , a projeção 1D dos pontos cubra bem o intervalo (cada “faixa” é usada uma vez).

---

### Por que LHS é útil para criar datasets para treinar DNNs

Quando você está gerando dados sintéticos (por simulação/solver/experimento virtual) para treinar uma DNN, LHS costuma ajudar porque:

1. **Melhor cobertura do domínio com menos amostras**
   * Em problemas caros (simulações CFD/FEM, química, etc.), você quer maximizar informação por amostra.
   * LHS reduz “buracos” e aglomerações típicas do aleatório puro.
2. **Menos viés de distribuição marginal**
   * Cada variável é bem coberta; isso tende a melhorar generalização quando o modelo precisa aprender dependências globais.
3. **Bom “ponto de partida” para estratégias adaptativas**
   * Você pode usar LHS para um *dataset inicial* e depois refinar com *active learning* (amostrar onde o erro/incerteza é maior).

---

### Receita prática: como usar LHS para dataset de DNN

#### 1) Defina o domínio e (se possível) uma distribuição alvo

* Se você quer uniformidade em um intervalo [a,b]: gere em [0,1] e transforme por x=a+(b−a)u.
* Se você quer seguir uma distribuição (ex.: normal, lognormal): use **transformação por quantil** (inversa da CDF):

  x=F−1(u), onde u∼LHS em [0,1].

Isso é comum em UQ ( *uncertainty quantification* ): LHS “uniforme” em probabilidade e depois mapeado para a distribuição física.

#### 2) Escolha N (nº de amostras) com noção de dimensão

* LHS melhora cobertura, mas  **não derrota a maldição da dimensionalidade** .
* Regra prática: quanto maior d, mais N precisa crescer para cobrir interações (não só marginais).
* Se o fenômeno tiver não linearidades fortes/interações, considere:
  * aumentar N,
  * ou usar amostragem adaptativa,
  * ou impor estrutura (ex.: reduzir dimensão com PCA/autoencoder/variáveis latentes).

#### 3) Use variantes “melhoradas” do LHS

O LHS “básico” garante marginais boas, mas pode ainda deixar pontos “perto demais” no espaço conjunto. Variantes úteis:

* **Maximin LHS** : escolhe a permutação para **maximizar a menor distância** entre pontos (melhor espalhamento global).
* **Centered LHS** : usa o centro de cada estrato (reduz aleatoriedade; às vezes melhora reprodutibilidade).
* **Orthogonal LHS (OLHS)** : tenta balancear melhor projeções em subespaços (útil quando interações 2D/3D importam).

Em datasets para DNN, *maximin* costuma ser uma boa escolha quando você quer cobertura geométrica.

---

### Pontos críticos (onde muita gente erra)

#### 1) Correlação entre variáveis

LHS padrão assume independência ao gerar as dimensões. Se suas entradas têm **correlação** (ex.: parâmetros físicos correlacionados), você precisa impor isso depois, por exemplo com:

* métodos tipo **Iman–Conover** (ajusta a correlação preservando marginais),
* ou amostrar diretamente de uma distribuição multivariada (ex.: normal multivariada) e então mapear.

Se você ignorar correlação real, a DNN pode aprender regiões “fisicamente impossíveis”.

#### 2) Variáveis discretas/categóricas

LHS é naturalmente contínuo. Para discretas:

* mapeie estratos para níveis discretos (com cuidado para manter balanceamento),
* ou use *design* misto (parte LHS contínua + plano fatorial/estratificação para categorias).

#### 3) Restrições e regiões inválidas

Se o domínio tem restrições (ex.: x1+x2≤1, estabilidade numérica, limites físicos):

* gerar LHS no hipercubo e depois rejeitar pode  **quebrar a estratificação** .
  Alternativas:
* reparametrizar para um espaço sem restrição,
* usar amostragem condicionada,
* ou gerar LHS e então aplicar um “repair”/projeção (avaliando se distorce demais).

#### 4) Treino da DNN: distribuição do dataset vs. distribuição de uso

Se no deploy você verá uma distribuição diferente, LHS “uniforme” pode não ser ideal. Às vezes o certo é:

* LHS para cobertura + *reweighting* na loss,
* ou mistura: parte LHS (exploração) + parte amostrada da distribuição real (exploração/uso).

---

### LHS vs alternativas (bem comum em datasets sintéticos)

* **Monte Carlo aleatório** : simples, mas precisa de mais pontos para a mesma cobertura marginal.
* **Sequências quasi-random (Sobol/Halton)** : muitas vezes dão cobertura global excelente e boa convergência para integração; Sobol é muito usado.
* **Design fatorial/grade** : explode com dimensão; bom só para d pequeno.

Na prática:

* se você quer “boa cobertura com N moderado”: **LHS (maximin)** é ótimo;
* se quer “preenchimento progressivo” (poder adicionar pontos sem refazer tudo): **Sobol** costuma ser mais conveniente;
* se tem restrições fortes: pode precisar de abordagem específica.

---

### Para eu te orientar melhor

Você está gerando dados para DNN em qual cenário?

1. Quantas variáveis de entrada (d) e quantas amostras você consegue (N)?
2. As entradas têm distribuição conhecida (uniforme/normal/log/etc.) ou é só intervalo?
3. Existem restrições/combinações inválidas?
4. O objetivo é regressão (aproximar um solver), classificação, ou surrogate + UQ?

Com isso eu consigo sugerir um desenho (LHS básico vs maximin vs OLHS vs Sobol, e como mapear distribuições/correlações) bem alinhado ao seu caso.
