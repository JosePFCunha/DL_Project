# DL_Project
Deep Learning Project

Workflow sugerido pelo Professor.


_**1)**_** Definição do problema e recolha de dados**. 
(que tipos de dados usar, que tipo de problema estamos a resolver, qual será o output do modelo);


**Definição do Problema:**

O problema é uma tarefa de classificação binária, onde o objetivo é distinguir entre imagens que contenham cavalos e imagens que contenham humanos.

**Recolha de Dados:**

Para este problema, foram recolhidas imagens que contenham cavalos e imagens que contenham humanos. Esses dados podem ser obtidos de diversas fontes, como bancos de imagens públicos, conjuntos de dados específicos para reconhecimento de objetos ou animais, ou até mesmo imagens coletadas manualmente.

**Tipos de Dados:**

Os tipos de dados utilizados neste problema são imagens digitais. Cada imagem é composta por pixels, onde cada pixel pode ter diferentes valores de intensidade de cor para os canais RGB (vermelho, verde e azul).

**Tipo de Problema:**

O problema é um problema de classificação binária, onde cada imagem é classificada em uma de duas classes: "cavalo" ou "humano".

**Output do Modelo:**

O output do modelo será uma previsão de probabilidade para cada classe. Em outras palavras, o modelo irá produzir uma probabilidade de que a imagem contenha um cavalo e uma probabilidade de que a imagem contenha um humano. A classe com a maior probabilidade será a previsão final do modelo. Por exemplo, se a probabilidade de conter um cavalo for maior que a probabilidade de conter um humano, o modelo irá classificar a imagem como contendo um cavalo, e vice-versa.

2) Definição do objetivo – qual a métrica a otimizar; qual o target para essa métrica;

3) Definição do protocolo de avaliação (hold-out, K-fold, etc);

4) Preparação dos dados (normalização dos dados, feature engineering);

5) Desenvolvimento de um modelo simples que sirva como baseline;

6) Adição de complexidade até gerar overfitting (acrescentar layers, aumentar o número
de neurónios, treinar por mais tempo);

7) Regularização e tuning dos híper-parâmetros (acrescentar regularização, dropout,
tentar diferentes valores de híper parâmetros);

8) Argumentar sobre a utilidade do modelo criado.


   
