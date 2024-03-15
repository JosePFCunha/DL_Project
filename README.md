# DL_Project
Deep Learning Project

Workflow sugerido pelo Professor.


_**1)**_

**Definição do problema e recolha de dados** 
(que tipos de dados usar, que tipo de problema estamos a resolver, qual será o output do modelo);


Definição do Problema:

O problema é uma tarefa de classificação binária, onde o objetivo é distinguir entre imagens que contenham cavalos e imagens que contenham humanos.

Recolha de Dados:

Para este problema, foram recolhidas imagens que contenham cavalos e imagens que contenham humanos. Esses dados podem ser obtidos de diversas fontes, como bancos de imagens públicos, conjuntos de dados específicos para reconhecimento de objetos ou animais, ou até mesmo imagens coletadas manualmente.

Tipos de Dados:

Os tipos de dados utilizados neste problema são imagens digitais. Cada imagem é composta por pixels, onde cada pixel pode ter diferentes valores de intensidade de cor para os canais RGB (vermelho, verde e azul).

Tipo de Problema:

O problema é um problema de classificação binária, onde cada imagem é classificada em uma de duas classes: "cavalo" ou "humano".

Output do Modelo:

O output do modelo será uma previsão de probabilidade para cada classe. Em outras palavras, o modelo irá produzir uma probabilidade de que a imagem contenha um cavalo e uma probabilidade de que a imagem contenha um humano. A classe com a maior probabilidade será a previsão final do modelo. Por exemplo, se a probabilidade de conter um cavalo for maior que a probabilidade de conter um humano, o modelo irá classificar a imagem como contendo um cavalo, e vice-versa.



_**2)**_ 

**Definição do objetivo**
(qual a métrica a otimizar; qual o target para essa métrica)

Objetico:
O objetivo deste código é treinar modelos de Deep Learning para classificar imagens de cavalos e humanos, e depois avaliar o desempenho desses modelos em relação aos dados de treino e validação.
A métrica a ser otimizada para este problema de classificação binária é a Accuracy. Quando escrevemos Accuracy, leia-se Precisão. A Accuracy é uma medida comum de desempenho em problemas de classificação e representa a proporção de exemplos classificados corretamente pelo modelo em relação ao total de exemplos.

Métrica a Otimizar: Accuracy

O objetivo é maximizar a Accuracy do modelo. Isso significa que o modelo deve ser capaz de classificar corretamente o maior número possível de imagens, distinguindo entre imagens que contenham cavalos e imagens que contenham humanos. Quanto maior a Accuracy, melhor será o desempenho do modelo na tarefa de classificação.



_**3)**_

**Definição do protocolo de avaliação**
(hold-out, K-fold, etc);

O protocolo de avaliação usado neste código é um tipo de hold-out, onde o conjunto de dados é dividido em três grupos: treino, teste e validação. O grupo de treino é usado para treinar os modelos, o grupo de validação é usado para ajustar e avaliar o desempenho durante o treino, e o grupo de teste é usado para avaliar o desempenho final do modelo após o treino.



_**4)**_

**Preparação dos dados**
(normalização dos dados, feature engineering);


A preparação dos dados neste código envolve várias etapas:

Carregamento do Dataset

O Dataset "horses_or_humans" é carregado usando a função tfds.load do TensorFlow Datasets. Este conjunto de dados contém imagens de cavalos e humanos.

Divisão do Dataset

O Dataset é dividido em três partes usando o método tfds.even_splits. Essas partes são utilizadas para formar os grupos de treino, teste e validação.

Normalização das Imagens

As imagens são normalizadas dividindo-se os valores dos pixels pelo valor máximo (255). Isso garante que os valores dos pixels estejam na faixa de 0 a 1, facilitando o treino da Neural Network.

Redimensionamento das Imagens

As imagens são redimensionadas para o tamanho desejado de 100x100 pixels usando a função tf.image.resize_with_pad. Isso garante que todas as imagens tenham as mesmas dimensões, o que é necessário para alimentar as imagens na Neural Network.

Agrupamento das Imagens em Batches

As imagens são agrupadas em lotes (batches) usando o método Batch. Isso permite que várias imagens sejam processadas simultaneamente durante o treino, melhorando a eficiência do processo.
Os dados são preparados com uma formatação correta, normalizados e redimensionados para serem alimentados nos modelos de Neural Network. As diferentes fases são cruciais para garantir que os dados estejam prontos para o treino e a avaliação dos modelos.



_**5)**_

**Desenvolvimento de um modelo simples que sirva como baseline**

O modelo baseline model é um Simple Neural Network Architecture e é composto por 4 Layers: Input layer, Flatten layer, Dense layer e Output layer.
Conforme mostrado abaixo.

baseline_model = Sequential([
  layers.Input(shape=(100, 100, 3)),
  layers.Flatten(),
  layers.Dense(300, activation='relu'),
  layers.Dense(100, activation='relu'),
  layers.Dense(10, activation='relu'),
  layers.Dense(1, activation='linear')
])



_**6)**_

**Adição de complexidade até gerar overfitting**
(acrescentar layers, aumentar o número de neurónios, treinar por mais tempo);

# Adicionar camadas adicionais ao modelo
baseline_model = Sequential([
  layers.Input(shape=(100, 100, 3)),
  layers.Flatten(),
  layers.Dense(300, activation='relu'),
  layers.Dense(200, activation='relu'),  # Adicione uma camada densa com mais neurônios
  layers.Dense(100, activation='relu'),  # Adicione uma camada densa com mais neurônios
  layers.Dense(10, activation='relu'),
  layers.Dense(1, activation='linear')
])

# Compilar o modelo
baseline_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Treinar o modelo por mais épocas
epochs=50  # Aumente o número de épocas de treinamento
history = baseline_model.fit(
  resized_ds_train,
  validation_data=resized_ds_val,
  epochs=epochs
)


_**7)**_

**Regularização e tuning dos híper-parâmetros**
(acrescentar regularização, dropout, tentar diferentes valores de híper parâmetros);

from tensorflow.keras import regularizers

# Adicionar regularização Dropout
baseline_model = Sequential([
  layers.Input(shape=(100, 100, 3)),
  layers.Flatten(),
  layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Adicionando regularização L2
  layers.Dropout(0.5),  # Adicionando Dropout com uma taxa de 0.5
  layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Adicionando regularização L2
  layers.Dropout(0.5),  # Adicionando Dropout com uma taxa de 0.5
  layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Adicionando regularização L2
  layers.Dropout(0.5),  # Adicionando Dropout com uma taxa de 0.5
  layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Adicionando regularização L2
  layers.Dropout(0.5),  # Adicionando Dropout com uma taxa de 0.5
  layers.Dense(1, activation='linear')
])

# Compilando o modelo
baseline_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Treinando o modelo com o número de épocas otimizado anteriormente
epochs = 50
history = baseline_model.fit(
  resized_ds_train,
  validation_data=resized_ds_val,
  epochs=epochs
)


_**8)**_

**Argumentar sobre a utilidade do modelo criado.**

O modelo apresentado satisfaz claramente o objetivo ao qual nos proposemos, com uma índice alto de Accuracy e um índice pequeno de Loss.
Apesar de na primeira fase do modelo a performance ao nível da Accuracy não ser alta (abaixo de 50% e uma loss grande), procedemos à alteração do modelo para CNN.
Desta forma, a eficácia do modelo evoluiu considerávelmente e o índice de Accuracy, melhorou de acordo, tornando o modelo mais preciso.

Podemos então inferir que é com alta precisão que conseguimos uma classificação e distinção clara entre imagens com humanos e imagens com cavalos, tal como pretendido.


   
