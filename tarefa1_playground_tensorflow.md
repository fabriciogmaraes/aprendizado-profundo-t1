# Tarefa 1 — Playground TensorFlow (Estudo Dirigido)

**Disciplina:** Aprendizado Profundo — PPgTI/UFRN
**Docente:** Prof. Josenalde Oliveira

## Exploração dos itens (a) a (g)

**a) Configuração inicial (dataset circular, 2 features, camadas 4→2 neurônios, tanh)**

A rede separa facilmente as classes porque o padrão é radialmente simples. A primeira camada aprende fronteiras lineares simples (retas), e a segunda camada combina essas retas em curvas mais complexas — quanto mais camadas, mais complexas as fronteiras que a rede consegue representar.

**b) Função de ativação: tanh → ReLU → Linear**

Trocando tanh por ReLU, a rede converge mais rápido (a curva de loss cai mais rapidamente e estabiliza em menos épocas), porque ReLU não satura para valores positivos (ao contrário de tanh, que satura em ±1 e sofre com gradientes pequenos). As fronteiras de decisão ficam mais "anguladas"/lineares por partes, em vez de suaves como com tanh.

Com ativação Linear, a rede não consegue separar bem padrões não-lineares (como o circular ou espiral): uma composição de transformações lineares continua sendo linear, então a rede perde a capacidade de aprender fronteiras curvas, não importa quantas camadas tenha.

**c) Variando a taxa de aprendizagem (tanh fixo)**

- Taxa muito alta: o treinamento fica instável, a loss oscila ou diverge, pode nunca convergir.
- Taxa muito baixa: a convergência fica muito lenta, precisa de muito mais épocas pra chegar a um resultado razoável.
- Existe uma faixa intermediária "ótima" onde a convergência é rápida e estável.

**d) 1 camada oculta, 3 neurônios — risco de mínimos locais**

O tempo de treinamento varia muito entre execuções porque a inicialização dos pesos é aleatória, e a rede pode ficar presa em mínimos locais (ou platôs) da função de perda — dependendo do ponto de partida, o gradiente descendente não encontra o caminho pra uma solução boa.

**e) Reduzindo para 2 neurônios**

A rede não consegue encontrar solução consistentemente. Isso ocorre porque o modelo tem capacidade insuficiente (poucos parâmetros/graus de liberdade) para representar a complexidade do padrão dos dados — é um caso de underfitting estrutural.

**f) Aumentando para 8 neurônios**

O treinamento fica rápido e não trava. Redes maiores (mais parâmetros) têm superfícies de perda com múltiplos caminhos viáveis até uma boa solução, então a chance de ficar presa em mínimo local ruim diminui — há mais "rotas" para o gradiente descer.

**g) Dataset espiral, 4 camadas ocultas × 8 neurônios**

O treinamento demora mais e pode ficar preso em platôs (regiões onde o gradiente é quase zero, mas a loss ainda não é boa). Aumentando a taxa de aprendizagem, ajuda a "escapar" de platôs mais rápido, mas se ficar alta demais causa instabilidade. Adicionando regularização L1/L2, penaliza pesos grandes, reduzindo overfitting, mas pode piorar levemente a capacidade de ajustar a espiral se a constante de regularização for alta demais (underfitting).

## Exercícios de fixação

**1) Três funções de ativação populares**

- **ReLU**: f(x) = max(0, x) — formato de "dobradiça", zero para x negativo e linear para x positivo.
- **Sigmoide**: f(x) = 1/(1+e⁻ˣ) — formato de "S", satura entre 0 e 1.
- **Tanh**: formato de "S" também, mas satura entre -1 e 1, centrada em zero.

**2) MLP: 10 entradas → 50 neurônios (oculta, ReLU) → 3 neurônios (saída, ReLU)**

Usando a notação de Géron (X: uma linha por instância, uma coluna por feature; W: uma linha por neurônio de entrada, uma coluna por neurônio na camada; b: um viés por neurônio):

- **a)** Formato de X: `(m, 10)`, onde m = número de instâncias no lote.
- **b)** Formato de Wh: `(10, 50)` — 10 entradas × 50 neurônios ocultos. Formato de bh: `(50,)` — um viés por neurônio oculto.
- **c)** Formato de Wo: `(50, 3)` — 50 entradas (saída da oculta) × 3 neurônios de saída. Formato de bo: `(3,)` — um viés por neurônio de saída.
- **d)** Formato de Y: `(m, 3)`.
- **e)** Equação da saída da rede, definindo Z = ReLU(X·Wh + bh) como saída da camada oculta:

  Y = ReLU(Z·Wo + bo) = ReLU(ReLU(X·Wh + bh)·Wo + bo)

**3) Neurônios e função de ativação na camada de saída**

- **Spam ou não-spam**: 1 neurônio de saída, função sigmoide (classificação binária).
- **MNIST** (10 dígitos): 10 neurônios de saída, função softmax (classificação multiclasse mutuamente exclusiva).
- **Cotação do dólar amanhã**: 1 neurônio de saída, ativação linear (identidade) — é um problema de regressão, não classificação.

**4) Hiperparâmetros de uma MLP básica e overfitting**

Hiperparâmetros ajustáveis: número de camadas ocultas, número de neurônios por camada, função de ativação, taxa de aprendizagem, número de épocas, tamanho do batch, tipo e taxa de regularização (L1/L2), otimizador (SGD, Adam, etc.), inicialização dos pesos, dropout.

Se a MLP sobreajustar (overfitting), soluções possíveis: reduzir a complexidade do modelo (menos camadas/neurônios), adicionar regularização L1/L2, usar dropout, aumentar o conjunto de treino (ou fazer data augmentation), usar early stopping, ou ajustar o batch size/número de épocas.
