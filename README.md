# Aprendizado Profundo — Trabalho 1: Redes Neurais e Convolucionais

**Disciplina:** Aprendizado Profundo — PPgTI/UFRN
**Docente:** Prof. Josenalde Oliveira
**Aluno:** Fabricio Guimarães Alcântara da Silva

## Tarefa 1 — Playground TensorFlow (Estudo Dirigido)

Relatório com a exploração dos hiperparâmetros no TensorFlow Playground e as respostas dos exercícios de fixação.

📄 [tarefa1_playground_tensorflow.md](./tarefa1_playground_tensorflow.md)

## Tarefa 2 — Otimização do MNIST com Optuna

Busca de hiperparâmetros com Optuna sobre uma MLP no MNIST, a partir do código base do professor, com treino final e avaliação no conjunto de teste.

- **Notebook (Colab):** https://colab.research.google.com/drive/1zL1PkCmRZAkhvXXAsTqGeJ_uOsMwXEu9?usp=sharing
- **Resultado:** melhor configuração encontrada (2 camadas ocultas, learning rate ≈ 0.000768, dropout ≈ 0.391, batch size 64), acurácia de validação 0,9810.
- **Acurácia no conjunto de teste:** 0,9814 — muito próxima da acurácia de treino/validação, indicando baixo overfitting. Erros concentrados em confusões visuais plausíveis entre dígitos (ex: 9↔3, 7↔2).

## Tarefa 3 — Reprodução das CNNs do artigo Silva Filho (2022) — Multiprova Corretor

Reprodução, em subconjuntos do EMNIST, das melhores arquiteturas de CNN identificadas no artigo *Classificação de Caracteres Manuscritos para Correção Automática do Sistema Multiprova* (Silva Filho, Medeiros e Maia, 2022), para os problemas de reconhecimento de dígitos (1-5), letras (A-E) e V/F.

- **Notebook (Colab):** https://colab.research.google.com/drive/14DHmamJTKI9qRPM295PAk4Rx02ZSVLwt?usp=sharing

### 3.1 Parâmetros treináveis e tamanho dos modelos

| Modelo | Parâmetros treináveis | Tamanho (MB) | Acurácia no teste |
|---|---|---|---|
| Dígitos (1-5) | 126.789 | 1,501 | 0,9965 |
| Letras (A-E) | 126.789 | 1,501 | 0,9550 |
| V ou F | 187.778 | 2,192 | 0,9932 |

Os três classificadores reproduziram bem as arquiteturas do artigo original, com acurácias próximas ou superiores às reportadas por Silva Filho (2022) — considerando que aqui foram usadas menos classes por classificador (5, 5 e 2, contra 11, 11 e 3 no artigo). O modelo de dígitos teve o melhor desempenho, com pouquíssima confusão entre classes. O de letras foi o mais desafiador, com confusões visuais esperadas (C↔E, B↔D). O de V/F teve ótimo desempenho, com apenas 7 erros no total.

### 3.2 Como o tamanho do modelo poderia ser reduzido?

- **Quantização** pós-treinamento (int8/float16 via `TFLiteConverter`) — reduz o tamanho em até 4x com perda mínima de acurácia.
- **Pruning**: remover pesos de baixa magnitude com `tensorflow_model_optimization`.
- **Redução de filtros/neurônios**: menos filtros convolucionais e camada densa menor, já que o número de classes de saída é pequeno.
- **Global Average Pooling** no lugar de `Flatten`, eliminando a maior parte dos parâmetros da camada densa final (principal gargalo de tamanho nessas CNNs).
- **Destilação de conhecimento**: treinar um modelo menor para imitar as saídas do modelo maior já treinado.
- **Conversão para TensorFlow Lite** (`.tflite`): formato otimizado para mobile — cenário real de uso do Multiprova Corretor (app em React Native).

## Estrutura do repositório
├── README.md
└── tarefa1_playground_tensorflow.md


Os notebooks das Tarefas 2 e 3 estão hospedados no Google Colab (links acima).
