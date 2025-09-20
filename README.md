
## 1. Dados

O dataset vem em dois arquivos principais:  
- `train_transaction.csv` (transações)  
- `train_identity.csv` (dados de identidade)  

Eles são juntados pela chave `TransactionID`.  
A variável alvo é `isFraud`, que representa se a transação foi fraude ou não. O problema é bem desbalanceado: só **~3,5%** das transações são fraudes.  

O tempo (`TransactionDT`) não é um timestamp real, mas segundos desde uma data inicial. Mesmo assim, dá para criar variáveis de hora, dia e dia da semana para explorar.

---

## 2. EDA (Exploração Rápida)

Primeiro olhei:
- **Dimensão**: ~590 mil linhas, mais de 400 colunas.  
- **Desbalanceamento**: já citado, fraudes são minoria.  
- **Nulos**: muita coluna com mais de 80% de valores ausentes.  
- **Valores monetários**: `TransactionAmt` tem cauda longa, então visualizei também em escala log.  
- **Temporal**: há variações interessantes da taxa de fraude por hora e dia da semana.  

### Correlações
Calcular a matriz de correlação inteira travava o Colab, então usei um esquema mais leve:
- Peguei uma **amostra de 200k linhas**.  
- Removi colunas constantes e converti tudo para `float32`.  
- Usei `corrwith` para calcular só a correlação de cada feature com `isFraud`.  

Com isso, selecionei as **20 variáveis mais correlacionadas** (todas da família `V*`), que foram usadas depois no modelo.

---

## 3. Preparação para LSTM

Como a base é transacional, faz sentido olhar sequências.  
- Defini um `user_id` como combinação de `card1` e `addr1` (proxy de usuário).  
- As features escolhidas foram as 20 mais correlacionadas.  
- Nulos viraram zero e escalei tudo com **RobustScaler** para lidar melhor com outliers.  

---

## 4. Construção das Sequências

Aqui precisei controlar a memória, senão o Colab travava. Algumas decisões:
- **Janela**: 10 transações por usuário.  
- **Split temporal**: últimas 20% das transações ficaram para validação.  
- **Limites**: usei parâmetros como `MAX_ROWS`, `MAX_USERS`, `MAX_WINDOWS` e `STRIDE` para não gerar mais sequências do que o Colab aguentava.  
- O rótulo de cada janela foi sempre o `isFraud` da última transação.  

No fim, fiquei com algo em torno de **75k janelas para treino** e **24k para validação**.

---

## 5. Modelo

A rede foi bem simples:


LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1, sigmoid)


- **Perda**: `binary_crossentropy`  
- **Otimizador**: Adam com lr=1e-3  
- **Classe desbalanceada**: usei `class_weight` para compensar  
- **Callbacks**: `ReduceLROnPlateau` e `EarlyStopping` monitorando AUC  

Treinei por até 15 épocas, batch size 512. O early stopping parou antes.

---

## 6. Resultados

- **AUC-ROC (validação)**: ~0,71  
- **Precision e Recall**: recall da classe fraude ainda baixo (~0,47), o que é esperado dado o desbalanceamento e a simplicidade das features.  

As curvas de loss e AUC mostram que o modelo não chegou a overfitar pesado, mas também não evoluiu muito além de certo ponto.

---

## 7. Observações e próximos passos

Esse baseline serve mais para mostrar o caminho do que para ganhar competição. Algumas ideias para melhorar:
- Trabalhar melhor o **threshold** (não ficar preso em 0,5).  
- Incluir **mais features relevantes**, não só as top-20 por correlação.  
- Testar **embeddings para variáveis categóricas** (`ProductCD`, domínios de email etc.).  
- Validar em blocos temporais diferentes.  
- Comparar com modelos **tree-based** (LightGBM, XGBoost), que são muito fortes nesse tipo de problema.  

---

## 8. Conclusão

O pipeline mostra que dá para estruturar as transações como sequências e aplicar um LSTM sem estourar a memória do Colab, desde que se use amostragem e limites de janelas. O desempenho (~0,71 AUC) é modesto, mas abre caminho para melhorias mais sofisticadas.
