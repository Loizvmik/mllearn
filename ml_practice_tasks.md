# Практические задачи по базе ML-интервью

Задачи сделаны по темам из `ml_interview_base.md`: classic ML, матстат, DL, NLP/Transformers, LLM/RAG, system design и coding. Данные лежат в `data/`, генератор — `scripts/generate_practice_data.py`.

## Как решать

1. На каждую задачу сначала напиши короткую гипотезу: какую метрику берёшь, какой split нужен, где может быть leakage.
2. Потом сделай минимальный baseline.
3. После baseline улучши решение только одним-двумя осмысленными изменениями.
4. В конце ответь на 3 вопроса: что получилось, почему так, что бы проверил дальше.

---

## 1. Метрики классификации без sklearn

**Данные:** `data/classification_predictions.csv`

Есть `y_true`, `model_a_score`, `model_b_score` для редкого положительного класса.

**Сделать:**
- реализовать вручную `TP`, `FP`, `TN`, `FN`, accuracy, precision, recall, F1;
- посчитать метрики для threshold `0.5`;
- подобрать threshold для recall не ниже `0.8`;
- сравнить Model A и Model B через PR-AUC и ROC-AUC.

**Контрольные вопросы:**
- Почему accuracy здесь может обманывать?
- Когда PR-AUC важнее ROC-AUC?
- Что изменится, если цена FN в 10 раз выше FP?

---

## 2. Churn prediction: полный classic ML pipeline

**Данные:** `data/churn_train.csv`, `data/churn_test.csv`

Нужно предсказывать `churn` для клиентов сервиса.

**Сделать:**
- EDA: доля churn, пропуски, категориальные признаки;
- train/validation split со стратификацией;
- preprocessing через `Pipeline` / `ColumnTransformer`: imputing, scaling, one-hot;
- baseline: logistic regression;
- улучшение: random forest или gradient boosting;
- сравнить F1, ROC-AUC, PR-AUC, log-loss;
- объяснить, какие признаки повышают риск churn.

**Контрольные вопросы:**
- Какие модели чувствительны к scaling?
- Почему log-loss полезен, если уже есть F1?
- Какой threshold выберешь, если бизнес хочет ловить максимум уходящих клиентов?

---

## 3. Fraud detection: imbalance, time split и leakage

**Данные:** `data/fraud_time_series.csv`

Транзакции идут по времени. Target — `is_fraud`. В датасете есть один специально опасный признак.

**Сделать:**
- сделать split по времени: первые 70% train, следующие 15% val, последние 15% test;
- измерить baseline с majority class и простой моделью;
- найти и убрать leakage-признак;
- сравнить результаты до и после удаления leakage;
- подобрать threshold под высокий recall;
- объяснить, почему SMOTE до split использовать нельзя.

**Контрольные вопросы:**
- Почему random split здесь некорректен?
- Чем class weights отличаются от oversampling?
- Почему `chargeback_seen_after_txn` нельзя использовать в production?

---

## 4. Регрессия: цена ноутбука и выбросы

**Данные:** `data/laptop_prices.csv`

Target — `price_usd`.

**Сделать:**
- построить baseline `median price`;
- обучить linear regression, ridge/lasso и tree-based модель;
- сравнить MAE, RMSE, R2;
- найти выбросы и проверить, как они влияют на MSE/RMSE;
- объяснить, почему MAE может быть основной метрикой.

**Контрольные вопросы:**
- Чем L1 отличается от L2?
- Когда R2 может быть отрицательным?
- Почему RMSE сильнее реагирует на выбросы?

---

## 5. PCA и k-means руками и через sklearn

**Данные:** `data/pca_clusters.csv`

Есть 5 числовых признаков и скрытая кластерная структура.

**Сделать:**
- стандартизировать признаки;
- реализовать PCA через covariance matrix + eigenvectors или SVD;
- спроецировать данные в 2D;
- запустить k-means для `k=2..6`;
- выбрать `k` по inertia/silhouette;
- сравнить кластеры с `true_cluster` только в конце, не во время выбора модели.

**Контрольные вопросы:**
- Почему PCA требует scaling?
- Как связаны PCA и SVD?
- Почему k-means чувствителен к инициализации?

---

## 6. Байес на медицинском тесте

**Данные:** `data/bayes_medical_test.csv`

Симуляция редкой болезни и несовершенного теста.

**Сделать:**
- оценить `P(disease)`, `P(test_positive | disease)`, `P(test_positive | no disease)`;
- посчитать `P(disease | test_positive)` по формуле Байеса;
- проверить результат прямым подсчётом по таблице;
- объяснить, почему положительный тест не означает почти 100% вероятность болезни.

**Контрольные вопросы:**
- Что такое prior, likelihood, posterior?
- Почему base rate важен?
- Как изменится posterior, если болезнь станет в 10 раз чаще?

---

## 7. BCE и log-loss с численной стабильностью

**Данные:** `data/classification_predictions.csv`

**Сделать:**
- реализовать BCE/log-loss вручную для вероятностей;
- добавить clipping вероятностей, чтобы не ловить `log(0)`;
- вывести BCE из MLE для Bernoulli;
- сравнить log-loss Model A и Model B.

**Контрольные вопросы:**
- Почему уверенная ошибка штрафуется сильнее?
- Чем BCE отличается от multiclass cross-entropy?
- Почему sigmoid + BCE часто объединяют в одну стабильную функцию?

---

## 8. PyTorch logistic regression training loop

**Данные:** `data/churn_train.csv`

Напиши минимальную логистическую регрессию на PyTorch.

**Сделать:**
- подготовить числовые признаки и one-hot encoding;
- сделать `Dataset`, `DataLoader`, `nn.Module`;
- написать цикл `zero_grad -> forward -> loss.backward -> step`;
- использовать `BCEWithLogitsLoss`;
- проверить `model.train()`, `model.eval()`, `torch.no_grad()`;
- сравнить качество с sklearn logistic regression.

**Контрольные вопросы:**
- Почему `BCEWithLogitsLoss`, а не `Sigmoid` + `BCELoss`?
- Зачем `optimizer.zero_grad()`?
- Почему dropout/BatchNorm требуют `train()` и `eval()`?

---

## 9. Attention в NumPy

**Данные:** `data/attention_qkv.npz`

В файле лежат `q`, `k`, `v` и эталонный `causal_output`.

**Сделать:**
- реализовать `softmax`;
- реализовать scaled dot-product attention;
- добавить causal mask;
- сравнить свой output с `causal_output` через `np.allclose`;
- объяснить роль делителя `sqrt(d_k)`.

**Контрольные вопросы:**
- Почему self-attention без positional encoding не знает порядок?
- Чем causal mask отличается от padding mask?
- Зачем multi-head attention?

---

## 10. NLP intent classifier

**Данные:** `data/text_intents_ru.csv`

Короткие русские обращения пользователей и intent.

**Сделать:**
- разбить на train/val stratified;
- baseline: `TfidfVectorizer` + logistic regression;
- посчитать macro-F1;
- посмотреть ошибки по confusion matrix;
- придумать, как улучшить модель через subword/transformer-подход.

**Контрольные вопросы:**
- Чем BPE/WordPiece полезны для редких словоформ?
- Почему macro-F1 лучше accuracy при перекосе классов?
- Чем статические embeddings отличаются от contextual embeddings?

---

## 11. Мини-RAG без LLM

**Данные:** `data/rag_documents.jsonl`, `data/rag_questions.csv`

Нужно найти релевантный документ для вопроса.

**Сделать:**
- распарсить jsonl с документами;
- сделать retrieval через TF-IDF или BM25;
- для каждого вопроса вернуть top-1/top-3 документов;
- посчитать recall@1 и recall@3 по `answer_doc_id`;
- вручную разобрать ошибки;
- описать, где в настоящем RAG добавились бы embeddings, reranker и LLM.

**Контрольные вопросы:**
- Что такое chunking и как выбрать размер чанка?
- Чем dense retrieval отличается от BM25?
- Что такое context precision/recall в RAGAS?

---

## 12. A/B test для рекомендаций

**Данные:** `data/ab_recommendations.csv`

Есть control/treatment и пользовательские метрики.

**Сделать:**
- выбрать primary metric: например `orders per user`;
- посчитать uplift treatment над control;
- сделать простой bootstrap confidence interval;
- проверить guardrail `revenue per user`;
- отдельно посмотреть сегменты `new`, `returning`, `vip`;
- сформулировать решение: раскатывать или нет.

**Контрольные вопросы:**
- Что такое p-value?
- Чем power отличается от confidence?
- Почему много сегментных проверок требуют осторожности?

---

## 13. ML system design: production scoring

**Данные:** используй `data/fraud_time_series.csv` или `data/churn_train.csv`

**Сделать письменно:**
- описать offline training pipeline;
- описать online inference pipeline;
- перечислить признаки, которые доступны на момент inference;
- придумать мониторинг drift/skew;
- описать план rollback, если модель ухудшила бизнес-метрику.

**Контрольные вопросы:**
- Что такое train/inference skew?
- Где хранить features?
- Какие метрики мониторить после релиза?

---

## 14. Mock-интервью по своему решению

Выбери любую задачу 2-4 или 10-12 и подготовь 5-минутный рассказ.

**Структура:**
- задача и метрика;
- данные и split;
- baseline;
- улучшение;
- ошибки и риски;
- что бы сделал дальше.

**Цель:** уметь объяснить решение без ноутбука и кода.
