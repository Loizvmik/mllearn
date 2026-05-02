# База для собесов junior ML / DS / GenAI

Шпаргалка-минимум, без которой не стоит идти на интервью в бигтехи (Сбер AI Lab, T-Банк, Яндекс, VK, Avito, Ozon, MTS AI).

## Как этим пользоваться

1. Прочитал тему — закрыл документ — попробовал **по памяти** написать суть на листке. Именно это тренирует то, что у тебя сейчас «провисает»: переход от «узнаю» к «могу объяснить».
2. На каждую тему — 5–10 «контрольных вопросов». Это твой mock-собес. Если на вопрос ответ не льётся за 30 секунд — белое пятно, возвращайся к теме.
3. Каждое определение — выпиши формулу/схему **руками**. Не копируй, не печатай — именно от руки. Связывает символы с пониманием.
4. После каждой темы — заведи Anki-карточки на формулы и термины. 10 минут в день.

Принцип: **не учить всё подряд**. Учить узлы, на которые потом легко цепляются детали.

---

## 1. Классический ML

Это всегда спрашивают, даже если позиция «GenAI». Это база.

**Bias–variance trade-off.** Bias — систематическая ошибка модели (слишком простая, недоучилась). Variance — чувствительность к выборке (слишком сложная, переучилась). Сумма + шум = ожидаемая ошибка. Простая модель → high bias / low variance, глубокая сеть → low bias / high variance.

**Overfit / underfit.** Underfit — train и val плохие. Overfit — train хороший, val плохой. Борьба с overfit: регуляризация, больше данных, аугментация, dropout, early stopping, упрощение модели.

**Регуляризация.**
- **L1 (Lasso):** штраф `λ∑|w|` → разреживает веса, делает feature selection.
- **L2 (Ridge):** штраф `λ∑w²` → ужимает все веса, не зануляет.
- **Elastic Net:** L1 + L2.

**Cross-validation.** k-fold: разбили на k частей, k раз обучили (на k−1) и проверили (на 1). Stratified k-fold — сохраняет долю классов (для классификации). Time-series CV — никогда не перемешиваем по времени.

**Метрики классификации.**
- **Accuracy** — доля верных. Бесполезна на imbalanced.
- **Precision** = TP / (TP + FP) — «из тех, кого назвал положительным, сколько действительно».
- **Recall (TPR)** = TP / (TP + FN) — «из всех положительных, сколько поймал».
- **F1** = 2·P·R / (P+R) — гармоническое среднее.
- **ROC-AUC** — устойчива к дисбалансу, но обманчива при сильном перекосе.
- **PR-AUC** — лучше для редких классов.
- **Log-loss** — штрафует уверенные ошибки.

Когда какую: imbalanced + важны редкие случаи (фрод, болезнь) → recall + PR-AUC. Балансированно → F1, ROC-AUC.

**Метрики регрессии.** MSE (квадратичная, штрафует выбросы), RMSE (та же, в исходных единицах), MAE (устойчива к выбросам), R² (доля объяснённой дисперсии, может быть отрицательной).

**Imbalanced data.** Class weights в loss, oversampling (SMOTE), undersampling, focal loss, изменение порога. SMOTE перед train/val split — это утечка.

**Алгоритмы (must explain).**
- **Linear regression**: МНК, нормальные уравнения, MSE.
- **Logistic regression**: сигмоида + log-loss, выпуклая задача.
- **Decision tree**: информационный выигрыш / Gini, переобучается легко.
- **Random Forest**: bagging + случайные подвыборки фич, борется с overfit.
- **Gradient Boosting**: последовательно строит слабые модели, каждая поправляет ошибки предыдущих. XGBoost / LightGBM / CatBoost — это его реализации.
- **kNN**: ленивый, чувствителен к scaling, проклятие размерности.
- **k-means**: кластеризация по центрам, чувствителен к инициализации (k-means++).
- **PCA**: проекция на оси максимальной дисперсии, через SVD ковариационной матрицы.

**Feature engineering.** Scaling (StandardScaler / MinMax) — обязателен для kNN, SVM, нейросетей; не нужен для деревьев. Encoding: one-hot для номинальных, target encoding для high-cardinality (с осторожностью — leakage), ordinal — для упорядоченных. Пропуски: mean/median, отдельная категория, MICE.

**Контрольные вопросы.**
1. В чём разница ROC-AUC и PR-AUC, когда какая лучше?
2. Что произойдёт, если применить SMOTE до train/val split?
3. Объясни bias–variance trade-off на примере полиномиальной регрессии.
4. Чем boosting отличается от bagging?
5. Зачем PCA нужен SVD?
6. Когда лучше MAE, а не MSE?
7. Что такое stratified k-fold и зачем?

---

## 2. Математика и статистика — минимум

**Теорема Байеса.** `P(A|B) = P(B|A) · P(A) / P(B)`. Posterior ∝ Likelihood · Prior. Должен уметь привести классический пример (тест на болезнь с ложноположительными).

**Likelihood, MLE.** Likelihood — вероятность данных при заданных параметрах. MLE — параметры, максимизирующие likelihood. Логарифмируем для удобства.

**Cross-entropy и log-loss.** Для классификации: `−∑ y log(p)`. Минимизация cross-entropy = MLE для категориального распределения. **BCE** (binary): `−(y·log p + (1−y)·log(1−p))`.

**KL-дивергенция.** `KL(P‖Q) = ∑ P log(P/Q)`. Несимметрична. Обоснование cross-entropy через KL: `CE = H(P) + KL(P‖Q)`. Везде в DL и diffusion.

**Линейная алгебра.** Матричное умножение размерности `(m×k)·(k×n) = (m×n)`. Eigenvectors / SVD: `A = UΣVᵀ`, столбцы V — оси PCA. Норма матрицы / вектора (L1, L2). Скалярное произведение, косинусная мера.

**Распределения.** Normal (μ, σ²), Bernoulli (p), Binomial (n, p), Poisson (λ). Для каждого — где встречается в ML.

**Expectation, variance.** `E[X] = ∑ x·P(x)`, `Var(X) = E[(X−μ)²]`. Линейность E. Закон больших чисел и ЦПТ — в чём суть.

**Контрольные вопросы.**
1. Выведи log-loss из MLE для логистической регрессии.
2. Зачем в VAE используется KL?
3. Покажи на пальцах, почему `KL(P‖Q) ≠ KL(Q‖P)`.
4. Что такое covariance, чем отличается от correlation?
5. Объясни связь PCA и SVD.
6. Что такое смещённая/несмещённая оценка дисперсии?

---

## 3. Deep Learning

**Forward / backward.** Forward — пропустили вход через слои, получили loss. Backward — вычислили `∂loss/∂w` по chain rule, обновили веса. **Backpropagation = chain rule + динамическое программирование**.

**Activations.**
- **ReLU**: `max(0, x)` — дёшев, dying ReLU проблема.
- **Leaky ReLU / GELU**: чинят умирание; GELU — стандарт в трансформерах.
- **Sigmoid**: `1/(1+e^−x)` — для бинарной классификации; vanishing gradient.
- **Tanh**: `(e^x − e^−x)/(e^x + e^−x)` — центрирована; тоже vanishing.
- **Softmax**: для мультиклассового выхода, нормализует в распределение.

**Loss-функции.**
- Регрессия: MSE, MAE, Huber.
- Бинарная классификация: BCE.
- Мультикласс: cross-entropy.
- Знай, какая loss использует softmax (CE), а какая sigmoid (BCE).

**Оптимизаторы.**
- **SGD**: `w ← w − lr·g`.
- **Momentum**: добавляет «инерцию» предыдущих градиентов.
- **Adam**: adaptive lr на параметр через 1-й и 2-й моменты градиента. Стандарт.
- **AdamW**: Adam с правильной weight decay (L2 на веса, не через градиент).

**Нормализация.**
- **BatchNorm**: нормализует по батчу, по фичам. Хорошо для CNN, хуже для маленьких батчей и RNN.
- **LayerNorm**: по последней размерности (фичам), независимо от батча. Стандарт в трансформерах.

**Регуляризация.** Dropout (случайно зануляет нейроны на train, на inference — нет), weight decay, label smoothing, data augmentation, early stopping.

**Vanishing / exploding gradients.** Глубокие сети → градиент затухает или взрывается. Лечение: ReLU, BatchNorm/LayerNorm, **residual connections** (ResNet), нормальные веса (Xavier, He).

**Transfer learning.** Берём предобученную модель → дообучаем на своих данных. Feature extraction = заморозили всё, кроме головы. Fine-tuning = размораживаем больше слоёв с маленьким lr.

**Контрольные вопросы.**
1. Объясни backprop на однослойной сети с MSE — выведи формулу.
2. Что такое dying ReLU и как лечить?
3. Чем LayerNorm отличается от BatchNorm и почему в трансформерах LN?
4. Зачем residual connections в глубоких сетях?
5. Чем AdamW лучше Adam?
6. Почему dropout отключается на inference?
7. Что такое gradient clipping и когда нужно?
8. Как выбрать learning rate? Что такое warmup?

---

## 4. NLP / Transformers

**Токенизация.** Разбивка текста на токены.
- **BPE (Byte-Pair Encoding)**: жадно сливает частые пары. Используется в GPT.
- **WordPiece**: вариант BPE с другим критерием. Используется в BERT.
- **SentencePiece**: модель токенизации поверх raw text, для языков без пробелов.

**Embeddings.** Векторные представления токенов. Раньше: word2vec (skip-gram, CBOW), GloVe — статические. Сейчас: контекстные (BERT/GPT) — каждое слово получает вектор в зависимости от контекста.

**Self-attention.** Сердце трансформера.
- Q, K, V — projections from embedding.
- `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`.
- Делитель `√d_k` — чтобы dot product не распёр softmax в один токен.
- **Multi-head**: несколько параллельных голов, конкатенация → линейный слой.
- Причинная (causal) маска — для decoder, чтобы не подглядывать в будущее.

**Архитектура трансформера.**
- **Encoder (BERT)**: bidirectional self-attention, обучается через MLM (masked LM). Для понимания текста — классификация, NER, поиск.
- **Decoder (GPT)**: causal self-attention, обучается через autoregressive next-token prediction. Для генерации.
- **Encoder-Decoder (T5, BART)**: encoder читает входной текст, decoder генерирует выходной с cross-attention. Для перевода, суммаризации.

**Positional encoding.** Self-attention не видит порядка токенов сама по себе. Решение: sinusoidal (классика), learned, RoPE (rotary — современная), ALiBi.

**FFN внутри блока.** Двухслойный MLP с GELU между attention-блоками. Расширение размерности (обычно ×4).

**Pre-LN vs Post-LN.** Современные модели — Pre-LN (LN до attention/FFN). Стабильнее в обучении.

**Контрольные вопросы.**
1. Выведи формулу self-attention. Зачем `√d_k`?
2. Чем BERT отличается от GPT по архитектуре и задаче обучения?
3. Зачем нужно multi-head attention, а не single-head?
4. Что такое causal mask и где используется?
5. Что такое RoPE и чем оно лучше sinusoidal?
6. Сколько параметров у трансформера с h голов, d_model, d_ff и L слоёв (примерно)?
7. Какой токенизатор у GPT-4 / Llama / BERT?

---

## 5. LLM-стек (must для NLP/GenAI вакансий)

**SFT (Supervised Fine-Tuning).** Дообучение на парах (prompt, target answer) с обычной cross-entropy на токенах ответа. Базовый шаг alignment-pipeline.

**LoRA / QLoRA.** Low-Rank Adaptation: вместо обучения всех весов W обучаем `ΔW = A·B`, где A `(d×r)` и B `(r×d)`, r ≪ d. Параметров на 1–2 порядка меньше. **QLoRA** — LoRA поверх квантизованной (4-bit) base-модели, чтобы влезло в одну GPU.

**RLHF.** Reinforcement Learning from Human Feedback. Этапы: SFT → reward model на парных предпочтениях `(prompt, win, lose)` → PPO с этим reward. Цель — выровнять модель по человеческим предпочтениям.

**DPO (Direct Preference Optimization).** Современная замена RLHF. Прямо из пар предпочтений минимизирует loss без отдельного reward model и PPO. Стабильнее, проще. Сейчас — стандарт в open-source.

**Prompting.**
- **Zero-shot**: «реши задачу X», без примеров.
- **Few-shot (in-context learning)**: даём 2–5 примеров в промпте.
- **Chain-of-Thought (CoT)**: «думай по шагам» — улучшает рассуждение.
- **Self-consistency**: сэмплируем несколько ответов, берём большинство.

**RAG (Retrieval-Augmented Generation).** Архитектура: запрос → embedder → vector DB → top-k релевантных кусков → склеили в промпт LLM → ответ. Решает проблему устаревания знаний и галлюцинаций.
- **Chunking**: режем документы на куски (по 200–500 токенов).
- **Embedding model**: OpenAI text-embedding-3, BGE, E5.
- **Vector DB**: FAISS (лёгкая), Chroma, Weaviate, Qdrant, Pinecone.
- **Re-ranking**: вторая модель (cross-encoder) переоценивает top-k для точности.
- **Hybrid search**: BM25 + dense embeddings.

**Hallucinations.** LLM уверенно генерит ложь. Меры: RAG, citation grounding, констрейнты на формат вывода (JSON schema), specialized eval (TruthfulQA, FActScore).

**LLM evaluation.**
- **Perplexity**: `exp(loss)`, для language modeling.
- **BLEU, ROUGE, METEOR**: для перевода/суммаризации.
- **Pass@k**: для кода (HumanEval).
- **MMLU, BIG-Bench, HellaSwag**: общие бенчмарки знаний/рассуждения.
- **RAGAS**: метрики специально для RAG (faithfulness, answer relevance, context precision/recall).
- **LLM-as-judge**: GPT-4 оценивает ответы. Дёшево, но biased.
- **LM-Eval-Harness**: фреймворк EleutherAI для прогона бенчей.

**Quantization.** Уменьшение точности весов (FP16 → INT8 → INT4) для inference. GPTQ, AWQ, bitsandbytes.

**Контрольные вопросы.**
1. Объясни LoRA — почему достаточно low-rank матриц?
2. Чем DPO отличается от RLHF и почему его выбирают?
3. Опиши пайплайн RAG от запроса до ответа. Где может сломаться?
4. Какие метрики используешь для оценки RAG-системы?
5. Что такое perplexity и зачем она нужна?
6. Как выбрать chunk size при индексации документов? Что зависит от размера?
7. Что такое hallucinations и какие способы борьбы знаешь?
8. Чем отличается embedding-модель от LLM?
9. Что такое temperature, top-k, top-p при генерации?

---

## 6. Diffusion (тебе особенно — у тебя боевой опыт)

**Forward process.** Постепенно добавляем гауссовский шум за T шагов: `x_t = √(α̅_t)·x_0 + √(1−α̅_t)·ε`. Конечное состояние — почти чистый шум.

**Reverse process.** Учим нейросеть `ε_θ(x_t, t)` предсказывать шум. Loss — простой MSE между предсказанным и истинным шумом. Это и есть DDPM.

**DDPM vs DDIM.** DDPM — стохастический сэмплинг, ~1000 шагов. DDIM — детерминистический, можно сэмплить за 20–50 шагов. То же качество, в 20× быстрее.

**Classifier-free guidance (CFG).** Учим модель и с условием, и без (с вероятностью dropout 10–20% на условие). На inference: `ε = ε_uncond + w·(ε_cond − ε_uncond)`. Параметр w (guidance scale) — насколько следовать условию.

**Latent diffusion (Stable Diffusion).** Диффузию делаем не в пиксельном пространстве, а в латентном (после VAE). На порядок дешевле.

**LoRA для диффузии.** Аналогично LLM, обучаем низкоранговые адаптеры в attention-слоях UNet/DiT. Стандарт для дообучения SD/SDXL/Flux.

**SFT для guard / safety.** Файнтюнинг на парах «запрос → отказ» или классификатор поверх диффузионной модели для фильтрации генераций.

**Контрольные вопросы.**
1. Объясни forward и reverse process в DDPM.
2. Что такое `α_t` и `α̅_t`?
3. Зачем DDIM, если есть DDPM?
4. Объясни classifier-free guidance — что такое w?
5. Чем latent diffusion дешевле пиксельной?
6. Как делается SFT диффузионки на маленьком датасете (роль LoRA)?

---

## 7. ML System Design / практика

**Train / val / test split.** Тест трогаем один раз в конце. Val — для подбора гиперпараметров. Test — финальная честная оценка. Стратифицировать для классификации.

**Data leakage.** Информация из теста просочилась в train. Источники: считали статистики (mean/std) на всём датасете до split, target encoding без CV, A/B тестовые юзеры в обоих сетах, time-leak.

**A/B тесты.** Контроль/тест, метрика, p-value, мощность. **p-value** — вероятность увидеть такую разницу при истинной H0 (никакой разницы нет). p < 0.05 — отвергаем H0. Знай: power, MDE (minimum detectable effect), Bonferroni-коррекция при множественных гипотезах.

**Train/inference skew.** Признаки/обработка различаются между train и prod. Решение — feature store, тот же pipeline.

**Когда выкинуть модель.** Простой бейзлайн (среднее, kNN, GBM) обыгрывает её. Метрики на test ниже бейзлайна. Цена ошибки выше профита. Не «спасайте» модель ради того, что вложили месяц.

**Контрольные вопросы.**
1. Что такое data leakage, приведи 3 примера.
2. Когда стоит применять stratified split?
3. Что такое p-value, что такое power?
4. Как бы ты дизайнил A/B тест для новой модели рекомендаций?
5. Что такое train/inference skew, как ловить?

---

## 8. Coding (что точно спросят)

**Python.**
- List comprehensions, dict comprehensions.
- Generators (`yield`), разница с list.
- Декораторы (`@`), что они делают.
- `*args`, `**kwargs`.
- Context managers (`with`).
- Mutable default arguments — известный pitfall.

**pandas.**
- `groupby` + `agg` / `transform`.
- `merge` / `join` (типы: inner/left/outer).
- `apply` vs векторизация — последняя быстрее на порядок.
- `pivot_table`, `melt`.
- Чтение CSV с типами, обработка NaN.

**numpy.**
- Broadcasting (правила).
- `axis=0` vs `axis=1`.
- Векторизованные операции вместо циклов.

**PyTorch — must be able from scratch:**
```
- nn.Module + forward
- DataLoader + Dataset
- Цикл: optimizer.zero_grad() → loss.backward() → optimizer.step()
- model.train() / model.eval(), torch.no_grad()
- .to(device)
```

**Алгоритмы (LeetCode easy/medium):**
- Two-pointers, sliding window.
- Hashmaps (anagrams, two sum).
- Strings, arrays.
- Binary search.
- BFS / DFS на графах и деревьях.
- Базовые DP (fibo, knapsack, longest substring).

Цель: 30–40 задач уровня easy/medium. Решай со словесным проговариванием решения вслух.

**Типовая live-coding задача:** напиши training loop для логистической регрессии на PyTorch. Или: посчитай precision/recall без sklearn. Или: реализуй attention(Q, K, V).

---

## 9. Софт-навыки и поведенческие вопросы

На junior часто тратят 15–20 минут на «расскажи о проекте». Подготовь **STAR-нарратив** для своих кейсов: Situation → Task → Action → Result.

**Тренируй три истории:**
1. Eval-фреймворк в ЦИИ МГУ — что было, что сделал, какой эффект.
2. Препринт arXiv — твой вклад в задачу black-box оптимизации.
3. Завод — как собирал требования, как проектировал БД, как один проблемный кейс решал.

В каждой — один пункт «что я бы сделал иначе сейчас». Это любят: показывает рефлексию.

---

## План на 14 дней

**Неделя 1 — фундамент.**
- День 1–2: §1 Классический ML. Cheat-sheet от руки. 7 контрольных вопросов вслух.
- День 3: §2 Мат/стат. Вывести log-loss из MLE.
- День 4–5: §3 DL. Backprop вывести от руки. PyTorch training loop с нуля.
- День 6: §7 ML system design. Прорешать 5 кейсов «как бы ты дизайнил…».
- День 7: §8 Coding — 5 задач LeetCode + типовая ML-задача.

**Неделя 2 — твоя специализация.**
- День 8–9: §4 NLP/Transformers. Вывести self-attention. Реализовать в numpy.
- День 10–11: §5 LLM-стек. Полный пайплайн RAG нарисовать. Список метрик с формулами.
- День 12: §6 Diffusion. Forward/reverse, CFG, LoRA — всё по памяти.
- День 13: §9 STAR-истории, прогон вслух. Прогон под зеркало.
- День 14: mock-интервью. Со мной или с человеком. Без подготовки. Разбор слабых мест.

После 14 дней: повторение через интервал (Anki-карты, проговаривание тем раз в 3 дня).

---

## Что игнорировать на этом этапе

- Глубокое понимание PPO / Bayesian DL / Causal inference / Multi-armed bandits — это не junior must.
- Архитектуры, которые не в трансформерах и не в диффузии (графовые сети, capsule, NeurIPS-флавор) — точечно, по необходимости позиции.
- DevOps и MLOps в глубине (Kubeflow, Triton, NVIDIA Dynamo) — для первой работы достаточно «слышал, понимаю зачем».

---

## После того как «база» уляжется

Возвращайся ко мне с:
1. «Готов под Сбер Model Validation» — пройдём по их специфике.
2. «Готов под Сбер Classic ML / LLM» — отработаем AI-ассистент / SFT.
3. Mock-интервью прямо в чате.

Удачи. Это всё реально учится за 2 недели плотной работы.
