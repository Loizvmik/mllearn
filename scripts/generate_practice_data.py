from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RNG = np.random.default_rng(42)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DATA_DIR / name, index=False)


def make_churn() -> None:
    n = 1800
    customer_id = np.arange(100_000, 100_000 + n)
    tenure_months = RNG.integers(1, 73, n)
    monthly_fee = RNG.normal(48, 15, n).clip(10, 120).round(2)
    total_usage_gb = RNG.gamma(3.0, 35.0, n).round(2)
    support_tickets = RNG.poisson(1.2, n)
    avg_latency_ms = RNG.normal(80, 25, n).clip(20, 220).round(1)
    discount_percent = RNG.choice([0, 5, 10, 15, 20, 30], n, p=[0.42, 0.15, 0.18, 0.1, 0.1, 0.05])
    contract_type = RNG.choice(["month_to_month", "one_year", "two_year"], n, p=[0.55, 0.3, 0.15])
    payment_method = RNG.choice(["card", "bank_transfer", "cash", "crypto"], n, p=[0.55, 0.25, 0.17, 0.03])
    region = RNG.choice(["central", "north", "south", "east", "west"], n)
    autopay = RNG.binomial(1, 0.58, n)

    logits = (
        -2.0
        + 0.032 * (monthly_fee - 45)
        - 0.035 * tenure_months
        + 0.18 * support_tickets
        + 0.009 * (avg_latency_ms - 80)
        - 0.045 * discount_percent
        + 0.75 * (contract_type == "month_to_month")
        - 0.55 * autopay
        + 0.32 * (payment_method == "cash")
        + RNG.normal(0, 0.7, n)
    )
    churn = RNG.binomial(1, sigmoid(logits))

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "monthly_fee": monthly_fee,
            "total_usage_gb": total_usage_gb,
            "support_tickets": support_tickets,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "region": region,
            "autopay": autopay,
            "avg_latency_ms": avg_latency_ms,
            "discount_percent": discount_percent,
            "churn": churn,
        }
    )

    for col, frac in {"total_usage_gb": 0.06, "avg_latency_ms": 0.04, "payment_method": 0.03}.items():
        idx = RNG.choice(df.index, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)
    write_csv(train.sort_values("customer_id"), "churn_train.csv")
    write_csv(test.sort_values("customer_id"), "churn_test.csv")


def make_laptops() -> None:
    n = 900
    brand = RNG.choice(["Aster", "Boreal", "Canyon", "Digma", "Evo"], n, p=[0.22, 0.18, 0.25, 0.2, 0.15])
    cpu_score = RNG.normal(52, 16, n).clip(10, 100)
    ram_gb = RNG.choice([4, 8, 16, 32, 64], n, p=[0.08, 0.32, 0.37, 0.18, 0.05])
    ssd_gb = RNG.choice([128, 256, 512, 1024, 2048], n, p=[0.08, 0.25, 0.37, 0.23, 0.07])
    age_months = RNG.integers(0, 73, n)
    condition = RNG.choice(["new", "excellent", "good", "fair"], n, p=[0.22, 0.34, 0.32, 0.12])
    warranty_months = RNG.choice([0, 3, 6, 12, 24], n, p=[0.28, 0.15, 0.2, 0.28, 0.09])
    seller_rating = RNG.normal(4.55, 0.35, n).clip(2.8, 5.0).round(2)
    brand_bonus = pd.Series(brand).map({"Aster": 180, "Boreal": 260, "Canyon": 90, "Digma": -60, "Evo": 40}).to_numpy()
    condition_bonus = pd.Series(condition).map({"new": 330, "excellent": 180, "good": 40, "fair": -160}).to_numpy()

    price_usd = (
        210
        + 16.5 * cpu_score
        + 21 * ram_gb
        + 0.23 * ssd_gb
        - 9.8 * age_months
        + 42 * warranty_months
        + 120 * (seller_rating - 4)
        + brand_bonus
        + condition_bonus
        + RNG.normal(0, 180, n)
    )
    outlier_idx = RNG.choice(np.arange(n), 18, replace=False)
    price_usd[outlier_idx] *= RNG.choice([0.45, 1.8], len(outlier_idx))

    df = pd.DataFrame(
        {
            "listing_id": np.arange(50_000, 50_000 + n),
            "brand": brand,
            "cpu_score": cpu_score.round(1),
            "ram_gb": ram_gb,
            "ssd_gb": ssd_gb,
            "age_months": age_months,
            "condition": condition,
            "warranty_months": warranty_months,
            "seller_rating": seller_rating,
            "price_usd": price_usd.clip(90, None).round(2),
        }
    )

    for col, frac in {"cpu_score": 0.035, "seller_rating": 0.025}.items():
        idx = RNG.choice(df.index, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    write_csv(df, "laptop_prices.csv")


def make_fraud() -> None:
    n = 5200
    ts = pd.date_range("2025-01-01", periods=n, freq="37min")
    user_id = RNG.integers(1_000, 1_850, n)
    amount = RNG.lognormal(mean=3.45, sigma=0.9, size=n).round(2)
    merchant_category = RNG.choice(["grocery", "electronics", "travel", "games", "pharmacy", "marketplace"], n)
    country = RNG.choice(["RU", "KZ", "AM", "TR", "AE"], n, p=[0.72, 0.1, 0.07, 0.07, 0.04])
    device_age_days = RNG.exponential(140, n).clip(0, 1500).round(1)
    transactions_last_24h = RNG.poisson(2.2, n)
    account_age_days = RNG.exponential(420, n).clip(1, 2500).round(1)
    hour = ts.hour.to_numpy()
    is_night = ((hour <= 5) | (hour >= 23)).astype(int)
    is_weekend = (ts.dayofweek >= 5).astype(int)

    logits = (
        -5.4
        + 0.95 * (amount > 180)
        + 0.85 * (merchant_category == "electronics")
        + 0.65 * (merchant_category == "games")
        + 0.55 * (country != "RU")
        + 0.8 * (device_age_days < 7)
        + 0.24 * transactions_last_24h
        - 0.002 * account_age_days
        + 0.45 * is_night
        + RNG.normal(0, 0.6, n)
    )
    is_fraud = RNG.binomial(1, sigmoid(logits))
    chargeback_seen_after_txn = np.where(is_fraud == 1, RNG.binomial(1, 0.82, n), RNG.binomial(1, 0.015, n))

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(900_000, 900_000 + n),
            "timestamp": ts.astype(str),
            "user_id": user_id,
            "amount": amount,
            "merchant_category": merchant_category,
            "country": country,
            "device_age_days": device_age_days,
            "transactions_last_24h": transactions_last_24h,
            "account_age_days": account_age_days,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "chargeback_seen_after_txn": chargeback_seen_after_txn,
            "is_fraud": is_fraud,
        }
    )
    write_csv(df, "fraud_time_series.csv")


def make_pca_clusters() -> None:
    n_per_cluster = 300
    centers = np.array(
        [
            [3.0, -1.2, 0.0],
            [-2.0, 2.8, 1.0],
            [0.4, 0.0, -2.5],
        ]
    )
    rows = []
    for cluster, center in enumerate(centers):
        z = RNG.normal(center, [0.7, 0.55, 0.8], size=(n_per_cluster, 3))
        x1 = z[:, 0] + 0.2 * z[:, 1] + RNG.normal(0, 0.2, n_per_cluster)
        x2 = -0.5 * z[:, 0] + z[:, 1] + RNG.normal(0, 0.25, n_per_cluster)
        x3 = z[:, 2] + RNG.normal(0, 0.15, n_per_cluster)
        x4 = 0.7 * z[:, 0] - 0.2 * z[:, 2] + RNG.normal(0, 0.2, n_per_cluster)
        x5 = z[:, 1] + 0.4 * z[:, 2] + RNG.normal(0, 0.2, n_per_cluster)
        part = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        part["true_cluster"] = cluster
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    write_csv(df.sample(frac=1, random_state=42).round(4), "pca_clusters.csv")


def make_predictions_and_bayes() -> None:
    n = 1000
    y_true = RNG.binomial(1, 0.085, n)
    base = RNG.normal(-2.4 + 3.6 * y_true, 1.0, n)
    model_a_score = sigmoid(base + RNG.normal(0, 0.55, n))
    model_b_score = sigmoid(base * 0.82 + RNG.normal(0, 0.85, n) - 0.15)
    df = pd.DataFrame(
        {
            "example_id": np.arange(n),
            "y_true": y_true,
            "model_a_score": model_a_score.round(6),
            "model_b_score": model_b_score.round(6),
        }
    )
    write_csv(df, "classification_predictions.csv")

    m = 10000
    disease = RNG.binomial(1, 0.012, m)
    test_positive = np.where(disease == 1, RNG.binomial(1, 0.94, m), RNG.binomial(1, 0.06, m))
    medical = pd.DataFrame(
        {
            "patient_id": np.arange(1, m + 1),
            "has_disease": disease,
            "test_positive": test_positive,
        }
    )
    write_csv(medical, "bayes_medical_test.csv")


def make_ab_test() -> None:
    n = 6000
    group = RNG.choice(["control", "treatment"], n)
    user_segment = RNG.choice(["new", "returning", "vip"], n, p=[0.42, 0.48, 0.1])
    impressions = RNG.poisson(np.where(user_segment == "vip", 12, 7), n).clip(1, None)
    click_rate = 0.07 + 0.015 * (group == "treatment") + 0.035 * (user_segment == "vip") - 0.01 * (user_segment == "new")
    clicks = RNG.binomial(impressions, np.clip(click_rate, 0.01, 0.4))
    order_rate = 0.06 + 0.012 * (group == "treatment") + 0.028 * (user_segment == "vip")
    orders = RNG.binomial(clicks, np.clip(order_rate, 0.01, 0.4))
    revenue = (orders * RNG.lognormal(3.5, 0.55, n)).round(2)
    session_minutes = RNG.gamma(2.2, 3.0, n) + 0.25 * clicks + 0.7 * orders
    df = pd.DataFrame(
        {
            "user_id": np.arange(200_000, 200_000 + n),
            "group": group,
            "user_segment": user_segment,
            "impressions": impressions,
            "clicks": clicks,
            "orders": orders,
            "revenue": revenue,
            "session_minutes": session_minutes.round(2),
        }
    )
    write_csv(df, "ab_recommendations.csv")


def make_attention() -> None:
    batch, seq_len, d_k, d_v = 2, 4, 3, 2
    q = RNG.normal(0, 1, (batch, seq_len, d_k)).round(4)
    k = RNG.normal(0, 1, (batch, seq_len, d_k)).round(4)
    v = RNG.normal(0, 1, (batch, seq_len, d_v)).round(4)
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    scores[:, np.triu_indices(seq_len, k=1)[0], np.triu_indices(seq_len, k=1)[1]] = -1e9
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    output = weights @ v
    np.savez(DATA_DIR / "attention_qkv.npz", q=q, k=k, v=v, causal_output=output.round(6))


def make_text_and_rag() -> None:
    intents = {
        "refund": [
            "верните деньги за заказ",
            "хочу оформить возврат",
            "товар не подошел, нужен возврат",
            "как получить деньги обратно",
        ],
        "delivery": [
            "где мой заказ",
            "когда приедет доставка",
            "курьер опаздывает",
            "изменить адрес доставки",
        ],
        "tech_support": [
            "приложение не открывается",
            "не приходит код подтверждения",
            "ошибка при оплате",
            "страница зависает после входа",
        ],
        "subscription": [
            "отключите подписку",
            "как поменять тариф",
            "продлить премиум на месяц",
            "списалась оплата за подписку",
        ],
    }
    rows = []
    for intent, templates in intents.items():
        for i in range(80):
            text = RNG.choice(templates)
            if RNG.random() < 0.35:
                text += " пожалуйста"
            if RNG.random() < 0.25:
                text = text.capitalize()
            rows.append({"text_id": f"{intent}_{i:03d}", "text": text, "intent": intent})
    write_csv(pd.DataFrame(rows).sample(frac=1, random_state=42), "text_intents_ru.csv")

    docs = [
        {
            "doc_id": "rag_001",
            "title": "ChurnGuard model card",
            "text": "ChurnGuard predicts customer churn. The production threshold is 0.62 because recall is prioritized over precision. The model must not use post-cancellation signals.",
        },
        {
            "doc_id": "rag_002",
            "title": "FraudRadar feature policy",
            "text": "FraudRadar is scored at transaction time. Chargeback fields are unavailable online and are forbidden features because they create data leakage.",
        },
        {
            "doc_id": "rag_003",
            "title": "SupportBot evaluation",
            "text": "SupportBot answers are evaluated with faithfulness, answer relevance, context precision, and manual refusal-quality checks.",
        },
        {
            "doc_id": "rag_004",
            "title": "LaptopPrice baseline",
            "text": "LaptopPrice uses MAE as the primary metric because several listings contain extreme outlier prices. RMSE is tracked as a secondary metric.",
        },
        {
            "doc_id": "rag_005",
            "title": "AB test rules",
            "text": "Recommendation experiments use user-level randomization. The primary metric is orders per user, and revenue per user is a guardrail metric.",
        },
        {
            "doc_id": "rag_006",
            "title": "Tokenizer note",
            "text": "The intent classifier uses subword tokenization so rare Russian word forms can be represented as combinations of known pieces.",
        },
    ]
    with (DATA_DIR / "rag_documents.jsonl").open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    questions = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "question": "Почему нельзя использовать chargeback_seen_after_txn в FraudRadar?",
                "answer_doc_id": "rag_002",
            },
            {
                "question_id": "q2",
                "question": "Какая основная метрика у LaptopPrice и почему?",
                "answer_doc_id": "rag_004",
            },
            {
                "question_id": "q3",
                "question": "Какие метрики качества использует SupportBot?",
                "answer_doc_id": "rag_003",
            },
            {
                "question_id": "q4",
                "question": "Какой threshold стоит у ChurnGuard?",
                "answer_doc_id": "rag_001",
            },
        ]
    )
    write_csv(questions, "rag_questions.csv")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    make_churn()
    make_laptops()
    make_fraud()
    make_pca_clusters()
    make_predictions_and_bayes()
    make_ab_test()
    make_attention()
    make_text_and_rag()
    print(f"Generated practice data in {DATA_DIR}")


if __name__ == "__main__":
    main()
