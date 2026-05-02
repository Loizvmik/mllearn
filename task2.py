import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "churn"
ID_COL = "customer_id"

NUM_COLS = [
    "tenure_months",
    "monthly_fee",
    "total_usage_gb",
    "support_tickets",
    "autopay",
    "avg_latency_ms",
    "discount_percent",
]

CAT_COLS = [
    "contract_type",
    "payment_method",
    "region",
]


def make_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]
    )

def make_pipeline_upgrade() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    )

def print_metrics(name: str, y_true, y_proba, threshold: float = 0.1) -> None:
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n{name}")
    print(f"threshold: {threshold}")
    print(f"accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"f1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"roc_auc:   {roc_auc_score(y_true, y_proba):.4f}")
    print(f"pr_auc:    {average_precision_score(y_true, y_proba):.4f}")
    print(f"log_loss:  {log_loss(y_true, y_proba):.4f}")


def main() -> None:
    train_df = pd.read_csv("data/churn_train.csv")
    test_df = pd.read_csv("data/churn_test.csv")

    X = train_df.drop(columns=[TARGET, ID_COL])
    y = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET, ID_COL])
    y_test = test_df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]
    pipe_up = make_pipeline_upgrade()
    pipe_up.fit(X_train, y_train)
    val_proba_up = pipe_up.predict_proba(X_val)[:, 1]
    test_proba_up = pipe_up.predict_proba(X_test)[:, 1]
    print_metrics("Validation", y_val, val_proba)
    print_metrics("Test", y_test, test_proba)
    print_metrics("Upgrade Validation", y_val, val_proba_up)
    print_metrics("Upgrade Test", y_test, test_proba_up)

if __name__ == "__main__":
    main()
