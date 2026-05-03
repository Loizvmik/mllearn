import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

data=pd.read_csv("data/fraud_time_series.csv")
data.sort_values("timestamp", inplace=True)
data.drop(columns=["transaction_id","user_id", "country", "timestamp","chargeback_seen_after_txn"], inplace=True)
X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

n = len(data)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

def func(F,X) -> Pipeline:
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
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", F(max_iter=1000, random_state=42, class_weight="balanced") if F == LogisticRegression else F(n_estimators=100, random_state=42)),
        ]
    )

def metrics(y_true, y_proba, threshold) -> tuple:
    y_pred = (y_proba >= threshold).astype(int)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)
    return recall, precision

def main():
    pipe = func(LogisticRegression, X_train)
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    pipe_up = func(GradientBoostingClassifier, X_train)
    pipe_up.fit(X_train, y_train)
    val_proba_up = pipe_up.predict_proba(X_val)[:, 1]
    test_proba_up = pipe_up.predict_proba(X_test)[:, 1]

    for t in [0.01,0.05,0.1, 0.2, 0.3]:
        recall_val, precision_val = metrics(y_val, val_proba, threshold=t)
        recall_up_val, precision_up_val = metrics(y_val, val_proba_up, threshold=t)
        recall_test, precision_test = metrics(y_test, test_proba, threshold=t)
        recall_up_test, precision_up_test = metrics(y_test, test_proba_up, threshold=t)
        print(f"Threshold: {t}, Recall Validation: {recall_val:.4f} -> {recall_up_val:.4f}, Precision Validation: {precision_val:.4f} -> {precision_up_val:.4f}")
        print(f"Threshold: {t}, Recall Test: {recall_test:.4f} -> {recall_up_test:.4f}, Precision Test: {precision_test:.4f} -> {precision_up_test:.4f}")

if __name__ == "__main__":
    main()

#метрики полная хуета залупа