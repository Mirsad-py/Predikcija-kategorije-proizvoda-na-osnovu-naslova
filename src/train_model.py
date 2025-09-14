import argparse, time, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def concat_text_1d(df_subset: pd.DataFrame):
    """Spaja više tekst kolona u 1D niz stringova (što TF-IDF očekuje)."""
    return df_subset.fillna("").astype(str).agg(" ".join, axis=1).values


def main():
    ap = argparse.ArgumentParser(description="Train product category model")
    ap.add_argument("--data", default="data/products_features.csv",
                    help="Putanja do CSV-a (prefer features; ako ne postoji, koristi data/products_clean.csv)")
    ap.add_argument("--target", default="Category Label", help="Naziv ciljne kolone")
    ap.add_argument("--out", default="models", help="Direktorijum za snimanje modela")
    args = ap.parse_args()

    # Učitaj podatke
    data_path = Path(args.data)
    if not data_path.exists():
        data_path = Path("data/products_clean.csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    assert args.target in df.columns, f"Nedostaje kolona '{args.target}'"
    df = df[~df[args.target].isna() & (df[args.target].astype(str).str.len() > 0)].copy()

    y = df[args.target].astype(str)
    X = df.drop(columns=[args.target])

    # Kolone za feature-e
    text_cols = [c for c in X.columns if "title" in c.lower()][:1]  # jedna tekst kolona je dosta
    num_cols = [c for c in [
        "Number_of_Views", "Merchant Rating",
        "title_length_chars", "title_length_words",
        "views_log", "year", "month"
    ] if c in X.columns]

    print("Text cols:", text_cols or "—")
    print("Num cols :", num_cols or "—")

    # Preprocessing
    transformers = []
    if text_cols:
        text_pipe = Pipeline([
            ("concat", FunctionTransformer(concat_text_1d, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ])
        transformers.append(("text", text_pipe, text_cols))

    if num_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ])
        transformers.append(("num", num_pipe, num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    # Train/test podela
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Kandidat modeli
    models = {
        "LogisticRegression": LogisticRegression(max_iter=400, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rows, trained = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        t0 = time.time()
        f1_cv = cross_val_score(pipe, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1)
        acc_cv = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "cv_f1_macro": float(f1_cv.mean()),
            "cv_acc": float(acc_cv.mean()),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "test_acc": float(accuracy_score(y_test, y_pred)),
            "train_time_s": round(time.time() - t0, 2),
        })
        trained[name] = (pipe, y_pred)

    results = pd.DataFrame(rows).sort_values(["test_f1_macro", "cv_f1_macro", "cv_acc"], ascending=False)
    print("\n=== Rezultati ===")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["model"]
    best_pipe, best_pred = trained[best_name]
    print(f"\n🏆 Najbolji model: {best_name}")
    print(classification_report(y_test, best_pred, zero_division=0))

    # Finalni refit na celom skupu i snimi
    final_pipe = Pipeline([("prep", preprocessor), ("clf", models[best_name])])
    final_pipe.fit(X, y)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": final_pipe,
        "target_col": args.target,
        "text_cols": text_cols,
        "num_cols": num_cols,
        "label_values": sorted(y.unique().tolist()),
        "trained_from": str(data_path),
        "best_model_name": best_name,
        "cv_summary": results.to_dict(orient="records"),
    }
    job_path = out_dir / "final_model.joblib"
    joblib.dump(bundle, job_path)
    print("\n💾 Sačuvan finalni model u:", job_path)


if __name__ == "__main__":
    main()
