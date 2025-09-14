import argparse, sys, re
import numpy as np
import pandas as pd
import joblib


def make_row(title: str, text_cols, num_cols, overrides: dict) -> pd.DataFrame:
    """Kreiraj jedan red sa svim očekivanim kolonama.
       Numeričke kolone se inicijalno postavljaju na NaN (pipeline će ih imputirati),
       a mogu se prepisati vrednostima iz 'overrides' (npr. views=1200 rating=4.5)."""
    cols = list(text_cols) + list(num_cols)
    row = {c: np.nan for c in cols}
    if text_cols:
        row[text_cols[0]] = title
    # upiši eventualne override vrednosti
    for k, v in overrides.items():
        if k in row:
            try:
                row[k] = float(v)  # probaj kao broj
            except ValueError:
                row[k] = v
    return pd.DataFrame([row])


def parse_keyvals(tokens):
    """Parsiraj key=value parove iz liste tokena."""
    out = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description="Predict product category interactively")
    ap.add_argument("--model", default="models/final_model.joblib", help="Putanja do snimljenog modela")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    text_cols = bundle.get("text_cols", [])
    num_cols = bundle.get("num_cols", [])
    target_col = bundle.get("target_col", "Category Label")

    print("Model učitan.")
    print(f"Očekivane kolone: text={text_cols or '-'} | num={num_cols or '-'}")
    print("Unos: naslov [opciono key=value parovi]\nPrimer:\n  Apple iPhone 12 64GB views=1200 rating=4.5\nCtrl+C za izlaz.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDoviđenja!")
            break
        if not line:
            continue

        tokens = line.split()
        # detektuj key=value parove na kraju
        kv = parse_keyvals([t for t in tokens if "=" in t])
        # naslov je pre prvog key=value tokena
        title_tokens = [t for t in tokens if "=" not in t]
        title = " ".join(title_tokens) if title_tokens else ""

        row = make_row(title=title, text_cols=text_cols, num_cols=num_cols, overrides=kv)
        pred = pipe.predict(row)[0]
        print(f"Predikcija {target_col}: {pred}")

if __name__ == "__main__":
    main()
