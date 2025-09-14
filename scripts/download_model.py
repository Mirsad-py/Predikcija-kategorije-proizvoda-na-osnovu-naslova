# scripts/download_model.py
import sys, requests, pathlib
URL = "https://github.com/<user>/<repo>/releases/download/<tag>/final_model.joblib"
out = pathlib.Path("models/final_model.joblib"); out.parent.mkdir(exist_ok=True, parents=True)
with requests.get(URL, stream=True) as r:
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(1<<20):  # 1MB
            f.write(chunk)
print("Saved to", out)
