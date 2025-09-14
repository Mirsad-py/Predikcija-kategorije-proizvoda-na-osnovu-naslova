Predikcija-kategorije-proizvoda-na-osnovu-naslova/
│
├── data/                  
│   ├── products.csv              # originalni podaci
│   ├── products_clean.csv        # očišćeni podaci
│   └── products_features.csv     # podaci sa dodatim feature-ima
│
├── models/                       # snimljeni modeli (ignorisan .gitignore)
│
├── notebooks/                    
│   └── Predikcija kategorije proizvoda.ipynb  # EDA + eksperimenti
│
├── scripts/                      
│   └── download_model.py         # preuzima model iz GitHub Releases
│
├── src/                          
│   ├── train_model.py            # treniranje i čuvanje finalnog modela
│   └── predict_category.py       # učitavanje i interaktivno testiranje
│
├── .gitignore
├── requirements.txt
└── README.md
# 📦 Predikcija kategorije proizvoda na osnovu naslova

Ovaj projekat razvija **machine learning model** koji predviđa kategoriju proizvoda na osnovu njegovog naslova i pratećih numeričkih osobina (npr. broj pregleda, ocena prodavca).

---

## 🚀 Struktura projekta

- **`data/`**
  - `products.csv` – sirovi podaci
  - `products_clean.csv` – očišćena verzija
  - `products_features.csv` – proširena verzija sa dodatim feature-ima
- **`models/`**
  - direktorijum za snimljene modele (`final_model.joblib`)  
  - ⚠️ ignorisan u `.gitignore` → modeli se čuvaju lokalno ili preko **GitHub Releases**
- **`notebooks/`**
  - eksperimenti i analize (EDA, testiranje algoritama)
- **`scripts/`**
  - pomoćni alati (npr. `download_model.py` za preuzimanje modela sa GitHub Releases)
- **`src/`**
  - `train_model.py` – trenira i snima najbolji model
  - `predict_category.py` – interaktivna predikcija kategorije proizvoda
- **`requirements.txt`**
  - potrebne biblioteke
- **`README.md`**
  - dokumentacija projekta

---

## ⚙️ Instalacija

1. Kloniraj repozitorijum:
   ```bash
   git clone https://github.com/Mirsad-py/Predikcija-kategorije-proizvoda-na-osnovu-naslova.git
   cd Predikcija-kategorije-proizvoda-na-osnovu-naslova

🛠️ Tehnologije

Python 3.9+

pandas, numpy

scikit-learn

matplotlib

joblib

requests

👥 Tim

Mirsad
 – razvoj i testiranje