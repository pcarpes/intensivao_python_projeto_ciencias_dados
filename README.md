
# Intensivão Python – Data Science Project

> **Predicting product _Sales_ from ad spend on TV, Radio and Newspaper**

This repo contains the hands‑on project developed during Hashtag Programação’s **“Intensivão de Python – Ciência de Dados”** boot‑camp.  
We build a complete machine‑learning pipeline – from raw CSV to production‑ready model – that helps a marketing team decide how much to invest in each advertising channel.

---

## 📂 Repository structure

| Path | Purpose |
|------|---------|
| `Arquivo Inicial - Aula 4.ipynb` | Step‑by‑step Jupyter Notebook (exploration ➜ training ➜ prediction) |
| `advertising.csv` | Historical dataset with spend and sales (200 rows) |
| `novos.csv` | New campaigns to be forecast |
| `README.md` | *You are here* |

---

## 🚀 Problem statement

Hashtag’s fictitious company wants to **forecast sales (in _millions of R$_)** given the planned spend (in _thousands of R$_) on:

1. **TV**
2. **Radio**
3. **Newspaper**

Accurate forecasts let the team re‑allocate the budget to the channels that produce the highest ROI.

---

## 🔎 Workflow overview

1. **Load data**  
   ```python
   import pandas as pd
   df = pd.read_csv("advertising.csv")
   ```
2. **Exploratory Data Analysis**  
   * Descriptive stats `df.describe()`  
   * Correlation heat‑map (`seaborn.heatmap`)  
   * Pairplots of each feature vs sales
3. **Split train / test**  
   ```python
   from sklearn.model_selection import train_test_split
   X = df[["TV", "Radio", "Newspaper"]]
   y = df["Sales"]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
   ```
4. **Model selection & training**  
   A simple **Multiple Linear Regression** was chosen for its interpretability:
   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression().fit(X_train, y_train)
   ```
5. **Evaluation**  
   * `R²` (coefficient of determination)  
   * `MAE`, `MSE`, `RMSE`  
   * Residual plots to confirm homoscedasticity
6. **Feature impact**  
   Inspect `model.coef_` to quantify marginal sales uplift per extra R$ 1 k in each medium.
7. **Predict on new data**  
   ```python
   novos = pd.read_csv("novos.csv")
   novos["Sales_Pred"] = model.predict(novos)
   novos.to_csv("novos_com_previsao.csv", index=False)
   ```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| R² (train) | ≈ 0.90 |
| R² (test)  | ≈ 0.88 |
| MAE        | ~0.7 million |
| RMSE       | ~1.0 million |

The model explains ~88 % of the variance in unseen data, providing actionable guidance for campaign planning.

---

## 🛠 Requirements

```bash
python >= 3.10
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install everything with:

```bash
pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 💻 How to run

```bash
git clone https://github.com/pcarpes/intensivao_python_projeto_ciencias_dados.git
cd intensivao_python_projeto_ciencias_dados
jupyter notebook "Arquivo Inicial - Aula 4.ipynb"
# run the notebook cells
```

To skip the notebook and **just generate forecasts**:

```python
import joblib, pandas as pd
model = joblib.load("modelo_sales.pkl")   # generated inside the notebook
novos = pd.read_csv("novos.csv")
print(model.predict(novos))
```

---

## 📝 Future work

* Grid‑search over Polynomial Regression and Random Forest
* Cross‑validation & learning curves
* Deploy a Streamlit dashboard for interactive “what‑if” analysis
* Add unit tests and GitHub Actions CI

---

## © License

This project is released under the **MIT License** – see `LICENSE` for details.
