# 🏠 Housing Prices Prediction - Kaggle

## 📊 Final Result
- **Rank:** Top 141 (out of ~3,900 participants)
- **MAE:** 14,552
- **Approach:** Stacking Ensemble with Feature Engineering

## 🎯 Overview
A complete project for predicting house prices using advanced machine learning techniques.
The model was developed incrementally from a simple baseline (MAE: 21,217) to an advanced Stacking model (MAE: 14,552).

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)
- **XGBoost, CatBoost, LightGBM** (Gradient Boosting)
- **Stacking** with Out-of-Fold Predictions
- **Feature Engineering** (5 new engineered features)

## 📈 Development Journey

| Stage | MAE | Rank |
|-------|-----|------|
| First model (Random Forest) | 21,217 | 3,614 |
| Data cleaning + RF | 17,675 | 474 |
| Feature Engineering + XGBoost | 15,831 | 300 |
| Stacking (RF + XGBoost) | 16,121 | 293 |
| Diverse XGBoost Stacking | 15,509 | 255 |
| Advanced Stacking (4 models) | 14,772 | 141 |
| **Final Stacking (3 models)** | **14,552** | **141** |

## 🧠 Methodology

### 1. Data Cleaning
- Dropped columns with more than 1000 missing values
- Handled missing values using **training statistics only** (avoiding Data Leakage)

### 2. Feature Engineering
```python
TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF
TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
Age = YrSold - YearBuilt
IsRemodeled = (YearRemodAdd != YearBuilt)
TotalPorch = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
```
| Model | MAE | Weight in Stacking |
|-------|-----|--------------------|
| CatBoost (depth=8) | 14,668 | 0.8748 |
| XGBoost (depth=3) | 15,453 | 0.1278 |
| KNN (k=10) | 25,067 | 0.0288 |

### 4. Stacking with Out-of-Fold
- Split data into 5 folds
- Trained models on 4 folds, predicted on the 5th
- Used Ridge Regression as the meta-model

## 💡 Key Learnings
1. **Simple models often win** on small datasets (XGBoost depth=3 > depth=8)
2. **Diversity matters more than strength** in Stacking (weak KNN still added value)
3. **Feature Engineering improves all models** at once
4. **Data Leakage** is a real risk when processing test data

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/04_stacking_model.ipynb
