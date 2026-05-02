# Titanic Survival Prediction

A binary classification project predicting passenger survival on the Titanic using feature engineering and gradient boosting.

## Dataset

| Property | Value |
|---|---|
| Source | `Classic Datasets/titanic.csv` |
| Rows | 891 |
| Raw Features | 12 |
| Final Features | 24 (after engineering) |
| Target | `Survived` — 0 or 1 |

**Survival rate:** 38.4% survived, 61.6% did not

**Missing values:**
| Column | Missing |
|---|---|
| Cabin | 77.1% |
| Age | 19.9% |
| Embarked | 0.2% |

## Workflow

1. **Data Loading & Inspection** — shape, dtypes, missing value audit
2. **EDA** — survival rates by sex, class, embarkation port, age, fare
3. **Feature Engineering** — 7 new features extracted from raw columns
4. **Preprocessing** — encoding, train/test split (80/20 stratified)
5. **Model Comparison** — 7 models with 5-fold stratified cross-validation
6. **Hyperparameter Tuning** — GridSearchCV on XGBoost
7. **Final Evaluation** — hold-out test set, confusion matrix, ROC curve, feature importance

## Feature Engineering

| Feature | Description |
|---|---|
| `Title` | Extracted from `Name` (Mr, Mrs, Miss, Master, Rare) — encodes social status + age |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | 1 if traveling alone |
| `FarePerPerson` | `Fare / FamilySize` — corrects for shared tickets |
| `AgeBand` | Age binned into 5 groups (child / teen / adult / middle-age / senior) |
| `FareBand` | Fare quartile bins |
| `Deck` | First letter of `Cabin`, `Unknown` if missing |

`Age` was imputed using the **median per title group** (more accurate than global median — `Master` passengers are young boys, `Mrs` are older).

## Models Compared (5-Fold CV on Training Set)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Gradient Boosting | 0.829 | 0.773 | **0.886** |
| XGBoost | 0.819 | 0.757 | 0.873 |
| LightGBM | 0.826 | 0.764 | 0.872 |
| Logistic Regression | 0.819 | 0.760 | 0.869 |
| Random Forest | 0.805 | 0.741 | 0.862 |
| KNN | 0.809 | 0.741 | 0.830 |
| Decision Tree | 0.806 | 0.741 | 0.790 |

## Best Model — Tuned XGBoost

Best params from GridSearchCV:

```
colsample_bytree=0.8, learning_rate=0.05, max_depth=4,
n_estimators=200, subsample=1.0
```

**Final test set results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8101 |
| F1 Score | 0.7424 |
| ROC-AUC | 0.8495 |

## Key Findings

1. **Sex** — strongest predictor; women survived at ~74% vs men at ~19%
2. **Pclass** — 1st class had 3× the survival rate of 3rd class
3. **Title** — captures social status and age signal beyond sex alone (`Master` = young boys, high survival)
4. **FarePerPerson** — proxy for wealth and cabin deck location
5. **FamilySize** — small families (2–4) survived better than solo travelers or large groups

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```
