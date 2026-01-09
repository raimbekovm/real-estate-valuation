# Model Training

## Current Results (Updated: 2026-01-10)

| Market | Model | MAE | MAPE | MedAPE | R² |
|--------|-------|-----|------|--------|-----|
| Bishkek | Ensemble + POI + Optuna | **$121.71/m²** | 7.81% | **5.49%** | **0.76** |
| Astana | XGBoost | 56,563₸/m² | 9.0% | - | 0.83 |

### Bishkek Model Improvements (2026-01-10)

| Version | MAE | R² | Changes |
|---------|-----|-----|---------|
| v1 (baseline) | $144/m² | 0.66 | 22 features, default params |
| **v3 (current)** | **$121.71/m²** | **0.76** | 39 features, POI, Optuna |
| Improvement | **-15.5%** | **+15.2%** | |

#### New Features Added (10 POI features)
- `dist_to_center` - distance to Ala-Too Square
- `dist_to_bazaars` - nearest bazaar (Osh, Dordoi, Ortosay, Alamedin)
- `dist_to_parks` - nearest park (5 parks)
- `dist_to_malls` - nearest mall (Bishkek Park, Dordoi Plaza, Vefa, TSUM)
- `dist_to_universities` - nearest university (AUCA, KRSU, BGU, KNU)
- `dist_to_hospitals` - nearest hospital
- `dist_to_transport` - nearest bus/train station
- `dist_to_admin` - administrative center (Jogorku Kenesh, Erkindik)
- `dist_to_premium` - nearest premium zone
- `is_premium_zone` - binary flag (within 1km of premium zone)

#### Optuna-Optimized Parameters (30 trials)
```python
# XGBoost (Best)
xgb_params = {
    'n_estimators': 907,
    'max_depth': 10,
    'learning_rate': 0.0147,
    'subsample': 0.893,
    'colsample_bytree': 0.691,
    'min_child_weight': 5,
    'reg_alpha': 0.00157,
    'reg_lambda': 5.27e-05
}

# LightGBM
lgb_params = {
    'n_estimators': 755,
    'max_depth': 11,
    'learning_rate': 0.075,
    'num_leaves': 50,
    'subsample': 0.963,
    'colsample_bytree': 0.609
}

# CatBoost
cat_params = {
    'iterations': 368,
    'depth': 8,
    'learning_rate': 0.216,
    'l2_leaf_reg': 0.825
}
```

#### Individual Model Results (After Optuna)
| Model | MAE | MedAPE | R² |
|-------|-----|--------|-----|
| XGBoost | $122.72 | 5.38% | 0.757 |
| LightGBM | $125.39 | 5.65% | 0.748 |
| CatBoost | $128.59 | 5.81% | 0.736 |
| **Ensemble** | **$121.71** | **5.49%** | **0.760** |

---

## Previous Results (Before 2026-01-10)

| Market | Model | MAE | MAPE | R² |
|--------|-------|-----|------|-----|
| Bishkek | Ensemble | $144/m² | 7.0% | 0.66 |
| Astana | XGBoost | 56,563₸/m² | 9.0% | 0.83 |

## Model Types

### Baseline Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'linear': LinearRegression(),
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.1),
    'rf': RandomForestRegressor(n_estimators=100),
    'gb': GradientBoostingRegressor(n_estimators=100)
}
```

### Boosting Models

```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

lgb = LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31
)

cat = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    verbose=0
)
```

### Ensemble Stacking

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

estimators = [
    ('xgb', XGBRegressor(**xgb_params)),
    ('lgb', LGBMRegressor(**lgb_params)),
    ('cat', CatBoostRegressor(**cat_params, verbose=0))
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5,
    n_jobs=-1
)
```

## Training Pipeline

### 1. Data Preparation

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/processed/bishkek_clean.csv')

# Define features
numeric_features = [
    'rooms', 'area', 'living_area', 'kitchen_area',
    'floor', 'total_floors', 'year_built', 'ceiling_height'
]

categorical_features = [
    'district', 'house_type', 'condition'
]

target = 'price_per_m2'

# Split
X = df[numeric_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 2. Preprocessing

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

### 3. Training

```python
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', stacking)
])

model.fit(X_train, y_train)
```

### 4. Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R²: {r2:.4f}')
```

## Hyperparameter Tuning

### With Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }

    model = XGBRegressor(**params)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='neg_mean_absolute_error'
    )
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
```

### Best Parameters (Bishkek)

```python
xgb_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

## Cross-Validation

### Time-Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    scores.append(mean_absolute_error(y_val, y_pred))

print(f'CV MAE: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
```

## Feature Importance

### SHAP Values

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Feature importance
importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)
```

### Built-in Importance

```python
# XGBoost
importance = model.feature_importances_

# Plot
import matplotlib.pyplot as plt

plt.barh(feature_names, importance)
plt.xlabel('Importance')
plt.title('Feature Importance')
```

## Model Persistence

### Save Model

```python
import joblib

joblib.dump(model, 'models/bishkek_model.pkl')
```

### Load Model

```python
model = joblib.load('models/bishkek_model.pkl')
predictions = model.predict(X_new)
```

## Metrics Comparison

| Model | Bishkek MAE | Bishkek R² | Astana MAE | Astana R² |
|-------|-------------|------------|------------|-----------|
| Linear | $210 | 0.45 | 85K₸ | 0.55 |
| Ridge | $205 | 0.47 | 82K₸ | 0.58 |
| Random Forest | $165 | 0.58 | 65K₸ | 0.72 |
| XGBoost | $150 | 0.63 | 58K₸ | 0.80 |
| LightGBM | $152 | 0.62 | 59K₸ | 0.79 |
| CatBoost | $148 | 0.64 | 57K₸ | 0.81 |
| Ensemble (baseline) | $144 | 0.66 | 56K₸ | 0.83 |
| **Ensemble + POI + Optuna** | **$121.71** | **0.76** | - | - |

## GPU Acceleration

The notebook supports automatic GPU detection on Kaggle:

```python
# XGBoost 2.0+ GPU
xgb_params['device'] = 'cuda'
xgb_params['tree_method'] = 'hist'

# LightGBM GPU
lgb_params['device'] = 'gpu'

# CatBoost GPU
cat_params['task_type'] = 'GPU'
```

## Kaggle Notebooks

- **Bishkek v3**: https://www.kaggle.com/code/muraraimbekov/bishkek-real-estate-price-prediction-v3
- **Dataset**: https://www.kaggle.com/datasets/muraraimbekov/bishkek-real-estate-2025

## Changelog

### 2026-01-10
- Added 10 POI distance features (bazaars, parks, malls, universities, hospitals, transport, admin, premium zones)
- Implemented Optuna hyperparameter tuning (30 trials per model)
- Added GPU auto-detection for XGBoost/LightGBM/CatBoost
- **Results**: MAE improved from $144 to $121.71 (-15.5%), R² from 0.66 to 0.76 (+15.2%)
