"""
Тюнинг гиперпараметров для Gradient Boosting моделей
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def load_data():
    """Загрузка и подготовка данных"""
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'bishkek_clean.csv'
    df = pd.read_csv(data_path)

    target = 'price_per_m2'

    numeric_features = [
        # Базовые
        'rooms', 'area', 'floor', 'total_floors', 'year_built',
        'latitude', 'longitude',
        # Дороги
        'dist_to_road_primary', 'dist_to_road_secondary',
        'dist_to_road_tertiary', 'dist_to_main_road',
        # POI
        'dist_to_bazaars', 'dist_to_parks', 'dist_to_malls',
        'dist_to_universities', 'dist_to_hospitals', 'dist_to_transport',
        'dist_to_admin', 'dist_to_premium',
        # Гео
        'distance_to_center', 'building_age', 'is_premium_zone',
        # Новые: из существующих данных
        'ceiling_height_clean', 'has_separate_bathroom', 'has_multiple_bathrooms',
        'has_balcony', 'has_glazed_balcony', 'has_parking',
        'is_elite', 'is_soviet_series',
        # Новые: производные
        'area_per_room', 'floor_ratio',
        'is_first_floor', 'is_last_floor', 'is_middle_floor',
        'is_new_building', 'is_highrise', 'is_lowrise'
    ]

    categorical_features = ['house_type', 'condition', 'heating', 'building_series_cat']

    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    X = df[numeric_features + categorical_features].copy()
    y = df[target].copy()

    mask = y.notna()
    X = X[mask]
    y = y[mask]

    for col in numeric_features:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_features:
        X[col] = X[col].fillna('unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y


def objective_xgboost(trial, X, y, cv):
    """Objective для XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0
    }

    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_lightgbm(trial, X, y, cv):
    """Objective для LightGBM"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }

    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_catboost(trial, X, y, cv):
    """Objective для CatBoost (ручной CV из-за несовместимости со sklearn)"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': 0
    }

    # Ручной cross-validation
    mae_scores = []
    X_np = X.values if hasattr(X, 'values') else X
    y_np = y.values if hasattr(y, 'values') else y

    for train_idx, val_idx in cv.split(X_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, y_pred))

    return np.mean(mae_scores)


def evaluate_final_model(model, X, y, cv, model_name):
    """Финальная оценка модели с CV"""
    is_catboost = 'catboost' in model_name.lower()

    if is_catboost:
        # Ручной CV для CatBoost
        mae_scores, rmse_scores, r2_scores = [], [], []
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y

        for train_idx, val_idx in cv.split(X_np):
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mae_scores.append(mean_absolute_error(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2_scores.append(r2_score(y_val, y_pred))

        mae_scores = np.array(mae_scores)
        rmse_scores = np.array(rmse_scores)
        r2_scores = np.array(r2_scores)
    else:
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    print(f"\n{'='*50}")
    print(f"{model_name} (5-Fold CV)")
    print(f"{'='*50}")
    print(f"MAE:  ${mae_scores.mean():.0f} ± ${mae_scores.std():.0f}")
    print(f"RMSE: ${rmse_scores.mean():.0f} ± ${rmse_scores.std():.0f}")
    print(f"R²:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

    return {
        'model': model_name,
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std()
    }


def main(n_trials=50):
    print("Загрузка данных...")
    X, y = load_data()
    print(f"Данные: {len(X)} записей, {len(X.columns)} признаков")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    sampler = TPESampler(seed=42)

    results = []
    best_models = {}

    # XGBoost
    print(f"\n{'='*60}")
    print("ТЮНИНГ XGBoost")
    print(f"{'='*60}")

    study_xgb = optuna.create_study(direction='minimize', sampler=sampler)
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X, y, cv), n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры XGBoost:")
    for key, value in study_xgb.best_params.items():
        print(f"  {key}: {value}")
    print(f"Лучший MAE: ${study_xgb.best_value:.0f}")

    best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
    metrics = evaluate_final_model(best_xgb, X, y, cv, "XGBoost (tuned)")
    results.append(metrics)
    best_models['xgboost'] = (best_xgb, study_xgb.best_params)

    # LightGBM
    print(f"\n{'='*60}")
    print("ТЮНИНГ LightGBM")
    print(f"{'='*60}")

    study_lgbm = optuna.create_study(direction='minimize', sampler=sampler)
    study_lgbm.optimize(lambda trial: objective_lightgbm(trial, X, y, cv), n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры LightGBM:")
    for key, value in study_lgbm.best_params.items():
        print(f"  {key}: {value}")
    print(f"Лучший MAE: ${study_lgbm.best_value:.0f}")

    best_lgbm = LGBMRegressor(**study_lgbm.best_params, random_state=42, verbose=-1)
    metrics = evaluate_final_model(best_lgbm, X, y, cv, "LightGBM (tuned)")
    results.append(metrics)
    best_models['lightgbm'] = (best_lgbm, study_lgbm.best_params)

    # CatBoost
    print(f"\n{'='*60}")
    print("ТЮНИНГ CatBoost")
    print(f"{'='*60}")

    study_cat = optuna.create_study(direction='minimize', sampler=sampler)
    study_cat.optimize(lambda trial: objective_catboost(trial, X, y, cv), n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры CatBoost:")
    for key, value in study_cat.best_params.items():
        print(f"  {key}: {value}")
    print(f"Лучший MAE: ${study_cat.best_value:.0f}")

    best_cat = CatBoostRegressor(**study_cat.best_params, random_state=42, verbose=0)
    metrics = evaluate_final_model(best_cat, X, y, cv, "CatBoost (tuned)")
    results.append(metrics)
    best_models['catboost'] = (best_cat, study_cat.best_params)

    # Сводка
    print(f"\n{'='*60}")
    print("СВОДНАЯ ТАБЛИЦА (после тюнинга)")
    print(f"{'='*60}")

    results_df = pd.DataFrame(results).sort_values('mae_mean')

    print(f"\n{'Модель':<25} {'MAE':<18} {'RMSE':<18} {'R²':<15}")
    print("-"*75)
    for _, row in results_df.iterrows():
        mae = f"${row['mae_mean']:.0f} ± ${row['mae_std']:.0f}"
        rmse = f"${row['rmse_mean']:.0f} ± ${row['rmse_std']:.0f}"
        r2 = f"{row['r2_mean']:.3f} ± {row['r2_std']:.3f}"
        print(f"{row['model']:<25} {mae:<18} {rmse:<18} {r2:<15}")

    best_model_name = results_df.iloc[0]['model']
    best_mae = results_df.iloc[0]['mae_mean']

    print(f"\n{'='*60}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
    print(f"MAE: ${best_mae:.0f}/м²")
    print(f"{'='*60}")

    return results_df, best_models


if __name__ == '__main__':
    results, best_models = main(n_trials=50)
