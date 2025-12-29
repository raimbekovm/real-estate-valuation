"""
Baseline модели для оценки недвижимости Бишкека
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Попытка импортировать продвинутые библиотеки
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except (ImportError, Exception):
    HAS_CATBOOST = False


def load_and_prepare_data(filepath: str):
    """Загрузка и подготовка данных"""
    df = pd.read_csv(filepath)
    print(f"Загружено: {len(df)} записей")

    # Целевая переменная - цена за м²
    target = 'price_per_m2'

    # Числовые признаки
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
        'distance_to_center', 'building_age', 'is_premium_zone'
    ]

    # Категориальные признаки
    categorical_features = [
        'house_type', 'condition', 'heating'
    ]

    # Фильтруем колонки которые есть
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    print(f"Числовые признаки: {len(numeric_features)}")
    print(f"Категориальные признаки: {len(categorical_features)}")

    # Подготовка данных
    X = df[numeric_features + categorical_features].copy()
    y = df[target].copy()

    # Удаляем строки с пропущенной целевой переменной
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Заполнение пропусков в числовых признаках
    for col in numeric_features:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Кодирование категориальных признаков
    label_encoders = {}
    for col in categorical_features:
        X[col] = X[col].fillna('unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    print(f"Финальный датасет: {len(X)} записей, {len(X.columns)} признаков")

    return X, y, numeric_features, categorical_features, label_encoders


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str):
    """Оценка модели"""
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'model': model_name,
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    }

    return metrics, model


def print_metrics(metrics: dict):
    """Вывод метрик"""
    print(f"\n{'='*50}")
    print(f"Модель: {metrics['model']}")
    print(f"{'='*50}")
    print(f"MAE  (train/test): ${metrics['train_mae']:.0f} / ${metrics['test_mae']:.0f}")
    print(f"RMSE (train/test): ${metrics['train_rmse']:.0f} / ${metrics['test_rmse']:.0f}")
    print(f"R²   (train/test): {metrics['train_r2']:.3f} / {metrics['test_r2']:.3f}")
    print(f"MAPE (test):       {metrics['mape']:.1f}%")


def print_feature_importance(model, feature_names, top_n=15):
    """Вывод важности признаков"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\nТоп-{top_n} важных признаков:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")


def main():
    # Загрузка данных
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'bishkek_clean.csv'
    X, y, numeric_features, categorical_features, label_encoders = load_and_prepare_data(str(data_path))

    feature_names = list(X.columns)

    # Разбиение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Масштабирование для линейных моделей
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    best_model = None
    best_score = float('inf')

    # 1. Linear Regression
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("="*60)

    lr = LinearRegression()
    metrics, _ = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")
    print_metrics(metrics)
    results.append(metrics)

    # 2. Ridge Regression
    ridge = Ridge(alpha=1.0)
    metrics, _ = evaluate_model(ridge, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge Regression")
    print_metrics(metrics)
    results.append(metrics)

    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    metrics, rf_model = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")
    print_metrics(metrics)
    print_feature_importance(rf_model, feature_names)
    results.append(metrics)
    if metrics['test_mae'] < best_score:
        best_score = metrics['test_mae']
        best_model = ('Random Forest', rf_model)

    # 4. Gradient Boosting (sklearn)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    metrics, gb_model = evaluate_model(gb, X_train, X_test, y_train, y_test, "Gradient Boosting")
    print_metrics(metrics)
    print_feature_importance(gb_model, feature_names)
    results.append(metrics)
    if metrics['test_mae'] < best_score:
        best_score = metrics['test_mae']
        best_model = ('Gradient Boosting', gb_model)

    # 5. XGBoost
    if HAS_XGBOOST:
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
        metrics, xgb_model = evaluate_model(xgb, X_train, X_test, y_train, y_test, "XGBoost")
        print_metrics(metrics)
        print_feature_importance(xgb_model, feature_names)
        results.append(metrics)
        if metrics['test_mae'] < best_score:
            best_score = metrics['test_mae']
            best_model = ('XGBoost', xgb_model)

    # 6. LightGBM
    if HAS_LIGHTGBM:
        lgbm = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
        metrics, lgbm_model = evaluate_model(lgbm, X_train, X_test, y_train, y_test, "LightGBM")
        print_metrics(metrics)
        print_feature_importance(lgbm_model, feature_names)
        results.append(metrics)
        if metrics['test_mae'] < best_score:
            best_score = metrics['test_mae']
            best_model = ('LightGBM', lgbm_model)

    # 7. CatBoost
    if HAS_CATBOOST:
        cat = CatBoostRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=0)
        metrics, cat_model = evaluate_model(cat, X_train, X_test, y_train, y_test, "CatBoost")
        print_metrics(metrics)
        print_feature_importance(cat_model, feature_names)
        results.append(metrics)
        if metrics['test_mae'] < best_score:
            best_score = metrics['test_mae']
            best_model = ('CatBoost', cat_model)

    # Сводная таблица
    print("\n" + "="*60)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mae')

    print(f"\n{'Модель':<25} {'MAE (test)':<12} {'RMSE (test)':<12} {'R² (test)':<10} {'MAPE':<8}")
    print("-"*70)
    for _, row in results_df.iterrows():
        print(f"{row['model']:<25} ${row['test_mae']:<11.0f} ${row['test_rmse']:<11.0f} {row['test_r2']:<10.3f} {row['mape']:<7.1f}%")

    print(f"\n{'='*60}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model[0]}")
    print(f"MAE на тесте: ${best_score:.0f}/м²")
    print(f"{'='*60}")

    # Интерпретация
    median_price = y.median()
    print(f"\nИнтерпретация:")
    print(f"  Медианная цена: ${median_price:.0f}/м²")
    print(f"  Ошибка модели: ${best_score:.0f}/м² ({best_score/median_price*100:.1f}% от медианы)")

    return results_df, best_model


if __name__ == '__main__':
    results, best_model = main()
