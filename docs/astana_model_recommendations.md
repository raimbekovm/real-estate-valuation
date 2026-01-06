# Рекомендации по модели для Астаны

На основе опыта разработки модели для Бишкека (house.kg).

---

## 1. Итоги модели Бишкека

### Лучший результат
| Метрика | Значение |
|---------|----------|
| Модель | XGBoost |
| MAE | $126/м² |
| MAPE | 8.1% |
| R² | 0.703 |
| Признаков | 42 |

### Этапы улучшения
| Этап | MAE | R² | Улучшение |
|------|-----|-----|-----------|
| Baseline (14 признаков) | $142 | 0.605 | — |
| + POI features | $145 | 0.600 | -2% |
| + Hyperparameter tuning | $129 | 0.682 | +9% |
| + Новые признаки (42) | $126 | 0.703 | +11% |

---

## 2. Рекомендуемые признаки для Астаны

### 2.1 Базовые признаки (обязательные)

| Признак | Источник | Важность (Бишкек) |
|---------|----------|-------------------|
| `latitude` | Координаты | 6.9% |
| `longitude` | Координаты | 21.2% |
| `area` | Площадь | 11.4% |
| `rooms` | Комнаты | 1.3% |
| `floor` | Этаж | 2.4% |
| `total_floors` | Этажность | 5.9% |
| `year_built` | Год постройки | 3.7% |

**Примечание для Астаны:** Координаты будут особенно важны из-за большой разницы цен между Левым берегом (Есильский р-н) и старыми районами.

### 2.2 Категориальные признаки

| Признак | Значения в Астане | Важность (Бишкек) |
|---------|-------------------|-------------------|
| `house_type` | монолит, кирпич, панель, иной | 0.7% |
| `condition` | свежий ремонт, аккуратный, черновая, требует ремонта | **28.2%** |
| `district` | 6 районов | — (через encoding) |

**Важно:** `condition` был главным предиктором в Бишкеке (28%). В Астане заполнено только 41%, но можно:
- Использовать `raw_состояние_квартиры`
- Заполнить NaN как отдельную категорию "не указано"

### 2.3 Производные признаки (feature engineering)

```python
# Этажность
df['floor_ratio'] = df['floor'] / df['total_floors']
df['is_first_floor'] = (df['floor'] == 1).astype(int)
df['is_last_floor'] = (df['floor'] == df['total_floors']).astype(int)
df['is_middle_floor'] = ((df['floor'] > 1) & (df['floor'] < df['total_floors'])).astype(int)

# Здание
df['building_age'] = 2025 - df['year_built']
df['is_new_building'] = (df['year_built'] >= 2020).astype(int)
df['is_highrise'] = (df['total_floors'] >= 10).astype(int)
df['is_lowrise'] = (df['total_floors'] <= 5).astype(int)

# Площадь
df['area_per_room'] = df['area'] / df['rooms']

# Декада постройки
df['building_decade'] = (df['year_built'] // 10) * 10
```

### 2.4 Бинарные признаки из категориальных

| Признак | Источник | Заполнение в Астане |
|---------|----------|---------------------|
| `has_balcony` | balcony | 51.3% |
| `has_parking` | parking | 68.7% |
| `has_furniture` | furniture | 49.5% |
| `is_monolith` | house_type | 51.7% |

```python
# Балкон
df['has_balcony'] = df['balcony'].notna().astype(int)

# Паркинг
df['has_parking'] = df['parking'].notna().astype(int)

# Мебель
df['has_furniture'] = df['furniture'].isin(['полностью', 'частично']).astype(int)

# Санузел
df['has_multiple_bathrooms'] = df['bathroom'].str.contains('2|более', na=False).astype(int)
df['has_separate_bathroom'] = (df['bathroom'] == 'раздельный').astype(int)
```

### 2.5 Target encoding по районам (важно!)

В Бишкеке district encoding дал **18% важности** (2-й после condition).

```python
from sklearn.model_selection import KFold

def target_encode_cv(df, col, target, n_splits=5):
    """Target encoding с кросс-валидацией для предотвращения утечки"""
    df[f'{col}_price_mean'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        df.iloc[val_idx, df.columns.get_loc(f'{col}_price_mean')] = \
            df.iloc[val_idx][col].map(means)

    # Сглаживание для малых выборок
    global_mean = df[target].mean()
    counts = df.groupby(col)[target].count()
    smoothing = 100  # минимум объектов для полного доверия

    df[f'{col}_price_smoothed'] = df[f'{col}_price_mean'] * (counts / (counts + smoothing)) + \
                                   global_mean * (smoothing / (counts + smoothing))

    return df
```

**Для Астаны ожидаемые средние цены по районам:**
| Район | Медиана ₸/м² | Относительно среднего |
|-------|--------------|----------------------|
| Есильский | 707,327 | +20% |
| Нура | 664,644 | +12% |
| Сарайшык | 571,584 | -3% |
| Алматы | 511,111 | -13% |
| Сарыарка | 470,404 | -20% |
| Байконур | 453,047 | -23% |

---

## 3. Рекомендуемая архитектура модели

### 3.1 Лучшая модель: XGBoost

В Бишкеке XGBoost показал лучший баланс качества и стабильности.

**Рекомендуемые гиперпараметры (стартовые):**
```python
xgb_params = {
    'n_estimators': 350,
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.65,
    'colsample_bytree': 0.65,
    'min_child_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 3.5,
    'random_state': 42
}
```

### 3.2 Альтернативы

| Модель | Плюсы | Минусы |
|--------|-------|--------|
| **XGBoost** | Лучший MAE, стабильный | Дольше обучается |
| CatBoost | Нативная работа с категориями | Немного хуже MAE |
| LightGBM | Быстрый, хороший R² | Чуть выше MAE |

### 3.3 Тюнинг гиперпараметров (Optuna)

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }

    model = XGBRegressor(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

## 4. Целевая переменная

### Рекомендация: `price_per_m2_kzt`

В Астане цены в тенге, поэтому используем `price_per_m2_kzt` (не USD).

**Характеристики:**
- Медиана: 590,909 ₸/м²
- Среднее: 627,000 ₸/м²
- Диапазон: 210k — 2M ₸/м²

### Нормализация (опционально)

```python
# Log-transform для более нормального распределения
y = np.log1p(df['price_per_m2_kzt'])

# Обратное преобразование при предсказании
predictions_kzt = np.expm1(model.predict(X))
```

---

## 5. Валидация

### 5.1 Кросс-валидация (5-Fold)

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"MAE: {-scores.mean():,.0f} ± {scores.std():,.0f} ₸/м²")
```

### 5.2 Метрики для оценки

| Метрика | Формула | Целевое значение |
|---------|---------|------------------|
| MAE | mean(\|y - ŷ\|) | < 60,000 ₸/м² (~10%) |
| MAPE | mean(\|y - ŷ\| / y) | < 10% |
| R² | 1 - SS_res/SS_tot | > 0.70 |

### 5.3 Ожидаемые результаты для Астаны

На основе Бишкека (MAPE 8.1%):
- **MAE**: ~48,000-60,000 ₸/м² (8-10% от медианы)
- **R²**: 0.65-0.75
- **Для квартиры 60м²**: ошибка ~3-3.6 млн ₸ (при средней цене 35 млн)

---

## 6. Особенности Астаны vs Бишкек

### 6.1 Что учесть

| Фактор | Бишкек | Астана | Рекомендация |
|--------|--------|--------|--------------|
| Валюта | USD | KZT | Использовать KZT |
| Год постройки | 1950-2024 | 1950-2026 | Корреляция +0.39 (выше!) |
| Новостройки | ~30% | 50% | Важный признак |
| Типы домов | монолит/кирпич/панель | монолит доминирует (52%) | Кодировать |
| Районы | много микрорайонов | 6 крупных районов | Target encoding |
| Этажность | разная | больше высоток | is_highrise важен |

### 6.2 Уникальные признаки для Астаны

1. **Левый берег** — can создать `is_left_bank = (district == 'Есильский')` (+20% к цене)

2. **Близость к Байтереку/центру** — если есть POI данные

3. **Элитные ЖК** — извлечь из `raw_жилой_комплекс`:
   ```python
   elite_jk = ['Хайвил', 'Гранд Астана', 'Абу-Даби Плаза', ...]
   df['is_elite'] = df['raw_жилой_комплекс'].isin(elite_jk).astype(int)
   ```

---

## 7. План действий

### Этап 1: Подготовка данных
1. Загрузить `data/processed/astana_clean.csv` (18,293 записи)
2. Создать базовые признаки (14 шт.)
3. Train/test split 80/20

### Этап 2: Baseline модели
1. Запустить Random Forest, XGBoost, LightGBM без тюнинга
2. Сравнить MAE и R²
3. Определить важность признаков

### Этап 3: Feature engineering
1. Добавить производные признаки (+10)
2. Target encoding по районам (+2)
3. Бинарные признаки (+8)
4. Запустить модели, сравнить

### Этап 4: Тюнинг гиперпараметров
1. Optuna с 50 trials
2. 5-Fold CV
3. Выбрать лучшую модель

### Этап 5: Финальная оценка
1. Test set evaluation
2. SHAP values для интерпретации
3. Сохранить модель

---

## 8. Код для старта

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor

# Загрузка данных
df = pd.read_csv('data/processed/astana_clean.csv')

# Базовые признаки
numeric_features = ['rooms', 'area', 'floor', 'total_floors', 'year_built',
                    'latitude', 'longitude']

# Feature engineering
df['floor_ratio'] = df['floor'] / df['total_floors']
df['is_first_floor'] = (df['floor'] == 1).astype(int)
df['is_last_floor'] = (df['floor'] == df['total_floors']).astype(int)
df['building_age'] = 2025 - df['year_built']
df['is_new_building'] = (df['year_built'] >= 2020).astype(int)
df['is_highrise'] = (df['total_floors'] >= 10).astype(int)
df['area_per_room'] = df['area'] / df['rooms']
df['has_parking'] = df['parking'].notna().astype(int)
df['has_balcony'] = df['balcony'].notna().astype(int)
df['is_monolith'] = (df['house_type'] == 'монолитный').astype(int)

# One-hot encoding для house_type
df = pd.get_dummies(df, columns=['house_type'], prefix='ht')

# Все признаки
features = numeric_features + [
    'floor_ratio', 'is_first_floor', 'is_last_floor',
    'building_age', 'is_new_building', 'is_highrise',
    'area_per_room', 'has_parking', 'has_balcony', 'is_monolith'
] + [c for c in df.columns if c.startswith('ht_')]

X = df[features].fillna(0)
y = df['price_per_m2_kzt']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline XGBoost
model = XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:,.0f} ₸/м²")
print(f"MAPE: {mape:.1f}%")
print(f"R²: {r2:.3f}")
```

---

*Документ создан: 6 января 2025 г.*
