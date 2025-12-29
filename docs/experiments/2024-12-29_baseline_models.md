# Baseline модели — 29 декабря 2024

## Данные
- **Датасет**: `data/processed/bishkek_clean.csv`
- **Записей**: 8,672
- **Признаков**: 14 (11 числовых + 3 категориальных)
- **Train/Test split**: 80/20 (6,937 / 1,735)

## Признаки

### Числовые
- rooms, area, floor, total_floors, year_built
- latitude, longitude
- dist_to_road_primary, dist_to_road_secondary, dist_to_road_tertiary, dist_to_main_road

### Категориальные
- house_type (3 значения: монолит, кирпич, панель)
- condition (5 значений)
- heating (8 значений)

## Целевая переменная
- **price_per_m2** — цена за квадратный метр в USD
- Медиана: $1,560/м²

## Результаты моделей

| Модель | MAE (test) | RMSE (test) | R² (test) | MAPE |
|--------|------------|-------------|-----------|------|
| **Random Forest** | $142 | $257 | 0.605 | **9.0%** |
| XGBoost | $149 | $263 | 0.587 | 9.4% |
| LightGBM | $150 | $263 | 0.586 | 9.5% |
| Gradient Boosting | $155 | $271 | 0.561 | 9.8% |
| CatBoost | $165 | $284 | 0.520 | 10.5% |
| Ridge Regression | $262 | $380 | 0.138 | 17.3% |
| Linear Regression | $262 | $380 | 0.138 | 17.3% |

## Лучшая модель: Random Forest
- **Параметры**: n_estimators=100, max_depth=15
- **MAE**: $142/м² (9.1% от медианной цены)
- **R²**: 0.605

## Важность признаков (Random Forest)

| Ранг | Признак | Важность |
|------|---------|----------|
| 1 | condition | 28.2% |
| 2 | longitude | 21.2% |
| 3 | area | 11.4% |
| 4 | latitude | 6.9% |
| 5 | total_floors | 5.9% |
| 6 | dist_to_road_secondary | 5.0% |
| 7 | dist_to_road_tertiary | 4.5% |
| 8 | dist_to_road_primary | 3.8% |
| 9 | year_built | 3.7% |
| 10 | dist_to_main_road | 3.7% |
| 11 | floor | 2.4% |
| 12 | rooms | 1.3% |
| 13 | heating | 1.2% |
| 14 | house_type | 0.7% |

## Наблюдения

1. **Линейные модели слабые** — R² = 0.14, зависимость нелинейная

2. **Координаты важны** — longitude + latitude дают 28% важности. Это логично: в координатах закодирован район, близость к центру, инфраструктура

3. **Состояние квартиры — главный фактор** — 28% важности. Ремонт сильно влияет на цену

4. **Дорожные признаки работают** — суммарно ~17% важности:
   - dist_to_road_secondary: 5.0%
   - dist_to_road_tertiary: 4.5%
   - dist_to_road_primary: 3.8%
   - dist_to_main_road: 3.7%

5. **Тип дома малозначим** — всего 0.7%. Возможно, эффект уже учтён в других признаках

6. **Random Forest переобучается** — R² train=0.908 vs test=0.605. Нужна регуляризация или меньше max_depth

## Следующие шаги

1. Добавить больше признаков:
   - POI distances (уже есть в данных, но не использованы)
   - is_premium_zone
   - building_age

2. Тюнинг гиперпараметров (GridSearch/Optuna)

3. Запустить XGBoost и LightGBM (после установки libomp)

4. Cross-validation вместо простого train/test split

5. Попробовать предсказывать log(price_per_m2) для лучшего распределения ошибок

---

## Дополнение: XGBoost и LightGBM (после установки libomp)

### XGBoost
- MAE: $149/м², R²: 0.587
- Интересно: condition получает 53% важности — намного больше чем у других моделей
- Меньше переобучение чем Random Forest (train R²: 0.834 vs 0.908)

### LightGBM
- MAE: $150/м², R²: 0.586
- Другая шкала важности (количество сплитов вместо gain)
- Area на первом месте вместо condition

### Вывод
Все Gradient Boosting модели показывают схожие результаты (MAE $149-165). Random Forest лидирует, но с переобучением. Для продакшена лучше XGBoost/LightGBM с тюнингом.
