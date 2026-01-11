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

---

## Computer Vision (Multimodal Model)

### Research Foundation

Our CV approach is based on state-of-the-art research in multimodal real estate valuation:

| Paper | Method | Key Finding |
|-------|--------|-------------|
| [MHPP (arXiv 2024)](https://arxiv.org/abs/2409.05335) | CLIP + ResNet50 | +21-26% MAE improvement |
| [PLOS One 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088074/) | ResNet-101 + t-SNE | R² 0.809→0.821 (+1.5%) |
| [NBER Study](https://www.nber.org/papers/w25174) | CNN embeddings | Images explain 11.7% of price variance |
| [Zillow Neural Zestimate](https://www.geekwire.com/2019/zillow-launches-retooled-zestimate-uses-ai-analyze-photographs-see-value-homes/) | CNN quality detection | +20% accuracy improvement |

**Key insights from literature:**
- Property photos capture quality signals not in tabular data (granite countertops, natural light, condition)
- Mean pooling across multiple images is robust to varying photo counts
- PCA reduction from 2048→64 dims preserves most information
- Simple concatenation fusion works as well as complex attention mechanisms

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐        ┌─────────────────────────────────┐    │
│  │ 64K Photos  │───────▶│  ResNet-50 (ImageNet pretrained)│    │
│  │ (~8/listing)│        │  Remove FC → 2048-dim embedding │    │
│  └─────────────┘        └──────────────┬──────────────────┘    │
│                                        │                        │
│                         ┌──────────────▼──────────────────┐    │
│                         │  Mean Pooling (per listing)     │    │
│                         │  Aggregates all photos → 2048d  │    │
│                         └──────────────┬──────────────────┘    │
│                                        │                        │
│                         ┌──────────────▼──────────────────┐    │
│                         │  PCA: 2048 → 64 dimensions      │    │
│                         │  Explained variance: ~85%       │    │
│                         └──────────────┬──────────────────┘    │
│                                        │                        │
│  ┌─────────────┐                       │                        │
│  │ 39 Tabular  │───────────────────────┼───────────────────┐   │
│  │ Features    │                       │                   │   │
│  └─────────────┘        ┌──────────────▼──────────────────┐│   │
│                         │  CONCATENATE: 39 + 64 = 103 dim ││   │
│                         └──────────────┬──────────────────┘│   │
│                                        │◄──────────────────┘   │
│                         ┌──────────────▼──────────────────┐    │
│                         │     ENSEMBLE + OPTUNA           │    │
│                         │  XGBoost + LightGBM + CatBoost  │    │
│                         │         Ridge Meta              │    │
│                         └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Image Feature Extraction

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageFeatureExtractor:
    """
    Extract embeddings using pretrained ResNet-50.
    Based on MHPP and PLOS One research.
    """

    def __init__(self, pca_components=64):
        # Load ResNet-50, remove classification head
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.pca = PCA(n_components=pca_components)

    def extract_single(self, image_path):
        """Extract 2048-dim embedding from single image"""
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.squeeze().numpy()

    def extract_listing(self, image_paths):
        """
        Extract mean embedding for all images of a listing.
        Mean pooling is robust to varying image counts (MHPP).
        """
        embeddings = [self.extract_single(p) for p in image_paths]
        return np.mean(embeddings, axis=0)
```

### Data Requirements

| Requirement | Our Dataset | Notes |
|-------------|-------------|-------|
| Images per listing | ~8 average | MHPP used ~5 |
| Total images | 64,451 | Sufficient for training |
| Image types | Interior + exterior | Kitchen, bathroom, rooms |
| Resolution | Variable | Resized to 224×224 |

### Memory Optimization

Key optimizations for processing 64K images:

```python
# 1. Batch processing with cleanup
for i, listing_id in enumerate(listing_ids):
    embedding = extract_listing(listing_id)
    embeddings.append(embedding)

    # Clear memory every 500 listings
    if i % 500 == 0:
        gc.collect()
        torch.cuda.empty_cache()

# 2. Close PIL images explicitly
img = Image.open(path).convert('RGB')
tensor = transform(img)
img.close()  # Prevent memory leak

# 3. Use torch.no_grad() for inference
with torch.no_grad():
    embedding = model(tensor)
```

### CV Experiment Results (2026-01-11)

We conducted an experiment to test whether adding image features improves price prediction.

#### Experiment Setup

| Parameter | Value |
|-----------|-------|
| Dataset | 8,727 apartments (after outlier removal) |
| Train/Test Split | 80/20 random split (6,981 / 1,746) |
| Listings with images | Train: 6,246/6,981 (89.5%), Test: 1,565/1,746 (89.6%) |
| Image model | ResNet-50 (ImageNet pretrained) |
| Embedding dimension | 2048 → 64 (PCA) |
| PCA explained variance | 81.36% |
| Total features | 103 (39 tabular + 64 image) |
| Optuna trials | 30 per model |
| Hardware | Kaggle GPU (Tesla T4) |

#### Results Comparison

| Model | v3 Tabular Only | v3 + CV (Multimodal) | Difference |
|-------|-----------------|----------------------|------------|
| **Features** | 39 | 103 (+64 image) | +164% |
| **XGBoost** | $122.72 | $126.06 | +$3.34 (worse) |
| **LightGBM** | $125.39 | $128.70 | +$3.31 (worse) |
| **CatBoost** | $128.59 | $130.89 | +$2.30 (worse) |
| **Ensemble MAE** | **$121.71** | $125.28 | **+$3.57 (worse)** |
| **Ensemble R²** | **0.760** | 0.7468 | **-0.013 (worse)** |
| **MedAPE** | **5.49%** | 5.76% | **+0.27% (worse)** |

#### Why CV Did Not Help

1. **Image Quality Issues**
   - Photos on market.kz are user-uploaded, not professional
   - Many photos are low quality, blurry, or poorly lit
   - Some listings have irrelevant photos (building exterior, floor plans, neighborhood)

2. **Weak Price-Image Correlation**
   - Unlike US markets (Zillow, Redfin), Bishkek apartment photos don't strongly correlate with price
   - Similar-looking apartments can have very different prices based on location
   - Tabular features (district, floor, area) already capture most price variance

3. **Curse of Dimensionality**
   - Adding 64 image features to 39 tabular features (103 total)
   - With only 7,000 training samples, model may overfit
   - Optuna found worse hyperparameters for the larger feature space

4. **Information Loss in PCA**
   - PCA retained only 81% of variance (lost 19%)
   - Important visual signals may have been compressed out

#### Conclusion

For the Bishkek real estate market, **tabular features alone are sufficient**. Computer Vision does not add predictive value because:
- Photo quality is inconsistent
- Visual appearance doesn't correlate with price as strongly as in Western markets
- Location and apartment characteristics (captured in tabular data) dominate price prediction

**Recommendation:** Use v3 tabular model ($121.71 MAE, 0.76 R²) as the production model.

#### Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| v3 (Best) | Tabular only, 39 features | [Kaggle](https://www.kaggle.com/code/muraraimbekov/bishkek-real-estate-price-prediction-v3) |
| v3 + CV | Multimodal experiment | [Kaggle](https://www.kaggle.com/code/muraraimbekov/bishkek-v3-computer-vision) |

### Dataset Sources

- **HuggingFace**: [raimbekovm/bishkek-real-estate](https://huggingface.co/datasets/raimbekovm/bishkek-real-estate) (64K images included)
- **Kaggle Tabular**: [bishkek-real-estate-2025](https://www.kaggle.com/datasets/muraraimbekov/bishkek-real-estate-2025)
- **Kaggle Images**: [bishkek-real-estate-images](https://www.kaggle.com/datasets/muraraimbekov/bishkek-real-estate-images) (9.6 GB, 64K photos)

---

## Road Network Features Experiment (2026-01-11)

### Hypothesis

Testing whether road network data from OpenStreetMap can improve price prediction:

1. **Accessibility Hypothesis**: Proximity to main roads → better transport → higher prices
2. **Noise Hypothesis**: Too close to main roads → noise pollution → lower prices
3. **Optimal Distance Theory**: Non-linear relationship, optimal at 100-400m from main roads
4. **Road Density**: Higher density = better infrastructure → higher prices

### Road Features Implemented

| Feature | Description |
|---------|-------------|
| `dist_to_main_road` | Distance to nearest primary/trunk road (km) |
| `dist_to_primary` | Distance to primary roads |
| `dist_to_secondary` | Distance to secondary roads |
| `dist_to_trunk` | Distance to trunk/highway roads |
| `road_density_500m` | Count of road points within 500m |
| `main_road_density_1km` | Main road points within 1km |
| `optimal_zone` | Binary: 100-400m from main road |
| `noise_zone` | Binary: <50m from main road |
| `road_hierarchy_score` | Weighted accessibility score |

### Road Feature Correlations

| Feature | Correlation with Price |
|---------|----------------------|
| main_road_density_1km | **+0.209** |
| road_density_500m | +0.095 |
| road_hierarchy_score | +0.089 |
| optimal_zone | +0.088 |
| dist_to_main_road | -0.051 |
| dist_to_primary | -0.052 |
| dist_to_secondary | -0.049 |
| dist_to_trunk | -0.033 |
| noise_zone | -0.017 |

### Zone Analysis

| Zone | Mean Price | vs Outside |
|------|------------|------------|
| Optimal (100-400m from main road) | $1,604/m² | **+$65/m²** |
| Noise (<50m from main road) | $1,543/m² | **-$31/m²** |

### Ablation Study Results

| Model | MAE | R² |
|-------|-----|-----|
| **Without road features (28 features)** | **$122.39/m²** | **0.7516** |
| With road features (36 features) | $124.33/m² | 0.7452 |
| **Difference** | **+$1.94/m² (worse)** | **-0.0063 (worse)** |

### Why Road Features Did Not Help

1. **Weak correlations** - Maximum correlation only 0.21 (main_road_density_1km)
2. **POI features overlap** - dist_to_center, dist_to_bazaars already capture location quality
3. **District encoding** - Target encoding of districts already captures infrastructure level
4. **Bishkek specifics** - City is relatively small, road access is not a differentiator

### Conclusion

Road network features **do not improve** the model for Bishkek. The existing POI and location features already capture the relevant geographic information. Adding 8 road features actually slightly degraded performance.

**Recommendation:** Do not include road features in production model.

### Resources

- **Notebook**: [Kaggle - Road Features Experiment](https://www.kaggle.com/code/muraraimbekov/bishkek-real-estate-v3-road-features)
- **Roads Dataset**: [Kaggle - Bishkek Roads](https://www.kaggle.com/datasets/muraraimbekov/bishkek-roads)
- **Source**: OpenStreetMap via Geofabrik (22,653 road segments)

---

## Changelog

### 2026-01-11 (Road Features Experiment)
- Tested road network features from OpenStreetMap (22,653 road segments)
- Implemented 8 road features: distances, density, optimal/noise zones
- **Result: Road features did not improve model** (MAE $124.33 vs $122.39 baseline)
- Uploaded roads dataset to Kaggle: [bishkek-roads](https://www.kaggle.com/datasets/muraraimbekov/bishkek-roads)
- Conclusion: POI features already capture location quality

### 2026-01-11 (CV Experiment)
- Conducted multimodal experiment: v3 + ResNet-50 image embeddings
- **Result: CV did not improve model** (MAE $125.28 vs $121.71 baseline)
- Documented findings and analysis in MODELS.md
- Conclusion: Tabular features sufficient for Bishkek market

### 2026-01-10 (CV Pipeline)
- Added multimodal Computer Vision pipeline based on research (MHPP, PLOS One, NBER)
- Implemented ResNet-50 image feature extraction with mean pooling
- Added PCA dimensionality reduction (2048→64)
- Created `notebooks/bishkek_cv_model.py` for multimodal training
- Uploaded 64K images to Kaggle (9.6 GB in 4 parts)

### 2026-01-10 (v3 Release)
- Added 10 POI distance features (bazaars, parks, malls, universities, hospitals, transport, admin, premium zones)
- Implemented Optuna hyperparameter tuning (30 trials per model)
- Added GPU auto-detection for XGBoost/LightGBM/CatBoost
- **Results**: MAE improved from $144 to $121.71 (-15.5%), R² from 0.66 to 0.76 (+15.2%)
