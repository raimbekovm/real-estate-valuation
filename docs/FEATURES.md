# Feature Engineering

## Overview

50+ features organized into categories:

| Category | Count | Description |
|----------|-------|-------------|
| Basic | 10 | rooms, area, floor, year |
| Location | 5 | district, coordinates |
| POI Distances | 10+ | distances to landmarks |
| Spatial | 6 | neighbor statistics |
| H3 Tiles | 3 | hexagonal encoding |
| Market Trends | 3 | rolling averages |
| Density | 2 | listing counts |
| Building | 5 | type, condition, amenities |

## Basic Features

```python
# From raw data
- rooms: int (1-10)
- area: float (m²)
- living_area: float (m²)
- kitchen_area: float (m²)
- floor: int
- total_floors: int
- year_built: int
- price_usd: int
- price_per_m2: int

# Derived
- floor_ratio: floor / total_floors
- building_age: current_year - year_built
- is_first_floor: floor == 1
- is_last_floor: floor == total_floors
```

## POI Distance Features (Updated: 2026-01-10)

Points of Interest for Bishkek (25 locations in 7 categories):

```python
BISHKEK_POI = {
    # Bazaars (4)
    'bazaars': [
        ('osh_bazaar', 42.874823, 74.569599),
        ('dordoi_bazaar', 42.939732, 74.620613),
        ('ortosay_bazaar', 42.836209, 74.615931),
        ('alamedin_bazaar', 42.88683, 74.637305),
    ],
    # Parks (5)
    'parks': [
        ('dubovy_park', 42.877681, 74.606759),
        ('ataturk_park', 42.839587, 74.595725),
        ('karagach_grove', 42.900362, 74.619652),
        ('victory_park', 42.826531, 74.604411),
        ('botanical_garden', 42.857152, 74.590671),
    ],
    # Malls (4)
    'malls': [
        ('bishkek_park', 42.875029, 74.590403),
        ('dordoi_plaza', 42.874685, 74.618469),
        ('vefa_center', 42.857078, 74.609628),
        ('tsum', 42.876813, 74.61499),
    ],
    # Universities (4)
    'universities': [
        ('auca', 42.81132, 74.627743),
        ('krsu', 42.874862, 74.627114),
        ('bhu', 42.850424, 74.585821),
        ('knu', 42.8822, 74.586638),
    ],
    # Hospitals (2)
    'hospitals': [
        ('national_hospital', 42.869973, 74.596739),
        ('city_hospital', 42.876149, 74.5619),
    ],
    # Transport (3)
    'transport': [
        ('west_bus_station', 42.873213, 74.406103),
        ('east_bus_station', 42.887128, 74.62894),
        ('railway_station', 42.864179, 74.605693),
    ],
    # Admin (3)
    'admin': [
        ('jogorku_kenesh', 42.876814, 74.600155),
        ('ala_too_square', 42.875039, 74.603604),
        ('erkindik_boulevard', 42.864402, 74.605287),
    ],
}

# Premium zones (4)
BISHKEK_PREMIUM_ZONES = {
    'golden_square': (42.8688, 74.6033),
    'voentorg': (42.8722, 74.5941),
    'railway_area': (42.8650, 74.6070),
    'mossovet': (42.8700, 74.6117),
}

# City center
BISHKEK_CENTER = (42.8746, 74.5698)  # Ala-Too Square
```

### Generated Features (10)

| Feature | Description |
|---------|-------------|
| `dist_to_center` | Distance to Ala-Too Square (km) |
| `dist_to_bazaars` | Distance to nearest bazaar (km) |
| `dist_to_parks` | Distance to nearest park (km) |
| `dist_to_malls` | Distance to nearest mall (km) |
| `dist_to_universities` | Distance to nearest university (km) |
| `dist_to_hospitals` | Distance to nearest hospital (km) |
| `dist_to_transport` | Distance to nearest bus/train station (km) |
| `dist_to_admin` | Distance to admin center (km) |
| `dist_to_premium` | Distance to nearest premium zone (km) |
| `is_premium_zone` | Binary: within 1km of premium zone |

### Usage

```python
from src.preprocessing.features import haversine_distance

# Calculate distance to center
distance_to_center = haversine_distance(
    lat1, lon1,
    42.8746, 74.5698
)
```

## Spatial Lag Features

Neighbor price statistics within radius.

```python
from src.features.advanced_features import SpatialLagFeatures

spatial = SpatialLagFeatures(radius_km=0.5)
spatial.fit(df)
df = spatial.transform(df)
```

### Output Columns

| Column | Description |
|--------|-------------|
| neighbor_price_mean | Average price of neighbors |
| neighbor_price_median | Median price of neighbors |
| neighbor_price_std | Std dev of neighbor prices |
| neighbor_price_min | Min neighbor price |
| neighbor_price_max | Max neighbor price |
| neighbor_count | Number of neighbors |
| price_to_neighbor_ratio | price / neighbor_mean |

### Parameters

```python
SpatialLagFeatures(
    radius_km=0.5,      # Search radius
    price_col='price_per_m2',
    lat_col='latitude',
    lon_col='longitude'
)
```

## H3 Hexagonal Features

Uber H3 tile encoding for location.

```python
from src.features.advanced_features import H3Features

h3 = H3Features(resolutions=[7, 8, 9])
df = h3.transform(df)
```

### Output Columns

| Column | Resolution | Hex Size |
|--------|------------|----------|
| h3_res7 | 7 | ~5 km |
| h3_res8 | 8 | ~2 km |
| h3_res9 | 9 | ~500 m |

### Why H3?

- Hierarchical: res9 fits inside res8 inside res7
- Uniform: all hexagons same size (unlike districts)
- Numeric: can be target-encoded

## Market Trend Features

Rolling price averages by district.

```python
from src.features.advanced_features import MarketTrendFeatures

trends = MarketTrendFeatures(windows=[30, 60, 90])
trends.fit(df)
df = trends.transform(df)
```

### Output Columns

| Column | Description |
|--------|-------------|
| district_price_30d_mean | 30-day rolling average |
| district_price_60d_mean | 60-day rolling average |
| district_price_90d_mean | 90-day rolling average |

### Requirements

DataFrame must have `parsed_at` datetime column.

## Density Features

Count of listings in radius.

```python
from src.features.advanced_features import DensityFeatures

density = DensityFeatures(radii=[0.5, 1.0])
density.fit(df)
df = density.transform(df)
```

### Output Columns

| Column | Description |
|--------|-------------|
| listings_500m | Listings within 500m |
| listings_1000m | Listings within 1km |

## Categorical Features

### District Encoding

```python
# Target encoding (mean price per district)
district_encoded = df.groupby('district')['price_per_m2'].transform('mean')

# One-hot encoding
district_dummies = pd.get_dummies(df['district'], prefix='district')
```

### Condition Mapping

```python
CONDITION_MAP = {
    'черновая': 1,
    'предчистовая': 2,
    'требует ремонта': 3,
    'средний ремонт': 4,
    'хороший ремонт': 5,
    'евроремонт': 6,
    'дизайнерский': 7
}
```

### Building Age Categories

```python
def categorize_age(year_built):
    age = 2025 - year_built
    if age <= 2:
        return 'новостройка'
    elif age <= 5:
        return 'новый'
    elif age <= 15:
        return 'средний'
    elif age <= 30:
        return 'старый'
    else:
        return 'очень_старый'
```

## Feature Pipeline Example

```python
import pandas as pd
from src.features.advanced_features import (
    SpatialLagFeatures,
    H3Features,
    MarketTrendFeatures,
    DensityFeatures
)

# Load data
df = pd.read_csv('data/processed/bishkek_clean.csv')

# Apply features
spatial = SpatialLagFeatures(radius_km=0.5)
h3 = H3Features(resolutions=[7, 8, 9])
trends = MarketTrendFeatures(windows=[30, 60, 90])
density = DensityFeatures(radii=[0.5, 1.0])

# Fit and transform
spatial.fit(df)
df = spatial.transform(df)
df = h3.transform(df)
trends.fit(df)
df = trends.transform(df)
density.fit(df)
df = density.transform(df)

# Result: 50+ features
print(df.columns.tolist())
```

## Feature Importance (Astana Model)

Top features by SHAP importance:

| Feature | Importance |
|---------|------------|
| raw_жилой_комплекс | 41% |
| area | 15% |
| floor | 8% |
| district | 7% |
| year_built | 6% |
| rooms | 5% |
| condition | 4% |
| ... | ... |

**Key insight**: Residential complex name is the strongest predictor.

## Changelog

### 2026-01-10
- Updated POI coordinates with 25 verified locations in 7 categories
- Added premium zones (Golden Square, Voentorg, Railway, Mossovet)
- Added 10 POI distance features to model (improved R² from 0.66 to 0.76)
