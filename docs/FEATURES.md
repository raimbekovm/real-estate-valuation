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

## POI Distance Features

Points of Interest for Bishkek:

```python
POI_COORDINATES = {
    # Center
    'center': (42.8746, 74.5698),

    # Bazaars
    'dordoi_bazaar': (42.9345, 74.6112),
    'osh_bazaar': (42.8678, 74.5923),
    'orto_sai_bazaar': (42.8234, 74.5567),

    # Malls
    'asia_mall': (42.8712, 74.5889),
    'dordoi_plaza': (42.8834, 74.5834),
    'bishkek_park': (42.8723, 74.6012),

    # Parks
    'oak_park': (42.8756, 74.5923),
    'ata_turk_park': (42.8689, 74.5845),

    # Universities
    'knu': (42.8734, 74.5878),
    'auca': (42.8645, 74.5912),
    'krsu': (42.8567, 74.6034),

    # Hospitals
    'national_hospital': (42.8623, 74.5789),

    # Transport
    'west_bus_station': (42.8456, 74.5234),
    'east_bus_station': (42.8789, 74.6234)
}
```

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
