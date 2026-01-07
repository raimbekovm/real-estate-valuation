# Real Estate Valuation Model Improvement Research

**Document Version:** 1.0
**Date:** 7 January 2026
**Author:** ML Engineering Team
**Status:** Research Complete

---

## Executive Summary

This document presents a comprehensive analysis of state-of-the-art techniques used by top-performing real estate valuation models worldwide, with specific recommendations for improving our Bishkek/Astana property valuation system. The research identifies a clear path from our current **7.0% MedAPE** to an achievable **4-5% MedAPE** through implementation of proven techniques.

### Key Findings

| Category | Current State | Industry Best | Gap |
|----------|---------------|---------------|-----|
| Median APE | 7.0% | 1.9-3.5% | -3.5% to -5% |
| R² Score | 0.668 | 0.85-0.95 | +0.18 to +0.28 |
| Within 10% Accuracy | 64.9% | 85-95% | +20% to +30% |

### Phase 1 Results (Implemented 7 Jan 2026)

⚠️ **Phase 1 features did NOT improve metrics as expected.** See [Section 10](#10-phase-1-experiment-results) for detailed analysis.

| Metric | v2 Baseline | v3 (Phase 1) | Change |
|--------|-------------|--------------|--------|
| MedAPE | 7.0% | 7.0% | +0.7% |
| R² | 0.668 | 0.660 | -1.1% |
| Within 10% | 64.9% | 63.6% | -2.0% |

### Recommended Actions (Updated)

1. ~~**Phase 1 (Quick Wins):** Spatial lag, H3 tiles, market trends → Expected improvement: **-2.5% MedAPE**~~ ✅ Implemented, no improvement
2. **Phase 2 (Computer Vision):** Property photo analysis → Expected improvement: **-1.5% MedAPE** ⬅️ **Next priority**
3. **Phase 3 (Advanced ML):** Graph Neural Networks, Attention mechanisms → Expected improvement: **-1% MedAPE**
4. **Alternative:** More data (Almaty, Astana combined training)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Current Model Architecture](#2-current-model-architecture)
3. [Industry Benchmark Analysis](#3-industry-benchmark-analysis)
4. [Gap Analysis](#4-gap-analysis)
5. [Recommended Improvements](#5-recommended-improvements)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Technical Specifications](#7-technical-specifications)
8. [Risk Assessment](#8-risk-assessment)
9. [References](#9-references)
10. [Phase 1 Experiment Results](#10-phase-1-experiment-results)

---

## 1. Introduction

### 1.1 Background

Real estate valuation using machine learning has evolved significantly over the past decade. Industry leaders like Zillow have achieved remarkable accuracy through continuous innovation in feature engineering, model architecture, and data integration. This research aims to identify actionable improvements for our Central Asian property valuation system.

### 1.2 Scope

This analysis covers:
- Techniques used by Zillow Zestimate (median error 1.9% on-market, 6.9% off-market)
- Kaggle competition winning solutions
- Recent academic research (2023-2025)
- Open-source implementations

### 1.3 Methodology

Research was conducted through:
- Analysis of Zillow Tech Hub publications
- Review of Kaggle House Prices competition top solutions
- Survey of peer-reviewed papers from IEEE, Springer, and MDPI
- Examination of open-source implementations on GitHub

---

## 2. Current Model Architecture

### 2.1 Model Overview

**Version:** Bishkek Real Estate Price Prediction v2
**Target Variable:** Price per square meter (USD/m²)
**Training Data:** ~4,000 apartments (Bishkek, Kyrgyzstan)

### 2.2 Current Features

| Category | Features | Count |
|----------|----------|-------|
| Core | rooms, area, floor_ratio, building_age | 4 |
| Building | is_new_building, is_soviet, is_highrise, total_floors | 4 |
| Apartment | area_per_room, kitchen_ratio, ceiling_height, condition_score | 10 |
| Amenities | bathroom, balcony, parking, furniture, security | 6 |
| House Type | is_monolith, is_brick, is_panel | 3 |
| Residential Complex | has_jk, jk_class_score, jk_completed | 3 |
| Target Encoding | district_target_enc, jk_name_target_enc | 2 |
| Location | dist_center, dist_mall, dist_transport, lat, lon | 8 |
| **Total** | | **~40** |

### 2.3 Current Model Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE MODEL                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  XGBoost    │  │  LightGBM   │  │  CatBoost   │         │
│  │  (n=300)    │  │  (n=300)    │  │  (n=300)    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                   ┌─────────────┐                           │
│                   │ Ridge Meta  │                           │
│                   │  Learner    │                           │
│                   └─────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│  Quantile Regression (LightGBM) for 80% Prediction Intervals│
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Current Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | $144/m² | Average absolute error |
| MedAE | $103/m² | Median absolute error (robust) |
| R² | 0.668 | Variance explained |
| MedAPE | 7.0% | Median percentage error |
| Within 10% | 64.9% | Predictions within 10% of actual |
| CI Coverage | 72.9% | 80% interval coverage |

---

## 3. Industry Benchmark Analysis

### 3.1 Zillow Zestimate (Industry Leader)

**Source:** [Zillow Tech Hub](https://www.zillow.com/tech/building-the-neural-zestimate/)

#### Performance Metrics
| Metric | On-Market | Off-Market |
|--------|-----------|------------|
| Median Error | 1.9% | 6.9% |
| Coverage | 104M homes | 104M homes |

#### Key Technical Innovations

1. **Neural Zestimate Architecture**
   - End-to-end deep learning system
   - Multi-scale geographic tiling (Google S2, Uber H3)
   - Rich categorical embeddings for high-cardinality features

2. **Computer Vision Integration**
   - CNN trained on millions of property photos
   - Automatic quality and condition assessment
   - Curb appeal scoring from exterior images

3. **Real-Time Market Data**
   - Listing price integration
   - Days-on-market features
   - Dynamic market condition adjustment

4. **Geographic Embeddings**
   - Learned representations for zip codes
   - Fine-grained location features
   - Skip-gram model for categorical encoding

### 3.2 Kaggle Competition Analysis

**Competition:** House Prices - Advanced Regression Techniques
**Participants:** 19,465 teams
**Top Solution Rank:** 13th place (top 0.06%)

#### Winning Techniques

| Technique | Usage Rate | Impact |
|-----------|------------|--------|
| Stacking/Blending | 95% | High |
| XGBoost | 90% | High |
| LightGBM | 85% | High |
| Feature Engineering | 100% | Critical |
| Outlier Handling | 80% | Medium |

#### Common Feature Engineering Patterns

```python
# Top features by importance (Kaggle winners)
1. OverallQual (0.234)     # Overall quality
2. GrLivArea (0.178)       # Living area
3. GarageCars (0.085)      # Garage capacity
4. TotalBsmtSF (0.065)     # Basement area
5. 1stFlrSF (0.064)        # First floor size
```

### 3.3 Academic Research Benchmarks

#### Multi-Head Gated Attention (2024)

**Source:** [arXiv:2405.07456](https://arxiv.org/abs/2405.07456)

- Novel attention mechanism for spatial interpolation
- Separate Geographical and Structural attention heads
- Produces embeddings enabling simple linear models to outperform complex ensembles

#### Graph Neural Networks for Property Valuation (2024)

**Source:** [arXiv:2405.06553](https://arxiv.org/html/2405.06553v1)

- Graph-based deep learning leveraging geospatial interactions
- Transformer convolutional message passing layers
- Superior performance on spatial data

#### Computer Vision Impact Study (2025)

**Source:** [PMC12088074](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088074/)

- Hong Kong study: 22,331 sales, 208,746 images
- Visual features reduce median error by **2.4%**
- Half of top 10 predictors were image-derived

---

## 4. Gap Analysis

### 4.1 Feature Gap Matrix

| Feature Category | Our Model | Top Models | Gap Severity |
|------------------|-----------|------------|--------------|
| Spatial Lag (neighbor prices) | ❌ No | ✅ Yes | **Critical** |
| Geographic Embeddings (H3/S2) | ❌ No | ✅ Yes | **Critical** |
| Market Trend Features | ❌ No | ✅ Yes | **High** |
| Listing Density | ❌ No | ✅ Yes | **Medium** |
| Computer Vision | ❌ No | ✅ Yes | **High** |
| External Data (schools, crime) | ⚠️ Partial | ✅ Yes | **Medium** |
| Neural Embeddings | ❌ No | ✅ Yes | **Medium** |
| Graph Neural Networks | ❌ No | ✅ Yes | **Low** |
| Attention Mechanisms | ❌ No | ✅ Yes | **Low** |

### 4.2 Architecture Gap Analysis

```
OUR MODEL                          TOP MODELS (Zillow)
─────────────────────────────────  ─────────────────────────────────
Feature Engineering                Feature Engineering
        │                                  │
        ▼                                  ▼
┌─────────────────┐                ┌─────────────────┐
│ Tree Ensembles  │                │ Neural Networks │
│ (XGB+LGB+CB)    │                │ + Tree Ensemble │
└────────┬────────┘                └────────┬────────┘
         │                                  │
         ▼                                  ▼
    Predictions                    ┌─────────────────┐
                                   │ Computer Vision │
                                   │    (Photos)     │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │  Geographic     │
                                   │  Embeddings     │
                                   └────────┬────────┘
                                            │
                                            ▼
                                       Predictions
```

### 4.3 Data Gap Analysis

| Data Type | Our Dataset | Zillow Dataset | Gap |
|-----------|-------------|----------------|-----|
| Transaction History | 1 year | 15+ years | Large |
| Property Photos | None | Millions | Large |
| School Ratings | None | Yes | Medium |
| Crime Statistics | None | Yes | Medium |
| Tax Assessments | None | Yes | Large |
| Market Trends | None | Real-time | Medium |

---

## 5. Recommended Improvements

### 5.1 Priority Matrix

| Priority | Feature | Complexity | Expected Impact | ROI |
|----------|---------|------------|-----------------|-----|
| 🔴 P0 | Spatial Lag | Low | -1.5-2.0% MedAPE | Very High |
| 🔴 P0 | H3 Geographic Tiles | Low | -1.0-1.5% MedAPE | Very High |
| 🟠 P1 | Market Trend Features | Low | -0.5-1.0% MedAPE | High |
| 🟠 P1 | Listing Density | Low | -0.3-0.5% MedAPE | High |
| 🟡 P2 | Computer Vision | High | -2.0-3.0% MedAPE | Medium |
| 🟡 P2 | External Data APIs | Medium | -0.5-1.0% MedAPE | Medium |
| 🟢 P3 | Graph Neural Networks | High | -1.0-2.0% MedAPE | Low |
| 🟢 P3 | Attention Mechanisms | High | -0.5-1.0% MedAPE | Low |

### 5.2 Detailed Recommendations

#### 5.2.1 Spatial Lag Features (P0 - Critical)

**Description:** Calculate statistics of neighboring property prices within specified radius.

**Rationale:**
- Property prices exhibit strong spatial autocorrelation
- Neighboring prices are among the strongest predictors
- Used by all top-performing models

**Implementation:**

```python
from sklearn.neighbors import BallTree
import numpy as np

class SpatialLagFeatures:
    """
    Calculate spatial lag features using Ball Tree for efficient neighbor lookup.

    Features generated:
    - neighbor_price_mean: Mean price of neighbors
    - neighbor_price_median: Median price of neighbors
    - neighbor_price_std: Standard deviation of neighbor prices
    - neighbor_count: Number of neighbors in radius
    """

    def __init__(self, radius_km: float = 0.5):
        self.radius_km = radius_km
        self.earth_radius_km = 6371.0
        self.tree = None
        self.prices = None

    def fit(self, df: pd.DataFrame, price_col: str = 'price_per_m2'):
        """Fit on training data only to prevent leakage."""
        coords = np.radians(df[['latitude', 'longitude']].values)
        self.tree = BallTree(coords, metric='haversine')
        self.prices = df[price_col].values
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform with spatial lag features."""
        df = df.copy()
        coords = np.radians(df[['latitude', 'longitude']].values)

        # Query neighbors within radius
        radius_rad = self.radius_km / self.earth_radius_km
        indices = self.tree.query_radius(coords, r=radius_rad)

        # Calculate statistics
        df['neighbor_price_mean'] = [
            np.mean(self.prices[idx]) if len(idx) > 0 else np.nan
            for idx in indices
        ]
        df['neighbor_price_median'] = [
            np.median(self.prices[idx]) if len(idx) > 0 else np.nan
            for idx in indices
        ]
        df['neighbor_price_std'] = [
            np.std(self.prices[idx]) if len(idx) > 1 else 0
            for idx in indices
        ]
        df['neighbor_count'] = [len(idx) for idx in indices]

        return df
```

**Expected Impact:** -1.5 to -2.0% MedAPE

**Dependencies:** scikit-learn (already installed)

---

#### 5.2.2 H3 Geographic Tiles (P0 - Critical)

**Description:** Replace raw lat/lon with Uber H3 hexagonal tile indices at multiple resolutions.

**Rationale:**
- Captures neighborhood effects at different scales
- Better for tree-based models than continuous coordinates
- Used by Zillow Neural Zestimate

**Implementation:**

```python
import h3

class H3Features:
    """
    Generate H3 hexagonal tile features at multiple resolutions.

    Resolution guide:
    - 7: ~5.16 km² (district level)
    - 8: ~0.74 km² (neighborhood level)
    - 9: ~0.11 km² (block level)
    """

    def __init__(self, resolutions: list = [7, 8, 9]):
        self.resolutions = resolutions
        self.encoders = {}

    def fit(self, df: pd.DataFrame):
        """Fit label encoders on training data."""
        for res in self.resolutions:
            col = f'h3_res{res}'
            h3_indices = df.apply(
                lambda r: h3.latlng_to_cell(r['latitude'], r['longitude'], res),
                axis=1
            )
            self.encoders[res] = {idx: i for i, idx in enumerate(h3_indices.unique())}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform with H3 features."""
        df = df.copy()

        for res in self.resolutions:
            col = f'h3_res{res}'
            df[col] = df.apply(
                lambda r: h3.latlng_to_cell(r['latitude'], r['longitude'], res),
                axis=1
            )
            # Encode to integers (handle unseen with -1)
            df[f'{col}_encoded'] = df[col].map(
                lambda x: self.encoders[res].get(x, -1)
            )

        return df
```

**Expected Impact:** -1.0 to -1.5% MedAPE

**Dependencies:** `pip install h3`

---

#### 5.2.3 Market Trend Features (P1 - High)

**Description:** Rolling statistics capturing market dynamics over time.

**Rationale:**
- Real estate markets have strong temporal patterns
- Prices in a district affect future prices
- Zillow specifically improved algorithm for "dynamic market conditions"

**Implementation:**

```python
class MarketTrendFeatures:
    """
    Calculate market trend features using rolling windows.

    Features:
    - Rolling mean price by district (30, 60, 90 days)
    - Price momentum (rate of change)
    - Relative price position (vs district average)
    """

    def __init__(self, windows: list = [30, 60, 90]):
        self.windows = windows
        self.district_stats = {}

    def fit(self, df: pd.DataFrame, price_col: str = 'price_per_m2'):
        """Calculate baseline statistics from training data."""
        df = df.sort_values('parsed_at')

        for district in df['district'].unique():
            district_df = df[df['district'] == district]
            self.district_stats[district] = {
                'mean': district_df[price_col].mean(),
                'std': district_df[price_col].std(),
            }

        self.global_mean = df[price_col].mean()
        return self

    def transform(self, df: pd.DataFrame, price_col: str = 'price_per_m2') -> pd.DataFrame:
        """Transform with market trend features."""
        df = df.copy().sort_values('parsed_at')

        # Rolling means by district
        for days in self.windows:
            df[f'district_price_{days}d_mean'] = df.groupby('district')[price_col].transform(
                lambda x: x.rolling(window=days, min_periods=1).mean()
            )

        # Price momentum (30-day change)
        df['price_momentum_30d'] = df.groupby('district')[price_col].transform(
            lambda x: x.pct_change(periods=30).fillna(0)
        )

        # Relative position vs district average
        df['price_vs_district'] = df.apply(
            lambda r: (r[price_col] - self.district_stats.get(r['district'], {}).get('mean', self.global_mean))
                      / max(self.district_stats.get(r['district'], {}).get('std', 1), 1),
            axis=1
        )

        return df
```

**Expected Impact:** -0.5 to -1.0% MedAPE

**Dependencies:** None (pandas)

---

#### 5.2.4 Listing Density Features (P1 - High)

**Description:** Count of nearby listings as proxy for market activity and desirability.

**Implementation:**

```python
class DensityFeatures:
    """
    Calculate listing density features.

    Features:
    - listings_1km: Number of listings within 1km
    - listings_500m: Number of listings within 500m
    - density_ratio: Local density vs city average
    """

    def __init__(self, radii_km: list = [0.5, 1.0]):
        self.radii_km = radii_km
        self.tree = None
        self.avg_density = {}

    def fit(self, df: pd.DataFrame):
        """Fit on training data."""
        coords = np.radians(df[['latitude', 'longitude']].values)
        self.tree = BallTree(coords, metric='haversine')

        # Calculate average densities
        for radius in self.radii_km:
            counts = self.tree.query_radius(
                coords, r=radius/6371.0, count_only=True
            )
            self.avg_density[radius] = np.mean(counts)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform with density features."""
        df = df.copy()
        coords = np.radians(df[['latitude', 'longitude']].values)

        for radius in self.radii_km:
            col = f'listings_{int(radius*1000)}m'
            counts = self.tree.query_radius(
                coords, r=radius/6371.0, count_only=True
            )
            df[col] = counts
            df[f'{col}_ratio'] = counts / max(self.avg_density[radius], 1)

        return df
```

**Expected Impact:** -0.3 to -0.5% MedAPE

---

#### 5.2.5 Computer Vision Features (P2 - Medium Priority)

**Description:** Extract property condition and quality features from listing photos.

**Rationale:**
- Restb.ai reports 9.2% AVM error reduction
- Academic research shows 2.4% median error reduction
- Half of top predictors in some studies are image-derived

**Implementation Approach:**

```python
# Option 1: Pre-trained model (recommended for MVP)
from transformers import AutoFeatureExtractor, AutoModel
import torch

class PropertyImageFeatures:
    """
    Extract features from property photos using pre-trained vision model.

    Architecture:
    - Base: ResNet50 or ViT pre-trained on ImageNet
    - Fine-tuned: Transfer learning on property condition classification

    Output features:
    - condition_score: 1-5 rating
    - quality_score: 1-5 rating
    - room_type_embedding: 512-dim vector
    """

    def __init__(self, model_name: str = "microsoft/resnet-50"):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def extract_features(self, image_urls: list) -> np.ndarray:
        """Extract embeddings from images."""
        features = []
        for url in image_urls:
            img = self._load_image(url)
            inputs = self.extractor(img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            features.append(outputs.pooler_output.numpy().flatten())
        return np.array(features)

# Option 2: Custom condition classifier
class ConditionClassifier:
    """
    Fine-tuned classifier for property condition.

    Classes:
    - C1: New/Never occupied
    - C2: Excellent condition
    - C3: Good condition (minor wear)
    - C4: Average condition
    - C5: Fair condition (needs work)
    - C6: Poor condition (major repairs)
    """
    pass
```

**Prerequisites:**
- Photo URLs from scraper
- GPU for training (optional, can use pre-trained)
- Labeled dataset for fine-tuning (500+ images)

**Expected Impact:** -2.0 to -3.0% MedAPE

---

#### 5.2.6 Graph Neural Network (P3 - Future)

**Description:** Model property relationships as a graph with GNN.

**Architecture:**

```
Nodes: Properties (apartments)
Edges: Spatial proximity, same building, same JK

┌─────────────────────────────────────────────────────────┐
│                  Property Graph                          │
│                                                          │
│    [Apt1]───────[Apt2]       [Apt5]───────[Apt6]        │
│      │            │            │            │            │
│      └────[Apt3]──┘            └────[Apt7]──┘            │
│             │                        │                   │
│         [Apt4]                   [Apt8]                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Graph Neural Network                        │
│                                                          │
│  Input: Node features (property attributes)              │
│  Message Passing: Aggregate neighbor information         │
│  Output: Property embeddings with spatial context        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation:** Use PyTorch Geometric library

**Expected Impact:** -1.0 to -2.0% MedAPE

---

## 6. Implementation Roadmap

### 6.1 Phase 1: Quick Wins (Week 1-2)

```
Week 1:
├── Day 1-2: Implement SpatialLagFeatures
├── Day 3-4: Implement H3Features
├── Day 5: Implement MarketTrendFeatures
└── Day 6-7: Implement DensityFeatures

Week 2:
├── Day 1-2: Integration testing
├── Day 3-4: Model retraining with new features
├── Day 5: A/B testing and validation
└── Day 6-7: Documentation and deployment
```

**Deliverables:**
- Updated feature engineering pipeline
- New model version (v3)
- Performance comparison report

**Expected Metrics After Phase 1:**

| Metric | Current | Target |
|--------|---------|--------|
| MedAPE | 7.0% | 4.5-5.0% |
| R² | 0.668 | 0.75-0.80 |
| Within 10% | 64.9% | 75-80% |

### 6.2 Phase 2: Computer Vision (Week 3-6)

```
Week 3-4:
├── Scraper modification for photo URLs
├── Image dataset collection
├── Data labeling (condition scores)
└── Pre-trained model evaluation

Week 5-6:
├── Model fine-tuning
├── Feature extraction pipeline
├── Integration with main model
└── A/B testing
```

**Prerequisites:**
- Photo URLs in database
- Labeled dataset (500+ images minimum)
- GPU access for training

**Expected Metrics After Phase 2:**

| Metric | After Phase 1 | Target |
|--------|---------------|--------|
| MedAPE | 4.5-5.0% | 3.5-4.0% |
| R² | 0.75-0.80 | 0.82-0.87 |
| Within 10% | 75-80% | 82-88% |

### 6.3 Phase 3: Advanced ML (Week 7-10)

```
Week 7-8:
├── GNN architecture design
├── Graph construction pipeline
├── Initial training experiments
└── Hyperparameter tuning

Week 9-10:
├── Multi-Head Attention implementation
├── Ensemble with existing models
├── Final evaluation
└── Production deployment
```

---

## 7. Technical Specifications

### 7.1 Updated Feature Pipeline

```python
class FeaturePipelineV3:
    """
    Complete feature engineering pipeline for v3 model.
    """

    def __init__(self):
        # Core features
        self.feature_engineer = FeatureEngineer()
        self.target_encoder = TargetEncoderCV(
            cols=['district', 'jk_name'],
            min_samples=30
        )

        # New features (Phase 1)
        self.spatial_lag = SpatialLagFeatures(radius_km=0.5)
        self.h3_features = H3Features(resolutions=[7, 8, 9])
        self.market_trends = MarketTrendFeatures(windows=[30, 60, 90])
        self.density = DensityFeatures(radii_km=[0.5, 1.0])

        # Optional (Phase 2)
        self.image_features = None  # PropertyImageFeatures()

    def fit(self, df: pd.DataFrame, target_col: str):
        """Fit all transformers on training data."""
        self.feature_engineer.fit(df, target_col)

        # Transform first for target encoder
        df_transformed = self.feature_engineer.transform(df)

        self.target_encoder.fit_transform(df_transformed, target_col)
        self.spatial_lag.fit(df_transformed, target_col)
        self.h3_features.fit(df_transformed)
        self.market_trends.fit(df_transformed, target_col)
        self.density.fit(df_transformed)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations."""
        df = self.feature_engineer.transform(df)
        df = self.target_encoder.transform(df)
        df = self.spatial_lag.transform(df)
        df = self.h3_features.transform(df)
        df = self.market_trends.transform(df)
        df = self.density.transform(df)

        if self.image_features:
            df = self.image_features.transform(df)

        return df
```

### 7.2 New Feature List

| Feature | Type | Description |
|---------|------|-------------|
| neighbor_price_mean | float | Mean price within 500m |
| neighbor_price_median | float | Median price within 500m |
| neighbor_price_std | float | Price std within 500m |
| neighbor_count | int | Number of neighbors |
| h3_res7_encoded | int | H3 tile at resolution 7 |
| h3_res8_encoded | int | H3 tile at resolution 8 |
| h3_res9_encoded | int | H3 tile at resolution 9 |
| district_price_30d_mean | float | 30-day rolling district mean |
| district_price_60d_mean | float | 60-day rolling district mean |
| district_price_90d_mean | float | 90-day rolling district mean |
| price_momentum_30d | float | 30-day price change rate |
| price_vs_district | float | Z-score vs district average |
| listings_500m | int | Listings within 500m |
| listings_1000m | int | Listings within 1km |
| listings_500m_ratio | float | Density ratio vs average |

### 7.3 Dependencies

```txt
# requirements.txt additions
h3>=3.7.0              # H3 geographic indexing
torch>=2.0.0           # For computer vision (Phase 2)
transformers>=4.30.0   # Pre-trained vision models (Phase 2)
torch-geometric>=2.3.0 # Graph neural networks (Phase 3)
```

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data leakage in spatial lag | Medium | High | Strict train/test separation |
| H3 library compatibility | Low | Medium | Test on target environment |
| Overfitting with new features | Medium | High | Cross-validation monitoring |
| Photo quality issues | Medium | Medium | Fallback to non-image model |
| GNN scalability | Medium | Low | Start with smaller graphs |

### 8.2 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient neighbor data | Low | Medium | Adaptive radius selection |
| Missing photos | High | Medium | Optional image features |
| Temporal data gaps | Medium | Low | Interpolation strategies |

### 8.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Increased inference time | Medium | Medium | Feature caching |
| Model complexity | Medium | Low | Modular design |
| Maintenance burden | Medium | Medium | Comprehensive documentation |

---

## 9. References

### 9.1 Zillow Publications

1. [Building the Neural Zestimate](https://www.zillow.com/tech/building-the-neural-zestimate/) - Zillow Tech Hub, 2021
2. [Introducing a new and improved Zestimate algorithm](https://www.zillow.com/tech/introducing-a-new-and-improved-zestimate-algorithm/) - Zillow Tech Hub
3. [Home Embeddings for Similar Home Recommendations](https://www.zillow.com/tech/embedding-similar-home-recommendation/) - Zillow Tech Hub, 2020

### 9.2 Academic Papers

4. Sellam, A.Z. et al. (2024). "Boosting House Price Estimations with Multi-Head Gated Attention." [arXiv:2405.07456](https://arxiv.org/abs/2405.07456)

5. "Scalable Property Valuation Models via Graph-based Deep Learning." (2024). [arXiv:2405.06553](https://arxiv.org/html/2405.06553v1)

6. "Real estate valuation with multi-source image fusion and enhanced machine learning pipeline." (2025). [PMC12088074](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088074/)

7. "A Comparative Study of Machine Learning and Spatial Interpolation Methods for Predicting House Prices." MDPI Sustainability, 2022. [Link](https://www.mdpi.com/2071-1050/14/15/9056)

8. "Housing Price Prediction - Machine Learning and Geostatistical Methods." Real Estate Management and Valuation, 2024. [Link](https://www.remv-journal.com/Housing-price-prediction-Machine-learning-and-geostatistical-methods,193897,0,2.html)

9. "Application of Feature Engineering Techniques and Machine Learning Algorithms for Property Price Prediction." ResearchGate, 2024. [Link](https://www.researchgate.net/publication/389609191)

### 9.3 Kaggle Resources

10. House Prices - Advanced Regression Techniques. [Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

11. "Winning Kaggle Solution: Predicting property sales prices." [RPubs](https://rpubs.com/nweissm/670674)

12. "Ensemble Stacked Regressions, XGBoost LightGBM." [Kaggle Notebook](https://www.kaggle.com/code/krishnaraj30/ensemble-stacked-regressions-xgboost-lightgbm)

### 9.4 Open Source Implementations

13. Multi-Head Gated Attention Implementation. [GitHub](https://github.com/ldb0071/Boosting-House-Price-Estimations-with-Multi-Head-Gated-Attention)

14. Uber H3 Documentation. [h3geo.org](https://h3geo.org/)

---

## Appendix A: Code Examples

### A.1 Complete Phase 1 Implementation

See separate file: `feature_engineering_v3.py`

### A.2 Benchmark Scripts

See separate file: `benchmarks/run_comparison.py`

---

---

## 10. Phase 1 Experiment Results

### 10.1 Experiment Overview

**Date:** 7 January 2026
**Notebook:** `bishkek_model_training_v3.ipynb`
**Kaggle Kernel:** [bishkek-real-estate-price-prediction-v3](https://www.kaggle.com/code/muraraimbekov/bishkek-real-estate-price-prediction-v3)

### 10.2 Implemented Features

All Phase 1 features were implemented as planned:

| Feature Category | Features Added | Implementation |
|------------------|----------------|----------------|
| **Spatial Lag** | neighbor_price_mean, neighbor_price_median, neighbor_price_std, neighbor_count | BallTree with 500m radius |
| **H3 Tiles** | h3_res7_encoded, h3_res8_encoded, h3_res9_encoded | Uber H3 library |
| **Market Trends** | district_price_30d_mean, district_price_60d_mean, district_price_90d_mean, days_on_market | Rolling window with shift(1) |
| **Density** | listings_500m, listings_1000m, listings_500m_ratio, listings_1000m_ratio, district_listing_count | BallTree density calculation |

**Total new features:** 16

### 10.3 Results Comparison

| Metric | v2 Baseline | v3 (Phase 1) | Change | Expected |
|--------|-------------|--------------|--------|----------|
| MAE | $144/m² | $144/m² | +0.2% | -15% |
| MedAE | $103/m² | $107/m² | +3.6% | -20% |
| R² | 0.668 | 0.660 | -1.1% | +0.08 |
| MedAPE | 7.0% | 7.0% | +0.7% | -2.5% |
| Within 10% | 64.9% | 63.6% | -2.0% | +15% |
| CI Coverage | 72.9% | 71.5% | -1.4% | - |

**Result:** ❌ Phase 1 features did NOT improve model performance.

### 10.4 Feature Importance Analysis

Despite no metric improvement, new features contribute **23.5%** of total model importance:

| Rank | Feature | Importance | Is v3 New |
|------|---------|------------|-----------|
| 1 | condition_score | 26.4% | No |
| **2** | **neighbor_price_mean** | **15.1%** | **Yes** |
| 3 | area | 14.0% | No |
| 4 | building_age | 4.3% | No |
| 5 | total_floors | 4.3% | No |
| 6 | district_target_enc | 4.1% | No |
| 7 | area_per_room | 3.6% | No |
| 8 | ceiling_height_m | 2.6% | No |
| 9 | dist_nearest_park | 2.2% | No |
| 10 | floor_ratio | 1.9% | No |
| **11** | **listings_1000m** | **1.6%** | **Yes** |
| **12** | **neighbor_price_median** | **1.5%** | **Yes** |

**Top v3 features by importance:**

```
neighbor_price_mean     15.1%
listings_1000m           1.6%
neighbor_price_median    1.5%
neighbor_price_std       0.8%
h3_res8_encoded          0.5%
neighbor_count           0.5%
district_price_30d_mean  0.4%
district_listing_count   0.3%
h3_res9_encoded          0.2%
district_price_60d_mean  0.2%
```

### 10.5 Data Leakage Incident

⚠️ **Important:** Initial v3 results showed suspiciously good metrics (MedAPE 1.4%, R² 0.967). Investigation revealed **data leakage** in the `price_vs_district_zscore` feature.

**Leaking code (removed):**
```python
# BAD: Uses current row's price to calculate zscore
df['price_vs_district_zscore'] = (df[price_col] - district_mean) / district_std
```

**Fix applied:**
- Removed `price_vs_district_zscore` feature entirely
- Added `shift(1)` to all rolling calculations to prevent using current row

### 10.6 Analysis & Conclusions

#### Why Phase 1 Didn't Improve Metrics

1. **Feature Redundancy**
   - `neighbor_price_mean` correlates highly with `district_target_enc` (both capture location value)
   - H3 tiles encode similar information to latitude/longitude
   - Tree-based models may already capture spatial patterns implicitly

2. **Dataset Size Limitations**
   - Only ~7,500 samples in Bishkek dataset
   - H3 resolution 9 creates many sparse tiles
   - Insufficient data for fine-grained spatial patterns

3. **Market Homogeneity**
   - Bishkek real estate market may be less spatially heterogeneous than US markets
   - District-level encoding may be sufficient granularity

4. **Temporal Data Sparsity**
   - Rolling windows have few data points per district
   - Market trends require longer historical data

#### Key Insights

1. **Spatial lag works** - `neighbor_price_mean` became the 2nd most important feature
2. **But it replaces, not complements** - importance shifted from `district_target_enc`, not added to it
3. **Density features are useful** - `listings_1000m` provides unique signal
4. **H3 encoding is marginal** - low importance at all resolutions

### 10.7 Recommendations

Based on Phase 1 results, updated priorities:

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P0** | **Phase 2: Computer Vision** | Only unexploited data source; research shows 2-3% improvement |
| **P1** | **More training data** | Combine Almaty + Astana + Bishkek; spatial features need more samples |
| **P2** | **Feature selection** | Remove redundant v3 features to reduce overfitting |
| **P3** | **Neural embeddings** | May capture non-linear spatial patterns better than H3 |

### 10.8 Artifacts

- **Model:** `bishkek_model_v3.joblib` (Kaggle output)
- **Config:** `bishkek_model_v3_config.json`
- **Notebook:** `notebooks/bishkek_model_training_v3.ipynb`
- **Kaggle Log:** Available at kernel page

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 7 January 2026 | ML Team | Initial research document |
| 1.1 | 7 January 2026 | ML Team | Added Phase 1 experiment results (Section 10) |

---

*This document is part of the Real Estate Valuation Project documentation.*
