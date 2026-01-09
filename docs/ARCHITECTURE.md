# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  house.kg    │  │  krisha.kz   │  │  krisha.kz   │           │
│  │  (Bishkek)   │  │  (Astana)    │  │  (Almaty)    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Scrapers                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ HouseKG      │  │ KrishaKZ     │  │ HouseKGJK    │           │
│  │ Scraper      │  │ Scraper      │  │ Scraper      │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Storage                                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                  SQLite Databases                     │       │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐        │       │
│  │  │bishkek.db  │ │ astana.db  │ │ almaty.db  │        │       │
│  │  └────────────┘ └────────────┘ └────────────┘        │       │
│  └──────────────────────────────────────────────────────┘       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                    Images                             │       │
│  │  data/images/{city}/{listing_id}/*.jpg               │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Engineering                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │  Spatial   │ │    H3      │ │  Market    │ │  Density   │   │
│  │    Lag     │ │   Tiles    │ │  Trends    │ │  Features  │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML Pipeline                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Ensemble Stacking                        │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐              │       │
│  │  │ XGBoost  │ │ LightGBM │ │ CatBoost │              │       │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘              │       │
│  │       └────────────┼────────────┘                     │       │
│  │                    ▼                                  │       │
│  │              ┌──────────┐                             │       │
│  │              │  Ridge   │ (meta-learner)              │       │
│  │              └──────────┘                             │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Output                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                   │
│  │  Kaggle    │ │ HuggingFace│ │   API      │                   │
│  │  Dataset   │ │  Dataset   │ │ (planned)  │                   │
│  └────────────┘ └────────────┘ └────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Scrapers (`src/scrapers/`)

| Component | Source | Features |
|-----------|--------|----------|
| `house_kg.py` | house.kg | Apartments, photos, 5 cities |
| `house_kg_jk.py` | house.kg | Residential complexes, URL linking |
| `krisha_kz.py` | krisha.kz | Kazakhstan market, proxy support |

### 2. Database (`src/database/`)

Normalized SQLite schema:
- `apartments` - main listings (30+ fields)
- `residential_complexes` - JK metadata
- `developers` - developer companies
- `parsing_queue` - scraping state

### 3. Features (`src/features/`)

| Class | Description | Output |
|-------|-------------|--------|
| `SpatialLagFeatures` | Neighbor price stats | 6 columns |
| `H3Features` | Hexagonal tiles | 3 columns |
| `MarketTrendFeatures` | Rolling averages | 3 columns |
| `DensityFeatures` | Listing density | 2 columns |

### 4. Models (`src/models/`)

- `baseline.py` - training pipeline
- `tuning.py` - Optuna HPO

## Data Flow

```
1. Scrape    → Raw HTML → Parse → Dict
2. Store     → Dict → SQLite → Normalized tables
3. Export    → SQLite → CSV → pandas DataFrame
4. Features  → DataFrame → Transform → 50+ features
5. Train     → Features → Ensemble → Model
6. Publish   → Model + Data → Kaggle/HuggingFace
```

## Directory Structure

```
real-estate-valuation/
├── configs/           # YAML configuration
├── data/
│   ├── raw/           # Scraped CSVs
│   ├── processed/     # Clean datasets
│   ├── databases/     # SQLite DBs
│   └── images/        # Photos by city
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
├── scripts/           # CLI utilities
├── src/
│   ├── scrapers/      # Web scrapers
│   ├── features/      # Feature engineering
│   ├── models/        # ML models
│   ├── database/      # DB management
│   └── preprocessing/ # Data preprocessing
└── tests/             # Unit tests
```
