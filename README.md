# Real Estate Valuation in Central Asia

Machine learning models for property price prediction in emerging Central Asian markets (Kyrgyzstan, Kazakhstan).

## Current Results

### Bishkek (house.kg)

| Metric | Value |
|--------|-------|
| MAE | $144/m² |
| MedAPE | 7.0% |
| R² | 0.66 |
| Within 10% | 64% |

Kaggle: [bishkek-real-estate-price-prediction-v3](https://www.kaggle.com/code/muraraimbekov/bishkek-real-estate-price-prediction-v3)

### Astana (krisha.kz)

| Metric | Value |
|--------|-------|
| MAE | 56,563 ₸/m² |
| MAPE | 9.0% |
| R² | **0.83** |
| Records | 18,293 |

Key insight: Residential complex name (`raw_жилой_комплекс`) contributes **41%** of prediction power.

Kaggle: [astana-real-estate-price-prediction](https://www.kaggle.com/code/muraraimbekov/astana-real-estate-price-prediction)

## Features

- **Data Collection**: Scraping from house.kg (Bishkek) and krisha.kz (Almaty, Astana)
- **Photo Scraping**: Download apartment photos for computer vision features
- **Rich Features**: 50+ features including spatial lag, H3 tiles, market trends, POI distances
- **Ensemble Model**: Stacking (XGBoost + LightGBM + CatBoost)
- **Kaggle Integration**: Upload datasets with photos directly to Kaggle

## Project Structure

```
real-estate-valuation/
├── data/
│   ├── raw/                 # Raw scraped CSV data
│   ├── images/              # Downloaded apartment photos
│   │   └── bishkek/         # Photos by listing_id
│   └── kaggle_upload/       # Prepared dataset for Kaggle
├── notebooks/
│   └── bishkek_model_training_v3.ipynb  # Main training notebook
├── src/
│   ├── scrapers/
│   │   ├── house_kg.py      # house.kg scraper (Kyrgyzstan)
│   │   └── krisha_kz.py     # krisha.kz scraper (Kazakhstan)
│   └── features/
│       └── advanced_features.py  # Feature engineering classes
├── scripts/
│   ├── enrich_listings.py   # Add photos & descriptions to existing data
│   └── upload_to_kaggle.py  # Upload dataset to Kaggle
├── docs/
│   └── experiments/         # Research and experiment logs
└── README.md
```

## Installation

```bash
git clone https://github.com/raimbekovm/real-estate-valuation.git
cd real-estate-valuation

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Data Collection

```bash
# Scrape Bishkek listings (house.kg)
python -m src.scrapers.house_kg

# Scrape with photo downloading
python -c "
from src.scrapers.house_kg import HouseKGScraper
scraper = HouseKGScraper(download_photos=True)
scraper.scrape(max_pages=10)
scraper.save()
"
```

### Enrich Existing Data

Add photos and descriptions to already scraped listings:

```bash
# Single thread (safe)
python scripts/enrich_listings.py --csv data/raw/listings.csv

# Multi-threaded (faster)
python scripts/enrich_listings.py --csv data/raw/listings.csv --workers 2
```

### Upload to Kaggle

```bash
# Prepare and upload dataset with photos
python scripts/upload_to_kaggle.py --csv data/raw/listings.csv --images data/images/bishkek

# Or full pipeline: scrape + upload
python scripts/upload_to_kaggle.py --scrape --max-pages 100
```

### Train Model

Run the Kaggle notebook locally or on Kaggle:

```bash
cd notebooks
kaggle kernels push -p .
kaggle kernels status muraraimbekov/bishkek-real-estate-price-prediction-v3
```

## Data Sources

| Source | Region | Records | Photos | Status |
|--------|--------|---------|--------|--------|
| [house.kg](https://house.kg) | Bishkek, Kyrgyzstan | ~10K | ~70K | Ready |
| [krisha.kz](https://krisha.kz) | Astana, Kazakhstan | ~22K | TBD | Collected |
| [krisha.kz](https://krisha.kz) | Almaty, Kazakhstan | TBD | TBD | Needs VPN |

**Kaggle Datasets**:
- [bishkek-real-estate-2025](https://www.kaggle.com/datasets/muraraimbekov/bishkek-real-estate-2025) - Bishkek with photos

## Model Architecture

```
Features (50+)
    ├── Basic: rooms, area, floor, year_built
    ├── Condition: condition_score, ceiling_height, bathroom_type
    ├── Location: district, latitude, longitude
    ├── POI Distances: center, malls, parks, bazaars, transport
    ├── Spatial Lag: neighbor_price_mean, neighbor_count (500m radius)
    ├── H3 Tiles: res7, res8, res9 encoded
    ├── Market Trends: district_price_30/60/90d rolling means
    └── Density: listings_500m, listings_1000m

Ensemble (Stacking)
    ├── XGBoost
    ├── LightGBM
    ├── CatBoost
    └── Ridge (meta-learner)
```

## Roadmap

- [x] Phase 1: Spatial features (H3, neighbor prices, density)
- [ ] Phase 2: Computer vision (photo-based quality scoring)
- [ ] Phase 3: Multi-city training (Bishkek + Almaty combined)
- [ ] Production API for real-time valuation

## Research

See [docs/experiments/](docs/experiments/) for detailed research notes:
- `2026-01-07_model_improvement_research.md` - Analysis of top models and improvement roadmap

## License

MIT

## Author

Murat Raimbekov
