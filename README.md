# Real Estate Valuation in Central Asia

Machine learning models for property price prediction in emerging Central Asian markets (Kyrgyzstan, Kazakhstan).

## Research Question

How do gradient boosting models perform on small emerging real estate markets compared to established markets, and what adaptations are necessary?

## Project Structure

```
real-estate-valuation/
├── configs/                 # Configuration files
│   └── config.yaml
├── data/
│   ├── raw/                # Raw scraped data
│   ├── processed/          # Cleaned and feature-engineered data
│   └── external/           # External datasets (Ames, etc.)
├── notebooks/              # Jupyter notebooks for EDA
├── src/
│   ├── scrapers/           # Data collection scripts
│   ├── preprocessing/      # Data cleaning and feature engineering
│   ├── models/             # Model training and evaluation
│   └── analysis/           # Results analysis and visualization
├── tests/                  # Unit tests
├── pyproject.toml          # Project configuration
├── Makefile               # Automation commands
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/real-estate-valuation.git
cd real-estate-valuation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
make install-dev
```

## Usage

### Data Collection

```bash
# Scrape data from house.kg
make scrape

# Test scrape (2 pages only)
make scrape-test
```

### Training

```bash
# Full pipeline
make experiment

# Or step by step
make preprocess
make train
```

## Models

- **Baseline**: Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-based**: Random Forest, Extra Trees
- **Boosting**: XGBoost, LightGBM, CatBoost

## Metrics

- MAE (Mean Absolute Error) - primary
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

## Data Sources

| Source | Region | Records |
|--------|--------|---------|
| house.kg | Bishkek, Kyrgyzstan | ~10K |
| krisha.kz | Almaty, Kazakhstan | ~15K |

## License

MIT

## Author

Murat Raimbekov
