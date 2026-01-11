# Real Estate Valuation

ML models for property price prediction in Central Asia (Kyrgyzstan, Kazakhstan).

## Results

| Market | Records | Model | MAE | R² | Notebook |
|--------|---------|-------|-----|-----|----------|
| Bishkek | 8,821 | Ensemble + Optuna | **$121.71/m²** | **0.76** | [Kaggle](https://www.kaggle.com/code/muraraimbekov/bishkek-real-estate-price-prediction-v3) |
| Astana | 18,388 | XGBoost | 56,563₸/m² | 0.83 | [Kaggle](https://www.kaggle.com/code/muraraimbekov/astana-real-estate-price-prediction) |

### Model Details

- **Ensemble**: XGBoost + LightGBM + CatBoost with Ridge meta-learner
- **Features**: 39 tabular features (POI distances, target encoding, spatial clusters)
- **Tuning**: Optuna hyperparameter optimization (30 trials per model)
- **CV Experiment**: ResNet-50 image embeddings tested but did not improve results

## Quick Start

```bash
# Clone
git clone https://github.com/raimbekovm/real-estate-valuation.git
cd real-estate-valuation
pip install -r requirements.txt

# Scrape data
python -m src.scrapers.house_kg --city bishkek --max-pages 100

# Train model
jupyter notebook notebooks/bishkek_v3_best.ipynb
```

## Project Structure

```
├── src/
│   ├── scrapers/      # Web scrapers (house.kg, krisha.kz)
│   ├── features/      # Feature engineering
│   ├── models/        # ML models
│   └── database/      # SQLite management
├── data/
│   ├── databases/     # SQLite DBs (bishkek.db, astana.db)
│   ├── processed/     # Clean CSV datasets
│   └── geo/           # GeoJSON data
├── notebooks/
│   ├── bishkek_v3_best.ipynb      # Best model (MAE $121.71)
│   ├── bishkek_cv_model.ipynb     # CV experiment
│   └── astana_model_training.ipynb
├── docs/              # Documentation
└── scripts/           # CLI utilities
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System overview
- [Database](docs/DATABASE.md) - Schema reference
- [Scraping](docs/SCRAPING.md) - Data collection
- [Features](docs/FEATURES.md) - Feature engineering
- [Models](docs/MODELS.md) - Training guide & results

## Datasets

**Kaggle:**
- [bishkek-real-estate-2025](https://www.kaggle.com/datasets/muraraimbekov/bishkek-real-estate-2025) - Tabular data
- [astana-real-estate-2025](https://www.kaggle.com/datasets/muraraimbekov/astana-real-estate-2025) - Tabular data
- [bishkek-real-estate-images](https://www.kaggle.com/datasets/muraraimbekov/bishkek-real-estate-images) - 64K photos (9.6 GB)

**HuggingFace:**
- [raimbekovm/bishkek-real-estate](https://huggingface.co/datasets/raimbekovm/bishkek-real-estate)
- [raimbekovm/astana-real-estate](https://huggingface.co/datasets/raimbekovm/astana-real-estate)

## License

MIT

## Author

Murat Raimbekov
