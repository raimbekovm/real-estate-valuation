# Real Estate Valuation

ML models for property price prediction in Central Asia (Kyrgyzstan, Kazakhstan).

## Results

| Market | Records | MAE | R² | Notebook |
|--------|---------|-----|-----|----------|
| Bishkek | 8,673 | $144/m² | 0.66 | [Kaggle](https://www.kaggle.com/code/muraraimbekov/bishkek-real-estate-price-prediction-v3) |
| Astana | 18,293 | 56K₸/m² | 0.83 | [Kaggle](https://www.kaggle.com/code/muraraimbekov/astana-real-estate-price-prediction) |

## Quick Start

```bash
# Install
git clone https://github.com/raimbekovm/real-estate-valuation.git
cd real-estate-valuation
pip install -e .

# Scrape data
python -m src.scrapers.house_kg --city bishkek --max-pages 100

# Train model
jupyter notebook notebooks/bishkek_model_training_v3.ipynb
```

## Project Structure

```
├── src/
│   ├── scrapers/      # Web scrapers (house.kg, krisha.kz)
│   ├── features/      # Feature engineering
│   ├── models/        # ML models
│   └── database/      # SQLite management
├── data/
│   ├── databases/     # SQLite DBs
│   └── images/        # Property photos
├── notebooks/         # Training notebooks
├── docs/              # Documentation
└── scripts/           # CLI utilities
```

## Documentation

See [docs/](docs/) for detailed documentation:

- [Architecture](docs/ARCHITECTURE.md) - System overview
- [Database](docs/DATABASE.md) - Schema reference
- [Scraping](docs/SCRAPING.md) - Data collection
- [Features](docs/FEATURES.md) - Feature engineering
- [Models](docs/MODELS.md) - Training guide

## Datasets

- [Kaggle: bishkek-real-estate-2025](https://www.kaggle.com/datasets/muraraimbekov/bishkek-real-estate-2025)
- [HuggingFace: raimbekovm/bishkek-real-estate](https://huggingface.co/datasets/raimbekovm/bishkek-real-estate)

## License

MIT

## Author

Murat Raimbekov
