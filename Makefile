.PHONY: help install install-dev clean lint format test scrape preprocess train

PYTHON := python3
PIP := pip

help:
	@echo "Real Estate Valuation Project"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make lint         Run linters (flake8, mypy)"
	@echo "  make format       Format code (black, isort)"
	@echo "  make test         Run tests"
	@echo "  make scrape       Run data scraping"
	@echo "  make preprocess   Run data preprocessing"
	@echo "  make train        Train models"
	@echo "  make clean        Clean temporary files"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,notebooks,tracking]"
	pre-commit install

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

scrape:
	$(PYTHON) -m src.scrapers.house_kg

scrape-test:
	$(PYTHON) -c "from src.scrapers.house_kg import HouseKGScraper; s = HouseKGScraper(); s.scrape(max_pages=2); s.save('test_scrape.csv')"

preprocess:
	$(PYTHON) -m src.preprocessing.pipeline

train:
	$(PYTHON) -m src.models.train

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/

# Data pipeline
data-pipeline: scrape preprocess
	@echo "Data pipeline completed"

# Full experiment
experiment: data-pipeline train
	@echo "Experiment completed"
