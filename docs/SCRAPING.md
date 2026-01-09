# Web Scraping Guide

## Scrapers Overview

| Scraper | Source | Cities | Features |
|---------|--------|--------|----------|
| `HouseKGScraper` | house.kg | Bishkek, Osh, Jalal-Abad, Karakol, Tokmok | Photos, resume |
| `HouseKGJKScraper` | house.kg | Bishkek | JK metadata, apartment linking |
| `KrishaKZScraper` | krisha.kz | Astana, Almaty | Proxies, Selenium |

## house.kg Scraper

### Basic Usage

```python
from src.scrapers.house_kg import HouseKGScraper

scraper = HouseKGScraper(city='bishkek')
scraper.scrape(max_pages=100)
scraper.save()
```

### With Photo Download

```python
scraper = HouseKGScraper(
    city='bishkek',
    download_photos=True,
    delay_range=(2, 4)
)
scraper.scrape(max_pages=50)
```

Photos saved to: `data/images/bishkek/{listing_id}/01.jpg, 02.jpg, ...`

### Resume Scraping

```python
# Automatically resumes from last checkpoint
scraper.scrape(resume=True)
```

### CLI Usage

```bash
python -m src.scrapers.house_kg --city bishkek --max-pages 100
python -m src.scrapers.house_kg --city bishkek --download-photos
```

## Residential Complex Scraper

### Scrape JK Metadata

```python
from src.scrapers.house_kg_jk import HouseKGJKScraper

scraper = HouseKGJKScraper(city='bishkek')

# Parse all JK pages
scraper.scrape()

# Link apartments to JK by URL
scraper.link_apartments_to_jk()
```

### Clear Wrong Links

```python
# Remove links to districts (non-real JK)
scraper.clear_wrong_links()
```

### CLI Usage

```bash
python -m src.scrapers.house_kg_jk --city bishkek
python -m src.scrapers.house_kg_jk --city bishkek --link-only
python -m src.scrapers.house_kg_jk --city bishkek --clear-links
```

## krisha.kz Scraper

### Basic Usage

```python
from src.scrapers.krisha_kz import KrishaKZScraper

scraper = KrishaKZScraper(city='astana')
scraper.scrape(max_pages=100)
scraper.save()
```

### With Proxies

```python
scraper = KrishaKZScraper(
    city='astana',
    use_proxy=True,
    proxy_list=['http://proxy1:8080', 'http://proxy2:8080']
)
```

### With Selenium (for JS-heavy pages)

```python
scraper = KrishaKZScraper(
    city='almaty',
    use_selenium=True
)
```

## Anti-Blocking Features

### Delays

```python
# Random delay between requests
scraper = HouseKGScraper(delay_range=(2, 4))  # 2-4 seconds
```

### User-Agent Rotation

Automatic via `fake-useragent` library.

### Retry Logic

```python
# Automatic retry on failure
max_retries = 3
timeout = 30  # seconds
```

### Incremental Saves

Data saved every N listings to prevent data loss:

```
data/raw/house_kg_bishkek_intermediate_20250109_143022.csv
data/raw/house_kg_bishkek_intermediate_20250109_150045.csv
```

## Data Fields Scraped

### Apartments

| Field | Type | Example |
|-------|------|---------|
| url | str | https://house.kg/details/abc123 |
| rooms | int | 3 |
| area | float | 85.5 |
| living_area | float | 55.0 |
| kitchen_area | float | 12.0 |
| floor | int | 5 |
| total_floors | int | 9 |
| price_usd | int | 75000 |
| price_per_m2 | int | 877 |
| year_built | int | 2020 |
| district | str | Магистраль |
| latitude | float | 42.8746 |
| longitude | float | 74.5698 |
| condition | str | евроремонт |
| house_type | str | кирпичный |
| photo_urls | list | ["url1", "url2"] |
| description | str | Продается... |

### Residential Complexes

| Field | Type | Example |
|-------|------|---------|
| name | str | Madison |
| class | str | business |
| developer_name | str | Империал Строй |
| price_from_per_m2 | int | 131400 |
| total_floors | int | 16 |
| has_parking | bool | True |
| has_gym | bool | True |
| latitude | float | 42.8812 |
| longitude | float | 74.5834 |

## Troubleshooting

### Rate Limiting

```
Error: 429 Too Many Requests
```

Solution: Increase delay

```python
scraper = HouseKGScraper(delay_range=(5, 10))
```

### Blocked IP

```
Error: 403 Forbidden
```

Solutions:
1. Use VPN
2. Use proxy (krisha.kz scraper)
3. Wait 24 hours

### Timeout Errors

```python
scraper = HouseKGScraper(timeout=60)  # Increase timeout
```

### Resume After Crash

```python
# Scraper auto-saves progress
scraper.scrape(resume=True)  # Continues from last save
```
