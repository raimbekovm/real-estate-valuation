# Database Schema

## Overview

Project uses SQLite databases stored in `data/databases/{city}.db`.

## Tables

### apartments

Main listing data.

```sql
CREATE TABLE apartments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id TEXT UNIQUE,
    url TEXT,
    source TEXT DEFAULT 'house.kg',

    -- Residential complex link
    residential_complex_id INTEGER REFERENCES residential_complexes(id),
    residential_complex_name TEXT,

    -- Basic info
    rooms INTEGER,
    area REAL,
    living_area REAL,
    kitchen_area REAL,
    floor INTEGER,
    total_floors INTEGER,

    -- Pricing
    price_usd INTEGER,
    price_local INTEGER,
    price_per_m2 INTEGER,

    -- Building
    house_type TEXT,
    year_built INTEGER,
    building_series TEXT,
    ceiling_height REAL,
    condition TEXT,

    -- Amenities
    heating TEXT,
    has_phone INTEGER,
    internet TEXT,
    bathroom TEXT,
    gas TEXT,
    balcony TEXT,
    parking TEXT,
    elevator TEXT,
    furnishing TEXT,

    -- Location
    district TEXT,
    address TEXT,
    latitude REAL,
    longitude REAL,

    -- Media
    photo_urls TEXT,  -- JSON array
    description TEXT,

    -- Metadata
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);
```

### residential_complexes

Жилые комплексы (ЖК).

```sql
CREATE TABLE residential_complexes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE,
    name TEXT,
    url TEXT,

    -- Classification
    class TEXT,  -- economy, comfort, business, elite
    house_type TEXT,
    status TEXT,  -- completed, under_construction

    -- Building specs
    total_floors INTEGER,
    ceiling_height REAL,
    year_built INTEGER,
    completion_date TEXT,

    -- Developer
    developer_id INTEGER REFERENCES developers(id),
    developer_name TEXT,

    -- Location
    address TEXT,
    district TEXT,
    latitude REAL,
    longitude REAL,

    -- Pricing
    price_from_per_m2 INTEGER,

    -- Amenities
    has_parking INTEGER,
    has_gym INTEGER,
    has_pool INTEGER,
    has_playground INTEGER,
    has_security INTEGER,
    has_concierge INTEGER,

    -- Stats
    rating REAL,
    reviews_count INTEGER,
    apartments_count INTEGER,

    -- Metadata
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);
```

### developers

Developer companies.

```sql
CREATE TABLE developers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE,
    name TEXT,
    url TEXT,
    projects_count INTEGER,
    rating REAL,
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);
```

### parsing_queue

Scraping state tracking.

```sql
CREATE TABLE parsing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    status TEXT DEFAULT 'pending',  -- pending, done, error
    attempts INTEGER DEFAULT 0,
    last_attempt TIMESTAMP,
    error_message TEXT
);
```

## Relationships

```
developers (1) ──── (N) residential_complexes (1) ──── (N) apartments
```

## Usage Examples

### Initialize Database

```python
from src.database.db_manager import RealEstateDB

db = RealEstateDB('bishkek')  # Creates bishkek.db if not exists
```

### Add Apartment

```python
apartment_id = db.add_apartment({
    'listing_id': 'abc123',
    'url': 'https://house.kg/details/abc123',
    'rooms': 3,
    'area': 85.0,
    'floor': 5,
    'total_floors': 9,
    'price_usd': 75000,
    'district': 'Магистраль',
    'latitude': 42.8746,
    'longitude': 74.5698
})
```

### Query with SQL

```python
# Get apartments with JK
results = db.conn.execute("""
    SELECT a.*, rc.name as jk_name, rc.class as jk_class
    FROM apartments a
    LEFT JOIN residential_complexes rc
        ON a.residential_complex_id = rc.id
    WHERE a.price_usd > 50000
""").fetchall()
```

### Export for ML

```python
df = db.export_for_ml('output.csv')
# Returns DataFrame with all apartments + joined JK data
```

### Get Statistics

```python
stats = db.get_stats()
# {
#     'total_apartments': 8821,
#     'apartments_with_jk': 1307,
#     'total_jk': 823,
#     'total_developers': 150
# }
```

## Indexes

Recommended indexes for performance:

```sql
CREATE INDEX idx_apartments_district ON apartments(district);
CREATE INDEX idx_apartments_price ON apartments(price_usd);
CREATE INDEX idx_apartments_rooms ON apartments(rooms);
CREATE INDEX idx_apartments_jk ON apartments(residential_complex_id);
CREATE INDEX idx_rc_slug ON residential_complexes(slug);
```

## Migrations

Currently manual. To add new column:

```python
db.conn.execute("ALTER TABLE apartments ADD COLUMN new_field TEXT")
db.conn.commit()
```
