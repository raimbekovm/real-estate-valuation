#!/usr/bin/env python3
"""
Скрипт обогащения существующих данных фотографиями и описаниями.

Быстрее полного парсинга - использует существующий CSV,
только добавляет фото и описания.

Использование:
    python scripts/enrich_listings.py --csv data/raw/house_kg_bishkek_latest.csv
    python scripts/enrich_listings.py --csv data/raw/listings.csv --workers 3
"""

import argparse
import json
import os
import re
import sys
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from fake_useragent import UserAgent

# Конфигурация
BASE_URL = "https://www.house.kg"
CDN_URL = "https://cdn.house.kg"


class ListingEnricher:
    """Обогащение объявлений фотографиями и описаниями"""

    def __init__(self, images_dir: Path, delay_range=(1.0, 2.5)):
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = requests.Session()
        self._lock = threading.Lock()

    def _get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        }

    def _parse_photos(self, soup):
        """Извлечение URL фотографий"""
        photo_urls = []

        def is_apartment_photo(url):
            exclude = ['/building-images/', '/banners/', '/logos/', '/avatars/']
            return all(p not in url for p in exclude)

        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'cdn.house.kg' in href and '/images/' in href and is_apartment_photo(href):
                if href not in photo_urls:
                    photo_urls.append(href)

        for img in soup.find_all('img', src=True):
            src = img['src']
            if 'cdn.house.kg' in src and '/images/' in src and is_apartment_photo(src):
                full_src = re.sub(r'_\d+x\d+\.', '_1200x900.', src)
                if full_src not in photo_urls:
                    photo_urls.append(full_src)

        # Исключаем последнюю фотку (реклама агентства)
        if len(photo_urls) > 3:
            photo_urls = photo_urls[:-1]

        return photo_urls

    def _parse_description(self, soup):
        """Извлечение описания"""
        description = None

        for section in soup.find_all(['div', 'section']):
            header = section.find(['h2', 'h3', 'h4', 'strong'])
            if header and 'описание' in header.get_text().lower():
                text_parts = []
                for sibling in header.find_next_siblings():
                    if sibling.name in ['h2', 'h3', 'h4', 'div']:
                        if sibling.find(['h2', 'h3', 'h4', 'strong']):
                            break
                    text = sibling.get_text(strip=True)
                    if text and len(text) > 10:
                        text_parts.append(text)
                if text_parts:
                    description = ' '.join(text_parts)
                    break

        if not description:
            for cls_pattern in ['description', 'text-content', 'details-text']:
                elem = soup.find(class_=re.compile(cls_pattern, re.I))
                if elem:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 50:
                        description = text
                        break

        if description:
            description = ' '.join(description.split())
            if len(description) > 5000:
                description = description[:5000] + '...'

        return description

    def _download_photos(self, listing_id: str, photo_urls: list) -> int:
        """Скачивание фото с CDN"""
        if not photo_urls:
            return 0

        listing_dir = self.images_dir / listing_id
        listing_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for i, url in enumerate(photo_urls):
            try:
                filename = f"{i+1:02d}.jpg"
                filepath = listing_dir / filename

                if filepath.exists():
                    downloaded += 1
                    continue

                response = requests.get(
                    url,
                    headers={'User-Agent': self.ua.random},
                    timeout=15
                )
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded += 1

                time.sleep(random.uniform(0.2, 0.5))

            except Exception as e:
                pass  # Молча пропускаем ошибки загрузки фото

        return downloaded

    def enrich_listing(self, listing_id: str, url: str = None) -> dict:
        """
        Обогащение одного объявления.
        Возвращает dict с description, photo_count, photos_downloaded
        """
        if not url:
            url = f"{BASE_URL}/details/{listing_id}"

        result = {
            'listing_id': listing_id,
            'description': None,
            'photo_count': 0,
            'photos_downloaded': 0,
            'error': None
        }

        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')

            # Описание
            result['description'] = self._parse_description(soup)

            # Фото
            photo_urls = self._parse_photos(soup)
            result['photo_count'] = len(photo_urls)

            # Скачиваем фото
            if photo_urls:
                result['photos_downloaded'] = self._download_photos(listing_id, photo_urls)

            time.sleep(random.uniform(*self.delay_range))

        except Exception as e:
            result['error'] = str(e)

        return result


def enrich_csv(csv_path: Path, images_dir: Path, output_path: Path = None,
               workers: int = 1, save_every: int = 100):
    """
    Обогащение CSV файла фотографиями и описаниями.
    """
    print(f"=== Обогащение данных ===")
    print(f"CSV: {csv_path}")
    print(f"Фото: {images_dir}")
    print(f"Воркеров: {workers}")

    # Загружаем данные
    df = pd.read_csv(csv_path)
    print(f"Записей: {len(df)}")

    if 'listing_id' not in df.columns:
        print("Ошибка: в CSV нет колонки listing_id")
        return

    # Добавляем колонки если их нет
    if 'description' not in df.columns:
        df['description'] = None
    if 'photo_count' not in df.columns:
        df['photo_count'] = 0
    if 'photos_downloaded' not in df.columns:
        df['photos_downloaded'] = 0

    # Определяем какие записи нужно обогатить
    # (нет описания ИЛИ нет фото)
    needs_enrichment = df[
        (df['description'].isna()) |
        (df['photo_count'] == 0) |
        (df['photo_count'].isna())
    ]

    print(f"Нужно обогатить: {len(needs_enrichment)} записей")

    if len(needs_enrichment) == 0:
        print("Все записи уже обогащены!")
        return df

    # Создаём enricher
    enricher = ListingEnricher(images_dir)

    # Прогресс
    enriched_count = 0
    errors_count = 0

    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_enriched.csv"

    def process_listing(row):
        listing_id = str(row['listing_id'])
        url = row.get('url')
        return enricher.enrich_listing(listing_id, url)

    # Обработка
    if workers == 1:
        # Последовательная обработка
        for idx, row in tqdm(needs_enrichment.iterrows(), total=len(needs_enrichment), desc="Обогащение"):
            result = process_listing(row)

            if result['error']:
                errors_count += 1
            else:
                df.loc[idx, 'description'] = result['description']
                df.loc[idx, 'photo_count'] = result['photo_count']
                df.loc[idx, 'photos_downloaded'] = result['photos_downloaded']
                enriched_count += 1

            # Промежуточное сохранение
            if enriched_count % save_every == 0:
                df.to_csv(output_path, index=False)
                tqdm.write(f"Сохранено: {enriched_count} записей")

    else:
        # Параллельная обработка
        rows_to_process = [(idx, row) for idx, row in needs_enrichment.iterrows()]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, row in rows_to_process:
                future = executor.submit(process_listing, row)
                futures[future] = idx

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Обогащение ({workers} потоков)"):
                idx = futures[future]
                try:
                    result = future.result()
                    if result['error']:
                        errors_count += 1
                    else:
                        with threading.Lock():
                            df.loc[idx, 'description'] = result['description']
                            df.loc[idx, 'photo_count'] = result['photo_count']
                            df.loc[idx, 'photos_downloaded'] = result['photos_downloaded']
                        enriched_count += 1
                except Exception as e:
                    errors_count += 1

                # Промежуточное сохранение
                if enriched_count % save_every == 0 and enriched_count > 0:
                    with threading.Lock():
                        df.to_csv(output_path, index=False)

    # Финальное сохранение
    df.to_csv(output_path, index=False)

    # Статистика
    print(f"\n=== Готово ===")
    print(f"Обогащено: {enriched_count}")
    print(f"Ошибок: {errors_count}")
    print(f"Сохранено: {output_path}")

    # Статистика по фото
    total_photos = df['photos_downloaded'].sum()
    photos_size = sum(f.stat().st_size for f in images_dir.rglob("*.jpg")) if images_dir.exists() else 0
    print(f"Фото: {int(total_photos)} шт, {photos_size / 1024 / 1024:.1f} MB")

    return df


def main():
    parser = argparse.ArgumentParser(description="Обогащение данных фотографиями и описаниями")
    parser.add_argument("--csv", type=str, required=True,
                        help="Путь к CSV файлу")
    parser.add_argument("--images", type=str, default=None,
                        help="Папка для фото (default: data/images/bishkek)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Путь для сохранения (default: {csv}_enriched.csv)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Количество потоков (1-5, default: 1)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Сохранять каждые N записей")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Файл не найден: {csv_path}")
        return

    project_root = Path(__file__).parent.parent

    if args.images:
        images_dir = Path(args.images)
    else:
        images_dir = project_root / "data" / "images" / "bishkek"

    output_path = Path(args.output) if args.output else None

    # Ограничиваем воркеров (чтобы не забанили)
    workers = min(args.workers, 5)

    enrich_csv(csv_path, images_dir, output_path, workers=workers, save_every=args.save_every)


if __name__ == "__main__":
    main()
