#!/usr/bin/env python3
"""
Скрипт для загрузки датасета с фотографиями на Kaggle.

Использование:
    python scripts/upload_to_kaggle.py --csv data/raw/house_kg_bishkek_latest.csv
    python scripts/upload_to_kaggle.py --csv data/raw/listings.csv --images data/images/bishkek
    python scripts/upload_to_kaggle.py --scrape --max-pages 10  # парсинг + загрузка

Структура датасета на Kaggle:
    bishkek-real-estate-2025/
    ├── listings.csv
    ├── images/
    │   ├── {listing_id}/
    │   │   ├── 01.jpg
    │   │   ├── 02.jpg
    │   │   └── ...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm


# Конфигурация Kaggle датасета
KAGGLE_DATASET = "muraraimbekov/bishkek-real-estate-2025"
KAGGLE_TITLE = "Bishkek Real Estate 2025"


def create_dataset_metadata(output_dir: Path, title: str, dataset_id: str):
    """Создание metadata файла для Kaggle API"""
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }

    metadata_path = output_dir / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def prepare_dataset(csv_path: Path, images_dir: Path, output_dir: Path):
    """
    Подготовка датасета для загрузки на Kaggle.

    Args:
        csv_path: Путь к CSV с данными
        images_dir: Папка с фотографиями (структура: {listing_id}/01.jpg)
        output_dir: Папка для подготовленного датасета
    """
    print(f"Подготовка датасета...")
    print(f"  CSV: {csv_path}")
    print(f"  Фото: {images_dir}")
    print(f"  Выход: {output_dir}")

    # Очищаем output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Копируем CSV
    df = pd.read_csv(csv_path)
    print(f"  Записей в CSV: {len(df)}")

    # Убираем photo_urls (они займут много места, а пути можно построить по listing_id)
    if 'photo_urls' in df.columns:
        df = df.drop(columns=['photo_urls'])

    # Сохраняем
    output_csv = output_dir / "listings.csv"
    df.to_csv(output_csv, index=False)
    print(f"  Сохранён: {output_csv}")

    # Копируем фотографии
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    if images_dir.exists():
        listing_ids = df['listing_id'].dropna().unique() if 'listing_id' in df.columns else []

        copied_listings = 0
        copied_photos = 0

        for listing_id in tqdm(listing_ids, desc="Копирование фото"):
            src_dir = images_dir / str(listing_id)
            if src_dir.exists() and src_dir.is_dir():
                dst_dir = output_images / str(listing_id)
                shutil.copytree(src_dir, dst_dir)
                copied_listings += 1
                copied_photos += len(list(dst_dir.glob("*.jpg")))

        print(f"  Скопировано: {copied_listings} папок, {copied_photos} фото")
    else:
        print(f"  Папка с фото не найдена: {images_dir}")

    # Создаём metadata
    create_dataset_metadata(output_dir, KAGGLE_TITLE, KAGGLE_DATASET)

    # Статистика
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\nГотово! Размер датасета: {total_size / 1024 / 1024:.1f} MB")

    return output_dir


def upload_to_kaggle(dataset_dir: Path, message: str = None):
    """Загрузка датасета на Kaggle"""
    if message is None:
        message = f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    print(f"\nЗагрузка на Kaggle...")
    print(f"  Датасет: {KAGGLE_DATASET}")
    print(f"  Сообщение: {message}")

    # Проверяем наличие kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ошибка: kaggle CLI не установлен. Установите: pip install kaggle")
        return False

    # Загружаем новую версию датасета
    cmd = [
        "kaggle", "datasets", "version",
        "-p", str(dataset_dir),
        "-m", message,
        "--dir-mode", "zip"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Успешно загружено!")
            print(f"URL: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
            return True
        else:
            print(f"Ошибка загрузки: {result.stderr}")
            return False
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def scrape_and_upload(max_pages: int = None, city: str = 'bishkek'):
    """Парсинг данных и загрузка на Kaggle"""
    from src.scrapers.house_kg import HouseKGScraper

    print(f"=== Парсинг {city.upper()} ===")

    # Папки
    project_root = Path(__file__).parent.parent
    images_dir = project_root / "data" / "images" / city
    output_dir = project_root / "data" / "kaggle_upload"

    # Парсим с загрузкой фото
    scraper = HouseKGScraper(
        city=city,
        download_photos=True,
        photos_dir=images_dir
    )

    print(f"Фото будут сохраняться в: {images_dir}")

    # Запускаем парсинг
    scraper.scrape(max_pages=max_pages, resume=True)
    csv_path = scraper.save()

    if not csv_path:
        print("Ошибка: нет данных для загрузки")
        return False

    # Подготовка и загрузка
    prepare_dataset(Path(csv_path), images_dir, output_dir)

    photo_count = sum(1 for _ in images_dir.rglob("*.jpg")) if images_dir.exists() else 0
    message = f"Update: {len(scraper.data)} listings, {photo_count} photos"

    return upload_to_kaggle(output_dir, message)


def main():
    parser = argparse.ArgumentParser(description="Загрузка датасета на Kaggle")

    # Режимы работы
    parser.add_argument("--scrape", action="store_true",
                        help="Парсинг + загрузка (полный цикл)")
    parser.add_argument("--csv", type=str,
                        help="Путь к готовому CSV файлу")
    parser.add_argument("--images", type=str,
                        help="Путь к папке с фото")

    # Опции
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Максимум страниц для парсинга")
    parser.add_argument("--city", type=str, default="bishkek",
                        help="Город (bishkek, osh, etc.)")
    parser.add_argument("--message", "-m", type=str,
                        help="Сообщение для версии датасета")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Только подготовить, не загружать")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "kaggle_upload"

    if args.scrape:
        # Полный цикл: парсинг + загрузка
        scrape_and_upload(max_pages=args.max_pages, city=args.city)

    elif args.csv:
        # Загрузка из готового CSV
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Файл не найден: {csv_path}")
            return

        # Папка с фото
        if args.images:
            images_dir = Path(args.images)
        else:
            images_dir = project_root / "data" / "images" / args.city

        # Подготовка
        prepare_dataset(csv_path, images_dir, output_dir)

        # Загрузка
        if not args.prepare_only:
            upload_to_kaggle(output_dir, args.message)
        else:
            print(f"\nДатасет подготовлен в: {output_dir}")
            print("Для загрузки выполните:")
            print(f"  kaggle datasets version -p {output_dir} -m 'Update'")

    else:
        parser.print_help()
        print("\nПримеры:")
        print("  # Парсинг 10 страниц + загрузка")
        print("  python scripts/upload_to_kaggle.py --scrape --max-pages 10")
        print("")
        print("  # Загрузка готового CSV с фото")
        print("  python scripts/upload_to_kaggle.py --csv data/raw/listings.csv --images data/images/bishkek")
        print("")
        print("  # Только подготовить (без загрузки)")
        print("  python scripts/upload_to_kaggle.py --csv data/raw/listings.csv --prepare-only")


if __name__ == "__main__":
    main()
