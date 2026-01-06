"""
Скрипт миграции данных из CSV в SQLite.

Мигрирует данные Бишкека из:
- data/raw/house_kg_bishkek_*.csv (сырые данные)
- data/processed/bishkek_clean.csv (очищенные данные)

В:
- data/databases/bishkek.db
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging

from src.database.db_manager import RealEstateDB, extract_listing_id_from_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_value(value):
    """Очистка значения для SQLite."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value) if isinstance(value, np.floating) else int(value)
    return value


def migrate_bishkek_raw(db: RealEstateDB, csv_path: str) -> int:
    """
    Миграция сырых данных Бишкека.

    Args:
        db: Экземпляр RealEstateDB
        csv_path: Путь к CSV файлу

    Returns:
        Количество добавленных записей
    """
    logger.info(f"Загрузка данных из {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Загружено {len(df)} записей")

    # Маппинг колонок CSV -> SQLite (только те что есть в схеме)
    column_mapping = {
        'url': 'url',
        'parsed_at': 'parsed_at',
        'rooms': 'rooms',
        'area': 'area',
        'price_usd': 'price_usd',
        'price_per_m2': 'price_per_m2',
        'offer_type': 'offer_type',
        'building_series': 'building_series',
        'house_type': 'house_type',
        'year_built': 'year_built',
        'floor': 'floor',
        'total_floors': 'total_floors',
        'living_area': 'living_area',
        'kitchen_area': 'kitchen_area',
        'heating': 'heating',
        'condition': 'condition',
        'internet': 'internet',
        'bathroom': 'bathroom',
        'balcony': 'balcony',
        'parking': 'parking',
        'furniture': 'furniture',
        'floor_type': 'floor_type',
        'security': 'security',
        'documents': 'documents',
        'address': 'address',
        'district': 'district',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'listing_id': 'listing_id',
        'ceiling_height': 'ceiling_height',
    }

    count = 0
    errors = 0

    for idx, row in df.iterrows():
        try:
            # Подготовка данных
            data = {}

            for csv_col, db_col in column_mapping.items():
                if csv_col in row.index:
                    data[db_col] = clean_value(row[csv_col])

            # Извлекаем listing_id из URL если нет
            if not data.get('listing_id') and data.get('url'):
                data['listing_id'] = extract_listing_id_from_url(data['url'])

            # Пропускаем записи без listing_id
            if not data.get('listing_id'):
                continue

            # Добавляем source
            data['source'] = 'house.kg'

            # Конвертируем числовые поля
            for field in ['rooms', 'floor', 'total_floors', 'year_built', 'price_usd', 'price_per_m2']:
                if data.get(field) is not None:
                    try:
                        data[field] = int(data[field])
                    except (ValueError, TypeError):
                        data[field] = None

            for field in ['area', 'living_area', 'kitchen_area', 'latitude', 'longitude', 'ceiling_height']:
                if data.get(field) is not None:
                    try:
                        data[field] = float(data[field])
                    except (ValueError, TypeError):
                        data[field] = None

            # Добавляем в БД
            db.add_apartment(data)
            count += 1

            if count % 1000 == 0:
                logger.info(f"Обработано {count} записей...")

        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning(f"Ошибка в строке {idx}: {e}")

    logger.info(f"Миграция завершена: {count} записей добавлено, {errors} ошибок")
    return count


def verify_migration(db: RealEstateDB, original_count: int) -> bool:
    """
    Проверка целостности миграции.

    Args:
        db: Экземпляр RealEstateDB
        original_count: Ожидаемое количество записей

    Returns:
        True если проверка пройдена
    """
    stats = db.get_stats()

    logger.info(f"\n{'='*50}")
    logger.info("ПРОВЕРКА МИГРАЦИИ")
    logger.info(f"{'='*50}")
    logger.info(f"Ожидалось записей: {original_count}")
    logger.info(f"В базе данных: {stats['apartments_total']}")

    # Проверяем основные поля
    df = db.export_for_ml(include_jk_features=False)

    checks = {
        'Записи с ценой': df['price_usd'].notna().sum(),
        'Записи с координатами': df['latitude'].notna().sum(),
        'Записи с площадью': df['area'].notna().sum(),
        'Уникальные районы': df['district'].nunique(),
    }

    logger.info("\nПроверки:")
    for name, value in checks.items():
        logger.info(f"  {name}: {value}")

    # Считаем миграцию успешной если > 90% записей
    success = stats['apartments_total'] >= original_count * 0.9
    if success:
        logger.info("\n✅ Миграция успешна!")
    else:
        logger.error("\n❌ Миграция не прошла проверку!")

    return success


def migrate_krisha_raw(db: RealEstateDB, csv_path: str) -> int:
    """
    Миграция данных krisha.kz (Астана/Алматы).

    Args:
        db: Экземпляр RealEstateDB
        csv_path: Путь к CSV файлу

    Returns:
        Количество добавленных записей
    """
    logger.info(f"Загрузка данных из {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Загружено {len(df)} записей")

    # Маппинг колонок krisha.kz CSV -> SQLite
    column_mapping = {
        'url': 'url',
        'parsed_at': 'parsed_at',
        'listing_id': 'listing_id',
        'rooms': 'rooms',
        'area': 'area',
        'price_usd': 'price_usd',
        # price_kzt не в схеме, используем только price_usd
        'price_per_m2_kzt': 'price_per_m2',
        'house_type': 'house_type',
        'year_built': 'year_built',
        'floor': 'floor',
        'total_floors': 'total_floors',
        'living_area': 'living_area',
        'kitchen_area': 'kitchen_area',
        'condition': 'condition',
        'internet': 'internet',
        'bathroom': 'bathroom',
        'balcony': 'balcony',
        'parking': 'parking',
        'furniture': 'furniture',
        'floor_type': 'floor_type',
        'security': 'security',
        'ceiling_height': 'ceiling_height',
        'address': 'address',
        'district': 'district',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'raw_жилой_комплекс': 'residential_complex_name',
    }

    count = 0
    errors = 0

    for idx, row in df.iterrows():
        try:
            data = {}

            for csv_col, db_col in column_mapping.items():
                if csv_col in row.index:
                    data[db_col] = clean_value(row[csv_col])

            # Извлекаем listing_id из URL если нет
            if not data.get('listing_id') and data.get('url'):
                data['listing_id'] = extract_listing_id_from_url(data['url'])

            if not data.get('listing_id'):
                continue

            # Источник
            data['source'] = 'krisha.kz'

            # Конвертируем числовые поля
            for field in ['rooms', 'floor', 'total_floors', 'year_built', 'price_usd', 'price_per_m2']:
                if data.get(field) is not None:
                    try:
                        data[field] = int(data[field])
                    except (ValueError, TypeError):
                        data[field] = None

            for field in ['area', 'living_area', 'kitchen_area', 'latitude', 'longitude', 'ceiling_height']:
                if data.get(field) is not None:
                    try:
                        data[field] = float(data[field])
                    except (ValueError, TypeError):
                        data[field] = None

            # Добавляем в БД
            db.add_apartment(data)
            count += 1

            if count % 2000 == 0:
                logger.info(f"Обработано {count} записей...")

        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning(f"Ошибка в строке {idx}: {e}")

    logger.info(f"Миграция завершена: {count} записей добавлено, {errors} ошибок")
    return count


def main():
    """Основная функция миграции."""
    import argparse

    parser = argparse.ArgumentParser(description='Миграция данных в SQLite')
    parser.add_argument('--city', type=str, default='bishkek',
                        choices=['bishkek', 'astana', 'almaty'],
                        help='Город для миграции')
    parser.add_argument('--source', type=str, default='raw',
                        choices=['raw', 'processed'],
                        help='Источник данных (raw или processed)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Только проверка без записи')

    args = parser.parse_args()

    # Определяем путь к исходному файлу
    data_dir = PROJECT_ROOT / 'data'

    if args.city == 'bishkek':
        if args.source == 'raw':
            raw_files = list((data_dir / 'raw').glob('house_kg_bishkek_2*.csv'))
            if not raw_files:
                logger.error("Не найдены сырые файлы Бишкека")
                return
            csv_path = max(raw_files, key=lambda f: f.stat().st_mtime)
        else:
            csv_path = data_dir / 'processed' / 'bishkek_clean.csv'
    elif args.city == 'astana':
        # Используем последний сырой файл krisha.kz
        raw_files = list((data_dir / 'raw').glob('krisha_kz_astana_2*.csv'))
        # Исключаем intermediate файлы
        raw_files = [f for f in raw_files if 'intermediate' not in f.name]
        if not raw_files:
            logger.error("Не найдены сырые файлы Астаны")
            return
        csv_path = max(raw_files, key=lambda f: f.stat().st_mtime)
    elif args.city == 'almaty':
        raw_files = list((data_dir / 'raw').glob('krisha_kz_almaty_2*.csv'))
        raw_files = [f for f in raw_files if 'intermediate' not in f.name]
        if not raw_files:
            logger.error("Не найдены сырые файлы Алматы")
            return
        csv_path = max(raw_files, key=lambda f: f.stat().st_mtime)
    else:
        logger.error(f"Неизвестный город: {args.city}")
        return

    if not csv_path.exists():
        logger.error(f"Файл не найден: {csv_path}")
        return

    # Считаем записи в исходном файле
    df_original = pd.read_csv(csv_path)
    original_count = len(df_original)
    logger.info(f"Исходный файл: {csv_path}")
    logger.info(f"Записей в исходном файле: {original_count}")

    if args.dry_run:
        logger.info("Режим dry-run: запись не производится")
        return

    # Создаём БД и мигрируем
    logger.info(f"\nСоздание базы данных для {args.city}...")

    with RealEstateDB(args.city) as db:
        # Проверяем, есть ли уже данные
        existing = db.get_apartments_count()
        if existing > 0:
            logger.warning(f"В базе уже есть {existing} записей!")
            response = input("Продолжить миграцию? Существующие записи будут обновлены. (y/n): ")
            if response.lower() != 'y':
                logger.info("Миграция отменена")
                return

        # Миграция - выбираем функцию в зависимости от источника
        if args.city == 'bishkek':
            count = migrate_bishkek_raw(db, str(csv_path))
        else:
            # Астана и Алматы используют krisha.kz
            count = migrate_krisha_raw(db, str(csv_path))

        # Проверка
        verify_migration(db, original_count)

        # Статистика
        db.print_stats()


if __name__ == '__main__':
    main()
