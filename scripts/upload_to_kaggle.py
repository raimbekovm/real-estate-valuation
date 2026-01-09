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


def get_free_space(path: Path) -> int:
    """Получить свободное место на диске в байтах"""
    import shutil as sh
    return sh.disk_usage(path).free


def get_dir_size(path: Path) -> int:
    """Получить размер директории в байтах (быстрый подсчёт)"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(Path(entry.path))
    except PermissionError:
        pass
    return total


def prepare_dataset(csv_path: Path, images_dir: Path, output_dir: Path, use_symlinks: bool = True):
    """
    Подготовка датасета для загрузки на Kaggle.

    Args:
        csv_path: Путь к CSV с данными
        images_dir: Папка с фотографиями (структура: {listing_id}/01.jpg)
        output_dir: Папка для подготовленного датасета
        use_symlinks: Использовать symlinks вместо копирования (экономит ~9GB)
    """
    print(f"Подготовка датасета...")
    print(f"  CSV: {csv_path}")
    print(f"  Фото: {images_dir}")
    print(f"  Выход: {output_dir}")
    print(f"  Режим: {'symlinks' if use_symlinks else 'копирование'}")

    # Проверяем свободное место
    free_space = get_free_space(output_dir.parent if output_dir.parent.exists() else Path.home())
    print(f"  Свободно на диске: {free_space / 1024 / 1024 / 1024:.1f} GB")

    if not use_symlinks:
        images_size = get_dir_size(images_dir) if images_dir.exists() else 0
        required_space = images_size * 1.1  # +10% запас
        if free_space < required_space:
            print(f"  ВНИМАНИЕ: Недостаточно места для копирования!")
            print(f"  Требуется: {required_space / 1024 / 1024 / 1024:.1f} GB")
            print(f"  Переключаюсь на режим symlinks...")
            use_symlinks = True

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

    # Обрабатываем фотографии
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    linked_listings = 0
    linked_photos = 0

    if images_dir.exists():
        listing_ids = df['listing_id'].dropna().unique() if 'listing_id' in df.columns else []

        for listing_id in tqdm(listing_ids, desc="Подготовка фото"):
            src_dir = images_dir / str(listing_id)
            if src_dir.exists() and src_dir.is_dir():
                dst_dir = output_images / str(listing_id)

                if use_symlinks:
                    # Создаём symlink на папку (не на отдельные файлы)
                    os.symlink(src_dir.resolve(), dst_dir)
                    linked_listings += 1
                    linked_photos += len(list(src_dir.glob("*.jpg")))
                else:
                    # Копируем (старый способ)
                    shutil.copytree(src_dir, dst_dir)
                    linked_listings += 1
                    linked_photos += len(list(dst_dir.glob("*.jpg")))

        action = "Слинковано" if use_symlinks else "Скопировано"
        print(f"  {action}: {linked_listings} папок, {linked_photos} фото")
    else:
        print(f"  Папка с фото не найдена: {images_dir}")

    # Создаём metadata
    create_dataset_metadata(output_dir, KAGGLE_TITLE, KAGGLE_DATASET)

    # Статистика (для symlinks показываем реальный размер данных)
    if use_symlinks:
        csv_size = output_csv.stat().st_size
        images_size = get_dir_size(images_dir) if images_dir.exists() else 0
        total_size = csv_size + images_size
        print(f"\nГотово! Размер датасета: {total_size / 1024 / 1024 / 1024:.2f} GB (symlinks → реальные данные)")
    else:
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        print(f"\nГотово! Размер датасета: {total_size / 1024 / 1024:.1f} MB")

    return output_dir


def create_tar_with_symlinks(dataset_dir: Path, output_tar: Path):
    """
    Создание tar архива с разыменованием symlinks.
    Использует tar -h (--dereference) для следования за symlinks.
    """
    import tarfile

    print(f"  Создание tar архива (следуя за symlinks)...")
    print(f"  Это может занять несколько минут...")

    # Подсчитаем общий размер для прогресса
    total_files = sum(1 for _ in dataset_dir.rglob("*") if _.is_file() or _.is_symlink())

    with tarfile.open(output_tar, "w") as tar:
        processed = 0
        for item in dataset_dir.rglob("*"):
            if item.is_file() or item.is_symlink():
                # Относительный путь внутри архива
                arcname = item.relative_to(dataset_dir)

                # Если это symlink - добавляем реальный файл
                if item.is_symlink():
                    real_path = item.resolve()
                    if real_path.exists():
                        if real_path.is_dir():
                            # Symlink на директорию - добавляем все файлы из неё
                            for sub_item in real_path.rglob("*"):
                                if sub_item.is_file():
                                    sub_arcname = item.relative_to(dataset_dir) / sub_item.relative_to(real_path)
                                    tar.add(sub_item, arcname=sub_arcname)
                                    processed += 1
                        else:
                            tar.add(real_path, arcname=arcname)
                            processed += 1
                else:
                    tar.add(item, arcname=arcname)
                    processed += 1

                # Прогресс каждые 1000 файлов
                if processed % 1000 == 0:
                    print(f"    Обработано: {processed} файлов...")

    tar_size = output_tar.stat().st_size / 1024 / 1024 / 1024
    print(f"  Создан: {output_tar.name} ({tar_size:.2f} GB)")
    return output_tar


def upload_to_kaggle(dataset_dir: Path, message: str = None):
    """Загрузка датасета на Kaggle (поддерживает symlinks)"""
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

    # Проверяем есть ли symlinks
    has_symlinks = any(p.is_symlink() for p in dataset_dir.rglob("*"))

    if has_symlinks:
        print("  Обнаружены symlinks, создаю tar архив вручную...")

        # Проверяем свободное место
        free_space = get_free_space(dataset_dir)
        images_dir = dataset_dir / "images"

        # Оцениваем размер реальных данных
        estimated_size = 0
        for item in images_dir.rglob("*"):
            if item.is_symlink():
                real_path = item.resolve()
                if real_path.exists() and real_path.is_dir():
                    estimated_size += get_dir_size(real_path)
            elif item.is_file():
                estimated_size += item.stat().st_size

        print(f"  Свободно на диске: {free_space / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Оценочный размер данных: {estimated_size / 1024 / 1024 / 1024:.1f} GB")

        if free_space < estimated_size * 1.1:
            print(f"\n  ОШИБКА: Недостаточно места для создания архива!")
            print(f"  Требуется минимум {estimated_size * 1.1 / 1024 / 1024 / 1024:.1f} GB")
            print(f"  Освободите место или используйте --no-symlinks с достаточным местом")
            return False

        # Создаём tar архив
        tar_path = dataset_dir / "dataset.tar.gz"
        try:
            create_tar_with_symlinks(dataset_dir, tar_path)
        except Exception as e:
            print(f"  Ошибка создания архива: {e}")
            return False

        # Удаляем symlinks и оставляем только tar + metadata
        print("  Подготовка к загрузке...")
        images_dir = dataset_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)

        # Загружаем
        cmd = [
            "kaggle", "datasets", "version",
            "-p", str(dataset_dir),
            "-m", message,
        ]
    else:
        # Нет symlinks - используем стандартный подход
        free_space = get_free_space(dataset_dir)
        print(f"  Свободно на диске: {free_space / 1024 / 1024 / 1024:.1f} GB")

        cmd = [
            "kaggle", "datasets", "version",
            "-p", str(dataset_dir),
            "-m", message,
            "--dir-mode", "zip"
        ]

    print(f"  Выполняю: {' '.join(cmd)}")
    print("  Это может занять несколько минут...")

    try:
        # Запускаем с выводом в реальном времени
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Выводим прогресс
        for line in process.stdout:
            print(f"  {line.rstrip()}")

        process.wait()

        if process.returncode == 0:
            print("\nУспешно загружено!")
            print(f"URL: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
            return True
        else:
            print(f"\nОшибка загрузки (код {process.returncode})")
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

    # Подготовка и загрузка (symlinks для экономии места)
    prepare_dataset(Path(csv_path), images_dir, output_dir, use_symlinks=True)

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
    parser.add_argument("--no-symlinks", action="store_true",
                        help="Копировать файлы вместо symlinks (требует много места)")

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

        # Подготовка (symlinks по умолчанию)
        use_symlinks = not args.no_symlinks
        prepare_dataset(csv_path, images_dir, output_dir, use_symlinks=use_symlinks)

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
