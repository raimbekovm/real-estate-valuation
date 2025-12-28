"""
Feature Engineering для данных недвижимости
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Tuple


# Координаты центров городов
CITY_CENTERS = {
    'bishkek': (42.8746, 74.5698),  # Бишкек, Ала-Тоо площадь
    'almaty': (43.2567, 76.9286),   # Алматы, центр
}


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Расстояние между двумя точками на сфере (в км)

    Args:
        lat1, lon1: координаты первой точки
        lat2, lon2: координаты второй точки

    Returns:
        Расстояние в километрах
    """
    R = 6371  # Радиус Земли в км

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def add_price_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет/пересчитывает цену за м²

    Args:
        df: DataFrame с колонками price_usd и area

    Returns:
        DataFrame с колонкой price_per_m2
    """
    df = df.copy()

    # Пересчитываем для всех записей где есть цена и площадь
    mask = (df['price_usd'].notna()) & (df['area'].notna()) & (df['area'] > 0)
    df.loc[mask, 'price_per_m2'] = df.loc[mask, 'price_usd'] / df.loc[mask, 'area']

    return df


def add_building_age(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    """
    Добавляет возраст здания

    Args:
        df: DataFrame с колонкой year_built
        current_year: текущий год

    Returns:
        DataFrame с колонкой building_age
    """
    df = df.copy()

    df['building_age'] = current_year - df['year_built']

    # Отрицательный возраст (новостройки) = 0
    df.loc[df['building_age'] < 0, 'building_age'] = 0

    # Категория возраста
    df['building_age_category'] = pd.cut(
        df['building_age'],
        bins=[-1, 0, 5, 15, 30, 50, 100],
        labels=['новостройка', 'новый', 'современный', 'советский', 'старый', 'очень_старый']
    )

    return df


def add_distance_to_center(
    df: pd.DataFrame,
    city: str = 'bishkek',
    center_coords: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Добавляет расстояние до центра города

    Args:
        df: DataFrame с колонками latitude и longitude
        city: название города ('bishkek' или 'almaty')
        center_coords: кастомные координаты центра (lat, lon)

    Returns:
        DataFrame с колонкой distance_to_center
    """
    df = df.copy()

    if center_coords is None:
        center_coords = CITY_CENTERS.get(city.lower(), CITY_CENTERS['bishkek'])

    center_lat, center_lon = center_coords

    def calc_distance(row):
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            return np.nan
        return haversine_distance(
            row['latitude'], row['longitude'],
            center_lat, center_lon
        )

    df['distance_to_center'] = df.apply(calc_distance, axis=1)

    # Категория расстояния
    df['distance_category'] = pd.cut(
        df['distance_to_center'],
        bins=[-1, 2, 5, 10, 20, 100],
        labels=['центр', 'близко', 'средне', 'далеко', 'пригород']
    )

    return df


def add_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признаки связанные с этажом

    Args:
        df: DataFrame с колонками floor и total_floors

    Returns:
        DataFrame с дополнительными признаками
    """
    df = df.copy()

    # Относительный этаж (0 = первый, 1 = последний)
    df['floor_ratio'] = df['floor'] / df['total_floors']

    # Первый этаж (обычно дешевле)
    df['is_first_floor'] = (df['floor'] == 1).astype(int)

    # Последний этаж (может быть дешевле из-за крыши)
    df['is_last_floor'] = (df['floor'] == df['total_floors']).astype(int)

    # Средние этажи (обычно дороже)
    df['is_middle_floor'] = (
        (df['floor'] > 1) & (df['floor'] < df['total_floors'])
    ).astype(int)

    # Высотность здания
    df['building_height_category'] = pd.cut(
        df['total_floors'],
        bins=[0, 5, 9, 16, 100],
        labels=['малоэтажный', 'среднеэтажный', 'высотный', 'небоскреб']
    )

    return df


def add_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признаки связанные с площадью

    Args:
        df: DataFrame с колонками area, living_area, kitchen_area, rooms

    Returns:
        DataFrame с дополнительными признаками
    """
    df = df.copy()

    # Площадь на комнату
    if 'rooms' in df.columns:
        df['area_per_room'] = df['area'] / df['rooms'].replace(0, np.nan)

    # Доля жилой площади
    if 'living_area' in df.columns:
        df['living_area_ratio'] = df['living_area'] / df['area']

    # Доля кухни
    if 'kitchen_area' in df.columns:
        df['kitchen_area_ratio'] = df['kitchen_area'] / df['area']

    # Категория размера
    df['size_category'] = pd.cut(
        df['area'],
        bins=[0, 35, 50, 70, 100, 150, 500],
        labels=['студия', 'маленькая', 'средняя', 'большая', 'очень_большая', 'элитная']
    )

    return df


def add_all_features(
    df: pd.DataFrame,
    city: str = 'bishkek',
    current_year: int = 2024
) -> pd.DataFrame:
    """
    Добавляет все признаки

    Args:
        df: исходный DataFrame
        city: город для расчета расстояния до центра
        current_year: текущий год

    Returns:
        DataFrame со всеми новыми признаками
    """
    df = add_price_per_m2(df)
    df = add_building_age(df, current_year)
    df = add_distance_to_center(df, city)
    df = add_floor_features(df)
    df = add_area_features(df)

    return df


if __name__ == '__main__':
    # Тест на примере данных
    import sys
    from pathlib import Path

    # Поиск последнего файла с данными
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    files = list(data_dir.glob('house_kg_intermediate_*.csv'))

    if not files:
        print("Нет файлов с данными для теста")
        sys.exit(1)

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Тестирование на: {latest_file.name}")

    df = pd.read_csv(latest_file)
    print(f"Исходные колонки: {len(df.columns)}")

    df = add_all_features(df, city='bishkek')
    print(f"После feature engineering: {len(df.columns)}")

    new_cols = [
        'price_per_m2', 'building_age', 'building_age_category',
        'distance_to_center', 'distance_category',
        'floor_ratio', 'is_first_floor', 'is_last_floor',
        'area_per_room', 'size_category'
    ]

    print("\nНовые признаки:")
    for col in new_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")

    print("\nПример данных:")
    print(df[['price_usd', 'area', 'price_per_m2', 'building_age', 'distance_to_center']].head(10))
