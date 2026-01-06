"""
Менеджер базы данных для проекта оценки недвижимости.

Архитектура:
- Один SQLite файл на город (bishkek.db, astana.db, almaty.db)
- Таблицы: apartments, residential_complexes, developers, parsing_queue
- Связи: apartments.residential_complex_id -> residential_complexes.id
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEstateDB:
    """
    Класс для работы с базой данных недвижимости.

    Использование:
        db = RealEstateDB('bishkek')
        db.add_apartment({...})
        df = db.export_for_ml()
    """

    # Путь к директории с базами данных
    DB_DIR = Path(__file__).parent.parent.parent / 'data' / 'databases'

    def __init__(self, city: str):
        """
        Инициализация базы данных для города.

        Args:
            city: Название города (bishkek, astana, almaty)
        """
        self.city = city.lower()
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = self.DB_DIR / f'{self.city}.db'
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Возвращать dict-like объекты
        self._init_schema()
        logger.info(f"База данных инициализирована: {self.db_path}")

    def _init_schema(self):
        """Создание схемы базы данных."""
        cursor = self.conn.cursor()

        # Таблица застройщиков
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS developers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE,
                name TEXT NOT NULL,
                url TEXT,
                projects_count INTEGER,
                rating REAL,
                parsed_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')

        # Таблица жилых комплексов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS residential_complexes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Идентификация
                slug TEXT UNIQUE,
                name TEXT NOT NULL,
                url TEXT,

                -- Основные характеристики
                class TEXT,
                house_type TEXT,
                total_floors INTEGER,
                ceiling_height REAL,

                -- Статус и сроки
                status TEXT,
                completion_date TEXT,
                year_built INTEGER,

                -- Застройщик
                developer_id INTEGER,
                developer_name TEXT,

                -- Локация
                address TEXT,
                district TEXT,
                latitude REAL,
                longitude REAL,

                -- Цены
                price_from_per_m2 INTEGER,

                -- Инфраструктура
                has_parking INTEGER DEFAULT 0,
                has_gym INTEGER DEFAULT 0,
                has_pool INTEGER DEFAULT 0,
                has_playground INTEGER DEFAULT 0,
                has_security INTEGER DEFAULT 0,
                has_concierge INTEGER DEFAULT 0,

                -- Рейтинг
                rating REAL,
                reviews_count INTEGER DEFAULT 0,

                -- Метаданные
                apartments_count INTEGER DEFAULT 0,
                parsed_at TIMESTAMP,
                updated_at TIMESTAMP,

                FOREIGN KEY (developer_id) REFERENCES developers(id)
            )
        ''')

        # Таблица квартир (объявлений)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS apartments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Идентификация
                listing_id TEXT UNIQUE,
                url TEXT,
                source TEXT,

                -- Связь с ЖК
                residential_complex_id INTEGER,
                residential_complex_name TEXT,

                -- Основные характеристики
                rooms INTEGER,
                area REAL,
                living_area REAL,
                kitchen_area REAL,
                floor INTEGER,
                total_floors INTEGER,

                -- Цена
                price_usd INTEGER,
                price_local INTEGER,
                price_per_m2 INTEGER,

                -- Дом
                house_type TEXT,
                year_built INTEGER,
                building_series TEXT,
                ceiling_height REAL,

                -- Состояние
                condition TEXT,

                -- Удобства
                bathroom TEXT,
                balcony TEXT,
                parking TEXT,
                furniture TEXT,
                floor_type TEXT,
                heating TEXT,
                security TEXT,
                internet TEXT,

                -- Документы
                documents TEXT,
                offer_type TEXT,

                -- Локация
                address TEXT,
                district TEXT,
                latitude REAL,
                longitude REAL,

                -- Метаданные
                parsed_at TIMESTAMP,
                updated_at TIMESTAMP,
                is_active INTEGER DEFAULT 1,

                FOREIGN KEY (residential_complex_id) REFERENCES residential_complexes(id)
            )
        ''')

        # Таблица очереди парсинга
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parsing_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                entity_type TEXT,
                status TEXT DEFAULT 'pending',
                worker_id INTEGER,
                attempts INTEGER DEFAULT 0,
                last_error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
        ''')

        # Индексы
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_apartments_jk ON apartments(residential_complex_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_apartments_district ON apartments(district)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_apartments_price ON apartments(price_per_m2)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_apartments_listing ON apartments(listing_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_apartments_active ON apartments(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_jk_class ON residential_complexes(class)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_jk_slug ON residential_complexes(slug)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_queue_status ON parsing_queue(status, entity_type)')

        self.conn.commit()

    def close(self):
        """Закрытие соединения с БД."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==========================================
    # МЕТОДЫ ДЛЯ КВАРТИР
    # ==========================================

    def add_apartment(self, data: Dict[str, Any]) -> int:
        """
        Добавление или обновление квартиры.

        Args:
            data: Словарь с данными квартиры

        Returns:
            ID добавленной/обновлённой записи
        """
        cursor = self.conn.cursor()

        # Проверяем существование по listing_id
        listing_id = data.get('listing_id')
        if listing_id:
            cursor.execute('SELECT id FROM apartments WHERE listing_id = ?', (listing_id,))
            existing = cursor.fetchone()
            if existing:
                # Обновляем существующую запись
                data['updated_at'] = datetime.now().isoformat()
                set_clause = ', '.join([f'{k} = ?' for k in data.keys()])
                values = list(data.values()) + [existing['id']]
                cursor.execute(f'UPDATE apartments SET {set_clause} WHERE id = ?', values)
                self.conn.commit()
                return existing['id']

        # Добавляем новую запись
        data['parsed_at'] = data.get('parsed_at', datetime.now().isoformat())
        data['source'] = data.get('source', 'house.kg' if self.city == 'bishkek' else 'krisha.kz')

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        cursor.execute(f'INSERT INTO apartments ({columns}) VALUES ({placeholders})', list(data.values()))
        self.conn.commit()

        return cursor.lastrowid

    def add_apartments_bulk(self, apartments: List[Dict[str, Any]]) -> int:
        """
        Массовое добавление квартир.

        Args:
            apartments: Список словарей с данными

        Returns:
            Количество добавленных записей
        """
        count = 0
        for apt in apartments:
            try:
                self.add_apartment(apt)
                count += 1
            except Exception as e:
                logger.warning(f"Ошибка добавления квартиры {apt.get('listing_id')}: {e}")
        return count

    def get_apartment(self, listing_id: str) -> Optional[Dict]:
        """Получение квартиры по listing_id."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM apartments WHERE listing_id = ?', (listing_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_apartments_without_jk(self) -> List[Dict]:
        """Получение квартир без привязки к ЖК."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM apartments
            WHERE residential_complex_id IS NULL AND is_active = 1
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def link_apartment_to_jk(self, listing_id: str, jk_id: int, jk_name: str = None):
        """Привязка квартиры к жилому комплексу."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE apartments
            SET residential_complex_id = ?,
                residential_complex_name = ?,
                updated_at = ?
            WHERE listing_id = ?
        ''', (jk_id, jk_name, datetime.now().isoformat(), listing_id))
        self.conn.commit()

    def get_apartments_count(self) -> int:
        """Количество квартир в БД."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM apartments')
        return cursor.fetchone()[0]

    # ==========================================
    # МЕТОДЫ ДЛЯ ЖИЛЫХ КОМПЛЕКСОВ
    # ==========================================

    def add_residential_complex(self, data: Dict[str, Any]) -> int:
        """
        Добавление или обновление ЖК.

        Args:
            data: Словарь с данными ЖК

        Returns:
            ID добавленного/обновлённого ЖК
        """
        cursor = self.conn.cursor()

        # Проверяем существование по slug
        slug = data.get('slug')
        if slug:
            cursor.execute('SELECT id FROM residential_complexes WHERE slug = ?', (slug,))
            existing = cursor.fetchone()
            if existing:
                # Обновляем существующую запись
                data['updated_at'] = datetime.now().isoformat()
                set_clause = ', '.join([f'{k} = ?' for k in data.keys()])
                values = list(data.values()) + [existing['id']]
                cursor.execute(f'UPDATE residential_complexes SET {set_clause} WHERE id = ?', values)
                self.conn.commit()
                return existing['id']

        # Добавляем новую запись
        data['parsed_at'] = data.get('parsed_at', datetime.now().isoformat())

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        cursor.execute(f'INSERT INTO residential_complexes ({columns}) VALUES ({placeholders})', list(data.values()))
        self.conn.commit()

        return cursor.lastrowid

    def get_residential_complex(self, slug: str) -> Optional[Dict]:
        """Получение ЖК по slug."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM residential_complexes WHERE slug = ?', (slug,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_residential_complex_by_id(self, jk_id: int) -> Optional[Dict]:
        """Получение ЖК по ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM residential_complexes WHERE id = ?', (jk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_residential_complexes(self) -> List[Dict]:
        """Получение всех ЖК."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM residential_complexes ORDER BY name')
        return [dict(row) for row in cursor.fetchall()]

    def get_jk_count(self) -> int:
        """Количество ЖК в БД."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM residential_complexes')
        return cursor.fetchone()[0]

    # ==========================================
    # МЕТОДЫ ДЛЯ ОЧЕРЕДИ ПАРСИНГА
    # ==========================================

    def add_to_queue(self, url: str, entity_type: str) -> bool:
        """
        Добавление URL в очередь парсинга.

        Args:
            url: URL для парсинга
            entity_type: Тип сущности ('apartment', 'residential_complex')

        Returns:
            True если добавлено, False если уже существует
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO parsing_queue (url, entity_type, status)
                VALUES (?, ?, 'pending')
            ''', (url, entity_type))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def add_urls_to_queue(self, urls: List[str], entity_type: str) -> int:
        """Массовое добавление URL в очередь."""
        count = 0
        for url in urls:
            if self.add_to_queue(url, entity_type):
                count += 1
        return count

    def claim_urls(self, worker_id: int, entity_type: str, batch_size: int = 50) -> List[str]:
        """
        Захват батча URL для обработки воркером.

        Args:
            worker_id: ID воркера
            entity_type: Тип сущности
            batch_size: Размер батча

        Returns:
            Список URL
        """
        cursor = self.conn.cursor()

        # Захватываем URL
        cursor.execute('''
            UPDATE parsing_queue
            SET status = 'processing', worker_id = ?
            WHERE id IN (
                SELECT id FROM parsing_queue
                WHERE status = 'pending' AND entity_type = ?
                LIMIT ?
            )
        ''', (worker_id, entity_type, batch_size))
        self.conn.commit()

        # Получаем захваченные URL
        cursor.execute('''
            SELECT url FROM parsing_queue
            WHERE worker_id = ? AND status = 'processing' AND entity_type = ?
        ''', (worker_id, entity_type))

        return [row['url'] for row in cursor.fetchall()]

    def mark_url_done(self, url: str):
        """Пометить URL как обработанный."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE parsing_queue
            SET status = 'done', processed_at = ?
            WHERE url = ?
        ''', (datetime.now().isoformat(), url))
        self.conn.commit()

    def mark_url_error(self, url: str, error: str):
        """Пометить URL как ошибочный."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE parsing_queue
            SET status = 'error', last_error = ?, attempts = attempts + 1
            WHERE url = ?
        ''', (error, url))
        self.conn.commit()

    def get_queue_stats(self, entity_type: str = None) -> Dict[str, int]:
        """Статистика очереди."""
        cursor = self.conn.cursor()

        where_clause = f"WHERE entity_type = '{entity_type}'" if entity_type else ""

        cursor.execute(f'''
            SELECT status, COUNT(*) as count
            FROM parsing_queue {where_clause}
            GROUP BY status
        ''')

        stats = {'pending': 0, 'processing': 0, 'done': 0, 'error': 0}
        for row in cursor.fetchall():
            stats[row['status']] = row['count']

        return stats

    # ==========================================
    # ЭКСПОРТ ДЛЯ ML
    # ==========================================

    def export_for_ml(self, include_jk_features: bool = True) -> pd.DataFrame:
        """
        Экспорт данных для машинного обучения.

        Args:
            include_jk_features: Включать ли фичи из ЖК

        Returns:
            DataFrame с данными для ML
        """
        if include_jk_features:
            query = '''
                SELECT
                    a.*,
                    jk.name as jk_name,
                    jk.class as jk_class,
                    jk.house_type as jk_house_type,
                    jk.total_floors as jk_total_floors,
                    jk.ceiling_height as jk_ceiling_height,
                    jk.status as jk_status,
                    jk.year_built as jk_year_built,
                    jk.has_parking as jk_has_parking,
                    jk.has_gym as jk_has_gym,
                    jk.has_pool as jk_has_pool,
                    jk.has_playground as jk_has_playground,
                    jk.has_security as jk_has_security,
                    jk.developer_name as jk_developer,
                    jk.rating as jk_rating
                FROM apartments a
                LEFT JOIN residential_complexes jk
                    ON a.residential_complex_id = jk.id
                WHERE a.is_active = 1
            '''
        else:
            query = 'SELECT * FROM apartments WHERE is_active = 1'

        return pd.read_sql_query(query, self.conn)

    def export_residential_complexes(self) -> pd.DataFrame:
        """Экспорт всех ЖК в DataFrame."""
        return pd.read_sql_query('SELECT * FROM residential_complexes', self.conn)

    # ==========================================
    # СТАТИСТИКА
    # ==========================================

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики по базе данных."""
        cursor = self.conn.cursor()

        stats = {
            'city': self.city,
            'db_path': str(self.db_path),
        }

        # Квартиры
        cursor.execute('SELECT COUNT(*) FROM apartments')
        stats['apartments_total'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM apartments WHERE is_active = 1')
        stats['apartments_active'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM apartments WHERE residential_complex_id IS NOT NULL')
        stats['apartments_with_jk'] = cursor.fetchone()[0]

        # ЖК
        cursor.execute('SELECT COUNT(*) FROM residential_complexes')
        stats['jk_total'] = cursor.fetchone()[0]

        # Очередь
        stats['queue'] = self.get_queue_stats()

        return stats

    def print_stats(self):
        """Печать статистики."""
        stats = self.get_stats()
        print(f"\n{'='*50}")
        print(f"База данных: {stats['city'].upper()}")
        print(f"{'='*50}")
        print(f"Путь: {stats['db_path']}")
        print(f"\nКвартиры:")
        print(f"  Всего: {stats['apartments_total']}")
        print(f"  Активных: {stats['apartments_active']}")
        print(f"  С привязкой к ЖК: {stats['apartments_with_jk']}")
        print(f"\nЖилые комплексы: {stats['jk_total']}")
        print(f"\nОчередь парсинга:")
        for status, count in stats['queue'].items():
            print(f"  {status}: {count}")


# ==========================================
# УТИЛИТЫ
# ==========================================

def extract_slug_from_url(url: str) -> Optional[str]:
    """Извлечение slug из URL ЖК."""
    match = re.search(r'/jilie-kompleksy/([^/?]+)', url)
    return match.group(1) if match else None


def extract_listing_id_from_url(url: str) -> Optional[str]:
    """Извлечение listing_id из URL объявления."""
    match = re.search(r'/details/([^/?]+)', url)
    return match.group(1) if match else None


# ==========================================
# CLI
# ==========================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Управление базой данных недвижимости')
    parser.add_argument('city', choices=['bishkek', 'astana', 'almaty'], help='Город')
    parser.add_argument('--stats', action='store_true', help='Показать статистику')
    parser.add_argument('--export', type=str, help='Экспортировать в CSV')

    args = parser.parse_args()

    with RealEstateDB(args.city) as db:
        if args.stats:
            db.print_stats()

        if args.export:
            df = db.export_for_ml()
            df.to_csv(args.export, index=False)
            print(f"Экспортировано {len(df)} записей в {args.export}")
