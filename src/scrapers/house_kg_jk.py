"""
Парсер жилых комплексов (ЖК) с house.kg/jilie-kompleksy
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from fake_useragent import UserAgent
import logging
import sys

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import RealEstateDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HouseKGJKScraper:
    """Парсер жилых комплексов с house.kg"""

    BASE_URL = "https://www.house.kg"
    JK_CATALOG_URL = "https://www.house.kg/jilie-kompleksy"

    CITIES = {
        'bishkek': {'region': 1, 'town': 2},
    }

    def __init__(self, city='bishkek', delay_range=(1.5, 3)):
        self.city = city.lower()
        if self.city not in self.CITIES:
            raise ValueError(f"Неизвестный город: {city}")

        self.city_params = self.CITIES[self.city]
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = requests.Session()
        self.db = None

    def _get_headers(self):
        # Используем стабильный Chrome UA (fake_useragent иногда возвращает проблемные)
        return {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
        }

    def _delay(self):
        time.sleep(random.uniform(*self.delay_range))

    def _get_page(self, url: str) -> BeautifulSoup | None:
        """Загрузка и парсинг страницы"""
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Ошибка загрузки {url}: {e}")
            return None

    def get_total_pages(self) -> int:
        """Получение общего количества страниц каталога"""
        url = f"{self.JK_CATALOG_URL}?region={self.city_params['region']}&town={self.city_params['town']}"
        soup = self._get_page(url)
        if not soup:
            return 0

        # Ищем последнюю страницу в пагинации
        pagination = soup.find('ul', class_='pagination')
        if pagination:
            # Ищем все ссылки с page=
            page_links = pagination.find_all('a', href=re.compile(r'page=\d+'))
            if page_links:
                pages = []
                for link in page_links:
                    match = re.search(r'page=(\d+)', link.get('href', ''))
                    if match:
                        pages.append(int(match.group(1)))
                if pages:
                    return max(pages)

        return 1

    def get_jk_urls_from_page(self, page: int) -> list[dict]:
        """Получение списка ЖК с одной страницы каталога"""
        url = f"{self.JK_CATALOG_URL}?region={self.city_params['region']}&town={self.city_params['town']}&page={page}"
        soup = self._get_page(url)
        if not soup:
            return []

        jk_list = []

        # Ищем карточки ЖК
        cards = soup.find_all('a', href=re.compile(r'/jilie-kompleksy/[^?]+$'))

        seen_urls = set()
        for card in cards:
            href = card.get('href', '')
            if not href or href in seen_urls:
                continue
            if '/jilie-kompleksy/' not in href:
                continue

            # Извлекаем slug из URL
            slug_match = re.search(r'/jilie-kompleksy/([^/?]+)', href)
            if not slug_match:
                continue

            slug = slug_match.group(1)
            full_url = f"{self.BASE_URL}/jilie-kompleksy/{slug}"

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Извлекаем название если есть
            name = card.get_text(strip=True) or slug

            jk_list.append({
                'url': full_url,
                'slug': slug,
                'name': name,
            })

        return jk_list

    def parse_jk_detail(self, url: str) -> dict | None:
        """Парсинг детальной страницы ЖК"""
        soup = self._get_page(url)
        if not soup:
            return None

        data = {
            'url': url,
            'parsed_at': datetime.now().isoformat(),
        }

        # Название ЖК
        title = soup.find('h1')
        if title:
            data['name'] = title.get_text(strip=True)

        # Slug из URL
        slug_match = re.search(r'/jilie-kompleksy/([^/?]+)', url)
        if slug_match:
            data['slug'] = slug_match.group(1)

        # Адрес
        address_elem = soup.find('div', class_='address') or soup.find(string=re.compile(r'Бишкек'))
        if address_elem:
            if hasattr(address_elem, 'get_text'):
                data['address'] = address_elem.get_text(strip=True)
            else:
                # NavigableString
                parent = address_elem.parent
                if parent:
                    data['address'] = parent.get_text(strip=True)

        # Ищем информацию в блоках характеристик
        # Типичные паттерны: "Этажей:", "Класс:", "Статус:", etc.
        page_text = soup.get_text()

        # Этажность
        floors_match = re.search(r'[Ээ]тажей[:\s]+(\d+)', page_text)
        if floors_match:
            data['total_floors'] = int(floors_match.group(1))

        # Класс жилья
        class_patterns = ['эконом', 'комфорт', 'бизнес', 'премиум', 'элит']
        for cls in class_patterns:
            if re.search(rf'\b{cls}', page_text.lower()):
                data['class'] = cls
                break

        # Тип дома
        type_patterns = {
            'монолит': 'монолитный',
            'панель': 'панельный',
            'кирпич': 'кирпичный',
            'каркасно-монолит': 'каркасно-монолитный',
        }
        for pattern, house_type in type_patterns.items():
            if pattern in page_text.lower():
                data['house_type'] = house_type
                break

        # Статус строительства
        status_patterns = {
            'сдан': 'completed',
            'завершен': 'completed',
            'строится': 'under_construction',
            'в стадии строительства': 'under_construction',
        }
        for pattern, status in status_patterns.items():
            if pattern in page_text.lower():
                data['status'] = status
                break

        # Застройщик
        developer_link = soup.find('a', href=re.compile(r'/builders/'))
        if developer_link:
            data['developer_name'] = developer_link.get_text(strip=True)
            data['developer_url'] = self.BASE_URL + developer_link.get('href', '')

        # Цена за м²
        price_match = re.search(r'от\s*([\d\s]+)\s*сом\s*/\s*м[²2]', page_text.replace(' ', '').replace('\xa0', ''))
        if price_match:
            price_str = price_match.group(1).replace(' ', '').replace('\xa0', '')
            try:
                data['price_from_per_m2'] = int(price_str)
            except ValueError:
                pass

        # Высота потолков
        ceiling_match = re.search(r'[Вв]ысота потолков[:\s]+([\d.,]+)\s*м', page_text)
        if ceiling_match:
            try:
                data['ceiling_height'] = float(ceiling_match.group(1).replace(',', '.'))
            except ValueError:
                pass

        # Координаты (ищем в скриптах или data-атрибутах)
        scripts = soup.find_all('script')
        for script in scripts:
            script_text = script.string or ''
            # Паттерны для координат
            lat_match = re.search(r'lat[itude]*["\']?\s*[:=]\s*["\']?([\d.]+)', script_text)
            lng_match = re.search(r'l[on]g[itude]*["\']?\s*[:=]\s*["\']?([\d.]+)', script_text)
            if lat_match and lng_match:
                try:
                    data['latitude'] = float(lat_match.group(1))
                    data['longitude'] = float(lng_match.group(1))
                except ValueError:
                    pass
                break

        # Также ищем в data-атрибутах карты
        map_elem = soup.find(attrs={'data-lat': True, 'data-lng': True})
        if map_elem:
            try:
                data['latitude'] = float(map_elem['data-lat'])
                data['longitude'] = float(map_elem['data-lng'])
            except (ValueError, KeyError):
                pass

        # Год сдачи
        year_match = re.search(r'(?:срок сдачи|сдача)[:\s]*(?:Q?\d?\s*)?(\d{4})', page_text, re.IGNORECASE)
        if year_match:
            data['year_built'] = int(year_match.group(1))

        return data

    def collect_all_jk_urls(self) -> list[dict]:
        """Сбор всех URL ЖК из каталога"""
        total_pages = self.get_total_pages()
        logger.info(f"Найдено {total_pages} страниц каталога ЖК")

        all_jks = []
        seen_slugs = set()

        for page in tqdm(range(1, total_pages + 1), desc="Сбор URL ЖК"):
            jks = self.get_jk_urls_from_page(page)
            for jk in jks:
                if jk['slug'] not in seen_slugs:
                    seen_slugs.add(jk['slug'])
                    all_jks.append(jk)
            self._delay()

        logger.info(f"Собрано {len(all_jks)} уникальных ЖК")
        return all_jks

    def scrape(self, resume: bool = True) -> int:
        """
        Основной метод парсинга всех ЖК.

        Args:
            resume: Продолжить с того места, где остановились

        Returns:
            Количество спарсенных ЖК
        """
        logger.info(f"Запуск парсера ЖК для {self.city}")

        # Открываем БД
        self.db = RealEstateDB(self.city)

        # Получаем уже спарсенные ЖК
        existing_slugs = set()
        if resume:
            existing = self.db.conn.execute(
                "SELECT slug FROM residential_complexes WHERE slug IS NOT NULL"
            ).fetchall()
            existing_slugs = {row[0] for row in existing}
            logger.info(f"В БД уже есть {len(existing_slugs)} ЖК")

        # Собираем все URL
        all_jks = self.collect_all_jk_urls()

        # Фильтруем уже спарсенные
        to_parse = [jk for jk in all_jks if jk['slug'] not in existing_slugs]
        logger.info(f"К парсингу: {len(to_parse)} ЖК")

        if not to_parse:
            logger.info("Все ЖК уже спарсены")
            return 0

        # Парсим детали каждого ЖК
        parsed_count = 0
        errors = 0

        for jk in tqdm(to_parse, desc="Парсинг ЖК"):
            try:
                data = self.parse_jk_detail(jk['url'])
                if data:
                    self.db.add_residential_complex(data)
                    parsed_count += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error(f"Ошибка парсинга {jk['url']}: {e}")
                errors += 1

            self._delay()

        logger.info(f"Парсинг завершён: {parsed_count} ЖК добавлено, {errors} ошибок")

        # Статистика
        self.db.print_stats()

        return parsed_count

    def link_apartments_to_jk(self) -> int:
        """
        Связывание квартир с ЖК по совпадению названий.

        Returns:
            Количество связанных квартир
        """
        if not self.db:
            self.db = RealEstateDB(self.city)

        logger.info("Связывание квартир с ЖК...")

        # Получаем все ЖК
        jks = self.db.conn.execute("""
            SELECT id, name, slug, address
            FROM residential_complexes
            WHERE name IS NOT NULL
        """).fetchall()

        if not jks:
            logger.warning("Нет ЖК для связывания")
            return 0

        logger.info(f"Найдено {len(jks)} ЖК для связывания")

        linked_count = 0

        for jk_id, jk_name, jk_slug, jk_address in tqdm(jks, desc="Связывание"):
            # Ищем квартиры с упоминанием ЖК в URL или адресе
            # house.kg хранит ЖК в URL как /jilie-kompleksy/slug

            # Метод 1: Поиск по URL
            result = self.db.conn.execute("""
                UPDATE apartments
                SET residential_complex_id = ?
                WHERE url LIKE ?
                AND residential_complex_id IS NULL
            """, (jk_id, f"%/jilie-kompleksy/{jk_slug}%"))
            linked_count += result.rowcount

            # Метод 2: Поиск по адресу (если название ЖК в адресе)
            if jk_name:
                # Экранируем спецсимволы для LIKE
                safe_name = jk_name.replace('%', '\\%').replace('_', '\\_')
                result = self.db.conn.execute("""
                    UPDATE apartments
                    SET residential_complex_id = ?
                    WHERE (address LIKE ? ESCAPE '\\' OR district LIKE ? ESCAPE '\\')
                    AND residential_complex_id IS NULL
                """, (jk_id, f"%{safe_name}%", f"%{safe_name}%"))
                linked_count += result.rowcount

        self.db.conn.commit()
        logger.info(f"Связано {linked_count} квартир с ЖК")

        # Статистика
        stats = self.db.get_stats()
        logger.info(f"Квартир с привязкой к ЖК: {stats['apartments_with_jk']}")

        return linked_count

    def close(self):
        """Закрытие соединений"""
        if self.db:
            self.db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Парсер ЖК с house.kg')
    parser.add_argument('--city', type=str, default='bishkek',
                        choices=['bishkek'],
                        help='Город для парсинга')
    parser.add_argument('--no-resume', action='store_true',
                        help='Начать парсинг с нуля')
    parser.add_argument('--link-only', action='store_true',
                        help='Только связать квартиры с ЖК (без парсинга)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Минимальная задержка между запросами (сек)')

    args = parser.parse_args()

    scraper = HouseKGJKScraper(
        city=args.city,
        delay_range=(args.delay, args.delay + 1.5)
    )

    try:
        if args.link_only:
            scraper.link_apartments_to_jk()
        else:
            scraper.scrape(resume=not args.no_resume)
            scraper.link_apartments_to_jk()
    finally:
        scraper.close()


if __name__ == '__main__':
    main()
