"""
Парсер объявлений о продаже квартир с house.kg
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from fake_useragent import UserAgent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HouseKGScraper:
    """Парсер для сбора данных о недвижимости с house.kg"""

    BASE_URL = "https://www.house.kg"

    # Доступные города (town ID)
    CITIES = {
        'bishkek': 2,
        'osh': 36,
        'jalal-abad': 27,
        'karakol': 18,
        'tokmok': 3,
    }

    def __init__(self, city='bishkek', delay_range=(1, 3)):
        self.city = city.lower()
        if self.city not in self.CITIES:
            raise ValueError(f"Неизвестный город: {city}. Доступные: {list(self.CITIES.keys())}")
        self.town_id = self.CITIES[self.city]
        self.listing_url = f"https://www.house.kg/kupit-kvartiru?town={self.town_id}"
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = requests.Session()
        self.data = []
        self.parsed_ids = set()

    def _get_latest_intermediate_file(self):
        """Получение последнего промежуточного файла"""
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        files = list(output_dir.glob(f'house_kg_{self.city}_intermediate_*.csv'))
        # Fallback для старых файлов (bishkek без префикса)
        if not files and self.city == 'bishkek':
            files = list(output_dir.glob('house_kg_intermediate_*.csv'))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    def load_progress(self):
        """Загрузка прогресса из последнего промежуточного файла"""
        latest_file = self._get_latest_intermediate_file()
        if not latest_file:
            logger.info("Промежуточные файлы не найдены, начинаем с нуля")
            return 0

        try:
            df = pd.read_csv(latest_file)
            self.data = df.to_dict('records')

            # Извлекаем listing_id для пропуска уже обработанных
            if 'listing_id' in df.columns:
                self.parsed_ids = set(df['listing_id'].dropna().astype(str))
            elif 'url' in df.columns:
                # Извлекаем ID из URL
                for url in df['url'].dropna():
                    match = re.search(r'/details/([^/?]+)', str(url))
                    if match:
                        self.parsed_ids.add(match.group(1))

            logger.info(f"Загружено {len(self.data)} записей из {latest_file.name}")
            logger.info(f"Найдено {len(self.parsed_ids)} уникальных ID для пропуска")
            return len(self.data)
        except Exception as e:
            logger.error(f"Ошибка загрузки прогресса: {e}")
            return 0

    def _get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
        }

    def _delay(self):
        time.sleep(random.uniform(*self.delay_range))

    def _safe_extract(self, element, default=None):
        """Безопасное извлечение текста из элемента"""
        if element:
            return element.get_text(strip=True)
        return default

    def _parse_price(self, price_str):
        """Извлечение цены в USD"""
        if not price_str:
            return None
        match = re.search(r'\$\s*([\d\s,]+)', price_str)
        if match:
            price = match.group(1).replace(' ', '').replace(',', '')
            try:
                return int(price)
            except ValueError:
                return None
        return None

    def _parse_area(self, area_str):
        """Извлечение площади в м²"""
        if not area_str:
            return None
        match = re.search(r'([\d.]+)\s*м', area_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _parse_rooms(self, rooms_str):
        """Извлечение количества комнат"""
        if not rooms_str:
            return None
        match = re.search(r'(\d+)', rooms_str)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _parse_floor(self, floor_str):
        """Извлечение этажа и этажности"""
        if not floor_str:
            return None, None
        match = re.search(r'(\d+)\s*(?:из|/)\s*(\d+)', floor_str)
        if match:
            try:
                return int(match.group(1)), int(match.group(2))
            except ValueError:
                return None, None
        return None, None

    def get_listing_urls(self, page=1):
        """Получение списка URL объявлений со страницы каталога"""
        url = f"{self.listing_url}&page={page}"

        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Ошибка при запросе страницы {page}: {e}")
            return []

        soup = BeautifulSoup(response.text, 'lxml')
        urls = []

        # Поиск ссылок на объявления
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/details/' in href:
                full_url = href if href.startswith('http') else self.BASE_URL + href
                if full_url not in urls:
                    urls.append(full_url)

        logger.info(f"Страница {page}: найдено {len(urls)} объявлений")
        return urls

    def _parse_info_rows(self, soup, data):
        """Парсинг всех info-row элементов (автоматически)"""
        info_rows = soup.find_all(class_='info-row')
        for row in info_rows:
            children = [c for c in row.children if str(c).strip()]
            if len(children) >= 2:
                label = children[0].get_text(strip=True).lower() if hasattr(children[0], 'get_text') else ''
                value = children[1].get_text(strip=True) if hasattr(children[1], 'get_text') else ''
            else:
                continue

            # Специальная обработка для полей с числами
            if label == 'этаж':
                floor_match = re.search(r'(\d+)\s*этаж\s*из\s*(\d+)', value)
                if floor_match:
                    data['floor'] = int(floor_match.group(1))
                    data['total_floors'] = int(floor_match.group(2))

            elif label == 'площадь':
                if 'area' not in data:
                    area_match = re.search(r'^([\d.,]+)\s*м', value)
                    if area_match:
                        data['area'] = float(area_match.group(1).replace(',', '.'))
                living_match = re.search(r'жилая[:\s]*([\d.,]+)', value)
                if living_match:
                    data['living_area'] = float(living_match.group(1).replace(',', '.'))
                kitchen_match = re.search(r'кухня[:\s]*([\d.,]+)', value)
                if kitchen_match:
                    data['kitchen_area'] = float(kitchen_match.group(1).replace(',', '.'))

            elif label == 'дом':
                for house_type in ['кирпич', 'панель', 'монолит', 'блочн']:
                    if house_type in value.lower():
                        data['house_type'] = house_type
                        break
                year_match = re.search(r'(\d{4})', value)
                if year_match:
                    data['year_built'] = int(year_match.group(1))

            elif label == 'высота потолков':
                height_match = re.search(r'([\d.,]+)', value)
                if height_match:
                    data['ceiling_height'] = float(height_match.group(1).replace(',', '.'))

            else:
                # Автоматическое добавление всех остальных полей
                col_name = self._label_to_column(label)
                if col_name and value:
                    clean_value = ' '.join(value.split())
                    data[col_name] = clean_value

    def _label_to_column(self, label):
        """Преобразование русского label в имя колонки"""
        # Маппинг русских названий в английские snake_case
        label_map = {
            'тип предложения': 'offer_type',
            'серия': 'building_series',
            'состояние': 'condition',
            'отопление': 'heating',
            'санузел': 'bathroom',
            'балкон': 'balcony',
            'мебель': 'furniture',
            'интернет': 'internet',
            'парковка': 'parking',
            'газ': 'gas',
            'пол': 'floor_type',
            'входная дверь': 'entrance_door',
            'телефон': 'has_phone',
            'безопасность': 'security',
            'разное': 'amenities',
            'правоустанавливающие документы': 'documents',
            'возможность рассрочки': 'installment',
            'возможность ипотеки': 'mortgage',
            'количество комнат': 'rooms_info',
            'комнаты': 'rooms_info',
            'окна': 'windows',
            'лифт': 'elevator',
            'охрана': 'security',
            'ремонт': 'renovation',
            'канализация': 'sewage',
            'вода': 'water',
            'электричество': 'electricity',
            'возможность обмена': 'exchange_possible',
        }

        if label in label_map:
            return label_map[label]

        # Если нет в маппинге - создаём колонку с префиксом raw_
        # чтобы потом можно было обработать
        if label and len(label) < 50:  # защита от мусора
            # Транслитерация или просто raw_
            return f'raw_{label.replace(" ", "_").replace("-", "_")[:30]}'

        return None

    def parse_listing(self, url):
        """Парсинг детальной страницы объявления"""
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Ошибка при запросе {url}: {e}")
            return None

        soup = BeautifulSoup(response.text, 'lxml')
        data = {'url': url, 'parsed_at': datetime.now().isoformat()}

        # Парсинг заголовка (details-header): "3-комн. кв., 90.8 м2"
        header = soup.find(class_='details-header')
        if header:
            header_text = header.get_text(strip=True)
            rooms_match = re.search(r'(\d+)-комн', header_text)
            if rooms_match:
                data['rooms'] = int(rooms_match.group(1))
            area_match = re.search(r'([\d.,]+)\s*м[²2]', header_text)
            if area_match:
                data['area'] = float(area_match.group(1).replace(',', '.'))

        # Цена
        price_elem = soup.find(class_='price-dollar')
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            price_match = re.search(r'([\d\s]+)', price_text)
            if price_match:
                price_str = price_match.group(1).replace(' ', '')
                if price_str.isdigit() and len(price_str) >= 4:
                    data['price_usd'] = int(price_str)

        price_elems = soup.find_all(class_='price-dollar')
        for elem in price_elems:
            text = elem.get_text(strip=True)
            if '/м' in text:
                price_per_m2_match = re.search(r'([\d\s]+)/м', text)
                if price_per_m2_match:
                    data['price_per_m2'] = int(price_per_m2_match.group(1).replace(' ', ''))
                break

        # Парсинг info-row (автоматически)
        self._parse_info_rows(soup, data)

        # Адрес - ищем в details-header
        address_elem = soup.select_one('.details-header .address')
        if address_elem:
            data['address'] = address_elem.get_text(strip=True)
        else:
            # Fallback
            address_elem = soup.find(class_='address')
            if address_elem:
                data['address'] = address_elem.get_text(strip=True)

        # Район из адреса
        if 'address' in data:
            addr = data['address']
            # Извлечение микрорайона (м-н, мкр, ж/м)
            district_match = re.search(r'([\w-]+\s*(?:м-н|мкр|ж/м))', addr)
            if district_match:
                data['district'] = district_match.group(1)
            else:
                # Попробуем извлечь район из начала адреса после города
                # Формат: "Бишкек, Район, улица..."
                parts = addr.split(',')
                if len(parts) >= 2:
                    potential_district = parts[1].strip()
                    # Если это не улица (не содержит типичных слов)
                    if not any(x in potential_district.lower() for x in ['ул.', 'улица', 'пер.', 'переулок', 'просп.']):
                        data['district'] = potential_district

        # Координаты из карты (наиболее точные)
        map_elem = soup.find(id='map2gis')
        if map_elem:
            lat = map_elem.get('data-lat')
            lon = map_elem.get('data-lon')
            if lat and lon:
                try:
                    data['latitude'] = float(lat)
                    data['longitude'] = float(lon)
                except ValueError:
                    pass

        # ID объявления из URL
        id_match = re.search(r'/details/([^/]+)', url)
        if id_match:
            data['listing_id'] = id_match.group(1)

        return data

    def get_total_pages(self):
        """Определение общего количества страниц через бинарный поиск"""

        def has_listings(page_num):
            """Проверка наличия объявлений на странице"""
            try:
                url = f"{self.listing_url}&page={page_num}"
                response = self.session.get(url, headers=self._get_headers(), timeout=30)
                soup = BeautifulSoup(response.text, 'lxml')
                listings = soup.find_all('a', href=re.compile(r'/details/'))
                return len(set(a['href'] for a in listings)) > 0
            except:
                return False

        # Бинарный поиск последней страницы (между 1 и 600)
        low, high = 1, 600

        # Сначала проверим верхнюю границу
        if has_listings(high):
            logger.info(f"Найдено более {high} страниц")
            return high

        # Бинарный поиск
        while high - low > 1:
            mid = (low + high) // 2
            if has_listings(mid):
                low = mid
            else:
                high = mid
            time.sleep(0.5)  # Небольшая задержка

        logger.info(f"Найдено страниц: {low}")
        return low

    def scrape(self, max_pages=None, save_every=50, resume=False):
        """
        Основной метод сбора данных

        Args:
            max_pages: Максимальное количество страниц для парсинга
            save_every: Сохранять промежуточные результаты каждые N объявлений
            resume: Продолжить с последней сохранённой точки
        """
        if resume:
            loaded_count = self.load_progress()
            logger.info(f"Режим возобновления: загружено {loaded_count} записей")

        total_pages = self.get_total_pages()
        if max_pages:
            total_pages = min(total_pages, max_pages)

        logger.info(f"Начинаем парсинг {total_pages} страниц")

        all_urls = []

        # Сбор всех URL
        for page in tqdm(range(1, total_pages + 1), desc="Сбор URL"):
            urls = self.get_listing_urls(page)
            all_urls.extend(urls)
            self._delay()

        logger.info(f"Собрано {len(all_urls)} уникальных URL")

        # Фильтрация уже обработанных URL
        if self.parsed_ids:
            original_count = len(all_urls)
            all_urls = [
                url for url in all_urls
                if not any(pid in url for pid in self.parsed_ids)
            ]
            skipped = original_count - len(all_urls)
            logger.info(f"Пропущено {skipped} уже обработанных, осталось {len(all_urls)}")

        # Парсинг каждого объявления
        for i, url in enumerate(tqdm(all_urls, desc="Парсинг объявлений")):
            listing_data = self.parse_listing(url)
            if listing_data and listing_data.get('price_usd'):
                self.data.append(listing_data)
                # Добавляем ID в set для избежания дублей
                if listing_data.get('listing_id'):
                    self.parsed_ids.add(listing_data['listing_id'])

            # Промежуточное сохранение
            if (i + 1) % save_every == 0:
                self._save_intermediate()

            self._delay()

        return self.data

    def _save_intermediate(self):
        """Сохранение промежуточных результатов"""
        if not self.data:
            return

        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = output_dir / f'house_kg_{self.city}_intermediate_{timestamp}.csv'
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Промежуточное сохранение: {len(df)} записей -> {filepath}")

    def save(self, filename=None):
        """Сохранение результатов в CSV"""
        if not self.data:
            logger.warning("Нет данных для сохранения")
            return None

        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.data)

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'house_kg_{self.city}_{timestamp}.csv'

        filepath = output_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8')

        logger.info(f"Сохранено {len(df)} записей в {filepath}")
        return filepath


def main():
    """Запуск парсера"""
    import argparse

    parser = argparse.ArgumentParser(description='Парсер house.kg')
    parser.add_argument('--city', type=str, default='bishkek',
                        choices=['bishkek', 'osh', 'jalal-abad', 'karakol', 'tokmok'],
                        help='Город для парсинга')
    parser.add_argument('--resume', action='store_true',
                        help='Продолжить с последней сохранённой точки')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Максимальное количество страниц')
    parser.add_argument('--delay-min', type=float, default=2,
                        help='Минимальная задержка между запросами (сек)')
    parser.add_argument('--delay-max', type=float, default=4,
                        help='Максимальная задержка между запросами (сек)')
    args = parser.parse_args()

    scraper = HouseKGScraper(city=args.city, delay_range=(args.delay_min, args.delay_max))

    print(f"Запуск парсера house.kg ({args.city.capitalize()})...")
    if args.resume:
        print("Режим возобновления: загрузка предыдущего прогресса...")
    print("Для полного сбора данных может потребоваться несколько часов")

    scraper.scrape(max_pages=args.max_pages, resume=args.resume)

    # Сохранение
    filepath = scraper.save()

    if filepath:
        df = pd.read_csv(filepath)
        print(f"\nСтатистика:")
        print(f"  Всего записей: {len(df)}")
        print(f"  Колонки: {list(df.columns)}")
        print(f"\nПример данных:")
        print(df.head())


if __name__ == '__main__':
    main()
