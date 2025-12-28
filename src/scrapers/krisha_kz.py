"""
Парсер объявлений о продаже квартир с krisha.kz (Казахстан)
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


class KrishaKZScraper:
    """Парсер для сбора данных о недвижимости с krisha.kz"""

    BASE_URL = "https://krisha.kz"

    # Доступные города
    CITIES = {
        'almaty': 'https://krisha.kz/prodazha/kvartiry/almaty/',
        'astana': 'https://krisha.kz/prodazha/kvartiry/astana/',
        'shymkent': 'https://krisha.kz/prodazha/kvartiry/shymkent/',
    }

    # Курс тенге к доллару (приблизительный)
    KZT_TO_USD = 0.002  # ~500 тенге = 1 USD

    # Маппинг data-name атрибутов в имена колонок
    DATA_NAME_MAP = {
        'flat.building': 'house_type',
        'house.year': 'year_built',
        'flat.floor': 'floor_info',
        'live.square': 'area',
        'flat.toilet': 'bathroom',
        'flat.balcony': 'balcony',
        'flat.balcony_g': 'balcony_glazed',
        'flat.door': 'entrance_door',
        'flat.parking': 'parking',
        'live.furniture': 'furniture',
        'flat.flooring': 'floor_type',
        'ceiling': 'ceiling_height',
        'flat.security': 'security',
        'flat.priv_dorm': 'former_dormitory',
        'has_change': 'exchange_possible',
        'flat.condition': 'condition',
        'flat.internet': 'internet',
        'flat.phone': 'has_phone',
    }

    # Маппинг русских названий в английские (для заголовков)
    LABEL_MAP = {
        'город': 'city',
        'тип дома': 'house_type',
        'год постройки': 'year_built',
        'этаж': 'floor_info',
        'площадь': 'area',
        'санузел': 'bathroom',
        'балкон': 'balcony',
        'балкон остеклён': 'balcony_glazed',
        'дверь': 'entrance_door',
        'парковка': 'parking',
        'квартира меблирована': 'furniture',
        'пол': 'floor_type',
        'высота потолков': 'ceiling_height',
        'безопасность': 'security',
        'бывшее общежитие': 'former_dormitory',
        'возможен обмен': 'exchange_possible',
        'состояние': 'condition',
        'интернет': 'internet',
        'телефон': 'has_phone',
        'кондиционер': 'air_conditioning',
        'сигнализация': 'alarm',
        'домофон': 'intercom',
        'видеонаблюдение': 'video_surveillance',
        'консьерж': 'concierge',
    }

    def __init__(self, city='almaty', delay_range=(2, 4)):
        self.city = city.lower()
        if self.city not in self.CITIES:
            raise ValueError(f"Неизвестный город: {city}. Доступные: {list(self.CITIES.keys())}")
        self.listing_url = self.CITIES[self.city]
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = requests.Session()
        self.data = []
        self.parsed_ids = set()

    def _get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Referer': 'https://krisha.kz/',
        }

    def _delay(self):
        time.sleep(random.uniform(*self.delay_range))

    def _get_latest_intermediate_file(self):
        """Получение последнего промежуточного файла"""
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        files = list(output_dir.glob(f'krisha_kz_{self.city}_intermediate_*.csv'))
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

            if 'listing_id' in df.columns:
                self.parsed_ids = set(df['listing_id'].dropna().astype(str))

            logger.info(f"Загружено {len(self.data)} записей из {latest_file.name}")
            logger.info(f"Найдено {len(self.parsed_ids)} уникальных ID для пропуска")
            return len(self.data)
        except Exception as e:
            logger.error(f"Ошибка загрузки прогресса: {e}")
            return 0

    def get_listing_urls(self, page=1):
        """Получение списка URL объявлений со страницы каталога"""
        url = f"{self.listing_url}?page={page}"

        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Ошибка при запросе страницы {page}: {e}")
            return []

        soup = BeautifulSoup(response.text, 'lxml')
        urls = []

        # Поиск ссылок на объявления (формат /a/show/ID)
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/a/show/' in href:
                full_url = href if href.startswith('http') else self.BASE_URL + href
                if full_url not in urls:
                    urls.append(full_url)

        logger.info(f"Страница {page}: найдено {len(urls)} объявлений")
        return urls

    def _extract_window_data(self, html):
        """Извлечение данных из JavaScript объекта window.data"""
        # Ищем window.data = {...}
        match = re.search(r'window\.data\s*=\s*(\{.*?\});', html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Альтернативный паттерн
        match = re.search(r'"advert"\s*:\s*(\{.*?\})\s*,\s*"photos"', html, re.DOTALL)
        if match:
            try:
                return {'advert': json.loads(match.group(1))}
            except json.JSONDecodeError:
                pass

        return None

    def parse_listing(self, url):
        """Парсинг детальной страницы объявления"""
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Ошибка при запросе {url}: {e}")
            return None

        html = response.text
        soup = BeautifulSoup(html, 'lxml')
        data = {'url': url, 'parsed_at': datetime.now().isoformat(), 'source': 'krisha.kz'}

        # Извлечение ID из URL
        id_match = re.search(r'/a/show/(\d+)', url)
        if id_match:
            data['listing_id'] = id_match.group(1)

        # Основной парсинг из HTML (как в house.kg)
        self._parse_html_fallback(soup, data)

        # Дополнительно: попытка извлечь координаты из window.data
        window_data = self._extract_window_data(html)
        if window_data:
            # Координаты из map
            if 'map' in window_data:
                map_data = window_data['map']
                if 'lat' in map_data and 'lon' in map_data:
                    data['latitude'] = map_data['lat']
                    data['longitude'] = map_data['lon']

            # Дополнительные данные из advert если не получены из HTML
            if 'advert' in window_data:
                advert = window_data['advert']
                advert_data = self._parse_advert_data(advert)
                # Добавляем только отсутствующие поля
                for key, value in advert_data.items():
                    if key not in data or data[key] is None:
                        data[key] = value

        # Конвертация цены в USD
        if 'price_kzt' in data and data['price_kzt']:
            data['price_usd'] = int(data['price_kzt'] * self.KZT_TO_USD)

        return data if data.get('price_kzt') or data.get('price_usd') else None

    def _parse_advert_data(self, advert):
        """Парсинг данных из объекта advert"""
        data = {}

        # Основные поля
        field_mapping = {
            'price': 'price_kzt',
            'rooms': 'rooms',
            'square': 'area',
            'floor': 'floor',
            'floorCount': 'total_floors',
            'buildingYear': 'year_built',
            'houseType': 'house_type',
            'condition': 'condition',
        }

        for src, dst in field_mapping.items():
            if src in advert and advert[src]:
                value = advert[src]
                if isinstance(value, str):
                    # Попытка преобразовать в число
                    num_match = re.search(r'[\d.]+', value.replace(',', '.'))
                    if num_match:
                        try:
                            if dst in ['rooms', 'floor', 'total_floors', 'year_built', 'price_kzt']:
                                value = int(float(num_match.group()))
                            else:
                                value = float(num_match.group())
                        except ValueError:
                            pass
                data[dst] = value

        # Адрес
        if 'addressTitle' in advert:
            data['address'] = advert['addressTitle']
        if 'cityTitle' in advert:
            data['city'] = advert['cityTitle']
        if 'districtTitle' in advert:
            data['district'] = advert['districtTitle']

        # Цена за м²
        if 'squarePrice' in advert:
            try:
                data['price_per_m2_kzt'] = int(advert['squarePrice'])
            except (ValueError, TypeError):
                pass

        return data

    def _label_to_column(self, label):
        """Преобразование русского label в имя колонки"""
        label_lower = label.lower().strip()

        if label_lower in self.LABEL_MAP:
            return self.LABEL_MAP[label_lower]

        # Если нет в маппинге - создаём колонку с префиксом raw_
        if label_lower and len(label_lower) < 50:
            return f'raw_{label_lower.replace(" ", "_").replace("-", "_")[:30]}'

        return None

    def _parse_offer_info_items(self, soup, data):
        """Парсинг элементов .offer__info-item (sidebar)"""
        for item in soup.find_all(class_='offer__info-item'):
            data_name = item.get('data-name', '')
            title_elem = item.find(class_='offer__info-title')
            value_elem = item.find(class_='offer__advert-short-info')

            if not value_elem:
                # Для города/локации
                value_elem = item.find(class_='offer__location')

            if not title_elem or not value_elem:
                continue

            title = title_elem.get_text(strip=True)
            value = value_elem.get_text(strip=True)

            # Определяем имя колонки
            if data_name and data_name in self.DATA_NAME_MAP:
                col_name = self.DATA_NAME_MAP[data_name]
            else:
                col_name = self._label_to_column(title)

            if not col_name or not value:
                continue

            # Специальная обработка для некоторых полей
            if col_name == 'floor_info':
                floor_match = re.search(r'(\d+)\s*из\s*(\d+)', value)
                if floor_match:
                    data['floor'] = int(floor_match.group(1))
                    data['total_floors'] = int(floor_match.group(2))
            elif col_name == 'year_built':
                year_match = re.search(r'(\d{4})', value)
                if year_match:
                    data['year_built'] = int(year_match.group(1))
            elif col_name == 'area' and 'area' not in data:
                # Извлекаем только первое число (основная площадь)
                area_match = re.search(r'^([\d.,]+)', value.strip())
                if area_match:
                    data['area'] = float(area_match.group(1).replace(',', '.'))
                # Дополнительно: площадь кухни
                kitchen_match = re.search(r'кухни?\s*[—-]\s*([\d.,]+)', value)
                if kitchen_match:
                    data['kitchen_area'] = float(kitchen_match.group(1).replace(',', '.'))
            elif col_name == 'ceiling_height':
                height_match = re.search(r'([\d.,]+)', value)
                if height_match:
                    data['ceiling_height'] = float(height_match.group(1).replace(',', '.'))
            elif col_name == 'city':
                # Убираем "показать на карте"
                clean_value = re.sub(r'показать на карте', '', value).strip()
                data['city'] = clean_value
                parts = clean_value.split(',')
                if len(parts) >= 2:
                    data['district'] = parts[1].strip()
            else:
                data[col_name] = value

    def _parse_offer_parameters(self, soup, data):
        """Парсинг элементов .offer__parameters dl (дополнительные параметры)"""
        params_section = soup.find(class_='offer__parameters')
        if not params_section:
            return

        for dl in params_section.find_all('dl'):
            dt = dl.find('dt')
            dd = dl.find('dd')

            if not dt or not dd:
                continue

            data_name = dt.get('data-name', '')
            title = dt.get_text(strip=True)
            value = dd.get_text(strip=True)

            # Определяем имя колонки
            if data_name and data_name in self.DATA_NAME_MAP:
                col_name = self.DATA_NAME_MAP[data_name]
            else:
                col_name = self._label_to_column(title)

            if not col_name or not value:
                continue

            # Специальная обработка
            if col_name == 'ceiling_height':
                height_match = re.search(r'([\d.,]+)', value)
                if height_match:
                    data['ceiling_height'] = float(height_match.group(1).replace(',', '.'))
            else:
                data[col_name] = value

    def _parse_html_fallback(self, soup, data):
        """Основной парсинг из HTML"""
        # Цена
        price_elem = soup.find(class_='offer__price')
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            # Убираем пробелы и nbsp
            price_clean = re.sub(r'[\s\xa0]+', '', price_text)
            price_match = re.search(r'(\d+)', price_clean)
            if price_match:
                try:
                    data['price_kzt'] = int(price_match.group(1))
                except ValueError:
                    pass

        # Заголовок (комнаты, площадь)
        title = soup.find('h1')
        if title:
            title_text = title.get_text(strip=True)
            rooms_match = re.search(r'(\d+)-комн', title_text)
            if rooms_match:
                data['rooms'] = int(rooms_match.group(1))
            # Площадь из заголовка (первое число перед "м²")
            area_match = re.search(r'([\d.,]+)\s*м[²2]?', title_text)
            if area_match:
                area_val = area_match.group(1).replace(',', '.')
                try:
                    data['area'] = float(area_val)
                except ValueError:
                    pass

        # Парсинг offer__info-item (основные параметры в sidebar)
        # Может перезаписать area более точным значением
        self._parse_offer_info_items(soup, data)

        # Парсинг offer__parameters (дополнительные параметры)
        self._parse_offer_parameters(soup, data)

        # Финальная очистка area если осталась строка
        if 'area' in data and isinstance(data['area'], str):
            area_match = re.search(r'^([\d.,]+)', str(data['area']))
            if area_match:
                data['area'] = float(area_match.group(1).replace(',', '.'))

        # Адрес из заголовка или breadcrumbs
        if 'address' not in data:
            address_elem = soup.find(class_='offer__location')
            if address_elem:
                addr_span = address_elem.find('span')
                if addr_span:
                    data['address'] = addr_span.get_text(strip=True)

    def get_total_pages(self, max_pages=1600):
        """Определение общего количества страниц"""

        def has_listings(page_num):
            try:
                url = f"{self.listing_url}?page={page_num}"
                response = self.session.get(url, headers=self._get_headers(), timeout=30)
                return '/a/show/' in response.text
            except:
                return False

        # Бинарный поиск
        low, high = 1, max_pages

        if has_listings(high):
            logger.info(f"Найдено более {high} страниц")
            return high

        while high - low > 1:
            mid = (low + high) // 2
            if has_listings(mid):
                low = mid
            else:
                high = mid
            time.sleep(0.5)

        logger.info(f"Найдено страниц: {low}")
        return low

    def scrape(self, max_pages=None, save_every=50, resume=False):
        """
        Основной метод сбора данных

        Args:
            max_pages: Максимальное количество страниц
            save_every: Сохранять каждые N объявлений
            resume: Продолжить с последней точки
        """
        if resume:
            loaded_count = self.load_progress()
            logger.info(f"Режим возобновления: загружено {loaded_count} записей")

        total_pages = self.get_total_pages()
        if max_pages:
            total_pages = min(total_pages, max_pages)

        logger.info(f"Начинаем парсинг {total_pages} страниц")

        all_urls = []

        # Сбор URL
        for page in tqdm(range(1, total_pages + 1), desc="Сбор URL"):
            urls = self.get_listing_urls(page)
            all_urls.extend(urls)
            self._delay()

        logger.info(f"Собрано {len(all_urls)} URL")

        # Фильтрация уже обработанных
        if self.parsed_ids:
            original_count = len(all_urls)
            all_urls = [
                url for url in all_urls
                if not any(pid in url for pid in self.parsed_ids)
            ]
            skipped = original_count - len(all_urls)
            logger.info(f"Пропущено {skipped} уже обработанных, осталось {len(all_urls)}")

        # Парсинг объявлений
        for i, url in enumerate(tqdm(all_urls, desc="Парсинг объявлений")):
            listing_data = self.parse_listing(url)
            if listing_data:
                self.data.append(listing_data)
                if listing_data.get('listing_id'):
                    self.parsed_ids.add(listing_data['listing_id'])

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
        filepath = output_dir / f'krisha_kz_{self.city}_intermediate_{timestamp}.csv'
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Промежуточное сохранение: {len(df)} записей -> {filepath}")

    def save(self, filename=None):
        """Сохранение результатов"""
        if not self.data:
            logger.warning("Нет данных для сохранения")
            return None

        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.data)

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'krisha_kz_{self.city}_{timestamp}.csv'

        filepath = output_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8')

        logger.info(f"Сохранено {len(df)} записей в {filepath}")
        return filepath


def main():
    """Запуск парсера"""
    import argparse

    parser = argparse.ArgumentParser(description='Парсер krisha.kz')
    parser.add_argument('--city', type=str, default='almaty',
                        choices=['almaty', 'astana', 'shymkent'],
                        help='Город для парсинга')
    parser.add_argument('--resume', action='store_true',
                        help='Продолжить с последней сохранённой точки')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Максимальное количество страниц')
    parser.add_argument('--delay-min', type=float, default=2,
                        help='Минимальная задержка (сек)')
    parser.add_argument('--delay-max', type=float, default=4,
                        help='Максимальная задержка (сек)')
    args = parser.parse_args()

    scraper = KrishaKZScraper(city=args.city, delay_range=(args.delay_min, args.delay_max))

    print(f"Запуск парсера krisha.kz ({args.city.capitalize()})...")
    if args.resume:
        print("Режим возобновления: загрузка предыдущего прогресса...")

    scraper.scrape(max_pages=args.max_pages, resume=args.resume)
    filepath = scraper.save()

    if filepath:
        df = pd.read_csv(filepath)
        print(f"\nСтатистика:")
        print(f"  Всего записей: {len(df)}")
        print(f"  Колонки: {list(df.columns)}")


if __name__ == '__main__':
    main()
