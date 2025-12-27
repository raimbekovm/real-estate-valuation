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
    LISTING_URL = "https://www.house.kg/kupit-kvartiru"

    def __init__(self, delay_range=(1, 3)):
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = requests.Session()
        self.data = []

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
        url = f"{self.LISTING_URL}?page={page}"

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

            # Комнаты
            rooms_match = re.search(r'(\d+)-комн', header_text)
            if rooms_match:
                data['rooms'] = int(rooms_match.group(1))

            # Площадь
            area_match = re.search(r'([\d.,]+)\s*м[²2]', header_text)
            if area_match:
                data['area'] = float(area_match.group(1).replace(',', '.'))

        # Цена - ищем элемент с классом price-dollar
        price_elem = soup.find(class_='price-dollar')
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            # Убираем $ и пробелы, извлекаем число
            price_match = re.search(r'([\d\s]+)', price_text)
            if price_match:
                price_str = price_match.group(1).replace(' ', '')
                if price_str.isdigit() and len(price_str) >= 4:  # минимум 4 цифры для цены
                    data['price_usd'] = int(price_str)

        # Цена за м² - второй элемент с классом price-dollar содержит /м2
        price_elems = soup.find_all(class_='price-dollar')
        for elem in price_elems:
            text = elem.get_text(strip=True)
            if '/м' in text:
                price_per_m2_match = re.search(r'([\d\s]+)/м', text)
                if price_per_m2_match:
                    data['price_per_m2'] = int(price_per_m2_match.group(1).replace(' ', ''))
                break

        # Парсинг info-row элементов
        info_rows = soup.find_all(class_='info-row')
        for row in info_rows:
            text = row.get_text(separator=' ', strip=True).lower()

            # Этаж
            if 'этаж' in text and 'floor' not in data:
                floor_match = re.search(r'(\d+)\s*из\s*(\d+)', text)
                if floor_match:
                    data['floor'] = int(floor_match.group(1))
                    data['total_floors'] = int(floor_match.group(2))

            # Площадь (если не нашли в заголовке)
            if ('площадь' in text or 'общая' in text) and 'area' not in data:
                area_match = re.search(r'([\d.,]+)\s*м', text)
                if area_match:
                    data['area'] = float(area_match.group(1).replace(',', '.'))

            # Жилая площадь
            if 'жилая' in text:
                living_match = re.search(r'([\d.,]+)\s*м', text)
                if living_match:
                    data['living_area'] = float(living_match.group(1).replace(',', '.'))

            # Кухня
            if 'кухня' in text:
                kitchen_match = re.search(r'([\d.,]+)\s*м', text)
                if kitchen_match:
                    data['kitchen_area'] = float(kitchen_match.group(1).replace(',', '.'))

            # Год постройки
            if 'год' in text and 'постройки' in text:
                year_match = re.search(r'(\d{4})', text)
                if year_match:
                    data['year_built'] = int(year_match.group(1))

            # Серия дома
            if 'серия' in text:
                series_text = text.replace('серия', '').strip()
                if series_text:
                    data['building_series'] = series_text

            # Тип дома
            if 'тип' in text and ('строен' in text or 'дом' in text):
                for house_type in ['кирпич', 'панель', 'монолит', 'блочн']:
                    if house_type in text:
                        data['house_type'] = house_type
                        break

            # Состояние
            if 'состояние' in text:
                for condition in ['евроремонт', 'хорошее', 'среднее', 'требует', 'черновая', 'без отделки']:
                    if condition in text:
                        data['condition'] = condition
                        break

            # Отопление
            if 'отопление' in text:
                for heating in ['центральное', 'газовое', 'электрическое', 'печное', 'автономное']:
                    if heating in text:
                        data['heating'] = heating
                        break

            # Санузел
            if 'санузел' in text or 'санузлы' in text:
                if 'раздельный' in text:
                    data['bathroom'] = 'раздельный'
                elif 'совмещенный' in text:
                    data['bathroom'] = 'совмещенный'
                elif '2' in text or 'два' in text:
                    data['bathroom'] = '2+'

        # Адрес
        address_elem = soup.find(class_='address')
        if not address_elem:
            # Попробуем найти в заголовке
            if header:
                address_spans = header.find_all('span')
                for span in address_spans:
                    span_text = span.get_text(strip=True)
                    if 'Бишкек' in span_text or 'м-н' in span_text or 'мкр' in span_text:
                        data['address'] = span_text
                        break
        else:
            data['address'] = address_elem.get_text(strip=True)

        # Район из адреса
        if 'address' in data:
            addr = data['address']
            # Извлечение микрорайона
            district_match = re.search(r'(\d+\s*м-н|\d+\s*мкр|[А-Яа-я]+\s*м-н)', addr)
            if district_match:
                data['district'] = district_match.group(1)

        # Координаты из JavaScript
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                # Ищем координаты в разных форматах
                lat_match = re.search(r'["\']?lat(?:itude)?["\']?\s*[:=]\s*["\']?([\d.]+)', script.string)
                lon_match = re.search(r'["\']?(?:lon|lng|longitude)["\']?\s*[:=]\s*["\']?([\d.]+)', script.string)
                if lat_match and lon_match:
                    lat = float(lat_match.group(1))
                    lon = float(lon_match.group(1))
                    # Проверка валидности координат (примерно Бишкек)
                    if 42 < lat < 43 and 74 < lon < 75:
                        data['latitude'] = lat
                        data['longitude'] = lon
                        break

        # ID объявления из URL
        id_match = re.search(r'/details/([^/]+)', url)
        if id_match:
            data['listing_id'] = id_match.group(1)

        return data

    def get_total_pages(self):
        """Определение общего количества страниц"""
        try:
            response = self.session.get(self.LISTING_URL, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Ошибка при определении количества страниц: {e}")
            return 1

        soup = BeautifulSoup(response.text, 'lxml')

        # Поиск пагинации
        pagination = soup.find('nav', class_=re.compile(r'pagination', re.I))
        if pagination:
            page_links = pagination.find_all('a', href=True)
            max_page = 1
            for link in page_links:
                match = re.search(r'page=(\d+)', link['href'])
                if match:
                    page_num = int(match.group(1))
                    max_page = max(max_page, page_num)
            return max_page

        # Альтернативный поиск
        all_links = soup.find_all('a', href=re.compile(r'page=\d+'))
        max_page = 1
        for link in all_links:
            match = re.search(r'page=(\d+)', link['href'])
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)

        return max_page if max_page > 1 else 100  # default fallback

    def scrape(self, max_pages=None, save_every=50):
        """
        Основной метод сбора данных

        Args:
            max_pages: Максимальное количество страниц для парсинга
            save_every: Сохранять промежуточные результаты каждые N объявлений
        """
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

        # Парсинг каждого объявления
        for i, url in enumerate(tqdm(all_urls, desc="Парсинг объявлений")):
            listing_data = self.parse_listing(url)
            if listing_data and listing_data.get('price_usd'):
                self.data.append(listing_data)

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
        filepath = output_dir / f'house_kg_intermediate_{timestamp}.csv'
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
            filename = f'house_kg_bishkek_{timestamp}.csv'

        filepath = output_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8')

        logger.info(f"Сохранено {len(df)} записей в {filepath}")
        return filepath


def main():
    """Запуск парсера"""
    scraper = HouseKGScraper(delay_range=(2, 4))

    # Парсинг (начнём с небольшого количества для теста)
    print("Запуск парсера house.kg...")
    print("Для полного сбора данных может потребоваться несколько часов")

    # Тестовый запуск - 5 страниц
    # Для полного сбора убрать max_pages или увеличить
    scraper.scrape(max_pages=5)

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
