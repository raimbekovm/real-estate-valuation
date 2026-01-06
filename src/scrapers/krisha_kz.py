"""
Парсер объявлений о продаже квартир с krisha.kz (Казахстан)

Улучшения для обхода блокировок:
- Прокси ротация
- Экспоненциальный backoff при ошибках
- Случайный порядок страниц
- Увеличенные случайные паузы
- Многопоточный парсинг (--workers N)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from fake_useragent import UserAgent
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProxyRotator:
    """Ротация прокси для обхода блокировок"""

    # Бесплатные прокси (нужно обновлять)
    FREE_PROXY_SOURCES = [
        'https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all',
        'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
        'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt',
    ]

    def __init__(self):
        self.proxies = []
        self.current_index = 0
        self.failed_proxies = set()

    def fetch_proxies(self):
        """Загрузка списка прокси из бесплатных источников"""
        all_proxies = []

        for source_url in self.FREE_PROXY_SOURCES:
            try:
                response = requests.get(source_url, timeout=10)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    for line in lines:
                        proxy = line.strip()
                        if proxy and ':' in proxy:
                            all_proxies.append(proxy)
            except Exception as e:
                logger.debug(f"Не удалось загрузить прокси из {source_url}: {e}")

        # Убираем дубликаты и failed
        self.proxies = [p for p in list(set(all_proxies)) if p not in self.failed_proxies]
        random.shuffle(self.proxies)
        logger.info(f"Загружено {len(self.proxies)} прокси")
        return len(self.proxies)

    def get_next(self):
        """Получить следующий прокси"""
        if not self.proxies:
            return None

        # Циклический перебор
        proxy = self.proxies[self.current_index % len(self.proxies)]
        self.current_index += 1
        return {'http': f'http://{proxy}', 'https': f'http://{proxy}'}

    def mark_failed(self, proxy_dict):
        """Пометить прокси как нерабочий"""
        if proxy_dict:
            proxy_str = proxy_dict.get('http', '').replace('http://', '')
            self.failed_proxies.add(proxy_str)
            if proxy_str in self.proxies:
                self.proxies.remove(proxy_str)
                logger.debug(f"Прокси {proxy_str} помечен как нерабочий, осталось {len(self.proxies)}")


class SeleniumDriver:
    """Управление Selenium WebDriver"""

    # Пул User-Agent для ротации
    USER_AGENTS = [
        # Chrome Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Chrome macOS
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        # Firefox Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        # Firefox macOS
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        # Edge
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        # Safari
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    ]

    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        self.user_agent = random.choice(self.USER_AGENTS)

    def start(self):
        """Запуск браузера"""
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        options = Options()
        if self.headless:
            options.add_argument('--headless=new')

        # Анти-детект настройки
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=ru-RU')
        options.add_argument(f'user-agent={self.user_agent}')

        logger.info(f"User-Agent: {self.user_agent[:50]}...")

        # Отключаем автоматизацию
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

        # Скрываем webdriver
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })

        # Устанавливаем таймауты
        self.driver.set_page_load_timeout(30)  # Таймаут загрузки страницы
        self.driver.implicitly_wait(10)  # Неявное ожидание элементов

        logger.info("Selenium WebDriver запущен")
        return self.driver

    def stop(self):
        """Остановка браузера"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Selenium WebDriver остановлен")

    def get(self, url, wait_time=10, max_retries=2):
        """Загрузка страницы с ожиданием и обработкой timeout"""
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import TimeoutException

        for attempt in range(max_retries):
            try:
                self.driver.get(url)
                # Ждём загрузки контента
                try:
                    WebDriverWait(self.driver, wait_time).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except:
                    pass
                return self.driver.page_source
            except TimeoutException:
                logger.warning(f"Timeout при загрузке {url}, попытка {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
        return self.driver.page_source


class KrishaSeleniumScraper:
    """Selenium-парсер для krisha.kz (обход блокировок)"""

    BASE_URL = "https://krisha.kz"

    CITIES = {
        'almaty': 'https://krisha.kz/prodazha/kvartiry/almaty/',
        'astana': 'https://krisha.kz/prodazha/kvartiry/astana/',
        'shymkent': 'https://krisha.kz/prodazha/kvartiry/shymkent/',
    }

    KZT_TO_USD = 0.002

    # Маппинг data-name атрибутов krisha.kz
    DATA_NAME_MAP = {
        'flat.building': 'house_type',
        'house.year': 'year_built',
        'flat.floor': 'floor_info',
        'live.square': 'area_info',
        'flat.renovation': 'condition',
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

    def __init__(self, city='almaty', delay_range=(3, 7), headless=True):
        self.city = city.lower()
        if self.city not in self.CITIES:
            raise ValueError(f"Неизвестный город: {city}")
        self.listing_url = self.CITIES[self.city]
        self.delay_range = delay_range
        self.headless = headless
        self.driver_manager = None
        self.data = []
        self.parsed_ids = set()

        # База данных для URL (общая с обычным парсером)
        self.db_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / f'krisha_kz_{self.city}_urls.db'
        self._db_lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Инициализация базы данных для хранения URL"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                parsed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Добавляем колонку worker_id если её нет (для параллельной обработки)
        try:
            cursor.execute('ALTER TABLE urls ADD COLUMN worker_id INTEGER DEFAULT NULL')
        except sqlite3.OperationalError:
            pass  # Колонка уже существует
        # Сбрасываем зависшие задачи (worker_id != NULL но parsed = 0)
        cursor.execute('UPDATE urls SET worker_id = NULL WHERE parsed = 0 AND worker_id IS NOT NULL')
        conn.commit()
        conn.close()
        logger.info(f"БД инициализирована: {self.db_path}")

    def _claim_urls_batch(self, worker_id, batch_size=100):
        """Захватить батч URL для обработки воркером"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Захватываем URL которые не обработаны и не заняты другим воркером
            cursor.execute('''
                UPDATE urls SET worker_id = ?
                WHERE id IN (
                    SELECT id FROM urls
                    WHERE parsed = 0 AND worker_id IS NULL
                    LIMIT ?
                )
            ''', (worker_id, batch_size))
            conn.commit()
            # Получаем захваченные URL
            cursor.execute('SELECT url FROM urls WHERE worker_id = ? AND parsed = 0', (worker_id,))
            urls = [row[0] for row in cursor.fetchall()]
            conn.close()
            return urls

    def _save_urls_to_db(self, urls):
        """Сохранение URL в базу данных"""
        if not urls:
            return 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        saved = 0
        for url in urls:
            try:
                cursor.execute('INSERT OR IGNORE INTO urls (url) VALUES (?)', (url,))
                if cursor.rowcount > 0:
                    saved += 1
            except sqlite3.Error:
                pass
        conn.commit()
        conn.close()
        return saved

    def _load_urls_from_db(self, only_unparsed=True):
        """Загрузка URL из базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if only_unparsed:
            cursor.execute('SELECT url FROM urls WHERE parsed = 0')
        else:
            cursor.execute('SELECT url FROM urls')
        urls = [row[0] for row in cursor.fetchall()]
        conn.close()
        return urls

    def _mark_url_parsed(self, url):
        """Пометить URL как обработанный"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE urls SET parsed = 1 WHERE url = ?', (url,))
        conn.commit()
        conn.close()

    def _get_db_stats(self):
        """Статистика по URL в БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM urls')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM urls WHERE parsed = 1')
        parsed = cursor.fetchone()[0]
        conn.close()
        return total, parsed

    def _delay(self):
        """Случайная задержка"""
        delay = random.uniform(*self.delay_range)
        if random.random() < 0.1:
            delay += random.uniform(3, 8)
        time.sleep(delay)

    def _get_latest_intermediate_file(self):
        """Получение последнего промежуточного файла"""
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        files = list(output_dir.glob(f'krisha_kz_{self.city}_intermediate_*.csv'))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    def load_progress(self):
        """Загрузка прогресса"""
        latest_file = self._get_latest_intermediate_file()
        if not latest_file:
            return 0
        try:
            df = pd.read_csv(latest_file)
            self.data = df.to_dict('records')
            if 'listing_id' in df.columns:
                self.parsed_ids = set(df['listing_id'].dropna().astype(str))
            logger.info(f"Загружено {len(self.data)} записей из {latest_file.name}")
            return len(self.data)
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            return 0

    def get_listing_urls(self, page=1):
        """Получение URL объявлений со страницы"""
        url = f"{self.listing_url}?page={page}"

        try:
            html = self.driver_manager.get(url, wait_time=2)
            soup = BeautifulSoup(html, 'lxml')

            urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/a/show/' in href:
                    full_url = href if href.startswith('http') else self.BASE_URL + href
                    if full_url not in urls:
                        urls.append(full_url)

            logger.info(f"Страница {page}: найдено {len(urls)} объявлений")
            return urls
        except Exception as e:
            logger.error(f"Ошибка страницы {page}: {e}")
            return []

    def parse_listing(self, url):
        """Полный парсинг объявления со всеми полями"""
        try:
            html = self.driver_manager.get(url, wait_time=8)

            # Дополнительное ожидание для загрузки карты
            time.sleep(0.5)
            html = self.driver_manager.driver.page_source

            soup = BeautifulSoup(html, 'lxml')

            data = {
                'url': url,
                'parsed_at': datetime.now().isoformat(),
                'source': 'krisha.kz'
            }

            # ID из URL
            id_match = re.search(r'/a/show/(\d+)', url)
            if id_match:
                data['listing_id'] = id_match.group(1)

            # Цена
            price_elem = soup.find(class_='offer__price')
            if price_elem:
                price_text = re.sub(r'[\s\xa0]+', '', price_elem.get_text())
                price_match = re.search(r'(\d+)', price_text)
                if price_match:
                    data['price_kzt'] = int(price_match.group(1))
                    data['price_usd'] = int(data['price_kzt'] * self.KZT_TO_USD)

            # Заголовок (комнаты, площадь)
            title = soup.find('h1')
            if title:
                title_text = title.get_text(strip=True)
                rooms_match = re.search(r'(\d+)-комн', title_text)
                if rooms_match:
                    data['rooms'] = int(rooms_match.group(1))
                area_match = re.search(r'([\d.,]+)\s*м', title_text)
                if area_match:
                    data['area'] = float(area_match.group(1).replace(',', '.'))

            # Парсинг offer__info-item (основные параметры)
            for item in soup.find_all(class_='offer__info-item'):
                data_name = item.get('data-name', '')
                title_elem = item.find(class_='offer__info-title')
                value_elem = item.find(class_='offer__advert-short-info')
                if not value_elem:
                    value_elem = item.find(class_='offer__location')

                if not title_elem or not value_elem:
                    continue

                value = value_elem.get_text(strip=True)
                col_name = self.DATA_NAME_MAP.get(data_name)

                # Специальная обработка
                if data_name == 'flat.floor' or col_name == 'floor_info':
                    floor_match = re.search(r'(\d+)\s*из\s*(\d+)', value)
                    if floor_match:
                        data['floor'] = int(floor_match.group(1))
                        data['total_floors'] = int(floor_match.group(2))
                elif data_name == 'house.year':
                    year_match = re.search(r'(\d{4})', value)
                    if year_match:
                        data['year_built'] = int(year_match.group(1))
                elif data_name == 'live.square' or col_name == 'area_info':
                    # Основная площадь
                    area_match = re.search(r'^([\d.,]+)', value)
                    if area_match:
                        data['area'] = float(area_match.group(1).replace(',', '.'))
                    # Площадь кухни
                    kitchen_match = re.search(r'кухн[яи]\s*[—-]?\s*([\d.,]+)', value)
                    if kitchen_match:
                        data['kitchen_area'] = float(kitchen_match.group(1).replace(',', '.'))
                    # Жилая площадь
                    living_match = re.search(r'жил[ая\.]+\s*[—-]?\s*([\d.,]+)', value)
                    if living_match:
                        data['living_area'] = float(living_match.group(1).replace(',', '.'))
                elif not data_name:  # Город (без data-name)
                    title_text = title_elem.get_text(strip=True).lower()
                    if 'город' in title_text:
                        clean_value = re.sub(r'показать на карте', '', value).strip()
                        data['city'] = clean_value
                        # Извлечение района
                        parts = clean_value.split(',')
                        if len(parts) >= 2:
                            data['district'] = parts[1].strip()
                        # Адрес
                        if len(parts) >= 3:
                            data['address'] = ', '.join(parts[2:]).strip()
                elif col_name:
                    data[col_name] = value
                else:
                    # Новое поле - создаём колонку raw_
                    title_text = title_elem.get_text(strip=True).lower()
                    if title_text and len(title_text) < 50:
                        raw_col = f'raw_{title_text.replace(" ", "_")[:30]}'
                        data[raw_col] = value

            # Парсинг offer__parameters dl (дополнительные параметры)
            params_section = soup.find(class_='offer__parameters')
            if params_section:
                for dl in params_section.find_all('dl'):
                    dt = dl.find('dt')
                    dd = dl.find('dd')
                    if not dt or not dd:
                        continue

                    data_name = dt.get('data-name', '')
                    value = dd.get_text(strip=True)
                    col_name = self.DATA_NAME_MAP.get(data_name)

                    if col_name == 'ceiling_height':
                        height_match = re.search(r'([\d.,]+)', value)
                        if height_match:
                            data['ceiling_height'] = float(height_match.group(1).replace(',', '.'))
                    elif col_name:
                        data[col_name] = value
                    else:
                        # Новое поле - создаём колонку raw_
                        title_text = dt.get_text(strip=True).lower()
                        if title_text and len(title_text) < 50:
                            raw_col = f'raw_{title_text.replace(" ", "_")[:30]}'
                            data[raw_col] = value

            # Адрес из JSON (addressTitle)
            address_match = re.search(r'"addressTitle"\s*:\s*"([^"]+)"', html)
            if address_match:
                data['address'] = address_match.group(1)

            # Координаты из JavaScript (несколько паттернов)
            lat, lon = None, None

            # Паттерн 1: "lat":51.123,"lon":71.456
            lat_match = re.search(r'"lat"\s*:\s*([\d.]+)', html)
            lon_match = re.search(r'"lon"\s*:\s*([\d.]+)', html)
            if lat_match and lon_match:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))

            # Паттерн 2: latitude: 51.123, longitude: 71.456
            if not lat:
                lat_match = re.search(r'latitude["\s:]+(\d+\.\d+)', html)
                lon_match = re.search(r'longitude["\s:]+(\d+\.\d+)', html)
                if lat_match and lon_match:
                    lat = float(lat_match.group(1))
                    lon = float(lon_match.group(1))

            # Паттерн 3: из data-атрибутов карты
            if not lat:
                map_match = re.search(r'data-lat=["\']?([\d.]+)["\']?\s+data-lon=["\']?([\d.]+)', html)
                if map_match:
                    lat = float(map_match.group(1))
                    lon = float(map_match.group(2))

            # Паттерн 4: coordinates: [lon, lat] (GeoJSON формат)
            if not lat:
                coords_match = re.search(r'"coordinates"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', html)
                if coords_match:
                    lon = float(coords_match.group(1))
                    lat = float(coords_match.group(2))

            if lat and lon and 40 < lat < 60 and 50 < lon < 90:  # Валидация для Казахстана
                data['latitude'] = lat
                data['longitude'] = lon

            # Цена за м²
            if data.get('price_kzt') and data.get('area'):
                data['price_per_m2_kzt'] = int(data['price_kzt'] / data['area'])

            return data if data.get('price_kzt') else None

        except Exception as e:
            logger.error(f"Ошибка парсинга {url}: {e}")
            return None

    def get_total_pages(self, max_check=1500):
        """Определение числа страниц"""
        def has_listings(page):
            try:
                url = f"{self.listing_url}?page={page}"
                html = self.driver_manager.get(url, wait_time=10)
                return '/a/show/' in html
            except:
                return False

        low, high = 1, max_check
        if has_listings(high):
            return high

        while high - low > 1:
            mid = (low + high) // 2
            if has_listings(mid):
                low = mid
            else:
                high = mid
            time.sleep(1)

        logger.info(f"Найдено страниц: {low}")
        return low

    def _save_intermediate(self):
        """Промежуточное сохранение"""
        if not self.data:
            return
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = output_dir / f'krisha_kz_{self.city}_intermediate_{timestamp}.csv'
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Сохранено {len(df)} записей -> {filepath.name}")

    def save(self, filename=None):
        """Финальное сохранение"""
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

    def scrape(self, max_pages=None, save_every=50, resume=False, collect_only=False):
        """Основной метод сбора данных"""
        if resume:
            self.load_progress()

        # Проверяем есть ли URL в БД
        total_in_db, parsed_in_db = self._get_db_stats()
        logger.info(f"В БД: {total_in_db} URL, из них обработано: {parsed_in_db}")

        # Если в БД есть необработанные URL - используем их
        all_urls = self._load_urls_from_db(only_unparsed=True)

        # Запуск браузера
        self.driver_manager = SeleniumDriver(headless=self.headless)
        self.driver_manager.start()

        try:
            if all_urls and not collect_only:
                logger.info(f"Загружено {len(all_urls)} необработанных URL из БД")
            else:
                # Собираем URL со страниц
                if max_pages:
                    total_pages = max_pages
                    logger.info(f"Используем max_pages={max_pages}")
                else:
                    total_pages = self.get_total_pages()

                logger.info(f"Сбор URL с {total_pages} страниц")

                # Случайный порядок страниц
                pages = list(range(1, total_pages + 1))
                random.shuffle(pages)

                collected_urls = []
                for page in tqdm(pages, desc="Сбор URL"):
                    urls = self.get_listing_urls(page)
                    collected_urls.extend(urls)

                    # Сохраняем в БД каждые 200 URL
                    if len(collected_urls) >= 200:
                        saved = self._save_urls_to_db(collected_urls)
                        logger.info(f"Сохранено {saved} новых URL в БД")
                        collected_urls = []

                    self._delay()

                # Сохраняем оставшиеся URL
                if collected_urls:
                    saved = self._save_urls_to_db(collected_urls)
                    logger.info(f"Сохранено {saved} новых URL в БД")

                total_in_db, _ = self._get_db_stats()
                logger.info(f"Всего URL в БД: {total_in_db}")

                # Если только сбор URL - выходим
                if collect_only:
                    logger.info("Режим collect_only: сбор URL завершён")
                    return []

                # Загружаем все необработанные URL
                all_urls = self._load_urls_from_db(only_unparsed=True)

            # Фильтрация обработанных по parsed_ids
            if self.parsed_ids:
                original_count = len(all_urls)
                all_urls = [u for u in all_urls if not any(pid in u for pid in self.parsed_ids)]
                skipped = original_count - len(all_urls)
                if skipped > 0:
                    logger.info(f"Пропущено {skipped} уже обработанных")

            random.shuffle(all_urls)
            logger.info(f"К парсингу: {len(all_urls)} объявлений")

            # Парсинг
            for i, url in enumerate(tqdm(all_urls, desc="Парсинг")):
                data = self.parse_listing(url)
                if data:
                    self.data.append(data)
                    if data.get('listing_id'):
                        self.parsed_ids.add(data['listing_id'])
                    # Помечаем URL как обработанный
                    self._mark_url_parsed(url)

                if (i + 1) % save_every == 0:
                    self._save_intermediate()

                self._delay()

        finally:
            self.driver_manager.stop()

        return self.data

    def _worker_parse(self, worker_id, pbar, results_list, results_lock):
        """Воркер для параллельного парсинга"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        def start_driver():
            d = SeleniumDriver(headless=self.headless)
            d.start()
            return d

        driver = start_driver()

        try:
            while True:
                # Захватываем батч URL
                urls = self._claim_urls_batch(worker_id, batch_size=50)
                if not urls:
                    logger.info(f"Воркер {worker_id}: нет больше URL")
                    break

                logger.info(f"Воркер {worker_id}: захватил {len(urls)} URL")

                for url in urls:
                    try:
                        # Парсим с коротким таймаутом
                        html = driver.get(url, wait_time=5, max_retries=1)
                        time.sleep(0.3)
                        html = driver.driver.page_source

                        soup = BeautifulSoup(html, 'lxml')
                        data = self._parse_listing_html(url, html, soup)

                        if data:
                            with results_lock:
                                results_list.append(data)
                            self._mark_url_parsed(url)

                        consecutive_errors = 0
                        pbar.update(1)

                        # Задержка для избежания блокировки
                        time.sleep(random.uniform(1.5, 3.0))

                    except Exception as e:
                        consecutive_errors += 1
                        logger.warning(f"Воркер {worker_id} ошибка #{consecutive_errors}: {type(e).__name__}")
                        pbar.update(1)

                        # Перезапуск браузера при серии ошибок
                        if consecutive_errors >= max_consecutive_errors:
                            logger.warning(f"Воркер {worker_id}: перезапуск браузера")
                            try:
                                driver.stop()
                            except:
                                pass
                            time.sleep(2)
                            driver = start_driver()
                            consecutive_errors = 0

                # Сохраняем промежуточные результаты
                with results_lock:
                    if len(results_list) % 100 == 0:
                        self.data = list(results_list)
                        self._save_intermediate()

        finally:
            try:
                driver.stop()
            except:
                pass
            logger.info(f"Воркер {worker_id} завершён")

    def _parse_listing_html(self, url, html, soup):
        """Парсинг HTML страницы объявления (вынесено для переиспользования)"""
        data = {
            'url': url,
            'parsed_at': datetime.now().isoformat(),
            'source': 'krisha.kz'
        }

        # ID из URL
        id_match = re.search(r'/a/show/(\d+)', url)
        if id_match:
            data['listing_id'] = id_match.group(1)

        # Цена
        price_elem = soup.find(class_='offer__price')
        if price_elem:
            price_text = re.sub(r'[\s\xa0]+', '', price_elem.get_text())
            price_match = re.search(r'(\d+)', price_text)
            if price_match:
                data['price_kzt'] = int(price_match.group(1))
                data['price_usd'] = int(data['price_kzt'] * self.KZT_TO_USD)

        # Заголовок (комнаты, площадь)
        title = soup.find('h1')
        if title:
            title_text = title.get_text(strip=True)
            rooms_match = re.search(r'(\d+)-комн', title_text)
            if rooms_match:
                data['rooms'] = int(rooms_match.group(1))
            area_match = re.search(r'([\d.,]+)\s*м', title_text)
            if area_match:
                data['area'] = float(area_match.group(1).replace(',', '.'))

        # Парсинг offer__info-item
        for item in soup.find_all(class_='offer__info-item'):
            data_name = item.get('data-name', '')
            title_elem = item.find(class_='offer__info-title')
            value_elem = item.find(class_='offer__advert-short-info')
            if not value_elem:
                value_elem = item.find(class_='offer__location')

            if not title_elem or not value_elem:
                continue

            value = value_elem.get_text(strip=True)
            col_name = self.DATA_NAME_MAP.get(data_name)

            if data_name == 'flat.floor' or col_name == 'floor_info':
                floor_match = re.search(r'(\d+)\s*из\s*(\d+)', value)
                if floor_match:
                    data['floor'] = int(floor_match.group(1))
                    data['total_floors'] = int(floor_match.group(2))
            elif data_name == 'house.year':
                year_match = re.search(r'(\d{4})', value)
                if year_match:
                    data['year_built'] = int(year_match.group(1))
            elif data_name == 'live.square' or col_name == 'area_info':
                area_match = re.search(r'^([\d.,]+)', value)
                if area_match:
                    data['area'] = float(area_match.group(1).replace(',', '.'))
                kitchen_match = re.search(r'кухн[яи]\s*[—-]?\s*([\d.,]+)', value)
                if kitchen_match:
                    data['kitchen_area'] = float(kitchen_match.group(1).replace(',', '.'))
                living_match = re.search(r'жил[ая\.]+\s*[—-]?\s*([\d.,]+)', value)
                if living_match:
                    data['living_area'] = float(living_match.group(1).replace(',', '.'))
            elif not data_name:
                title_text = title_elem.get_text(strip=True).lower()
                if 'город' in title_text:
                    clean_value = re.sub(r'показать на карте', '', value).strip()
                    data['city'] = clean_value
                    parts = clean_value.split(',')
                    if len(parts) >= 2:
                        data['district'] = parts[1].strip()
                    if len(parts) >= 3:
                        data['address'] = ', '.join(parts[2:]).strip()
            elif col_name:
                data[col_name] = value
            else:
                title_text = title_elem.get_text(strip=True).lower()
                if title_text and len(title_text) < 50:
                    raw_col = f'raw_{title_text.replace(" ", "_")[:30]}'
                    data[raw_col] = value

        # Парсинг offer__parameters dl
        params_section = soup.find(class_='offer__parameters')
        if params_section:
            for dl in params_section.find_all('dl'):
                dt = dl.find('dt')
                dd = dl.find('dd')
                if not dt or not dd:
                    continue

                data_name = dt.get('data-name', '')
                value = dd.get_text(strip=True)
                col_name = self.DATA_NAME_MAP.get(data_name)

                if col_name == 'ceiling_height':
                    height_match = re.search(r'([\d.,]+)', value)
                    if height_match:
                        data['ceiling_height'] = float(height_match.group(1).replace(',', '.'))
                elif col_name:
                    data[col_name] = value
                else:
                    title_text = dt.get_text(strip=True).lower()
                    if title_text and len(title_text) < 50:
                        raw_col = f'raw_{title_text.replace(" ", "_")[:30]}'
                        data[raw_col] = value

        # Адрес из JSON
        address_match = re.search(r'"addressTitle"\s*:\s*"([^"]+)"', html)
        if address_match:
            data['address'] = address_match.group(1)

        # Координаты
        lat, lon = None, None
        lat_match = re.search(r'"lat"\s*:\s*([\d.]+)', html)
        lon_match = re.search(r'"lon"\s*:\s*([\d.]+)', html)
        if lat_match and lon_match:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))

        if not lat:
            lat_match = re.search(r'latitude["\s:]+(\d+\.\d+)', html)
            lon_match = re.search(r'longitude["\s:]+(\d+\.\d+)', html)
            if lat_match and lon_match:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))

        if lat and lon and 40 < lat < 60 and 50 < lon < 90:
            data['latitude'] = lat
            data['longitude'] = lon

        # Цена за м²
        if data.get('price_kzt') and data.get('area'):
            data['price_per_m2_kzt'] = int(data['price_kzt'] / data['area'])

        return data if data.get('price_kzt') else None

    def scrape_parallel(self, num_workers=3, resume=True):
        """Параллельный парсинг несколькими воркерами"""
        if resume:
            self.load_progress()

        total_in_db, parsed_in_db = self._get_db_stats()
        remaining = total_in_db - parsed_in_db
        logger.info(f"В БД: {total_in_db} URL, обработано: {parsed_in_db}, осталось: {remaining}")

        if remaining == 0:
            logger.info("Все URL обработаны!")
            return self.data

        results_list = list(self.data)  # Копируем существующие данные
        results_lock = threading.Lock()

        with tqdm(total=remaining, desc=f"Парсинг ({num_workers} воркеров)") as pbar:
            threads = []
            for worker_id in range(num_workers):
                t = threading.Thread(
                    target=self._worker_parse,
                    args=(worker_id, pbar, results_list, results_lock)
                )
                t.start()
                threads.append(t)
                time.sleep(2)  # Небольшая задержка между запуском воркеров

            for t in threads:
                t.join()

        self.data = results_list
        self._save_intermediate()
        return self.data


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

    def __init__(self, city='almaty', delay_range=(3, 7), use_proxy=True):
        self.city = city.lower()
        if self.city not in self.CITIES:
            raise ValueError(f"Неизвестный город: {city}. Доступные: {list(self.CITIES.keys())}")
        self.listing_url = self.CITIES[self.city]
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = requests.Session()
        self.data = []
        self.parsed_ids = set()

        # Прокси ротация
        self.use_proxy = use_proxy
        self.proxy_rotator = ProxyRotator() if use_proxy else None
        self.current_proxy = None

        # Экспоненциальный backoff
        self.base_backoff = 30  # начальная задержка при ошибке (сек)
        self.max_backoff = 300  # максимальная задержка (5 мин)
        self.consecutive_errors = 0

        # База данных для URL
        self.db_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / f'krisha_kz_{self.city}_urls.db'
        self._init_db()

    def _init_db(self):
        """Инициализация базы данных для хранения URL"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                parsed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"БД инициализирована: {self.db_path}")

    def _save_urls_to_db(self, urls):
        """Сохранение URL в базу данных"""
        if not urls:
            return 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        saved = 0
        for url in urls:
            try:
                cursor.execute('INSERT OR IGNORE INTO urls (url) VALUES (?)', (url,))
                if cursor.rowcount > 0:
                    saved += 1
            except sqlite3.Error:
                pass
        conn.commit()
        conn.close()
        return saved

    def _load_urls_from_db(self, only_unparsed=True):
        """Загрузка URL из базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if only_unparsed:
            cursor.execute('SELECT url FROM urls WHERE parsed = 0')
        else:
            cursor.execute('SELECT url FROM urls')
        urls = [row[0] for row in cursor.fetchall()]
        conn.close()
        return urls

    def _mark_url_parsed(self, url):
        """Пометить URL как обработанный"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE urls SET parsed = 1 WHERE url = ?', (url,))
        conn.commit()
        conn.close()

    def _get_db_stats(self):
        """Статистика по URL в БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM urls')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM urls WHERE parsed = 1')
        parsed = cursor.fetchone()[0]
        conn.close()
        return total, parsed

    def _get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Referer': 'https://krisha.kz/',
        }

    def _delay(self):
        """Случайная задержка между запросами"""
        base_delay = random.uniform(*self.delay_range)
        # Иногда делаем длинную паузу (имитация человека)
        if random.random() < 0.1:  # 10% шанс
            base_delay += random.uniform(5, 15)
            logger.debug(f"Длинная пауза: {base_delay:.1f}с")
        time.sleep(base_delay)

    def _exponential_backoff(self):
        """Экспоненциальная задержка при ошибках"""
        delay = min(self.base_backoff * (2 ** self.consecutive_errors), self.max_backoff)
        # Добавляем jitter (случайность)
        delay = delay * (0.5 + random.random())
        logger.warning(f"Backoff: ожидание {delay:.0f}с (ошибок подряд: {self.consecutive_errors})")
        time.sleep(delay)

    def _rotate_proxy(self):
        """Сменить прокси"""
        if self.proxy_rotator:
            self.current_proxy = self.proxy_rotator.get_next()
            if self.current_proxy:
                logger.debug(f"Используем прокси: {self.current_proxy.get('http', 'None')}")

    def _make_request(self, url, max_retries=3):
        """Выполнить запрос с retry и backoff"""
        for attempt in range(max_retries):
            try:
                # Меняем прокси при каждой попытке
                if self.use_proxy and attempt > 0:
                    self._rotate_proxy()

                proxies = self.current_proxy if self.use_proxy else None
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    proxies=proxies,
                    timeout=30
                )
                response.raise_for_status()

                # Успех - сбрасываем счётчик ошибок
                self.consecutive_errors = 0
                return response

            except requests.RequestException as e:
                self.consecutive_errors += 1
                logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась: {e}")

                # Помечаем прокси как нерабочий
                if self.proxy_rotator and self.current_proxy:
                    self.proxy_rotator.mark_failed(self.current_proxy)

                if attempt < max_retries - 1:
                    self._exponential_backoff()
                    self._rotate_proxy()

        return None

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

        response = self._make_request(url)
        if not response:
            logger.error(f"Не удалось загрузить страницу {page}")
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
        response = self._make_request(url)
        if not response:
            logger.error(f"Не удалось загрузить объявление: {url}")
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

    def scrape(self, max_pages=None, save_every=50, resume=False, shuffle_pages=True, collect_only=False):
        """
        Основной метод сбора данных

        Args:
            max_pages: Максимальное количество страниц
            save_every: Сохранять каждые N объявлений
            resume: Продолжить с последней точки
            shuffle_pages: Случайный порядок страниц (обход блокировок)
            collect_only: Только собрать URL, без парсинга
        """
        # Загрузка прокси если включено
        if self.use_proxy and self.proxy_rotator:
            proxy_count = self.proxy_rotator.fetch_proxies()
            if proxy_count > 0:
                self._rotate_proxy()  # Выбираем первый прокси
            else:
                logger.warning("Прокси не загружены, работаем без прокси")
                self.use_proxy = False

        if resume:
            loaded_count = self.load_progress()
            logger.info(f"Режим возобновления: загружено {loaded_count} записей")

        # Проверяем есть ли URL в БД
        total_in_db, parsed_in_db = self._get_db_stats()
        logger.info(f"В БД: {total_in_db} URL, из них обработано: {parsed_in_db}")

        # Если collect_only или нет URL в БД - собираем URL
        all_urls = self._load_urls_from_db(only_unparsed=True)

        if all_urls and not collect_only:
            logger.info(f"Загружено {len(all_urls)} необработанных URL из БД")
        else:
            # Собираем URL со страниц
            total_pages = self.get_total_pages()
            if max_pages:
                total_pages = min(total_pages, max_pages)

            logger.info(f"Начинаем сбор URL с {total_pages} страниц")

            # Случайный порядок страниц
            pages = list(range(1, total_pages + 1))
            if shuffle_pages:
                random.shuffle(pages)
                logger.info("Страницы перемешаны для обхода блокировок")

            collected_urls = []

            # Сбор URL с сохранением каждые 200 штук
            for page in tqdm(pages, desc="Сбор URL"):
                urls = self.get_listing_urls(page)
                collected_urls.extend(urls)

                # Сохраняем в БД каждые 200 URL
                if len(collected_urls) >= 200:
                    saved = self._save_urls_to_db(collected_urls)
                    logger.info(f"Сохранено {saved} новых URL в БД (всего собрано: {len(collected_urls)})")
                    collected_urls = []

                self._delay()

            # Сохраняем оставшиеся URL
            if collected_urls:
                saved = self._save_urls_to_db(collected_urls)
                logger.info(f"Сохранено {saved} новых URL в БД")

            total_in_db, _ = self._get_db_stats()
            logger.info(f"Всего URL в БД: {total_in_db}")

            # Если только сбор URL - выходим
            if collect_only:
                logger.info("Режим collect_only: сбор URL завершён")
                return []

            # Загружаем все необработанные URL
            all_urls = self._load_urls_from_db(only_unparsed=True)
            logger.info(f"К парсингу: {len(all_urls)} URL")

        # Фильтрация уже обработанных (по parsed_ids из CSV)
        if self.parsed_ids:
            original_count = len(all_urls)
            all_urls = [
                url for url in all_urls
                if not any(pid in url for pid in self.parsed_ids)
            ]
            skipped = original_count - len(all_urls)
            if skipped > 0:
                logger.info(f"Пропущено {skipped} уже обработанных, осталось {len(all_urls)}")

        # Перемешиваем URLs для случайного порядка парсинга
        if shuffle_pages:
            random.shuffle(all_urls)

        # Парсинг объявлений
        for i, url in enumerate(tqdm(all_urls, desc="Парсинг объявлений")):
            listing_data = self.parse_listing(url)
            if listing_data:
                self.data.append(listing_data)
                if listing_data.get('listing_id'):
                    self.parsed_ids.add(listing_data['listing_id'])
                # Помечаем URL как обработанный
                self._mark_url_parsed(url)

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

    parser = argparse.ArgumentParser(description='Парсер krisha.kz с обходом блокировок')
    parser.add_argument('--city', type=str, default='almaty',
                        choices=['almaty', 'astana', 'shymkent'],
                        help='Город для парсинга')
    parser.add_argument('--resume', action='store_true',
                        help='Продолжить с последней сохранённой точки')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Максимальное количество страниц')
    parser.add_argument('--delay-min', type=float, default=3,
                        help='Минимальная задержка (сек)')
    parser.add_argument('--delay-max', type=float, default=7,
                        help='Максимальная задержка (сек)')
    parser.add_argument('--no-proxy', action='store_true',
                        help='Отключить использование прокси')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Отключить случайный порядок страниц')
    parser.add_argument('--selenium', action='store_true',
                        help='Использовать Selenium (headless браузер)')
    parser.add_argument('--no-headless', action='store_true',
                        help='Показывать браузер (только с --selenium)')
    parser.add_argument('--collect-only', action='store_true',
                        help='Только собрать URL без парсинга')
    parser.add_argument('--workers', type=int, default=1,
                        help='Количество параллельных воркеров (только с --selenium)')
    args = parser.parse_args()

    print(f"Запуск парсера krisha.kz ({args.city.capitalize()})...")

    if args.selenium:
        # Selenium парсер
        headless = not args.no_headless
        scraper = KrishaSeleniumScraper(
            city=args.city,
            delay_range=(args.delay_min, args.delay_max),
            headless=headless
        )
        print(f"  Режим: Selenium ({'headless' if headless else 'видимый'})")
        print(f"  Задержка: {args.delay_min}-{args.delay_max} сек")
        if args.workers > 1:
            print(f"  Воркеров: {args.workers} (параллельный режим)")
        if args.resume:
            print("  Режим возобновления: да")

        if args.workers > 1 and not args.collect_only:
            # Параллельный парсинг
            scraper.scrape_parallel(num_workers=args.workers, resume=args.resume)
        else:
            scraper.scrape(max_pages=args.max_pages, save_every=50, resume=args.resume, collect_only=args.collect_only)

        if not args.collect_only:
            filepath = scraper.save()
    else:
        # Обычный парсер с прокси
        use_proxy = not args.no_proxy
        shuffle_pages = not args.no_shuffle

        scraper = KrishaKZScraper(
            city=args.city,
            delay_range=(args.delay_min, args.delay_max),
            use_proxy=use_proxy
        )

        print(f"  Режим: requests + прокси")
        print(f"  Прокси: {'включены' if use_proxy else 'отключены'}")
        print(f"  Случайный порядок: {'да' if shuffle_pages else 'нет'}")
        print(f"  Задержка: {args.delay_min}-{args.delay_max} сек")
        if args.resume:
            print("  Режим возобновления: да")

        scraper.scrape(max_pages=args.max_pages, resume=args.resume, shuffle_pages=shuffle_pages)
        filepath = scraper.save()

    if filepath:
        df = pd.read_csv(filepath)
        print(f"\nСтатистика:")
        print(f"  Всего записей: {len(df)}")
        print(f"  Колонки: {list(df.columns)}")


if __name__ == '__main__':
    main()
