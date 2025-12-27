"""Tests for house.kg scraper"""

import pytest
from src.scrapers.house_kg import HouseKGScraper


class TestHouseKGScraper:
    """Test suite for HouseKGScraper"""

    def test_init(self):
        """Test scraper initialization"""
        scraper = HouseKGScraper()
        assert scraper.BASE_URL == "https://www.house.kg"
        assert scraper.data == []

    def test_parse_price(self):
        """Test price parsing"""
        scraper = HouseKGScraper()

        assert scraper._parse_price("$130,000") == 130000
        assert scraper._parse_price("$ 50 000") == 50000
        assert scraper._parse_price("$1,500,000") == 1500000
        assert scraper._parse_price(None) is None
        assert scraper._parse_price("invalid") is None

    def test_parse_area(self):
        """Test area parsing"""
        scraper = HouseKGScraper()

        assert scraper._parse_area("90.8 м²") == 90.8
        assert scraper._parse_area("100м2") == 100.0
        assert scraper._parse_area("45.5 м") == 45.5
        assert scraper._parse_area(None) is None

    def test_parse_rooms(self):
        """Test rooms parsing"""
        scraper = HouseKGScraper()

        assert scraper._parse_rooms("3-комн.") == 3
        assert scraper._parse_rooms("2 комнаты") == 2
        assert scraper._parse_rooms("1") == 1
        assert scraper._parse_rooms(None) is None

    def test_parse_floor(self):
        """Test floor parsing"""
        scraper = HouseKGScraper()

        assert scraper._parse_floor("4 из 9") == (4, 9)
        assert scraper._parse_floor("1/5") == (1, 5)
        assert scraper._parse_floor("этаж 3 из 12") == (3, 12)
        assert scraper._parse_floor(None) == (None, None)


class TestScraperIntegration:
    """Integration tests (require network)"""

    @pytest.mark.slow
    def test_get_listing_urls(self):
        """Test URL collection from listing page"""
        scraper = HouseKGScraper()
        urls = scraper.get_listing_urls(page=1)

        assert isinstance(urls, list)
        assert len(urls) > 0
        assert all('/details/' in url for url in urls)
