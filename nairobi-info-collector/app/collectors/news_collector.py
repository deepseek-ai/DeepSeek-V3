"""
News collector for various Kenyan news sources
"""
import logging
from typing import List, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import feedparser

from app.collectors.base_collector import BaseCollector
from app.models.data_models import InformationItem, CategoryType, ReliabilityLevel
from app.config import DATA_SOURCES

logger = logging.getLogger(__name__)


class NewsCollector(BaseCollector):
    """
    Collector for news sources

    Supports:
    - Nation Africa
    - Standard Media
    - Citizen Digital
    - BBC Africa
    - Business Daily
    """

    def __init__(self, db, news_source: str = "all"):
        """
        Initialize news collector

        Args:
            db: Database session
            news_source: Specific news source or "all"
        """
        super().__init__(db, "News Collector", "news")
        self.news_source = news_source
        self.sources_config = DATA_SOURCES.get("news", {})

    def collect(self) -> List[InformationItem]:
        """Collect news from configured sources"""
        all_items = []

        if self.news_source == "all":
            sources = self.sources_config.items()
        else:
            source_config = self.sources_config.get(self.news_source)
            if source_config:
                sources = [(self.news_source, source_config)]
            else:
                logger.error(f"Unknown news source: {self.news_source}")
                return []

        for source_name, config in sources:
            if not config.get("enabled", False):
                logger.info(f"Skipping disabled source: {source_name}")
                continue

            logger.info(f"Collecting from {source_name}")

            try:
                items = self._collect_from_source(source_name, config)
                all_items.extend(items)
            except Exception as e:
                logger.error(f"Error collecting from {source_name}: {e}")

        return all_items

    def _collect_from_source(
        self,
        source_name: str,
        config: dict
    ) -> List[InformationItem]:
        """
        Collect from a specific news source

        Args:
            source_name: Name of the source
            config: Source configuration

        Returns:
            List of collected items
        """
        items = []
        url = config.get("url")
        reliability = config.get("reliability", 0.5)

        # Try RSS feed first
        rss_url = config.get("rss_url")
        if rss_url:
            items.extend(self._collect_from_rss(rss_url, source_name, reliability))

        # Try web scraping if RSS not available or failed
        if not items and url:
            items.extend(self._collect_from_web(url, source_name, reliability))

        return items

    def _collect_from_rss(
        self,
        rss_url: str,
        source_name: str,
        reliability: float
    ) -> List[InformationItem]:
        """
        Collect news from RSS feed

        Args:
            rss_url: RSS feed URL
            source_name: Name of the source
            reliability: Reliability score

        Returns:
            List of collected items
        """
        items = []

        try:
            feed = feedparser.parse(rss_url)

            for entry in feed.entries[:self.settings.max_items_per_source]:
                try:
                    # Parse published date
                    published_at = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_at = datetime(*entry.published_parsed[:6])

                    # Extract summary
                    summary = ""
                    if hasattr(entry, 'summary'):
                        summary = BeautifulSoup(entry.summary, 'html.parser').get_text()

                    # Determine category
                    category = self._categorize_content(
                        entry.title,
                        summary
                    )

                    item_data = {
                        'title': entry.title,
                        'summary': summary[:500] if summary else None,
                        'url': entry.link,
                        'category': category,
                        'published_at': published_at,
                        'reliability_level': self._reliability_to_enum(reliability),
                        'tags': self._extract_tags(entry.title, summary),
                        'is_verified': reliability >= 0.8
                    }

                    item = self._save_item(item_data)
                    if item:
                        items.append(item)

                except Exception as e:
                    logger.error(f"Error processing RSS entry: {e}")

        except Exception as e:
            logger.error(f"Error fetching RSS feed {rss_url}: {e}")

        return items

    def _collect_from_web(
        self,
        url: str,
        source_name: str,
        reliability: float
    ) -> List[InformationItem]:
        """
        Collect news by web scraping

        Args:
            url: Website URL
            source_name: Name of the source
            reliability: Reliability score

        Returns:
            List of collected items
        """
        items = []

        try:
            response = self._make_request(url)
            if not response:
                return items

            soup = self._parse_html(response.text)

            # Generic article extraction
            articles = soup.find_all(['article', 'div'], class_=lambda x: x and (
                'article' in x.lower() or
                'story' in x.lower() or
                'post' in x.lower()
            ))

            for article in articles[:self.settings.max_items_per_source]:
                try:
                    # Extract title
                    title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    # Extract link
                    link_elem = article.find('a', href=True)
                    if not link_elem:
                        continue

                    link = link_elem['href']
                    if link.startswith('/'):
                        from urllib.parse import urljoin
                        link = urljoin(url, link)

                    # Extract summary
                    summary_elem = article.find(['p', 'div'], class_=lambda x: x and (
                        'summary' in x.lower() or
                        'excerpt' in x.lower() or
                        'description' in x.lower()
                    ))
                    summary = summary_elem.get_text(strip=True) if summary_elem else ""

                    # Extract image
                    image_url = None
                    img_elem = article.find('img', src=True)
                    if img_elem:
                        image_url = img_elem['src']
                        if image_url.startswith('/'):
                            from urllib.parse import urljoin
                            image_url = urljoin(url, image_url)

                    # Categorize
                    category = self._categorize_content(title, summary)

                    item_data = {
                        'title': title,
                        'summary': summary[:500] if summary else None,
                        'url': link,
                        'image_url': image_url,
                        'category': category,
                        'reliability_level': self._reliability_to_enum(reliability),
                        'tags': self._extract_tags(title, summary),
                        'is_verified': reliability >= 0.8
                    }

                    item = self._save_item(item_data)
                    if item:
                        items.append(item)

                except Exception as e:
                    logger.error(f"Error processing article: {e}")

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

        return items

    def _categorize_content(self, title: str, content: str) -> CategoryType:
        """
        Categorize content based on title and content

        Args:
            title: Article title
            content: Article content

        Returns:
            CategoryType enum
        """
        text = f"{title} {content}".lower()

        # Breaking news
        if any(word in text for word in ['breaking', 'urgent', 'just in', 'alert']):
            return CategoryType.BREAKING

        # Events
        if any(word in text for word in ['event', 'concert', 'festival', 'exhibition']):
            return CategoryType.EVENTS

        # Economy/Business
        if any(word in text for word in ['economy', 'business', 'market', 'trade', 'investment']):
            return CategoryType.ECONOMY

        # Food/Nightlife
        if any(word in text for word in ['restaurant', 'food', 'dining', 'nightlife']):
            return CategoryType.FOOD

        # Travel/Transport
        if any(word in text for word in ['traffic', 'transport', 'road', 'airport']):
            return CategoryType.TRAVEL

        # Default to news
        return CategoryType.NEWS

    def _extract_tags(self, title: str, content: str) -> list:
        """
        Extract relevant tags from content

        Args:
            title: Article title
            content: Article content

        Returns:
            List of tags
        """
        tags = []
        text = f"{title} {content}".lower()

        # Common Nairobi locations
        locations = [
            'westlands', 'kileleshwa', 'karen', 'ngong', 'cbd',
            'kilimani', 'lavington', 'parklands', 'eastleigh'
        ]
        for loc in locations:
            if loc in text:
                tags.append(loc)

        # Topics
        topics = [
            'politics', 'sports', 'entertainment', 'technology',
            'health', 'education', 'crime', 'weather'
        ]
        for topic in topics:
            if topic in text:
                tags.append(topic)

        return list(set(tags))

    @staticmethod
    def _reliability_to_enum(score: float) -> ReliabilityLevel:
        """Convert reliability score to enum"""
        if score >= 0.9:
            return ReliabilityLevel.VERIFIED
        elif score >= 0.7:
            return ReliabilityLevel.HIGH
        elif score >= 0.5:
            return ReliabilityLevel.MEDIUM
        elif score >= 0.3:
            return ReliabilityLevel.LOW
        else:
            return ReliabilityLevel.UNVERIFIED
