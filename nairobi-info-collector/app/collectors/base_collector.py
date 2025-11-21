"""
Base collector class for all data collection operations
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.data_models import (
    InformationItem, Source, CategoryType, ReliabilityLevel
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseCollector(ABC):
    """
    Base class for all data collectors

    Provides common functionality for:
    - HTTP requests with retries
    - Rate limiting
    - Caching
    - Data normalization
    - Error handling
    """

    def __init__(self, db: Session, source_name: str, source_type: str):
        """
        Initialize collector

        Args:
            db: Database session
            source_name: Name of the source
            source_type: Type of source (news, social_media, etc.)
        """
        self.db = db
        self.source_name = source_name
        self.source_type = source_type
        self.settings = settings

        # Get or create source in database
        self.source = self._get_or_create_source()

        # Request session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.user_agent
        })

        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 60 / settings.rate_limit_requests_per_minute

    def _get_or_create_source(self) -> Source:
        """Get or create source in database"""
        source = self.db.query(Source).filter(
            Source.name == self.source_name
        ).first()

        if not source:
            source = Source(
                name=self.source_name,
                source_type=self.source_type,
                reliability_score=0.5,
                is_active=True
            )
            self.db.add(source)
            self.db.commit()
            self.db.refresh(source)
            logger.info(f"Created new source: {self.source_name}")

        return source

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _make_request(
        self,
        url: str,
        method: str = "GET",
        **kwargs
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic and rate limiting

        Args:
            url: URL to request
            method: HTTP method
            **kwargs: Additional arguments for requests

        Returns:
            Response object or None if failed
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        try:
            logger.debug(f"Requesting: {url}")

            response = self.session.request(
                method=method,
                url=url,
                timeout=settings.request_timeout_seconds,
                **kwargs
            )
            response.raise_for_status()

            self.last_request_time = time.time()
            self.request_count += 1

            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    def _parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML content

        Args:
            html: HTML string

        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html, 'lxml')

    def _generate_item_hash(self, title: str, url: str) -> str:
        """
        Generate unique hash for an item

        Args:
            title: Item title
            url: Item URL

        Returns:
            Hash string
        """
        content = f"{title}{url}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _item_exists(self, title: str, url: str) -> bool:
        """
        Check if item already exists in database

        Args:
            title: Item title
            url: Item URL

        Returns:
            True if exists, False otherwise
        """
        existing = self.db.query(InformationItem).filter(
            InformationItem.title == title,
            InformationItem.url == url
        ).first()

        return existing is not None

    def _save_item(self, item_data: Dict[str, Any]) -> Optional[InformationItem]:
        """
        Save information item to database

        Args:
            item_data: Dictionary with item data

        Returns:
            Saved InformationItem or None
        """
        try:
            # Check if already exists
            if self._item_exists(item_data.get('title', ''), item_data.get('url', '')):
                logger.debug(f"Item already exists: {item_data.get('title')}")
                return None

            # Create item
            item = InformationItem(
                title=item_data.get('title'),
                summary=item_data.get('summary'),
                content=item_data.get('content'),
                category=item_data.get('category', CategoryType.NEWS),
                url=item_data.get('url'),
                image_url=item_data.get('image_url'),
                source_id=self.source.id,
                source_name=self.source_name,
                reliability_level=item_data.get(
                    'reliability_level',
                    ReliabilityLevel.MEDIUM
                ),
                published_at=item_data.get('published_at'),
                location=item_data.get('location'),
                coordinates=item_data.get('coordinates'),
                tags=item_data.get('tags', []),
                entities=item_data.get('entities', {}),
                is_verified=item_data.get('is_verified', False),
                is_alert=item_data.get('is_alert', False)
            )

            self.db.add(item)
            self.db.commit()
            self.db.refresh(item)

            logger.info(f"Saved item: {item.title[:50]}...")
            return item

        except Exception as e:
            logger.error(f"Error saving item: {e}")
            self.db.rollback()
            return None

    @abstractmethod
    def collect(self) -> List[InformationItem]:
        """
        Collect data from source

        Must be implemented by subclasses

        Returns:
            List of collected InformationItem objects
        """
        pass

    def run(self) -> Dict[str, Any]:
        """
        Run the collector

        Returns:
            Dictionary with collection results
        """
        start_time = time.time()
        logger.info(f"Starting collection from {self.source_name}")

        try:
            items = self.collect()

            elapsed = time.time() - start_time

            result = {
                'source': self.source_name,
                'items_collected': len(items),
                'elapsed_seconds': round(elapsed, 2),
                'success': True
            }

            logger.info(
                f"Collection completed: {len(items)} items in {elapsed:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Collection failed for {self.source_name}: {e}")

            return {
                'source': self.source_name,
                'items_collected': 0,
                'elapsed_seconds': 0,
                'success': False,
                'error': str(e)
            }
