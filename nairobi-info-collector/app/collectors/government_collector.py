"""
Government and public services data collector
"""
import logging
from typing import List
from datetime import datetime

from app.collectors.base_collector import BaseCollector
from app.models.data_models import (
    InformationItem, Alert, CategoryType, ReliabilityLevel
)
from app.config import DATA_SOURCES

logger = logging.getLogger(__name__)


class GovernmentCollector(BaseCollector):
    """
    Collector for government and public service information

    Sources:
    - Nairobi City County
    - Kenya Open Data Portal
    - NTSA (traffic/road updates)
    - Public service announcements
    """

    def __init__(self, db):
        super().__init__(db, "Government Collector", "government")
        self.config = DATA_SOURCES.get("government", {})

    def collect(self) -> List[InformationItem]:
        """Collect government and public data"""
        all_items = []

        all_items.extend(self._collect_nairobi_county())
        all_items.extend(self._collect_open_data())

        return all_items

    def _collect_nairobi_county(self) -> List[InformationItem]:
        """
        Collect from Nairobi City County website

        Returns:
            List of information items
        """
        items = []
        config = self.config.get("nairobi_county", {})

        if not config.get("enabled"):
            return items

        url = config.get("url")

        try:
            response = self._make_request(url)
            if not response:
                return items

            soup = self._parse_html(response.text)

            # Find announcements and news
            announcements = soup.find_all(['div', 'article'], class_=lambda x: x and (
                'announcement' in x.lower() or
                'news' in x.lower() or
                'notice' in x.lower()
            ))

            for announcement in announcements[:self.settings.max_items_per_source]:
                try:
                    # Extract title
                    title_elem = announcement.find(['h1', 'h2', 'h3', 'h4'])
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    # Extract content
                    content_elem = announcement.find(['p', 'div'], class_=lambda x: x and 'content' in x.lower())
                    content = content_elem.get_text(strip=True) if content_elem else ""

                    # Extract link
                    link_elem = announcement.find('a', href=True)
                    link = link_elem['href'] if link_elem else url
                    if link.startswith('/'):
                        from urllib.parse import urljoin
                        link = urljoin(url, link)

                    # Check if it's an alert
                    is_alert = any(word in title.lower() for word in [
                        'alert', 'urgent', 'warning', 'closure', 'disruption'
                    ])

                    # Categorize
                    category = self._categorize_government_content(title, content)

                    item_data = {
                        'title': title,
                        'summary': content[:500] if content else None,
                        'content': content,
                        'url': link,
                        'category': category,
                        'reliability_level': ReliabilityLevel.VERIFIED,
                        'tags': ['government', 'nairobi county'],
                        'is_verified': True,
                        'is_alert': is_alert
                    }

                    item = self._save_item(item_data)
                    if item:
                        items.append(item)

                        # Create alert if necessary
                        if is_alert:
                            self._create_alert(title, content, link)

                except Exception as e:
                    logger.error(f"Error processing announcement: {e}")

        except Exception as e:
            logger.error(f"Error collecting from Nairobi County: {e}")

        return items

    def _collect_open_data(self) -> List[InformationItem]:
        """
        Collect from Kenya Open Data Portal

        Returns:
            List of information items
        """
        items = []
        config = self.config.get("kenya_open_data", {})

        if not config.get("enabled"):
            return items

        # Kenya Open Data typically provides datasets via API
        # This is a simplified example - you'd want to use their API properly

        logger.info("Kenya Open Data collection - placeholder for API integration")

        return items

    def _categorize_government_content(self, title: str, content: str) -> CategoryType:
        """Categorize government content"""
        text = f"{title} {content}".lower()

        if any(word in text for word in ['traffic', 'road', 'transport', 'closure']):
            return CategoryType.TRAVEL

        if any(word in text for word in ['event', 'ceremony', 'launch']):
            return CategoryType.EVENTS

        if any(word in text for word in ['business', 'permit', 'license', 'tender']):
            return CategoryType.ECONOMY

        return CategoryType.NEWS

    def _create_alert(self, title: str, message: str, url: str) -> None:
        """
        Create a public alert

        Args:
            title: Alert title
            message: Alert message
            url: Source URL
        """
        try:
            # Determine alert type and severity
            alert_type = "general"
            severity = "medium"

            text = f"{title} {message}".lower()

            if any(word in text for word in ['traffic', 'road']):
                alert_type = "traffic"

            if any(word in text for word in ['water', 'electricity', 'power']):
                alert_type = "utility"

            if any(word in text for word in ['security', 'safety']):
                alert_type = "security"

            if any(word in text for word in ['urgent', 'critical', 'emergency']):
                severity = "high"

            # Check if alert already exists
            existing = self.db.query(Alert).filter(
                Alert.title == title,
                Alert.is_active == True
            ).first()

            if not existing:
                alert = Alert(
                    title=title,
                    message=message,
                    alert_type=alert_type,
                    severity=severity,
                    source_name="Nairobi City County",
                    url=url,
                    is_active=True
                )

                self.db.add(alert)
                self.db.commit()

                logger.info(f"Created alert: {title}")

        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            self.db.rollback()
