"""
Business and economy data collector
"""
import logging
from typing import List
from datetime import datetime

from app.collectors.base_collector import BaseCollector
from app.models.data_models import InformationItem, CategoryType, ReliabilityLevel
from app.config import DATA_SOURCES

logger = logging.getLogger(__name__)


class BusinessCollector(BaseCollector):
    """
    Collector for business and economy information

    Sources:
    - TechCabal
    - Business Daily
    - Startup news
    - Investment announcements
    """

    def __init__(self, db):
        super().__init__(db, "Business Collector", "business")
        self.config = DATA_SOURCES.get("business", {})

    def collect(self) -> List[InformationItem]:
        """Collect business news"""
        all_items = []

        all_items.extend(self._collect_techcabal())

        return all_items

    def _collect_techcabal(self) -> List[InformationItem]:
        """
        Collect tech and startup news from TechCabal

        Returns:
            List of information items
        """
        items = []
        config = self.config.get("techcabal", {})

        if not config.get("enabled"):
            return items

        url = config.get("url")

        try:
            response = self._make_request(url)
            if not response:
                return items

            soup = self._parse_html(response.text)

            # Find articles
            articles = soup.find_all(['article', 'div'], class_=lambda x: x and (
                'article' in x.lower() or
                'post' in x.lower() or
                'story' in x.lower()
            ))

            for article in articles[:self.settings.max_items_per_source]:
                try:
                    # Extract title
                    title_elem = article.find(['h1', 'h2', 'h3'])
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    # Filter for Nairobi/Kenya related content
                    if not any(word in title.lower() for word in [
                        'nairobi', 'kenya', 'kenyan', 'east africa'
                    ]):
                        continue

                    # Extract link
                    link_elem = article.find('a', href=True)
                    if not link_elem:
                        continue

                    link = link_elem['href']
                    if link.startswith('/'):
                        from urllib.parse import urljoin
                        link = urljoin(url, link)

                    # Extract excerpt
                    excerpt_elem = article.find(['p', 'div'], class_=lambda x: x and (
                        'excerpt' in x.lower() or
                        'summary' in x.lower()
                    ))
                    excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                    # Extract image
                    image_url = None
                    img_elem = article.find('img', src=True)
                    if img_elem:
                        image_url = img_elem['src']
                        if image_url.startswith('/'):
                            from urllib.parse import urljoin
                            image_url = urljoin(url, image_url)

                    # Extract date
                    date_elem = article.find(['time', 'span'], class_=lambda x: x and 'date' in x.lower())
                    published_at = None
                    if date_elem and date_elem.get('datetime'):
                        try:
                            published_at = datetime.fromisoformat(
                                date_elem['datetime'].replace('Z', '+00:00')
                            )
                        except:
                            pass

                    # Extract tags
                    tags = ['business', 'tech', 'startup']
                    if 'investment' in title.lower() or 'funding' in excerpt.lower():
                        tags.append('investment')
                    if 'startup' in title.lower() or 'startup' in excerpt.lower():
                        tags.append('startup')

                    item_data = {
                        'title': title,
                        'summary': excerpt[:500] if excerpt else None,
                        'url': link,
                        'image_url': image_url,
                        'category': CategoryType.ECONOMY,
                        'published_at': published_at,
                        'reliability_level': ReliabilityLevel.HIGH,
                        'tags': tags,
                        'is_verified': True
                    }

                    item = self._save_item(item_data)
                    if item:
                        items.append(item)

                except Exception as e:
                    logger.error(f"Error processing TechCabal article: {e}")

        except Exception as e:
            logger.error(f"Error collecting from TechCabal: {e}")

        return items
