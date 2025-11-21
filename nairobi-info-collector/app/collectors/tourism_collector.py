"""
Tourism and hospitality data collector
"""
import logging
from typing import List, Optional
from datetime import datetime

from app.collectors.base_collector import BaseCollector
from app.models.data_models import InformationItem, CategoryType, ReliabilityLevel
from app.config import DATA_SOURCES, get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TourismCollector(BaseCollector):
    """
    Collector for tourism and hospitality information

    Sources:
    - Google Maps/Places API (restaurants, hotels, attractions)
    - TripAdvisor
    - Tourism websites
    """

    def __init__(self, db):
        super().__init__(db, "Tourism Collector", "tourism")
        self.config = DATA_SOURCES.get("tourism", {})

    def collect(self) -> List[InformationItem]:
        """Collect tourism data"""
        all_items = []

        all_items.extend(self._collect_google_places())
        all_items.extend(self._collect_tripadvisor())

        return all_items

    def _collect_google_places(self) -> List[InformationItem]:
        """
        Collect new places and reviews from Google Maps

        Returns:
            List of information items
        """
        items = []

        if not settings.google_maps_api_key:
            logger.warning("Google Maps API key not configured")
            return items

        try:
            import googlemaps

            gmaps = googlemaps.Client(key=settings.google_maps_api_key)

            # Nairobi coordinates
            location = (-1.286389, 36.817223)

            # Search for different types of places
            place_types = [
                'restaurant',
                'cafe',
                'bar',
                'hotel',
                'tourist_attraction',
                'museum'
            ]

            for place_type in place_types:
                try:
                    # Search for recently added places
                    results = gmaps.places_nearby(
                        location=location,
                        radius=10000,  # 10km radius
                        type=place_type,
                        keyword='new OR opening'
                    )

                    for place in results.get('results', [])[:20]:
                        try:
                            place_id = place.get('place_id')

                            # Get place details
                            details = gmaps.place(
                                place_id=place_id,
                                fields=[
                                    'name', 'rating', 'formatted_address',
                                    'opening_hours', 'photos', 'reviews', 'website'
                                ]
                            ).get('result', {})

                            name = details.get('name', '')
                            rating = details.get('rating', 0)
                            address = details.get('formatted_address', '')
                            website = details.get('website')

                            # Get photo URL
                            image_url = None
                            photos = details.get('photos', [])
                            if photos:
                                photo_reference = photos[0].get('photo_reference')
                                image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={settings.google_maps_api_key}"

                            # Get recent review
                            reviews = details.get('reviews', [])
                            recent_review = reviews[0].get('text', '') if reviews else ''

                            # Determine category
                            category = CategoryType.PLACES
                            if place_type in ['restaurant', 'cafe']:
                                category = CategoryType.FOOD

                            item_data = {
                                'title': f"New {place_type.replace('_', ' ').title()}: {name}",
                                'summary': f"Rating: {rating}/5.0 - {address}",
                                'content': recent_review[:500] if recent_review else None,
                                'url': website or f"https://www.google.com/maps/place/?q=place_id:{place_id}",
                                'image_url': image_url,
                                'category': category,
                                'location': address,
                                'coordinates': {
                                    'lat': place.get('geometry', {}).get('location', {}).get('lat'),
                                    'lng': place.get('geometry', {}).get('location', {}).get('lng')
                                },
                                'reliability_level': ReliabilityLevel.HIGH,
                                'tags': [place_type, 'new opening'],
                                'is_verified': True
                            }

                            item = self._save_item(item_data)
                            if item:
                                items.append(item)

                        except Exception as e:
                            logger.error(f"Error processing place: {e}")

                except Exception as e:
                    logger.error(f"Error searching for {place_type}: {e}")

        except ImportError:
            logger.error("googlemaps not installed. Run: pip install googlemaps")
        except Exception as e:
            logger.error(f"Error in Google Places collection: {e}")

        return items

    def _collect_tripadvisor(self) -> List[InformationItem]:
        """
        Collect reviews and updates from TripAdvisor

        Note: TripAdvisor API access is limited. This is a web scraping approach.

        Returns:
            List of information items
        """
        items = []
        config = self.config.get("tripadvisor", {})

        if not config.get("enabled"):
            return items

        url = config.get("url")

        try:
            response = self._make_request(url)
            if not response:
                return items

            soup = self._parse_html(response.text)

            # Find attraction/restaurant listings
            listings = soup.find_all(['div'], class_=lambda x: x and (
                'listing' in x.lower() or
                'attraction' in x.lower()
            ))

            for listing in listings[:self.settings.max_items_per_source]:
                try:
                    # Extract name
                    name_elem = listing.find(['h2', 'h3'], class_=lambda x: x and 'title' in x.lower())
                    if not name_elem:
                        continue

                    name = name_elem.get_text(strip=True)

                    # Extract rating
                    rating_elem = listing.find(class_=lambda x: x and 'rating' in x.lower())
                    rating = rating_elem.get_text(strip=True) if rating_elem else ""

                    # Extract link
                    link_elem = listing.find('a', href=True)
                    link = link_elem['href'] if link_elem else ""
                    if link.startswith('/'):
                        link = f"https://www.tripadvisor.com{link}"

                    # Extract review snippet
                    review_elem = listing.find(class_=lambda x: x and 'review' in x.lower())
                    review = review_elem.get_text(strip=True) if review_elem else ""

                    item_data = {
                        'title': name,
                        'summary': f"{rating} - {review[:200]}",
                        'url': link,
                        'category': CategoryType.PLACES,
                        'reliability_level': ReliabilityLevel.MEDIUM,
                        'tags': ['tripadvisor', 'tourism'],
                        'is_verified': False
                    }

                    item = self._save_item(item_data)
                    if item:
                        items.append(item)

                except Exception as e:
                    logger.error(f"Error processing TripAdvisor listing: {e}")

        except Exception as e:
            logger.error(f"Error collecting from TripAdvisor: {e}")

        return items
