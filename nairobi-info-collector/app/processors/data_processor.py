"""
Data processing and brief generation
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.data_models import (
    InformationItem, InformationBrief, TrendingTopic,
    Alert, CategoryType
)
from app.config import CATEGORIES

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes collected data and generates intelligence briefs
    """

    def __init__(self, db: Session):
        """
        Initialize data processor

        Args:
            db: Database session
        """
        self.db = db

    def generate_brief(self, hours: int = 24) -> InformationBrief:
        """
        Generate an intelligence brief for a time period

        Args:
            hours: Number of hours to include in the brief

        Returns:
            Generated InformationBrief
        """
        logger.info(f"Generating intelligence brief for last {hours} hours")

        period_end = datetime.utcnow()
        period_start = period_end - timedelta(hours=hours)

        # Get items from the period
        items = self.db.query(InformationItem).filter(
            InformationItem.collected_at >= period_start,
            InformationItem.collected_at <= period_end
        ).all()

        # Organize by category
        breaking_updates = self._get_items_by_category(items, CategoryType.BREAKING)
        city_life = self._get_items_by_category(items, CategoryType.NEWS)
        culture_events = self._get_items_by_category(items, CategoryType.EVENTS)
        business_economy = self._get_items_by_category(items, CategoryType.ECONOMY)
        food_nightlife = self._get_items_by_category(items, CategoryType.FOOD)
        new_places = self._get_items_by_category(items, CategoryType.PLACES)
        community_stories = self._get_items_by_category(items, CategoryType.COMMUNITY)

        # Get social media trends
        social_trends = self._get_social_trends(period_start)

        # Get travel/movement info
        travel_movement = self._get_travel_info(items, period_start)

        # Count unique sources
        sources = set(item.source_name for item in items if item.source_name)
        sources_count = len(sources)

        # Generate markdown content
        markdown = self._generate_markdown(
            period_start,
            period_end,
            breaking_updates,
            city_life,
            culture_events,
            business_economy,
            food_nightlife,
            social_trends,
            travel_movement,
            new_places,
            community_stories
        )

        # Create brief
        brief = InformationBrief(
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            breaking_updates=breaking_updates,
            city_life=city_life,
            culture_events=culture_events,
            business_economy=business_economy,
            food_nightlife=food_nightlife,
            social_trends=social_trends,
            travel_movement=travel_movement,
            new_places=new_places,
            community_stories=community_stories,
            total_items=len(items),
            sources_count=sources_count,
            markdown_content=markdown
        )

        self.db.add(brief)
        self.db.commit()
        self.db.refresh(brief)

        logger.info(f"Generated brief with {len(items)} items from {sources_count} sources")

        return brief

    def _get_items_by_category(
        self,
        items: List[InformationItem],
        category: CategoryType,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get items for a specific category

        Args:
            items: List of all items
            category: Category to filter by
            limit: Maximum number of items

        Returns:
            List of item dictionaries
        """
        category_items = [
            item for item in items
            if item.category == category
        ]

        # Sort by importance/recency
        category_items.sort(
            key=lambda x: (
                x.importance_score or 0,
                x.collected_at
            ),
            reverse=True
        )

        return [
            {
                'title': item.title,
                'summary': item.summary or '',
                'source': item.source_name or '',
                'url': item.url or '',
                'date': item.published_at.isoformat() if item.published_at else item.collected_at.isoformat()
            }
            for item in category_items[:limit]
        ]

    def _get_social_trends(self, since: datetime) -> Dict[str, Any]:
        """
        Get social media trends

        Args:
            since: Start date

        Returns:
            Dictionary with social trends
        """
        # Get trending topics
        topics = self.db.query(TrendingTopic).filter(
            TrendingTopic.last_updated >= since
        ).order_by(
            TrendingTopic.mention_count.desc()
        ).limit(10).all()

        # Get top social posts
        social_items = self.db.query(InformationItem).filter(
            InformationItem.category == CategoryType.SOCIAL,
            InformationItem.collected_at >= since
        ).order_by(
            InformationItem.importance_score.desc()
        ).limit(5).all()

        trending_hashtags = [
            {
                'topic': t.topic,
                'platform': t.platform,
                'mentions': t.mention_count
            }
            for t in topics
        ]

        viral_content = [
            {
                'title': item.title,
                'summary': item.summary or '',
                'url': item.url or ''
            }
            for item in social_items
        ]

        return {
            'trending_hashtags': trending_hashtags,
            'viral_content': viral_content
        }

    def _get_travel_info(
        self,
        items: List[InformationItem],
        since: datetime
    ) -> Dict[str, Any]:
        """
        Get travel and movement information

        Args:
            items: All items
            since: Start date

        Returns:
            Dictionary with travel info
        """
        travel_items = [
            item for item in items
            if item.category == CategoryType.TRAVEL
        ]

        # Get active alerts related to travel
        alerts = self.db.query(Alert).filter(
            Alert.is_active == True,
            Alert.alert_type.in_(['traffic', 'transport', 'road']),
            Alert.created_at >= since
        ).all()

        traffic_alerts = [
            {
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity,
                'area': alert.area_affected or ''
            }
            for alert in alerts
        ]

        transit_info = [
            {
                'title': item.title,
                'summary': item.summary or '',
                'source': item.source_name or ''
            }
            for item in travel_items[:5]
        ]

        return {
            'traffic_alerts': traffic_alerts,
            'transit_information': transit_info
        }

    def _generate_markdown(
        self,
        start: datetime,
        end: datetime,
        breaking: List[Dict],
        city_life: List[Dict],
        culture: List[Dict],
        economy: List[Dict],
        food: List[Dict],
        social: Dict,
        travel: Dict,
        places: List[Dict],
        community: List[Dict]
    ) -> str:
        """
        Generate markdown formatted brief

        Returns:
            Markdown string
        """
        md = f"# Nairobi Intelligence Brief\n\n"
        md += f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        md += f"**Period:** {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}\n\n"
        md += "---\n\n"

        # Breaking Updates
        if breaking:
            md += "## 🚨 Breaking Updates\n\n"
            for item in breaking:
                md += f"- **{item['title']}** — {item['summary']} — [{item['source']}]({item['url']})\n"
            md += "\n"

        # City Life & Alerts
        if city_life:
            md += "## 🏙️ City Life & Alerts\n\n"
            for item in city_life:
                md += f"- **{item['title']}** — {item['summary']} — [{item['source']}]({item['url']})\n"
            md += "\n"

        # Culture & Events
        if culture:
            md += "## 🎭 Culture & Events\n\n"
            for item in culture:
                md += f"- **{item['title']}** — {item['summary']} — [{item['source']}]({item['url']})\n"
            md += "\n"

        # Business & Economy
        if economy:
            md += "## 💼 Business & Economy\n\n"
            for item in economy:
                md += f"- **{item['title']}** — {item['summary']} — [{item['source']}]({item['url']})\n"
            md += "\n"

        # Food & Nightlife
        if food:
            md += "## 🍽️ Food & Nightlife\n\n"
            for item in food:
                md += f"- **{item['title']}** — {item['summary']} — [{item['source']}]({item['url']})\n"
            md += "\n"

        # Social Media Trends
        if social.get('trending_hashtags') or social.get('viral_content'):
            md += "## 📱 Social Media Trends\n\n"

            if social.get('trending_hashtags'):
                md += "### Trending Hashtags:\n"
                for tag in social['trending_hashtags']:
                    md += f"- **{tag['topic']}** ({tag['platform']}) — {tag['mentions']} mentions\n"
                md += "\n"

            if social.get('viral_content'):
                md += "### Viral Content:\n"
                for content in social['viral_content']:
                    md += f"- [{content['title']}]({content['url']}) — {content['summary']}\n"
                md += "\n"

        # Travel & Movement
        if travel.get('traffic_alerts') or travel.get('transit_information'):
            md += "## 🚗 Travel & Movement\n\n"

            if travel.get('traffic_alerts'):
                md += "### Traffic Alerts:\n"
                for alert in travel['traffic_alerts']:
                    md += f"- **{alert['title']}** ({alert['severity']}) — {alert['message']}\n"
                md += "\n"

            if travel.get('transit_information'):
                md += "### Transit Information:\n"
                for info in travel['transit_information']:
                    md += f"- {info['title']} — {info['summary']}\n"
                md += "\n"

        # New Places / Reviews
        if places:
            md += "## 📍 New Places / Reviews\n\n"
            for item in places:
                md += f"- **{item['title']}** — {item['summary']} — [{item['source']}]({item['url']})\n"
            md += "\n"

        # Community Stories
        if community:
            md += "## 👥 Community Stories\n\n"
            for item in community:
                md += f"- **{item['title']}** — {item['summary']} — [{item['source']}]({item['url']})\n"
            md += "\n"

        md += "---\n\n"
        md += "*End of brief.*\n"

        return md
