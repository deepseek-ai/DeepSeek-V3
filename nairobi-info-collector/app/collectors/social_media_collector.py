"""
Social media collector for Twitter, Instagram, TikTok, etc.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from app.collectors.base_collector import BaseCollector
from app.models.data_models import (
    InformationItem, TrendingTopic, CategoryType, ReliabilityLevel
)
from app.config import DATA_SOURCES, get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SocialMediaCollector(BaseCollector):
    """
    Collector for social media platforms

    Supports:
    - Twitter/X (via API)
    - Instagram (via unofficial API)
    - TikTok trending
    - Facebook (via Graph API)
    """

    def __init__(self, db, platform: str = "all"):
        """
        Initialize social media collector

        Args:
            db: Database session
            platform: Specific platform or "all"
        """
        super().__init__(db, "Social Media Collector", "social_media")
        self.platform = platform
        self.config = DATA_SOURCES.get("social_media", {})

    def collect(self) -> List[InformationItem]:
        """Collect social media data"""
        all_items = []

        if self.platform == "all" or self.platform == "twitter":
            all_items.extend(self._collect_twitter())

        if self.platform == "all" or self.platform == "instagram":
            all_items.extend(self._collect_instagram())

        if self.platform == "all" or self.platform == "tiktok":
            all_items.extend(self._collect_tiktok())

        return all_items

    def _collect_twitter(self) -> List[InformationItem]:
        """
        Collect trending topics and posts from Twitter/X

        Returns:
            List of information items
        """
        items = []

        if not settings.twitter_bearer_token:
            logger.warning("Twitter API credentials not configured")
            return items

        try:
            import tweepy

            # Initialize Twitter API client
            client = tweepy.Client(bearer_token=settings.twitter_bearer_token)

            hashtags = self.config.get("twitter", {}).get("hashtags", [])

            for hashtag in hashtags:
                try:
                    # Search recent tweets
                    tweets = client.search_recent_tweets(
                        query=f"{hashtag} -is:retweet lang:en",
                        max_results=20,
                        tweet_fields=['created_at', 'public_metrics', 'entities']
                    )

                    if not tweets.data:
                        continue

                    for tweet in tweets.data:
                        # Skip if low engagement
                        metrics = tweet.public_metrics
                        engagement = (
                            metrics.get('like_count', 0) +
                            metrics.get('retweet_count', 0) * 2 +
                            metrics.get('reply_count', 0)
                        )

                        if engagement < 10:  # Minimum engagement threshold
                            continue

                        # Extract entities
                        entities = {}
                        if hasattr(tweet, 'entities'):
                            if 'hashtags' in tweet.entities:
                                entities['hashtags'] = [
                                    tag['tag'] for tag in tweet.entities['hashtags']
                                ]
                            if 'mentions' in tweet.entities:
                                entities['mentions'] = [
                                    m['username'] for m in tweet.entities['mentions']
                                ]

                        # Determine if trending
                        is_trending = engagement > 100

                        item_data = {
                            'title': f"Tweet: {tweet.text[:100]}...",
                            'summary': tweet.text,
                            'url': f"https://twitter.com/i/status/{tweet.id}",
                            'category': CategoryType.SOCIAL,
                            'published_at': tweet.created_at,
                            'reliability_level': ReliabilityLevel.MEDIUM,
                            'tags': [hashtag.replace('#', '')],
                            'entities': entities,
                            'is_featured': is_trending
                        }

                        item = self._save_item(item_data)
                        if item:
                            items.append(item)

                        # Track trending topic
                        if is_trending:
                            self._track_trending_topic(
                                hashtag,
                                'twitter',
                                engagement,
                                {'tweet_id': tweet.id, 'text': tweet.text}
                            )

                except Exception as e:
                    logger.error(f"Error collecting Twitter data for {hashtag}: {e}")

        except ImportError:
            logger.error("tweepy not installed. Run: pip install tweepy")
        except Exception as e:
            logger.error(f"Error in Twitter collection: {e}")

        return items

    def _collect_instagram(self) -> List[InformationItem]:
        """
        Collect trending posts from Instagram

        Returns:
            List of information items
        """
        items = []

        if not settings.instagram_username or not settings.instagram_password:
            logger.warning("Instagram credentials not configured")
            return items

        try:
            from instagrapi import Client

            client = Client()
            client.login(settings.instagram_username, settings.instagram_password)

            hashtags = self.config.get("instagram", {}).get("hashtags", [])

            for hashtag in hashtags:
                try:
                    # Get top posts for hashtag
                    medias = client.hashtag_medias_top(hashtag, amount=20)

                    for media in medias:
                        # Get media info
                        like_count = media.like_count
                        comment_count = media.comment_count

                        # Skip low engagement
                        if like_count < 50:
                            continue

                        item_data = {
                            'title': f"Instagram Post: {media.caption_text[:100] if media.caption_text else 'No caption'}",
                            'summary': media.caption_text[:500] if media.caption_text else "",
                            'url': f"https://www.instagram.com/p/{media.code}/",
                            'image_url': media.thumbnail_url,
                            'category': CategoryType.SOCIAL,
                            'published_at': media.taken_at,
                            'reliability_level': ReliabilityLevel.MEDIUM,
                            'tags': [hashtag],
                            'is_featured': like_count > 500
                        }

                        item = self._save_item(item_data)
                        if item:
                            items.append(item)

                except Exception as e:
                    logger.error(f"Error collecting Instagram data for {hashtag}: {e}")

        except ImportError:
            logger.error("instagrapi not installed. Run: pip install instagrapi")
        except Exception as e:
            logger.error(f"Error in Instagram collection: {e}")

        return items

    def _collect_tiktok(self) -> List[InformationItem]:
        """
        Collect trending videos from TikTok

        Returns:
            List of information items
        """
        items = []

        # Note: TikTok API access is limited. This is a placeholder for future implementation
        # You would need TikTok API credentials and use their official API

        logger.info("TikTok collection not yet implemented")

        return items

    def _track_trending_topic(
        self,
        topic: str,
        platform: str,
        mention_count: int,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Track a trending topic in the database

        Args:
            topic: The trending topic/hashtag
            platform: Social media platform
            mention_count: Number of mentions
            metadata: Additional metadata
        """
        try:
            # Check if topic already exists
            existing = self.db.query(TrendingTopic).filter(
                TrendingTopic.topic == topic,
                TrendingTopic.platform == platform
            ).first()

            if existing:
                # Update existing
                existing.mention_count += mention_count
                existing.last_updated = datetime.utcnow()
                if existing.related_content:
                    existing.related_content.append(metadata)
                else:
                    existing.related_content = [metadata]
            else:
                # Create new
                trending = TrendingTopic(
                    topic=topic,
                    platform=platform,
                    mention_count=mention_count,
                    related_content=[metadata]
                )
                self.db.add(trending)

            self.db.commit()

        except Exception as e:
            logger.error(f"Error tracking trending topic: {e}")
            self.db.rollback()

    def get_trending_topics(self, platform: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Get current trending topics

        Args:
            platform: Filter by platform
            limit: Maximum number of topics to return

        Returns:
            List of trending topics
        """
        query = self.db.query(TrendingTopic)

        if platform:
            query = query.filter(TrendingTopic.platform == platform)

        # Get topics from last 24 hours
        since = datetime.utcnow() - timedelta(days=1)
        query = query.filter(TrendingTopic.last_updated >= since)

        # Order by mention count
        topics = query.order_by(
            TrendingTopic.mention_count.desc()
        ).limit(limit).all()

        return [
            {
                'topic': t.topic,
                'platform': t.platform,
                'mention_count': t.mention_count,
                'first_seen': t.first_seen.isoformat() if t.first_seen else None,
                'last_updated': t.last_updated.isoformat() if t.last_updated else None
            }
            for t in topics
        ]
