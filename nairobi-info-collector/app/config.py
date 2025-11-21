"""
Configuration management for Nairobi Information Collector
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "Nairobi Information Collector"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./nairobi_info.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None

    # API Keys - News
    news_api_key: Optional[str] = None

    # API Keys - Social Media
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None

    instagram_username: Optional[str] = None
    instagram_password: Optional[str] = None

    # API Keys - Maps
    google_maps_api_key: Optional[str] = None
    foursquare_api_key: Optional[str] = None

    # API Keys - NLP
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Collection Settings
    collection_interval_seconds: int = 300
    max_items_per_source: int = 100
    request_timeout_seconds: int = 30
    max_retries: int = 3

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000

    # Scraping
    user_agent: str = "Mozilla/5.0 (compatible; NairobiInfoBot/1.0)"
    respect_robots_txt: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Data Processing
    enable_nlp_processing: bool = True
    enable_sentiment_analysis: bool = True
    enable_auto_categorization: bool = True
    min_reliability_score: float = 0.5

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/nairobi_collector.log"

    # Security
    secret_key: str = "change-this-in-production"
    api_key_header: str = "X-API-Key"
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"

    # Monitoring
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Feature Flags
    enable_social_media_collection: bool = True
    enable_news_collection: bool = True
    enable_government_collection: bool = True
    enable_tourism_collection: bool = True
    enable_business_collection: bool = True

    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_recipients: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse allowed origins into a list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def alert_recipients_list(self) -> List[str]:
        """Parse alert recipients into a list"""
        if not self.alert_email_recipients:
            return []
        return [email.strip() for email in self.alert_email_recipients.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Data source configurations
DATA_SOURCES = {
    "news": {
        "nation_africa": {
            "url": "https://nation.africa/kenya/counties/nairobi",
            "enabled": True,
            "reliability": 0.9
        },
        "standard_media": {
            "url": "https://www.standardmedia.co.ke/nairobi",
            "enabled": True,
            "reliability": 0.9
        },
        "citizen_digital": {
            "url": "https://www.citizen.digital/news",
            "enabled": True,
            "reliability": 0.85
        },
        "bbc_africa": {
            "url": "https://www.bbc.com/news/topics/c302m85q53mt",
            "enabled": True,
            "reliability": 0.95
        },
        "business_daily": {
            "url": "https://www.businessdailyafrica.com/bd/economy",
            "enabled": True,
            "reliability": 0.9
        }
    },
    "government": {
        "nairobi_county": {
            "url": "https://nairobi.go.ke",
            "enabled": True,
            "reliability": 1.0
        },
        "kenya_open_data": {
            "url": "https://www.opendata.go.ke",
            "enabled": True,
            "reliability": 1.0
        }
    },
    "tourism": {
        "tripadvisor": {
            "url": "https://www.tripadvisor.com/Tourism-g294207-Nairobi-Vacations.html",
            "enabled": True,
            "reliability": 0.8
        },
        "google_maps": {
            "api_url": "https://maps.googleapis.com/maps/api/place",
            "enabled": True,
            "reliability": 0.85
        }
    },
    "social_media": {
        "twitter": {
            "hashtags": [
                "#Nairobi", "#NairobiKenya", "#VisitNairobi",
                "#NairobiLife", "#254", "#KenyaNews"
            ],
            "enabled": True,
            "reliability": 0.6
        },
        "instagram": {
            "hashtags": [
                "nairobi", "nairobidiaries", "nairobikenya",
                "visitnairobi", "nairobilife"
            ],
            "enabled": True,
            "reliability": 0.6
        }
    },
    "business": {
        "techcabal": {
            "url": "https://techcabal.com/category/kenya/",
            "enabled": True,
            "reliability": 0.85
        }
    }
}

# Information categories
CATEGORIES = {
    "breaking": {
        "name": "Breaking Updates",
        "keywords": ["breaking", "urgent", "alert", "just in", "developing"],
        "priority": 1
    },
    "news": {
        "name": "City Life & Alerts",
        "keywords": ["news", "update", "announcement", "report"],
        "priority": 2
    },
    "events": {
        "name": "Culture & Events",
        "keywords": ["event", "concert", "festival", "exhibition", "show"],
        "priority": 3
    },
    "economy": {
        "name": "Business & Economy",
        "keywords": ["business", "economy", "startup", "investment", "market"],
        "priority": 4
    },
    "food": {
        "name": "Food & Nightlife",
        "keywords": ["restaurant", "food", "dining", "nightlife", "bar", "cafe"],
        "priority": 5
    },
    "social": {
        "name": "Social Media Trends",
        "keywords": ["trending", "viral", "hashtag"],
        "priority": 6
    },
    "travel": {
        "name": "Travel & Movement",
        "keywords": ["traffic", "transport", "airport", "road", "transit"],
        "priority": 7
    },
    "places": {
        "name": "New Places / Reviews",
        "keywords": ["opening", "new", "review", "rating"],
        "priority": 8
    },
    "community": {
        "name": "Community Stories",
        "keywords": ["community", "story", "people", "charity", "initiative"],
        "priority": 9
    }
}
