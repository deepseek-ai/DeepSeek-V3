"""
SQLAlchemy models and Pydantic schemas for data structures
"""
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean,
    ForeignKey, JSON, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from enum import Enum

Base = declarative_base()


# Enums
class CategoryType(str, Enum):
    """Information category types"""
    BREAKING = "breaking"
    NEWS = "news"
    EVENTS = "events"
    ECONOMY = "economy"
    FOOD = "food"
    SOCIAL = "social"
    TRAVEL = "travel"
    PLACES = "places"
    COMMUNITY = "community"


class ReliabilityLevel(str, Enum):
    """Source reliability levels"""
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


# SQLAlchemy Models (Database Tables)

class Source(Base):
    """Data source information"""
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    url = Column(String(500))
    source_type = Column(String(50))  # news, social_media, government, etc.
    reliability_score = Column(Float, default=0.5)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    information_items = relationship("InformationItem", back_populates="source")


class InformationItem(Base):
    """Individual piece of information collected"""
    __tablename__ = "information_items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text)
    content = Column(Text)
    category = Column(SQLEnum(CategoryType), nullable=False)
    url = Column(String(1000))
    image_url = Column(String(1000))

    # Source information
    source_id = Column(Integer, ForeignKey("sources.id"))
    source_name = Column(String(255))
    reliability_level = Column(SQLEnum(ReliabilityLevel), default=ReliabilityLevel.MEDIUM)

    # Metadata
    published_at = Column(DateTime)
    collected_at = Column(DateTime, default=datetime.utcnow)
    location = Column(String(255))  # Specific location in Nairobi
    coordinates = Column(JSON)  # {"lat": -1.286389, "lng": 36.817223}

    # Processing
    sentiment_score = Column(Float)  # -1 to 1
    importance_score = Column(Float)  # 0 to 1
    tags = Column(JSON)  # List of tags
    entities = Column(JSON)  # Extracted entities (people, places, organizations)

    # Flags
    is_verified = Column(Boolean, default=False)
    is_featured = Column(Boolean, default=False)
    is_alert = Column(Boolean, default=False)

    # Relationships
    source = relationship("Source", back_populates="information_items")

    # Indexes
    __table_args__ = (
        {'extend_existing': True}
    )


class Alert(Base):
    """High-priority alerts and notifications"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    alert_type = Column(String(50))  # traffic, weather, security, utility, etc.
    severity = Column(String(20))  # low, medium, high, critical
    area_affected = Column(String(255))
    coordinates = Column(JSON)
    source_name = Column(String(255))
    url = Column(String(1000))

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    metadata = Column(JSON)


class TrendingTopic(Base):
    """Trending topics and hashtags"""
    __tablename__ = "trending_topics"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String(255), nullable=False)
    platform = Column(String(50))  # twitter, instagram, tiktok, etc.
    mention_count = Column(Integer, default=0)
    sentiment_score = Column(Float)

    first_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    related_content = Column(JSON)  # Sample posts/content
    metadata = Column(JSON)


class InformationBrief(Base):
    """Generated intelligence briefs"""
    __tablename__ = "information_briefs"

    id = Column(Integer, primary_key=True, index=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    period_start = Column(DateTime)
    period_end = Column(DateTime)

    # Brief sections (stored as JSON)
    breaking_updates = Column(JSON)
    city_life = Column(JSON)
    culture_events = Column(JSON)
    business_economy = Column(JSON)
    food_nightlife = Column(JSON)
    social_trends = Column(JSON)
    travel_movement = Column(JSON)
    new_places = Column(JSON)
    community_stories = Column(JSON)

    # Metadata
    total_items = Column(Integer)
    sources_count = Column(Integer)

    # Export
    markdown_content = Column(Text)
    html_content = Column(Text)


# Pydantic Schemas (API Request/Response)

class SourceSchema(BaseModel):
    """Source schema for API"""
    id: Optional[int] = None
    name: str
    url: Optional[str] = None
    source_type: str
    reliability_score: float = Field(ge=0, le=1)
    is_active: bool = True
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class InformationItemSchema(BaseModel):
    """Information item schema for API"""
    id: Optional[int] = None
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    category: CategoryType
    url: Optional[str] = None
    image_url: Optional[str] = None

    source_name: str
    reliability_level: ReliabilityLevel = ReliabilityLevel.MEDIUM

    published_at: Optional[datetime] = None
    collected_at: Optional[datetime] = None
    location: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None

    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    importance_score: Optional[float] = Field(None, ge=0, le=1)
    tags: Optional[List[str]] = []
    entities: Optional[Dict[str, List[str]]] = {}

    is_verified: bool = False
    is_featured: bool = False
    is_alert: bool = False

    class Config:
        from_attributes = True


class AlertSchema(BaseModel):
    """Alert schema for API"""
    id: Optional[int] = None
    title: str
    message: str
    alert_type: str
    severity: str
    area_affected: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    source_name: str
    url: Optional[str] = None

    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True

    metadata: Optional[Dict[str, Any]] = {}

    class Config:
        from_attributes = True


class TrendingTopicSchema(BaseModel):
    """Trending topic schema for API"""
    id: Optional[int] = None
    topic: str
    platform: str
    mention_count: int = 0
    sentiment_score: Optional[float] = None

    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    related_content: Optional[List[Dict[str, Any]]] = []
    metadata: Optional[Dict[str, Any]] = {}

    class Config:
        from_attributes = True


class BriefSection(BaseModel):
    """Schema for a brief section"""
    items: List[Dict[str, str]]


class InformationBriefSchema(BaseModel):
    """Information brief schema for API"""
    id: Optional[int] = None
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    breaking_updates: Optional[List[Dict[str, str]]] = []
    city_life: Optional[List[Dict[str, str]]] = []
    culture_events: Optional[List[Dict[str, str]]] = []
    business_economy: Optional[List[Dict[str, str]]] = []
    food_nightlife: Optional[List[Dict[str, str]]] = []
    social_trends: Optional[Dict[str, Any]] = {}
    travel_movement: Optional[Dict[str, Any]] = {}
    new_places: Optional[List[Dict[str, str]]] = []
    community_stories: Optional[List[Dict[str, str]]] = []

    total_items: int
    sources_count: int

    markdown_content: Optional[str] = None

    class Config:
        from_attributes = True


class SearchQuery(BaseModel):
    """Search query parameters"""
    q: str = Field(..., min_length=1)
    category: Optional[CategoryType] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    min_reliability: Optional[float] = Field(None, ge=0, le=1)
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class CollectionStats(BaseModel):
    """Statistics about data collection"""
    total_items: int
    items_by_category: Dict[str, int]
    items_by_source: Dict[str, int]
    latest_collection: Optional[datetime]
    active_alerts: int
    trending_topics_count: int
