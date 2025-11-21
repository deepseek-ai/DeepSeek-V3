"""
API routes for Nairobi Information Collector
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.database import get_db
from app.models.data_models import (
    InformationItem, InformationBrief, Alert, TrendingTopic,
    InformationItemSchema, InformationBriefSchema, AlertSchema,
    TrendingTopicSchema, SearchQuery, CollectionStats,
    CategoryType
)
from app.processors.data_processor import DataProcessor

router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Nairobi Information Collector API",
        "version": "1.0.0",
        "endpoints": {
            "brief": "/api/v1/brief/latest",
            "info": "/api/v1/info/{category}",
            "search": "/api/v1/search",
            "trending": "/api/v1/trending",
            "alerts": "/api/v1/alerts",
            "stats": "/api/v1/stats"
        }
    }


@router.get("/brief/latest", response_model=InformationBriefSchema)
async def get_latest_brief(db: Session = Depends(get_db)):
    """
    Get the latest intelligence brief

    Returns:
        The most recent intelligence brief
    """
    brief = db.query(InformationBrief).order_by(
        InformationBrief.generated_at.desc()
    ).first()

    if not brief:
        # Generate a new brief if none exists
        processor = DataProcessor(db)
        brief = processor.generate_brief()

    return brief


@router.get("/brief/generate", response_model=InformationBriefSchema)
async def generate_new_brief(
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Generate a new intelligence brief

    Args:
        hours: Number of hours to include in the brief (default: 24)

    Returns:
        Newly generated brief
    """
    processor = DataProcessor(db)
    brief = processor.generate_brief(hours=hours)
    return brief


@router.get("/info/{category}", response_model=List[InformationItemSchema])
async def get_info_by_category(
    category: CategoryType,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Get information items by category

    Args:
        category: Category type (news, events, economy, etc.)
        limit: Maximum number of items to return
        offset: Number of items to skip
        hours: Look back this many hours (default: 24)

    Returns:
        List of information items
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    query = db.query(InformationItem).filter(
        InformationItem.category == category,
        InformationItem.collected_at >= since
    )

    items = query.order_by(
        InformationItem.collected_at.desc()
    ).offset(offset).limit(limit).all()

    return items


@router.get("/info/all", response_model=List[InformationItemSchema])
async def get_all_info(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    hours: int = Query(24, ge=1, le=168),
    min_reliability: Optional[float] = Query(None, ge=0, le=1),
    db: Session = Depends(get_db)
):
    """
    Get all information items

    Args:
        limit: Maximum number of items to return
        offset: Number of items to skip
        hours: Look back this many hours
        min_reliability: Minimum reliability score

    Returns:
        List of information items
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    query = db.query(InformationItem).filter(
        InformationItem.collected_at >= since
    )

    if min_reliability is not None:
        # Filter by reliability (would need to add mapping)
        pass

    items = query.order_by(
        InformationItem.collected_at.desc()
    ).offset(offset).limit(limit).all()

    return items


@router.get("/search", response_model=List[InformationItemSchema])
async def search_info(
    q: str = Query(..., min_length=1),
    category: Optional[CategoryType] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Search information items

    Args:
        q: Search query
        category: Filter by category
        from_date: Start date
        to_date: End date
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        List of matching information items
    """
    query = db.query(InformationItem)

    # Text search in title and summary
    search_filter = (
        InformationItem.title.ilike(f"%{q}%") |
        InformationItem.summary.ilike(f"%{q}%")
    )
    query = query.filter(search_filter)

    # Category filter
    if category:
        query = query.filter(InformationItem.category == category)

    # Date filters
    if from_date:
        query = query.filter(InformationItem.collected_at >= from_date)
    if to_date:
        query = query.filter(InformationItem.collected_at <= to_date)

    # Order and paginate
    items = query.order_by(
        InformationItem.collected_at.desc()
    ).offset(offset).limit(limit).all()

    return items


@router.get("/trending", response_model=List[TrendingTopicSchema])
async def get_trending(
    platform: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Get trending topics

    Args:
        platform: Filter by platform (twitter, instagram, etc.)
        limit: Maximum number of topics
        hours: Look back this many hours

    Returns:
        List of trending topics
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    query = db.query(TrendingTopic).filter(
        TrendingTopic.last_updated >= since
    )

    if platform:
        query = query.filter(TrendingTopic.platform == platform)

    topics = query.order_by(
        TrendingTopic.mention_count.desc()
    ).limit(limit).all()

    return topics


@router.get("/alerts", response_model=List[AlertSchema])
async def get_alerts(
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get current alerts

    Args:
        alert_type: Filter by type (traffic, weather, security, etc.)
        severity: Filter by severity (low, medium, high, critical)
        active_only: Only return active alerts

    Returns:
        List of alerts
    """
    query = db.query(Alert)

    if active_only:
        query = query.filter(Alert.is_active == True)

    if alert_type:
        query = query.filter(Alert.alert_type == alert_type)

    if severity:
        query = query.filter(Alert.severity == severity)

    alerts = query.order_by(Alert.created_at.desc()).all()

    return alerts


@router.get("/stats", response_model=CollectionStats)
async def get_stats(db: Session = Depends(get_db)):
    """
    Get collection statistics

    Returns:
        Statistics about collected data
    """
    # Total items
    total_items = db.query(InformationItem).count()

    # Items by category
    items_by_category = {}
    for category in CategoryType:
        count = db.query(InformationItem).filter(
            InformationItem.category == category
        ).count()
        items_by_category[category.value] = count

    # Items by source
    from sqlalchemy import func
    items_by_source_query = db.query(
        InformationItem.source_name,
        func.count(InformationItem.id)
    ).group_by(InformationItem.source_name).all()

    items_by_source = {
        source: count for source, count in items_by_source_query
    }

    # Latest collection
    latest = db.query(InformationItem).order_by(
        InformationItem.collected_at.desc()
    ).first()

    latest_collection = latest.collected_at if latest else None

    # Active alerts
    active_alerts = db.query(Alert).filter(Alert.is_active == True).count()

    # Trending topics
    trending_count = db.query(TrendingTopic).count()

    return CollectionStats(
        total_items=total_items,
        items_by_category=items_by_category,
        items_by_source=items_by_source,
        latest_collection=latest_collection,
        active_alerts=active_alerts,
        trending_topics_count=trending_count
    )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
