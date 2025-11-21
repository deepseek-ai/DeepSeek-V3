"""
Scheduled tasks for data collection
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from app.database import SessionLocal
from app.collectors import (
    NewsCollector,
    SocialMediaCollector,
    GovernmentCollector,
    TourismCollector,
    BusinessCollector
)
from app.processors import DataProcessor
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

scheduler = BackgroundScheduler()


def run_all_collectors():
    """
    Run all data collectors

    This function is executed on a schedule
    """
    logger.info("Starting scheduled data collection")
    start_time = datetime.utcnow()

    db = SessionLocal()
    results = []

    try:
        # Run collectors based on feature flags
        if settings.enable_news_collection:
            logger.info("Running news collector...")
            news_collector = NewsCollector(db, "all")
            result = news_collector.run()
            results.append(result)

        if settings.enable_social_media_collection:
            logger.info("Running social media collector...")
            social_collector = SocialMediaCollector(db, "all")
            result = social_collector.run()
            results.append(result)

        if settings.enable_government_collection:
            logger.info("Running government collector...")
            gov_collector = GovernmentCollector(db)
            result = gov_collector.run()
            results.append(result)

        if settings.enable_tourism_collection:
            logger.info("Running tourism collector...")
            tourism_collector = TourismCollector(db)
            result = tourism_collector.run()
            results.append(result)

        if settings.enable_business_collection:
            logger.info("Running business collector...")
            business_collector = BusinessCollector(db)
            result = business_collector.run()
            results.append(result)

        # Calculate totals
        total_items = sum(r.get('items_collected', 0) for r in results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Collection completed: {total_items} items from {successful} sources "
            f"in {elapsed:.2f}s ({failed} failed)"
        )

    except Exception as e:
        logger.error(f"Error in scheduled collection: {e}")

    finally:
        db.close()


def generate_brief():
    """
    Generate a new intelligence brief

    This function is executed on a schedule
    """
    logger.info("Generating intelligence brief")

    db = SessionLocal()

    try:
        processor = DataProcessor(db)
        brief = processor.generate_brief(hours=24)

        logger.info(
            f"Brief generated with {brief.total_items} items "
            f"from {brief.sources_count} sources"
        )

    except Exception as e:
        logger.error(f"Error generating brief: {e}")

    finally:
        db.close()


def start_scheduler():
    """
    Start the background scheduler with all tasks
    """
    logger.info("Starting task scheduler")

    # Schedule data collection
    scheduler.add_job(
        func=run_all_collectors,
        trigger=IntervalTrigger(seconds=settings.collection_interval_seconds),
        id='collect_data',
        name='Collect data from all sources',
        replace_existing=True
    )

    # Schedule brief generation (every 6 hours)
    scheduler.add_job(
        func=generate_brief,
        trigger=IntervalTrigger(hours=6),
        id='generate_brief',
        name='Generate intelligence brief',
        replace_existing=True
    )

    # Start the scheduler
    scheduler.start()

    logger.info(
        f"Scheduler started. Collection interval: {settings.collection_interval_seconds}s"
    )


def stop_scheduler():
    """Stop the background scheduler"""
    logger.info("Stopping task scheduler")
    scheduler.shutdown()
