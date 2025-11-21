#!/usr/bin/env python3
"""
Command-line interface for Nairobi Information Collector
"""
import argparse
import logging
from datetime import datetime

from app.database import SessionLocal, init_db
from app.collectors import (
    NewsCollector,
    SocialMediaCollector,
    GovernmentCollector,
    TourismCollector,
    BusinessCollector
)
from app.processors import DataProcessor
from app.scheduler.tasks import run_all_collectors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def collect_news(args):
    """Collect news from all sources"""
    logger.info("Collecting news...")
    db = SessionLocal()
    try:
        collector = NewsCollector(db, args.source or "all")
        result = collector.run()
        print(f"✓ Collected {result['items_collected']} items in {result['elapsed_seconds']}s")
    finally:
        db.close()


def collect_social(args):
    """Collect social media data"""
    logger.info("Collecting social media data...")
    db = SessionLocal()
    try:
        collector = SocialMediaCollector(db, args.platform or "all")
        result = collector.run()
        print(f"✓ Collected {result['items_collected']} items in {result['elapsed_seconds']}s")
    finally:
        db.close()


def collect_government(args):
    """Collect government data"""
    logger.info("Collecting government data...")
    db = SessionLocal()
    try:
        collector = GovernmentCollector(db)
        result = collector.run()
        print(f"✓ Collected {result['items_collected']} items in {result['elapsed_seconds']}s")
    finally:
        db.close()


def collect_tourism(args):
    """Collect tourism data"""
    logger.info("Collecting tourism data...")
    db = SessionLocal()
    try:
        collector = TourismCollector(db)
        result = collector.run()
        print(f"✓ Collected {result['items_collected']} items in {result['elapsed_seconds']}s")
    finally:
        db.close()


def collect_business(args):
    """Collect business data"""
    logger.info("Collecting business data...")
    db = SessionLocal()
    try:
        collector = BusinessCollector(db)
        result = collector.run()
        print(f"✓ Collected {result['items_collected']} items in {result['elapsed_seconds']}s")
    finally:
        db.close()


def collect_all(args):
    """Collect from all sources"""
    logger.info("Collecting from all sources...")
    run_all_collectors()
    print("✓ Collection completed")


def generate_brief(args):
    """Generate an intelligence brief"""
    logger.info(f"Generating brief for last {args.hours} hours...")
    db = SessionLocal()
    try:
        processor = DataProcessor(db)
        brief = processor.generate_brief(hours=args.hours)

        print(f"\n✓ Brief generated:")
        print(f"  - Period: {brief.period_start} to {brief.period_end}")
        print(f"  - Total items: {brief.total_items}")
        print(f"  - Sources: {brief.sources_count}")

        if args.output:
            with open(args.output, 'w') as f:
                f.write(brief.markdown_content)
            print(f"  - Saved to: {args.output}")
        else:
            print("\n" + brief.markdown_content)

    finally:
        db.close()


def setup_database(args):
    """Initialize the database"""
    logger.info("Initializing database...")
    try:
        init_db()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Nairobi Information Collector CLI'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Collect commands
    collect_parser = subparsers.add_parser('collect', help='Collect data from sources')
    collect_subparsers = collect_parser.add_subparsers(dest='source_type')

    # News
    news_parser = collect_subparsers.add_parser('news', help='Collect news')
    news_parser.add_argument('--source', help='Specific news source')
    news_parser.set_defaults(func=collect_news)

    # Social media
    social_parser = collect_subparsers.add_parser('social', help='Collect social media')
    social_parser.add_argument('--platform', help='Specific platform (twitter, instagram, etc.)')
    social_parser.set_defaults(func=collect_social)

    # Government
    gov_parser = collect_subparsers.add_parser('government', help='Collect government data')
    gov_parser.set_defaults(func=collect_government)

    # Tourism
    tourism_parser = collect_subparsers.add_parser('tourism', help='Collect tourism data')
    tourism_parser.set_defaults(func=collect_tourism)

    # Business
    business_parser = collect_subparsers.add_parser('business', help='Collect business data')
    business_parser.set_defaults(func=collect_business)

    # All
    all_parser = collect_subparsers.add_parser('all', help='Collect from all sources')
    all_parser.set_defaults(func=collect_all)

    # Brief generation
    brief_parser = subparsers.add_parser('brief', help='Generate intelligence brief')
    brief_parser.add_argument('--hours', type=int, default=24, help='Hours to include in brief')
    brief_parser.add_argument('--output', help='Output file for markdown')
    brief_parser.set_defaults(func=generate_brief)

    # Database setup
    db_parser = subparsers.add_parser('init-db', help='Initialize database')
    db_parser.set_defaults(func=setup_database)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
