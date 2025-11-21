#!/usr/bin/env python3
"""
Example usage of Nairobi Information Collector

This script demonstrates how to use the collector programmatically
"""

from app.database import SessionLocal, init_db
from app.collectors import NewsCollector
from app.processors import DataProcessor
from app.models.data_models import InformationItem, CategoryType
from datetime import datetime, timedelta


def example_1_collect_news():
    """Example 1: Collect news from all sources"""
    print("=" * 60)
    print("Example 1: Collecting News")
    print("=" * 60)

    db = SessionLocal()

    try:
        # Create news collector
        collector = NewsCollector(db, "all")

        # Run collection
        result = collector.run()

        print(f"\nCollection Results:")
        print(f"  - Items collected: {result['items_collected']}")
        print(f"  - Time taken: {result['elapsed_seconds']}s")
        print(f"  - Success: {result['success']}")

    finally:
        db.close()


def example_2_query_data():
    """Example 2: Query collected data"""
    print("\n" + "=" * 60)
    print("Example 2: Querying Data")
    print("=" * 60)

    db = SessionLocal()

    try:
        # Get total items
        total = db.query(InformationItem).count()
        print(f"\nTotal items in database: {total}")

        # Get items by category
        print("\nItems by category:")
        for category in CategoryType:
            count = db.query(InformationItem).filter(
                InformationItem.category == category
            ).count()
            print(f"  - {category.value}: {count}")

        # Get latest items
        print("\nLatest 5 items:")
        latest = db.query(InformationItem).order_by(
            InformationItem.collected_at.desc()
        ).limit(5).all()

        for item in latest:
            print(f"  - [{item.category.value}] {item.title[:60]}...")

    finally:
        db.close()


def example_3_generate_brief():
    """Example 3: Generate an intelligence brief"""
    print("\n" + "=" * 60)
    print("Example 3: Generating Intelligence Brief")
    print("=" * 60)

    db = SessionLocal()

    try:
        # Create processor
        processor = DataProcessor(db)

        # Generate brief for last 24 hours
        brief = processor.generate_brief(hours=24)

        print(f"\nBrief generated:")
        print(f"  - Period: {brief.period_start} to {brief.period_end}")
        print(f"  - Total items: {brief.total_items}")
        print(f"  - Sources: {brief.sources_count}")

        # Save to file
        output_file = f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(output_file, 'w') as f:
            f.write(brief.markdown_content)

        print(f"  - Saved to: {output_file}")

        # Print preview
        print("\nBrief preview:")
        print("-" * 60)
        lines = brief.markdown_content.split('\n')
        print('\n'.join(lines[:20]))
        print("...")
        print("-" * 60)

    finally:
        db.close()


def example_4_search():
    """Example 4: Search for specific information"""
    print("\n" + "=" * 60)
    print("Example 4: Searching Information")
    print("=" * 60)

    db = SessionLocal()

    try:
        # Search for items containing "restaurant"
        query = "restaurant"

        results = db.query(InformationItem).filter(
            (InformationItem.title.ilike(f"%{query}%")) |
            (InformationItem.summary.ilike(f"%{query}%"))
        ).limit(5).all()

        print(f"\nSearch results for '{query}':")
        print(f"Found {len(results)} items\n")

        for i, item in enumerate(results, 1):
            print(f"{i}. {item.title}")
            print(f"   Category: {item.category.value}")
            print(f"   Source: {item.source_name}")
            print(f"   URL: {item.url}")
            print()

    finally:
        db.close()


def example_5_api_usage():
    """Example 5: Using the REST API"""
    print("\n" + "=" * 60)
    print("Example 5: Using REST API")
    print("=" * 60)

    import requests

    base_url = "http://localhost:8000/api/v1"

    print("\nMake sure the API server is running!")
    print("Run: python -m app.main\n")

    try:
        # Get stats
        print("Getting statistics...")
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"  - Total items: {stats['total_items']}")
            print(f"  - Active alerts: {stats['active_alerts']}")
        else:
            print("  ✗ API not available")

        # Search
        print("\nSearching via API...")
        response = requests.get(
            f"{base_url}/search",
            params={"q": "nairobi", "limit": 3},
            timeout=5
        )
        if response.status_code == 200:
            results = response.json()
            print(f"  - Found {len(results)} results")

    except requests.exceptions.ConnectionError:
        print("  ✗ Could not connect to API server")
        print("    Start the server with: python -m app.main")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Nairobi Information Collector" + " " * 19 + "║")
    print("║" + " " * 19 + "Example Usage" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Initialize database if needed
    print("Initializing database...")
    try:
        init_db()
        print("✓ Database ready\n")
    except:
        pass

    # Run examples
    try:
        # Only run data query example if we have data
        db = SessionLocal()
        item_count = db.query(InformationItem).count()
        db.close()

        if item_count > 0:
            example_2_query_data()
            example_3_generate_brief()
            example_4_search()
        else:
            print("\nNo data in database. Running collection first...\n")
            example_1_collect_news()
            example_2_query_data()

        # API example (may fail if server not running)
        example_5_api_usage()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - README.md")
    print("  - QUICKSTART.md")
    print("  - API docs: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
