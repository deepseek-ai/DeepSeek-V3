# Nairobi Information Collector

An advanced intelligence retrieval system designed to collect, verify, and synthesize comprehensive information about Nairobi, Kenya from multiple reliable digital sources.

## Features

- **Multi-Source Data Collection**: Gathers information from news sites, social media, government portals, tourism platforms, and business sources
- **Real-Time Updates**: Continuously collects and updates information
- **Structured Data**: Organizes information into categories (News, Events, Culture, Economy, etc.)
- **RESTful API**: Easy-to-use API endpoints for accessing collected data
- **Automated Scheduling**: Runs collectors at scheduled intervals
- **Data Verification**: Tracks sources and reliability levels
- **Categorization**: Automatically categorizes information by type

## Architecture

```
nairobi-info-collector/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── models/                 # Data models
│   ├── collectors/             # Source-specific data collectors
│   ├── processors/             # Data processing and NLP
│   ├── api/                    # API endpoints
│   ├── database/               # Database connection and setup
│   └── scheduler/              # Task scheduling
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── docker-compose.yml          # Docker setup
```

## Installation

### Prerequisites

- Python 3.9+
- PostgreSQL (or SQLite for development)
- Redis (for caching and task queue)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd nairobi-info-collector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python -m app.database.db init
```

6. Run the application:
```bash
uvicorn app.main:app --reload
```

### Using Docker

```bash
docker-compose up -d
```

## API Endpoints

### Get Latest Brief
```
GET /api/v1/brief/latest
```
Returns the most recent intelligence brief.

### Get Information by Category
```
GET /api/v1/info/{category}
```
Categories: `news`, `events`, `culture`, `economy`, `food`, `social`, `travel`, `places`, `community`

### Search Information
```
GET /api/v1/search?q={query}&category={category}&from={date}&to={date}
```

### Get Trending Topics
```
GET /api/v1/trending
```

### Get Real-Time Alerts
```
GET /api/v1/alerts
```

## Data Sources

### News & Media
- Nation Africa
- Standard Media
- Citizen Digital
- BBC Africa
- Business Daily Africa

### Government & Public
- Nairobi City County
- Kenya Open Data Portal
- NTSA, KCAA, KNBS

### Tourism
- TripAdvisor
- Google Maps
- Airbnb Experiences

### Social Media
- Twitter/X (via API)
- Instagram (via unofficial APIs)
- TikTok trending
- YouTube

### Business
- TechCabal
- StartUp Kenya
- LinkedIn insights

## Configuration

Edit `.env` file to configure:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/nairobi_info

# API Keys
TWITTER_API_KEY=your_key
GOOGLE_MAPS_API_KEY=your_key
OPENAI_API_KEY=your_key  # For NLP processing

# Collection Settings
COLLECTION_INTERVAL=300  # seconds
MAX_ITEMS_PER_SOURCE=100

# Cache
REDIS_URL=redis://localhost:6379
```

## Usage Examples

### Python Client

```python
import requests

# Get latest brief
response = requests.get("http://localhost:8000/api/v1/brief/latest")
brief = response.json()

# Search for specific information
response = requests.get(
    "http://localhost:8000/api/v1/search",
    params={"q": "restaurant opening", "category": "food"}
)
results = response.json()
```

### CLI

```bash
# Trigger manual collection
python -m app.collectors.run --source news

# Generate brief
python -m app.processors.generate_brief
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Ethical Considerations

- Respects robots.txt
- Implements rate limiting
- Uses official APIs where available
- Caches responses to minimize requests
- Only collects publicly available information

## License

MIT License

## Support

For issues and questions, please open a GitHub issue.
