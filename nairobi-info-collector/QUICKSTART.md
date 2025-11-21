# Quick Start Guide

Get the Nairobi Information Collector up and running in minutes!

## Prerequisites

- Python 3.9+ or Docker
- PostgreSQL (optional, SQLite works for development)
- API keys for various services (optional but recommended)

## Installation

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd nairobi-info-collector

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env

# Start with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f app
```

The API will be available at `http://localhost:8000`

### Option 2: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd nairobi-info-collector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP model
python -m spacy download en_core_web_sm

# Copy and configure environment
cp .env.example .env
nano .env

# Initialize database
python cli.py init-db

# Run the application
python -m app.main
```

## Configuration

### Required API Keys

Edit `.env` and add your API keys:

```env
# Social Media (optional but recommended)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
GOOGLE_MAPS_API_KEY=your_google_maps_key

# NLP Processing (optional)
OPENAI_API_KEY=your_openai_key

# Database (for production)
DATABASE_URL=postgresql://user:password@localhost:5432/nairobi_info
```

### Free Tier Options

You can start without API keys:
- News collection works without keys (web scraping)
- Government data works without keys
- Social media requires API keys

## Usage

### Web API

1. **Access the API documentation:**
   - Open `http://localhost:8000/docs` in your browser
   - Interactive Swagger UI with all endpoints

2. **Get the latest brief:**
   ```bash
   curl http://localhost:8000/api/v1/brief/latest
   ```

3. **Search for information:**
   ```bash
   curl "http://localhost:8000/api/v1/search?q=restaurant&category=food"
   ```

4. **Get trending topics:**
   ```bash
   curl http://localhost:8000/api/v1/trending
   ```

### Command Line Interface

```bash
# Collect news
python cli.py collect news

# Collect from all sources
python cli.py collect all

# Generate a brief
python cli.py brief --hours 24 --output brief.md

# Collect social media (requires API keys)
python cli.py collect social --platform twitter
```

## Testing

### Manual Collection Test

```bash
# Test news collection
python cli.py collect news

# Check the database
python -c "from app.database import SessionLocal; from app.models.data_models import InformationItem; db = SessionLocal(); print(f'Items collected: {db.query(InformationItem).count()}')"
```

### Generate a Brief

```bash
# Generate and save brief
python cli.py brief --output my_brief.md

# View the brief
cat my_brief.md
```

## Accessing the Data

### Via API

```python
import requests

# Get latest brief
response = requests.get("http://localhost:8000/api/v1/brief/latest")
brief = response.json()

# Search
response = requests.get(
    "http://localhost:8000/api/v1/search",
    params={"q": "nairobi", "limit": 10}
)
results = response.json()
```

### Via Database

```python
from app.database import SessionLocal
from app.models.data_models import InformationItem

db = SessionLocal()
items = db.query(InformationItem).limit(10).all()

for item in items:
    print(f"{item.title} - {item.category}")
```

## Automation

The application automatically:
- Collects data every 5 minutes (configurable)
- Generates briefs every 6 hours
- Updates trending topics in real-time

To change collection frequency:
```env
# In .env
COLLECTION_INTERVAL_SECONDS=300  # 5 minutes
```

## Troubleshooting

### Database connection errors
```bash
# Check PostgreSQL is running
docker-compose ps

# Reset database
docker-compose down -v
docker-compose up -d
```

### No data being collected
1. Check logs: `docker-compose logs -f app`
2. Verify network connectivity
3. Check API keys in `.env`
4. Try manual collection: `python cli.py collect news`

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. **Add API Keys:** Configure Twitter, Google Maps, etc. for more data sources
2. **Customize Sources:** Edit `app/config.py` to add/remove sources
3. **Set Up Monitoring:** Configure Sentry for error tracking
4. **Deploy to Production:** Use Docker Compose with proper environment variables

## API Documentation

Full API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Support

For issues and questions:
- Check logs: `tail -f logs/app.log`
- View API health: `http://localhost:8000/api/v1/health`
- See stats: `http://localhost:8000/api/v1/stats`
