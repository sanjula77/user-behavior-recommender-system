# Sample Data Generator

Generate realistic web behavior events for testing ML models.

## Features

- ✅ Generates realistic user sessions with page views and clicks
- ✅ Creates multiple users with varying behavior patterns
- ✅ Simulates realistic browsing patterns (navigation, clicks, time spent)
- ✅ Saves to MongoDB and/or CSV
- ✅ Configurable: number of users, days, sessions per user

## Usage

### Basic Usage (Save to CSV only)

```bash
# Activate virtual environment first
# Then run:
python ml/generate_sample_data.py --users 50 --days 7 --save-csv
```

### Save to MongoDB

```bash
python ml/generate_sample_data.py --users 100 --days 30 --save-mongo --save-csv
```

### Quick Test (Small dataset)

```bash
python ml/generate_sample_data.py --users 5 --days 1 --save-csv
```

## Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--users` | `-u` | 50 | Number of unique users |
| `--days` | `-d` | 30 | Number of days back to generate data |
| `--sessions-per-user` | `-s` | random (2-10) | Average sessions per user |
| `--save-mongo` | | False | Save events to MongoDB |
| `--save-csv` | | True | Save events to CSV file |
| `--no-csv` | | False | Don't save to CSV |

## Examples

### Generate sample data for testing (50 users, 7 days)
```bash
python ml/generate_sample_data.py --users 50 --days 7
```

### Generate large dataset for training (500 users, 90 days)
```bash
python ml/generate_sample_data.py --users 500 --days 90 --save-mongo
```

### Quick test with minimal data (10 users, 1 day)
```bash
python ml/generate_sample_data.py --users 10 --days 1 --sessions-per-user 2
```

## Generated Data Structure

Each event contains:
- `event_id`: Unique event identifier
- `user_id`: User identifier (e.g., "user-1234")
- `session_id`: Session identifier (e.g., "user-1234_sess_456")
- `event_type`: "page_view" or "click"
- `page_url`: Full URL of the page
- `timestamp`: ISO format timestamp
- `metadata`: Optional metadata (for clicks: tag, id, classes)

## Output Location

- **CSV**: `ml/outputs/events_export.csv`
- **MongoDB**: Events collection in configured database

## What Gets Generated

1. **Users**: Unique users with realistic IDs
2. **Sessions**: Multiple sessions per user spread over time
3. **Page Views**: Navigation through different pages
4. **Clicks**: Click events on buttons, links, etc.
5. **Timestamps**: Realistic timing between actions

## Usage in Model Training

After generating sample data, you can run model training:

```bash
# The training pipeline will automatically use events_export.csv or MongoDB
python -m pytest ml/phase2_hybrid/tests/test_phase2_pipeline.py
```

Or directly:
```bash
python ml/phase2_hybrid/run_train_all.py
```

## Notes

- MongoDB connection requires `.env` file with `MONGO_URI` and `MONGO_DB`
- CSV export works without any database connection
- Generated data is realistic but synthetic (for testing purposes)
- Timestamps are in UTC

