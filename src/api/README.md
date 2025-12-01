# User Behavior Insights API

FastAPI-based REST API for accessing user behavior insights, trends, predictions, and anomalies.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Run the API

```bash
# From project root
python -m src.api.main
```

Or using uvicorn directly:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📋 API Endpoints

### Health & Info

- `GET /` - Root endpoint (health check)
- `GET /health` - Health check
- `GET /api/info` - API information and available endpoints

### Trends

- `GET /api/trends/daily` - Get daily trends data
  - Query params: `start_date`, `end_date`, `limit`
- `GET /api/trends/weekly` - Get weekly trends data
  - Query params: `limit`
- `GET /api/trends/daily/summary` - Get daily trends summary

### Segments

- `GET /api/segments/trends` - Get segment weekly trends
  - Query params: `segment`, `start_date`, `end_date`, `limit`
- `GET /api/segments/list` - Get list of available segments
- `GET /api/segments/summary` - Get segment summary
  - Query params: `segment`

### Predictions

- `GET /api/predictions/segments` - Get ML segment summary
- `GET /api/predictions/high-value/{model}` - Get high-value users
  - Path params: `model` (rf, xgb, or lr)
  - Query params: `limit`, `min_probability`
- `GET /api/predictions/at-risk` - Get at-risk users
  - Query params: `limit`, `min_probability`
- `GET /api/predictions/summary` - Get predictions summary

### Anomalies

- `GET /api/anomalies` - Get anomaly detection data
  - Query params: `start_date`, `end_date`, `limit`, `anomalies_only`
- `GET /api/anomalies/summary` - Get anomalies summary
- `GET /api/anomalies/recent` - Get recent anomalies
  - Query params: `limit`

### Summary

- `GET /api/summary/llm` - Get LLM-generated summary

## 📝 Example Requests

### Get Daily Trends

```bash
curl http://localhost:8000/api/trends/daily?limit=10
```

### Get Segment Trends

```bash
curl http://localhost:8000/api/segments/trends?segment=High-Value%20Buyers
```

### Get High-Value Users

```bash
curl http://localhost:8000/api/predictions/high-value/rf?limit=50
```

### Get Recent Anomalies

```bash
curl http://localhost:8000/api/anomalies/recent?limit=5
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in project root (optional):

```env
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### CORS Configuration

Update `src/api/main.py` to restrict CORS origins in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📊 Response Formats

All endpoints return JSON responses with consistent structure:

### Success Response

```json
{
  "data": [...],
  "total_records": 100,
  "date_range": {
    "start": "2025-06-03",
    "end": "2025-11-30"
  }
}
```

### Error Response

```json
{
  "error": "Not Found",
  "detail": "No data found",
  "timestamp": "2025-01-XX..."
}
```

## 🧪 Testing

### Using Swagger UI

1. Go to http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get daily trends
curl http://localhost:8000/api/trends/daily?limit=5

# Get segment trends
curl http://localhost:8000/api/segments/trends?segment=High-Value%20Buyers
```

### Using Python requests

```python
import requests

# Get daily trends
response = requests.get("http://localhost:8000/api/trends/daily", params={"limit": 10})
data = response.json()
print(data)
```

## 📦 Project Structure

```
src/api/
├── main.py              # FastAPI application
├── routers/
│   ├── trends.py        # Trends endpoints
│   ├── segments.py      # Segment endpoints
│   ├── predictions.py   # Prediction endpoints
│   ├── anomalies.py     # Anomaly endpoints
│   └── summary.py       # Summary endpoints
├── models/
│   └── schemas.py       # Pydantic models
└── utils/
    └── data_loader.py   # Data loading utilities
```

## 🔒 Best Practices

1. **Data Caching**: Data is cached for 5 minutes to improve performance
2. **Error Handling**: All endpoints have proper error handling
3. **Validation**: Request/response validation using Pydantic
4. **Documentation**: Auto-generated API documentation
5. **Type Safety**: Full type hints throughout
6. **Status Codes**: Proper HTTP status codes

## 🚀 Production Deployment

### Using Gunicorn + Uvicorn

```bash
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

