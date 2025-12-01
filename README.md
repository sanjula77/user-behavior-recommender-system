# User Behavior Analytics Platform

<div align="center">

**A comprehensive analytics platform for user behavior analysis, predictive modeling, and intelligent recommendations**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The User Behavior Analytics Platform is an end-to-end solution for analyzing user interactions, predicting behaviors, and providing intelligent recommendations. The system combines machine learning models, real-time analytics, and an interactive dashboard to deliver actionable insights.

### Key Capabilities

- **Data Processing**: Synthetic data generation, feature engineering, and data pipeline management
- **User Segmentation**: Unsupervised clustering to identify distinct user behavior patterns
- **Predictive Analytics**: ML models for conversion and bounce prediction
- **Recommendation System**: Hybrid recommender with content-based and collaborative filtering
- **Anomaly Detection**: Automated detection of unusual patterns and behaviors
- **Interactive Dashboard**: Real-time visualization and insights
- **RESTful API**: Comprehensive API for programmatic access

---

## ✨ Features

### 📊 Analytics & Insights
- **Daily & Weekly Trends**: Track user behavior patterns over time
- **Segment Analysis**: Understand behavior differences across user segments
- **Anomaly Detection**: Identify unusual patterns and outliers
- **LLM-Powered Summaries**: AI-generated insights using Google Gemini

### 🤖 Machine Learning
- **User Clustering**: K-means and hierarchical clustering for segmentation
- **Conversion Prediction**: Multiple models (RF, XGBoost, Logistic Regression)
- **Bounce Prediction**: Identify at-risk users
- **Model Evaluation**: Comprehensive metrics and validation

### 🧠 Recommendation System
- **Content-Based Filtering**: Semantic similarity using sentence transformers
- **Collaborative Filtering**: User-user similarity recommendations
- **Hybrid Recommendations**: Combined approach with fallback mechanisms
- **Admin Insights**: Actionable recommendations for site optimization

### 📈 Dashboard
- **Real-time Visualizations**: Interactive charts and graphs
- **User Recommendations**: Personalized page recommendations
- **Admin Dashboard**: Optimization suggestions and insights
- **Trend Analysis**: Historical and predictive analytics

---

## 🛠 Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.12+** - Core programming language
- **Pandas/NumPy** - Data manipulation and numerical computing
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Sentence Transformers** - Semantic embeddings for recommendations

### Frontend
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive data visualization

### Data & Storage
- **Parquet** - Efficient columnar data format
- **CSV/JSON** - Data interchange formats
- **Pickle** - Model serialization

### AI/ML
- **Google Gemini API** - LLM-powered insights
- **TensorFlow** - Deep learning framework (for embeddings)

---

## 📁 Project Structure

```
user_behavior_project/
├── data/                          # Data storage (gitignored)
│   ├── raw/                       # Raw input data
│   ├── processed/                 # Processed features
│   ├── insights/                  # Generated insights
│   ├── predictions/               # ML predictions
│   └── recommendations/           # Recommendation outputs
│
├── models/                        # Trained ML models (gitignored)
│   ├── *.pkl                     # Pickled model files
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_synthetic_data_generator.ipynb
│   ├── 02_data_preprocessing_feature_engineering.ipynb
│   ├── 03_user_clustering.ipynb
│   └── 04_supervised_ml_models.ipynb
│
├── src/                           # Source code
│   ├── api/                       # FastAPI application
│   │   ├── routers/               # API route handlers
│   │   ├── models/                # Pydantic schemas
│   │   └── utils/                 # API utilities
│   │
│   ├── dashboard/                 # Streamlit dashboard
│   │   ├── pages/                 # Dashboard pages
│   │   ├── components/            # Reusable components
│   │   └── services/              # API client
│   │
│   ├── insights/                  # Insight generation
│   │   ├── daily_trends.py
│   │   ├── weekly_trends.py
│   │   ├── segment_insights.py
│   │   ├── ml_insights.py
│   │   ├── anomaly_detection.py
│   │   └── llm_summaries.py
│   │
│   ├── recommendations/           # Recommendation system
│   │   ├── data_pipeline.py       # Data preparation
│   │   ├── content_based.py       # Content-based filtering
│   │   ├── collaborative.py       # Collaborative filtering
│   │   ├── hybrid.py              # Hybrid recommendations
│   │   ├── admin_insights.py      # Admin recommendations
│   │   ├── evaluation.py          # Model evaluation
│   │   └── runner.py              # Pipeline runner
│   │
│   └── config/                    # Configuration files
│
├── tests/                         # Test suite
│   ├── test_recommendations.py
│   ├── validate_recommendation_quality.py
│   ├── check_data_leakage.py
│   └── verify_no_leakage.py
│
├── requirements.txt               # Core dependencies
├── requirements_api.txt           # API dependencies
├── requirements_dashboard.txt     # Dashboard dependencies
├── run_api.py                     # API server launcher
├── run_dashboard.py               # Dashboard launcher
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.12+** ([Download](https://www.python.org/downloads/))
- **Git** (for cloning the repository)
- **Google Gemini API Key** (optional, for LLM summaries)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd user_behavior_project
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv ubehavior-env
ubehavior-env\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv ubehavior-env
source ubehavior-env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install API dependencies
pip install -r requirements_api.txt

# Install dashboard dependencies
pip install -r requirements_dashboard.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the project root (optional, for LLM summaries):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note**: LLM summaries are optional. The system works without this configuration.

---

## 🏃 Quick Start

### 1. Generate Synthetic Data

```bash
# Open and run Jupyter notebook
jupyter notebook notebooks/01_synthetic_data_generator.ipynb
```

### 2. Run Data Processing Pipeline

```bash
# Feature engineering and preprocessing
jupyter notebook notebooks/02_data_preprocessing_feature_engineering.ipynb

# User clustering
jupyter notebook notebooks/03_user_clustering.ipynb

# ML model training
jupyter notebook notebooks/04_supervised_ml_models.ipynb
```

### 3. Generate Insights

```bash
# Generate all insights
python -m src.insights.daily_trends
python -m src.insights.weekly_trends
python -m src.insights.segment_insights
python -m src.insights.ml_insights
python -m src.insights.anomaly_detection

# Generate LLM summary (requires API key)
python -m src.insights.llm_summaries
```

### 4. Run Recommendation Pipeline

```bash
# Generate recommendations
python -m src.recommendations.runner
```

### 5. Start Services

**Terminal 1 - Start API Server:**
```bash
python run_api.py
```
API will be available at: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Terminal 2 - Start Dashboard:**
```bash
python run_dashboard.py
```
Dashboard will be available at: `http://localhost:8501`

---

## 📖 Usage

### API Endpoints

#### Trends
- `GET /api/trends/daily` - Get daily trend data
- `GET /api/trends/weekly` - Get weekly trend data

#### Segments
- `GET /api/segments/trends` - Get segment trend data
- `GET /api/segments/list` - List all segments
- `GET /api/segments/summary` - Get segment summary

#### Predictions
- `GET /api/predictions/segments` - Get segment predictions
- `GET /api/predictions/high-value/{model}` - Get high-value users
- `GET /api/predictions/at-risk` - Get at-risk users

#### Recommendations
- `GET /api/recommendations/users/{user_id}` - Get user recommendations
- `GET /api/recommendations/pages/{page_id}` - Get similar pages
- `GET /api/recommendations/admin` - Get admin recommendations
- `GET /api/recommendations/metrics` - Get evaluation metrics

#### Anomalies
- `GET /api/anomalies` - Get anomaly data
- `GET /api/anomalies/recent` - Get recent anomalies

#### Summary
- `GET /api/summary/llm` - Get LLM-generated summary

**Full API Documentation**: Visit `http://localhost:8000/docs` when API is running.

### Dashboard Usage

1. **Overview Page**: High-level KPIs, AI insights, and daily trends
2. **Trends Page**: Historical trends and patterns
3. **Segments Page**: User segment analysis and behavior
4. **Predictions Page**: ML model predictions and probabilities
5. **Anomalies Page**: Detected anomalies and alerts
6. **Recommendations Page**: User and page recommendations

### Command Line Usage

#### Recommendation System

```bash
# Run full recommendation pipeline
python -m src.recommendations.runner

# Run individual components
python -m src.recommendations.data_pipeline
python -m src.recommendations.content_based
python -m src.recommendations.collaborative
python -m src.recommendations.hybrid
python -m src.recommendations.admin_insights
python -m src.recommendations.evaluation
```

---

## 🧪 Testing

### Run Test Suite

```bash
# Functional tests
python tests/test_recommendations.py

# Quality validation
python tests/validate_recommendation_quality.py

# Data leakage checks
python tests/check_data_leakage.py
python tests/verify_no_leakage.py
```

### Test Coverage

- ✅ Data pipeline outputs
- ✅ Recommendation quality
- ✅ Semantic accuracy
- ✅ Ranking optimization
- ✅ Data leakage detection
- ✅ API endpoint validation

---

## ⚙️ Configuration

### API Configuration

Default API settings in `src/api/main.py`:
- **Port**: 8000
- **Host**: 127.0.0.1
- **CORS**: Enabled for all origins (configure for production)

### Dashboard Configuration

Default dashboard settings in `src/dashboard/config.py`:
- **Port**: 8501
- **API URL**: http://localhost:8000

### Recommendation System Configuration

Tunable parameters in `src/recommendations/hybrid.py`:
- `content_weight`: Weight for content-based scores (default: 0.4)
- `collab_weight`: Weight for collaborative scores (default: 0.6)
- `fallback_weight`: Weight for fallback recommendations (default: 0.3)

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running scripts
- **Solution**: Ensure virtual environment is activated and dependencies are installed

**Issue**: API returns 500 errors
- **Solution**: Check that data files exist in `data/` directory. Run data pipeline first.

**Issue**: Dashboard shows "API Error"
- **Solution**: Ensure API server is running on port 8000

**Issue**: Sentence transformers import fails
- **Solution**: Install with `pip install sentence-transformers tf-keras`

**Issue**: LLM summaries fail
- **Solution**: Check `.env` file has valid `GEMINI_API_KEY`. LLM summaries are optional.

### Performance Optimization

- Use Parquet files for faster I/O
- Enable caching in API (`data_loader` has built-in caching)
- For large datasets, consider chunking data processing

---

## 📝 Data Flow

```
Raw Data
    ↓
[Data Preprocessing & Feature Engineering]
    ↓
Processed Features
    ↓
[User Clustering] → User Segments
    ↓
[ML Model Training] → Predictions
    ↓
[Insight Generation] → Analytics
    ↓
[Recommendation System] → Recommendations
    ↓
[API + Dashboard] → User Interface
```

---

## 🏗 Architecture

### System Components

1. **Data Layer**: Raw data storage and processing
2. **ML Layer**: Model training and inference
3. **Insight Layer**: Analytics and anomaly detection
4. **Recommendation Layer**: Hybrid recommendation engine
5. **API Layer**: RESTful API for data access
6. **Presentation Layer**: Interactive Streamlit dashboard

### Key Design Patterns

- **Modular Architecture**: Separate concerns (data, ML, API, dashboard)
- **Pipeline Pattern**: Data flows through standardized pipelines
- **Factory Pattern**: Model creation and evaluation
- **Repository Pattern**: Data access abstraction

---

## 🤝 Contributing

### Development Workflow

1. Create a feature branch
2. Make your changes
3. Add tests for new features
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to functions and classes
- Keep functions focused and small

---

## 📊 Performance Metrics

### Recommendation System

- **Hit Rate**: 29.0%
- **NDCG**: 0.173
- **Coverage**: 100% of users
- **Semantic Accuracy**: 100%

### Model Performance

- **Conversion Prediction**: Multiple models (RF, XGBoost, LR)
- **Bounce Prediction**: Random Forest classifier
- **Evaluation**: Cross-validation with temporal splits

---

## 📚 Additional Documentation

- [API README](src/api/README.md) - Detailed API documentation
- [Clustering Results](notebooks/03_USER_CLUSTERING_RESULTS_AND_USAGE.md)
- [ML Results](notebooks/04_SUPERVISED_ML_RESULTS_AND_USAGE.md)
- [Data Leakage Analysis](tests/DATA_LEAKAGE_ANALYSIS.md)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**User Behavior Analytics Team**

---

## 🙏 Acknowledgments

- **FastAPI** for the excellent web framework
- **Streamlit** for rapid dashboard development
- **scikit-learn** for ML algorithms
- **Google Gemini** for AI-powered insights
- **Sentence Transformers** for semantic embeddings

---

## 📞 Support

For issues, questions, or contributions, please open an issue in the repository.

---

<div align="center">

**Built with ❤️ using Python, FastAPI, and Streamlit**

⭐ Star this repo if you find it useful!

</div>

