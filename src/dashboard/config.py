import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# UI Configuration
APP_TITLE = "User Behavior Insights"
APP_ICON = "📊"

# Chart Colors
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#262730",
    "accent": "#FF4B4B",
    "success": "#00CC96",
    "warning": "#FFA15A",
    "danger": "#EF553B",
    "info": "#636EFA",
}

# Date Formats
DATE_FORMAT = "%Y-%m-%d"
