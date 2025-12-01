"""
Simple script to run the FastAPI server
Run from project root: python run_api.py
"""

import uvicorn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("=" * 60)
    print("Starting User Behavior Insights API")
    print("=" * 60)
    print("API will be available at: http://localhost:8000")
    print("Swagger UI: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

