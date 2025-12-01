"""
Run the Streamlit dashboard
"""
import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent

# Run streamlit
if __name__ == "__main__":
    app_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    
    print("🚀 Starting Streamlit Dashboard...")
    print(f"📁 App path: {app_path}")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("\n⚠️  Make sure the FastAPI server is running on http://localhost:8000")
    print("   Run: python run_api.py\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

