#!/usr/bin/env python3
"""
Startup script for the FastAPI backend server
"""
import uvicorn
import sys
import os
import signal
import time
from pathlib import Path

# Get port from environment variable (Render provides this)
PORT = int(os.getenv("PORT", 8001))


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down backend server...")
    sys.exit(0)


def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import fastapi
        import pandas
        import numpy
        import sklearn
        import yfinance
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r backend/requirements.txt")
        return False


def check_backend_structure():
    """Check if backend directory structure is correct"""
    backend_dir = Path("backend")
    required_files = [
        "backend/app/main.py",
        "backend/models/lasso_model.py",
        "backend/models/news_prediction.py",
        "backend/requirements.txt"
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing required file: {file_path}")
            return False

    print("✅ Backend structure verified")
    return True


def main():
    """Main function to start the backend server"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    print("🚀 Starting XAU/USD Real-time Data API...")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("backend").exists():
        print("❌ Error: Backend directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check backend structure
    if not check_backend_structure():
        sys.exit(1)

    print(f"📊 Backend will be available at: http://0.0.0.0:{PORT}")
    print(f"📡 WebSocket endpoint: ws://0.0.0.0:{PORT}/ws/xauusd")
    print(f"🌐 API docs: http://0.0.0.0:{PORT}/docs")
    print("🔄 Auto-reload: Disabled (Production Mode)")
    print("📝 Logs: Check terminal output")
    print(f"🔧 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"🔢 Port: {PORT}")
    print("\n" + "=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        # Ensure backend directory is on sys.path for imports
        project_root = Path(__file__).resolve().parent
        backend_path = project_root / "backend"

        # Add backend to sys.path for model imports
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=PORT,
            reload=False,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
