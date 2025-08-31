#!/usr/bin/env python3
"""
Convenience script to run the Crypto Analysis API server
"""

import os
import sys
import subprocess

def main():
    """Run the FastAPI server"""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    print("🚀 Starting Crypto Analysis API Server...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📖 API docs at: http://127.0.0.1:8000/docs")
    print("─" * 50)
    
    try:
        # Run the app.py serve command
        cmd = [sys.executable, app_path, "serve"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 API server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    except FileNotFoundError:
        print(f"❌ Python not found. Make sure Python is in your PATH.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
