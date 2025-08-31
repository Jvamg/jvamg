#!/usr/bin/env python3
"""
Convenience script to run the Streamlit interface
"""

import os
import sys
import subprocess

def main():
    """Run the Streamlit interface"""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_path = os.path.join(script_dir, "streamlit_interface.py")
    
    print("🎨 Starting Streamlit Interface...")
    print("📍 Interface will be available at: http://localhost:8501")
    print("ℹ️ Make sure the API server is running first!")
    print("─" * 50)
    
    try:
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", streamlit_path]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return 1
    except FileNotFoundError:
        print(f"❌ Streamlit not found. Install it with: pip install streamlit")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
