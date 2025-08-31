#!/usr/bin/env python3
"""
Test script to verify the setup and integration
"""

import subprocess
import sys
import time
import requests
import os

def check_dependencies():
    """Check if required packages are installed"""
    
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "streamlit",
        "requests",
        "plotly",
        "pandas"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install -r requirements_api.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True

def test_api_import():
    """Test if the app.py can be imported"""
    
    print("\n🔍 Testing app.py import...")
    
    try:
        # Change to the correct directory
        import app
        print("✅ app.py imports successfully")
        
        # Check if FastAPI app exists
        if hasattr(app, 'app'):
            print("✅ FastAPI app instance found")
        else:
            print("❌ FastAPI app instance not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_streamlit_import():
    """Test if streamlit_interface.py can be imported"""
    
    print("\n🔍 Testing streamlit_interface.py import...")
    
    try:
        import streamlit_interface
        print("✅ streamlit_interface.py imports successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🧪 Crypto Analysis API & Interface Setup Test")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Test 1: Dependencies
    deps_ok = check_dependencies()
    
    # Test 2: App import
    app_ok = test_api_import()
    
    # Test 3: Streamlit import
    streamlit_ok = test_streamlit_import()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"FastAPI App: {'✅ PASS' if app_ok else '❌ FAIL'}")
    print(f"Streamlit Interface: {'✅ PASS' if streamlit_ok else '❌ FAIL'}")
    
    if deps_ok and app_ok and streamlit_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n🚀 Ready to run:")
        print("1. Start API: python run_api.py")
        print("2. Start Streamlit: python run_streamlit.py")
        print("3. Access interface at: http://localhost:8501")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())
