#!/usr/bin/env python3
"""
BiscuitIQ — Biscuit Recommendation Website
Run this file to start the web server.
"""
import subprocess
import sys
import os

def install_deps():
    print("Installing Flask...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "--break-system-packages", "-q"])
    print("Flask ready!")

if __name__ == '__main__':
    try:
        import flask
    except ImportError:
        install_deps()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from app import app
    print("\n" + "="*50)
    print("  GreenPath is running....")
    print("  Open your browser: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
