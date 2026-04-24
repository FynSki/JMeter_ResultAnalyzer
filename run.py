#!/usr/bin/env python3
"""
JMeter Performance Analyzer — uruchamiacz
Uruchom: python run.py
Następnie otwórz: http://localhost:5050
"""
import subprocess
import sys
import os

def check_deps():
    try:
        import flask, pandas, numpy
    except ImportError as e:
        print(f"Brakuje zależności: {e}")
        print("Instaluję...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask', 'pandas', 'numpy'])

if __name__ == '__main__':
    check_deps()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 50)
    print("  JMeter Performance Analyzer")
    print("  Otwórz: http://localhost:5050")
    print("=" * 50)
    from app import app
    app.run(debug=False, port=5050, host='0.0.0.0')
