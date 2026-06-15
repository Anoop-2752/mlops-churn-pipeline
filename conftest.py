import os
import sys

# Ensure the project root is on sys.path so `from src.X import Y` works
# regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))