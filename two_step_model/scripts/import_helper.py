import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

def add_models():
    """Add the model directory to sys.path"""
    if ROOT_DIR not in [os.path.abspath(p) for p in sys.path]:
        sys.path.insert(1, ROOT_DIR)