# src/fpvgen/__init__.py

# Define package version
__version__ = "0.1.0"

# Import core components for easier access from `fpvgen`
from .flamelet_table_generator import FlameletTableGenerator

# Specify what gets imported with `from fpvgen import *`
__all__ = ["FlameletTableGenerator"]
