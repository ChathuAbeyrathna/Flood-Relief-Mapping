"""
modules/flood_detection/__init__.py
Flood Detection Module — Team Trivia
University of Moratuwa · 2026

Exports the main processing function for use by the backend.
"""

from .processor import FloodDetectionProcessor
from .config import FloodDetectionConfig

__all__ = ['FloodDetectionProcessor', 'FloodDetectionConfig']