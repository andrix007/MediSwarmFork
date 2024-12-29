"""
This package initializes the necessary modules and classes for the project.
"""

from .all_datasets import AllDatasetsShared

__all__ = [name for name in dir() if not name.startswith('_')]
