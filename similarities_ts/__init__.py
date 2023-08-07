"""
Similarities Time Series.
A simple package for dimensionality reduction of multivariate multiple time series, and clustering it.
"""

__author__ = """Marcin Dąbrowski"""
__email__ = 'mrcndabrowski@gmail.com'
__version__ = '0.1.0'

from .clustering import *
from .torch import *
from .utils import *

__all__ = ['clustering', 'torch', 'utils']
