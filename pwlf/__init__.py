from .pwlf import PiecewiseLinFit
from .ga import genetic_algorithm
import os

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
