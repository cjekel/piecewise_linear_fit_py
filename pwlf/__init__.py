from .pwlf import piecewise_lin_fit
from .pwlf import PiecewiseLinFit
import os

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
