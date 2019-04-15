from .pwlf import PiecewiseLinFit
try:
    import tensorflow as tf
    from .pwlftf import PiecewiseLinFitTF
except ImportError:
    print('Warning: Install tensorflow to have access to PiecewiseLinFitTF.')
import os

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
