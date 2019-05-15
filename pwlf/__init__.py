from .pwlf import PiecewiseLinFit  # noqa F401
try:
    import tensorflow as tf  # noqa F401
    from .pwlftf import PiecewiseLinFitTF  # noqa F401
except ImportError:
    pass
import os  # noqa F401

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
