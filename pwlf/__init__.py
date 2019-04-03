from .pwlf import PiecewiseLinFit
try:
    from .pwlftf import PiecewiseLinFitTF
except ImportError:
    class PiecewiseLinFitTF(object):

        def __init__(self, *args, **kwargs):
            ImportWarning('Possible error with importing TensorFlow')
import os

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
