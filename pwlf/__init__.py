from .pwlf import PiecewiseLinFit  # noqa F401
try:
    import tensorflow as tf  # noqa F401
    if tf.__version__ < '2.0.0':
        from .pwlftf import PiecewiseLinFitTF  # noqa F401
except ImportError:
    pass
import pwlf.version
__version__ = pwlf.version.__version__
