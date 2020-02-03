from .pwlf import PiecewiseLinFit  # noqa F401
try:
    import cupy as cp  # noqa F401
    if cp.__version__ >= '7.0.0':
        from .pwlfcp import PiecewiseLinFitCp  # noqa F401
except ImportError:
    pass
from .version import __version__  # noqa F401
