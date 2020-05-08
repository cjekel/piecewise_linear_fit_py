try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata

try:
    __version__ = metadata.version('pwlf')
except Exception:
    __version__ = 'unknown'
