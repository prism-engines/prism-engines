'''
PRISM Engine
'''

try:
    from .config import __version__, __author__, __project__
except ImportError:
    # When run as a script or imported directly (not as a package)
    __version__ = "0.1.0"
    __author__ = "PRISM Team"
    __project__ = "PRISM Engine"

__all__ = ['__version__', '__author__', '__project__']
