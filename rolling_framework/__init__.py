"""
rolling_framework
~~~~~~~~~~~~~~~~~
High-level rolling-window prediction engine.

Version history
---------------
0.1.0   2025-06-21  Initial public release
"""

__version__ = "0.1.0"          # ← 버전 문자열

from .machine import Machine

__all__ = ["Machine", "__version__"]