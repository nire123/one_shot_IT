"""Put the src/ layout on sys.path so `import fbl` works under pytest
without an editable install."""
import os, sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
