"""Make `lawgraph` importable when pytest is started from anywhere (python -m pytest or plain pytest)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
