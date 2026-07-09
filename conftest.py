# conftest.py
import sys
from pathlib import Path

root_path = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_path))