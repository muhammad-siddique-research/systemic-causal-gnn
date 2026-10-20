"""Backward-compatible entry point for the verified research benchmark.

Usage: python src/run_simulation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.research.pipeline import main

if __name__ == "__main__":
    main()
