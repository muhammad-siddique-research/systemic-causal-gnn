"""Run held-out GAT/GRU prediction, without claiming causal identification."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.research.gnn_benchmark import main
if __name__ == "__main__":
    main()
