# =============================================================================
# run_test.py — kompatibilitas dengan versi lama.
# Setara dengan:  python -m fact_checker demo
# =============================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact_checker.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main(["demo", *sys.argv[1:]])
