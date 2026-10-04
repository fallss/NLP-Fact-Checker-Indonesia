# =============================================================================
# main.py — kompatibilitas dengan versi lama.
# Setara dengan:  python -m fact_checker interactive
# =============================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact_checker.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main(["interactive", *sys.argv[1:]])
