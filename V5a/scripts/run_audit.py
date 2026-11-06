#!/usr/bin/env python
"""Deprecated compatibility wrapper for the canonical reference audit."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_reference_audit import main as canonical_main


def main() -> int:
    warnings.warn(
        "scripts/run_audit.py is deprecated; use scripts/run_reference_audit.py",
        DeprecationWarning,
        stacklevel=2,
    )
    return canonical_main()


if __name__ == "__main__":
    raise SystemExit(main())
