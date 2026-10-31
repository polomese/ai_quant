"""Pre-commit guard: refuse to commit licensed data or oversized files.

Blocks any staged file that lives under ``data/``, has a data-file extension
(.parquet, .duckdb, .csv.zip) or is a ``.env`` file, and any file larger than 5 MB.
Usable standalone: ``python scripts/check_no_data.py $(git ls-files)``.
"""

from __future__ import annotations

import os
import sys
from pathlib import PurePosixPath

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = (".parquet", ".duckdb", ".duckdb.wal", ".csv.zip")


def problems(paths: list[str]) -> list[str]:
    out = []
    for raw in paths:
        p = PurePosixPath(raw.replace("\\", "/"))
        if p.parts and p.parts[0] == "data":
            out.append(f"{raw}: files under data/ are licensed data and must never be committed")
        elif str(p).endswith(BLOCKED_SUFFIXES):
            out.append(f"{raw}: data-file extension is blocked")
        elif p.name == ".env":
            out.append(f"{raw}: .env holds secrets")
        elif os.path.isfile(raw) and os.path.getsize(raw) > MAX_BYTES:
            out.append(f"{raw}: larger than 5 MB")
    return out


def main(argv: list[str]) -> int:
    found = problems(argv)
    for msg in found:
        print(f"BLOCKED  {msg}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
