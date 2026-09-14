# -*- coding: utf-8 -*-
"""A file lock and CSV header helpers, for every file more than one process writes."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

# How long to wait for another process's lock before giving up.
WAIT_S = 120


class FileLock:
    """Exclusive lock through a sidecar `<file>.lock`. `O_EXCL` is atomic on Windows and POSIX."""

    def __init__(self, target, wait_s: float | None = None):
        target = Path(target)
        self.path = target.with_suffix(target.suffix + ".lock")
        self.wait_s = WAIT_S if wait_s is None else wait_s

    def __enter__(self):
        deadline = time.time() + self.wait_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"lock {self.path.name} held by another process for {self.wait_s}s. "
                        f"Most likely two processes write the same file; if one died, "
                        f"delete the lock file.") from None
                time.sleep(0.1)
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return self

    def __exit__(self, *_):
        self.path.unlink(missing_ok=True)
        return False


def csv_header(path) -> list[str] | None:
    """The header row of a CSV, or None when the file is missing or empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return next(csv.reader(f), None)


def rewrite_csv(path, columns) -> int:
    """Rewrite a CSV under a new column list and return its row count.

    Columns the file lacks come back empty. Values are copied as text and never parsed,
    so a short git sha like `0123456` keeps its leading zero and a float keeps its digits.
    """
    path = Path(path)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return len(rows)
