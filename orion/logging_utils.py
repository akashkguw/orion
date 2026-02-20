from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: Path, *, wall_time_offset: float | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if wall_time_offset is None:
            self._t0 = time.time()
        else:
            self._t0 = time.time() - wall_time_offset

    def log(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("wall_time_s", time.time() - self._t0)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
