from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("wall_time_s", time.time() - self._t0)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

