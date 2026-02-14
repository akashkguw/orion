from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(prog="orion")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Run training")
    t.add_argument("--config", required=True)
    t.add_argument("--device", default=None)

    e = sub.add_parser("eval", help="Run evaluation")
    e.add_argument("--config", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--device", default=None)

    args = p.parse_args()

    if args.cmd == "train":
        cmd = [sys.executable, "-m", "orion.train", "--config", args.config]
        if args.device:
            cmd += ["--device", args.device]
        raise SystemExit(subprocess.call(cmd))

    if args.cmd == "eval":
        cmd = [
            sys.executable,
            "-m",
            "orion.eval",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
        ]
        if args.device:
            cmd += ["--device", args.device]
        raise SystemExit(subprocess.call(cmd))
