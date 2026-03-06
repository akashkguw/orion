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
    t.add_argument("--resume", nargs="?", const="auto", default=None)
    t.add_argument("--save-every", type=int, default=None)

    e = sub.add_parser("eval", help="Run evaluation")
    e.add_argument("--config", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--device", default=None)

    r = sub.add_parser("run", help="Run train/eval with environment-aware run paths")
    r.add_argument("--config", required=True)
    r.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    r.add_argument("--device", default=None)
    r.add_argument("--base-dir", default=None)
    r.add_argument("--run-id", default=None)
    r.add_argument("--resume", nargs="?", const="auto", default=None)
    r.add_argument("--save-every", type=int, default=None)
    r.add_argument("--checkpoint", default=None)

    args = p.parse_args()

    if args.cmd == "train":
        cmd = [sys.executable, "-m", "orion.train", "--config", args.config]
        if args.device:
            cmd += ["--device", args.device]
        if args.resume is not None:
            cmd += ["--resume"]
            if args.resume != "auto":
                cmd += [args.resume]
        if args.save_every is not None:
            cmd += ["--save-every", str(args.save_every)]
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

    if args.cmd == "run":
        cmd = [
            sys.executable,
            "-m",
            "orion.run",
            "--config",
            args.config,
            "--mode",
            args.mode,
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.base_dir is not None:
            cmd += ["--base-dir", args.base_dir]
        if args.run_id is not None:
            cmd += ["--run-id", args.run_id]
        if args.resume is not None:
            cmd += ["--resume"]
            if args.resume != "auto":
                cmd += [args.resume]
        if args.save_every is not None:
            cmd += ["--save-every", str(args.save_every)]
        if args.checkpoint is not None:
            cmd += ["--checkpoint", args.checkpoint]
        raise SystemExit(subprocess.call(cmd))
