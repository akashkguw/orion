from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser(prog="orion")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--device", default=None)

    e = sub.add_parser("eval")
    e.add_argument("--config", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--device", default=None)

    b = sub.add_parser("bench")
    b.add_argument("--config", required=True)
    b.add_argument("--device", default=None)

    args = p.parse_args()

    if args.cmd == "train":
        from orion.train.loop import main as train_main

        train_main()
    elif args.cmd == "eval":
        from orion.eval.eval import main as eval_main

        eval_main()
    else:
        from orion.bench.bench import main as bench_main

        bench_main()
