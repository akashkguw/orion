import sys

import pytest

from orion import cli


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    calls: list[list[str]] = []

    def fake_call(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli.subprocess, "call", fake_call)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert len(calls) == 1
    return calls[0]


def test_cli_train_basic(monkeypatch: pytest.MonkeyPatch):
    cmd = _run_cli(monkeypatch, ["orion", "train", "--config", "configs/golden.yaml"])
    assert cmd == [
        sys.executable,
        "-m",
        "orion.train",
        "--config",
        "configs/golden.yaml",
    ]


def test_cli_train_with_resume_auto_and_save_every(monkeypatch: pytest.MonkeyPatch):
    cmd = _run_cli(
        monkeypatch,
        [
            "orion",
            "train",
            "--config",
            "configs/golden.yaml",
            "--device",
            "cpu",
            "--resume",
            "--save-every",
            "7",
        ],
    )

    assert cmd == [
        sys.executable,
        "-m",
        "orion.train",
        "--config",
        "configs/golden.yaml",
        "--device",
        "cpu",
        "--resume",
        "--save-every",
        "7",
    ]


def test_cli_train_with_resume_path(monkeypatch: pytest.MonkeyPatch):
    cmd = _run_cli(
        monkeypatch,
        [
            "orion",
            "train",
            "--config",
            "configs/golden.yaml",
            "--resume",
            "runs/latest/checkpoint.pt",
        ],
    )

    assert cmd == [
        sys.executable,
        "-m",
        "orion.train",
        "--config",
        "configs/golden.yaml",
        "--resume",
        "runs/latest/checkpoint.pt",
    ]


def test_cli_eval_with_device(monkeypatch: pytest.MonkeyPatch):
    cmd = _run_cli(
        monkeypatch,
        [
            "orion",
            "eval",
            "--config",
            "configs/golden.yaml",
            "--checkpoint",
            "runs/latest/checkpoint.pt",
            "--device",
            "cpu",
        ],
    )

    assert cmd == [
        sys.executable,
        "-m",
        "orion.eval",
        "--config",
        "configs/golden.yaml",
        "--checkpoint",
        "runs/latest/checkpoint.pt",
        "--device",
        "cpu",
    ]


def test_cli_run_with_all_options(monkeypatch: pytest.MonkeyPatch):
    cmd = _run_cli(
        monkeypatch,
        [
            "orion",
            "run",
            "--config",
            "configs/golden.yaml",
            "--mode",
            "both",
            "--device",
            "cpu",
            "--base-dir",
            "/tmp/orion",
            "--run-id",
            "demo123",
            "--resume",
            "--save-every",
            "5",
            "--checkpoint",
            "runs/latest/checkpoint.pt",
        ],
    )

    assert cmd == [
        sys.executable,
        "-m",
        "orion.run",
        "--config",
        "configs/golden.yaml",
        "--mode",
        "both",
        "--device",
        "cpu",
        "--base-dir",
        "/tmp/orion",
        "--run-id",
        "demo123",
        "--resume",
        "--save-every",
        "5",
        "--checkpoint",
        "runs/latest/checkpoint.pt",
    ]
