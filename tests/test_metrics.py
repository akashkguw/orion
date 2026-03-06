"""Tests for the metrics tracking system."""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from orion.logging_utils import JsonlLogger
from orion.metrics import (
    EvalMetrics,
    MetricsTracker,
    RunMetrics,
    StepMetrics,
    WindowMetrics,
    metrics_to_dict,
)
from orion.model import TinyDecoderOnly


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_compute_throughput(self):
        """Test throughput computation."""
        tracker = MetricsTracker()

        # 32 tokens in 0.01 seconds = 3200 tokens/sec
        throughput = tracker.compute_throughput(batch_size=2, seq_len=16, step_time_sec=0.01)
        assert throughput == pytest.approx(3200.0, rel=1e-5)

    def test_compute_throughput_zero_time(self):
        """Test throughput with zero time returns 0."""
        tracker = MetricsTracker()
        throughput = tracker.compute_throughput(batch_size=2, seq_len=16, step_time_sec=0.0)
        assert throughput == 0.0

    def test_compute_grad_norm_pre_clip(self):
        """Test gradient norm computation using clip_grad_norm_."""
        model = TinyDecoderOnly(vocab_size=256, d_model=64, n_layers=2, n_heads=4, mlp_mult=4)

        # Create dummy gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        # Compute norm using clip_grad_norm_ (which returns pre-clip norm)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        assert grad_norm > 0.0
        assert isinstance(grad_norm, float)

    def test_compute_grad_norm_no_gradients(self):
        """Test gradient norm with no gradients."""
        model = TinyDecoderOnly(vocab_size=256, d_model=64, n_layers=2, n_heads=4, mlp_mult=4)

        # With no gradients, clip_grad_norm_ returns 0.0
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        assert grad_norm == 0.0

    def test_check_divergence_nan_loss(self):
        """Test divergence detection with NaN loss."""
        tracker = MetricsTracker()
        diverged = tracker.check_divergence(loss=float("nan"), grad_norm=1.0)
        assert diverged is True

    def test_check_divergence_inf_loss(self):
        """Test divergence detection with Inf loss."""
        tracker = MetricsTracker()
        diverged = tracker.check_divergence(loss=float("inf"), grad_norm=1.0)
        assert diverged is True

    def test_check_divergence_nan_grad(self):
        """Test divergence detection with NaN grad norm."""
        tracker = MetricsTracker()
        diverged = tracker.check_divergence(loss=5.0, grad_norm=float("nan"))
        assert diverged is True

    def test_check_divergence_inf_grad(self):
        """Test divergence detection with Inf grad norm."""
        tracker = MetricsTracker()
        diverged = tracker.check_divergence(loss=5.0, grad_norm=float("inf"))
        assert diverged is True

    def test_check_divergence_normal(self):
        """Test divergence detection with normal values."""
        tracker = MetricsTracker()
        diverged = tracker.check_divergence(loss=5.0, grad_norm=1.0)
        assert diverged is False

    def test_compute_activation_norm(self):
        """Test activation norm computation."""
        tracker = MetricsTracker()
        x = torch.randn(2, 16, 64)  # [B, T, D]
        norm = tracker.compute_activation_norm(x)
        assert norm > 0.0
        assert isinstance(norm, float)

    def test_compute_activation_norm_empty(self):
        """Test activation norm with empty tensor."""
        tracker = MetricsTracker()
        x = torch.tensor([])
        norm = tracker.compute_activation_norm(x)
        assert norm == 0.0

    def test_compute_activation_norm_none(self):
        """Test activation norm with None."""
        tracker = MetricsTracker()
        norm = tracker.compute_activation_norm(None)
        assert norm == 0.0

    def test_compute_attention_entropy(self):
        """Test attention entropy computation."""
        tracker = MetricsTracker()
        # Create uniform attention weights
        attn_weights = torch.ones(2, 4, 16, 8) / 8  # [B, H, T, degree]
        entropy, entropy_norm = tracker.compute_attention_entropy(attn_weights, degree=8)

        # Uniform distribution should have max entropy
        assert entropy > 0.0
        assert entropy_norm > 0.0
        assert entropy_norm <= 1.0

    def test_compute_attention_entropy_single_neighbor(self):
        """Test attention entropy with single neighbor."""
        tracker = MetricsTracker()
        attn_weights = torch.ones(2, 4, 16, 1)  # [B, H, T, degree=1]
        entropy, entropy_norm = tracker.compute_attention_entropy(attn_weights, degree=1)

        # Single neighbor should have zero entropy
        assert entropy == pytest.approx(0.0, abs=1e-5)
        assert entropy_norm == pytest.approx(0.0, abs=1e-5)

    def test_record_step_metrics(self):
        """Test step metrics recording."""
        tracker = MetricsTracker()
        metrics = tracker.record_step_metrics(
            step=1,
            loss=5.0,
            grad_norm=1.5,
            throughput=1000.0,
        )

        assert isinstance(metrics, StepMetrics)
        assert metrics.step == 1
        assert metrics.loss == 5.0
        assert metrics.ppl == pytest.approx(math.exp(5.0), rel=1e-5)
        assert metrics.grad_norm == 1.5
        assert metrics.grad_clipped is True
        assert metrics.diverged is False

    def test_record_step_metrics_no_clip(self):
        """Test step metrics when gradient not clipped."""
        tracker = MetricsTracker()
        metrics = tracker.record_step_metrics(
            step=1,
            loss=5.0,
            grad_norm=0.5,
            throughput=1000.0,
        )

        assert metrics.grad_clipped is False

    def test_record_step_metrics_diverged(self):
        """Test step metrics with divergence."""
        tracker = MetricsTracker()
        metrics = tracker.record_step_metrics(
            step=1,
            loss=float("nan"),
            grad_norm=1.0,
            throughput=1000.0,
        )

        assert metrics.diverged is True

    def test_record_window_metrics(self):
        """Test window metrics recording."""
        tracker = MetricsTracker()

        # Record some step metrics to populate diverged_steps
        for i in range(50):
            tracker.record_step_metrics(
                step=i + 1,
                loss=5.0,
                grad_norm=1.0,
                throughput=1000.0,
            )

        metrics = tracker.record_window_metrics(
            step=50,
            vram_peak_mib=2048,
            activation_norm=0.5,
            attention_entropy=1.5,
            attention_entropy_normalized=0.5,
        )

        assert isinstance(metrics, WindowMetrics)
        assert metrics.step == 50
        assert metrics.vram_peak_mib == 2048
        assert metrics.activation_norm_rms == 0.5
        assert metrics.attention_entropy == 1.5
        assert metrics.attention_entropy_normalized == 0.5
        assert metrics.divergence_rate == 0.0

    def test_record_window_metrics_with_divergence(self):
        """Test window metrics with diverged steps."""
        tracker = MetricsTracker()

        # Record 50 steps, 10 diverged
        for i in range(50):
            loss = float("nan") if i % 5 == 0 else 5.0
            tracker.record_step_metrics(
                step=i + 1,
                loss=loss,
                grad_norm=1.0,
                throughput=1000.0,
            )

        metrics = tracker.record_window_metrics(
            step=50,
            vram_peak_mib=2048,
            activation_norm=0.5,
            attention_entropy=1.5,
            attention_entropy_normalized=0.5,
        )

        # 10 out of 50 diverged = 0.2 rate
        assert metrics.divergence_rate == pytest.approx(0.2, rel=1e-5)

    def test_record_run_metrics(self):
        """Test run metrics recording."""
        tracker = MetricsTracker()
        metrics = tracker.record_run_metrics(
            step=1,
            attention_backend="sparse",
            window_size=64,
            expander_degree=8,
            batch_size=16,
            seq_len=256,
            n_layers=4,
            n_heads=8,
        )

        assert isinstance(metrics, RunMetrics)
        assert metrics.step == 1
        assert metrics.attention_degree == 72  # 64 + 8
        assert metrics.compute_proxy_per_token == 72
        assert metrics.compute_proxy_per_seq == 256 * 72
        assert metrics.compute_proxy_per_step == 16 * 8 * 256 * 72

    def test_record_run_metrics_dense_backend(self):
        """Dense backend should use full sequence length as degree proxy."""
        tracker = MetricsTracker()
        metrics = tracker.record_run_metrics(
            step=1,
            attention_backend="dense",
            batch_size=4,
            seq_len=128,
            n_layers=2,
            n_heads=4,
        )

        assert metrics.attention_degree == 128
        assert metrics.compute_proxy_per_token == 128
        assert metrics.compute_proxy_per_seq == 128 * 128
        assert metrics.compute_proxy_per_step == 4 * 4 * 128 * 128

    def test_record_run_metrics_window_backend(self):
        """Window backend degree should be capped by sequence length."""
        tracker = MetricsTracker()
        metrics = tracker.record_run_metrics(
            step=1,
            attention_backend="window",
            window_size=256,
            batch_size=2,
            seq_len=64,
            n_layers=2,
            n_heads=2,
        )

        assert metrics.attention_degree == 64
        assert metrics.compute_proxy_per_token == 64
        assert metrics.compute_proxy_per_seq == 64 * 64
        assert metrics.compute_proxy_per_step == 2 * 2 * 64 * 64

    def test_record_eval_metrics(self):
        """Test eval metrics recording."""
        tracker = MetricsTracker()
        metrics = tracker.record_eval_metrics(
            step=1000,
            eval_ppl_512=10.5,
            eval_ppl_1024=12.3,
            eval_ppl_2048=15.2,
            eval_ppl_4096=18.9,
        )

        assert isinstance(metrics, EvalMetrics)
        assert metrics.step == 1000
        assert metrics.eval_ppl_512 == 10.5
        assert metrics.eval_ppl_1024 == 12.3
        assert metrics.eval_ppl_2048 == 15.2
        assert metrics.eval_ppl_4096 == 18.9


class TestMetricsDataclasses:
    """Test metrics dataclasses."""

    def test_step_metrics_to_dict(self):
        """Test StepMetrics conversion to dict."""
        metrics = StepMetrics(
            step=1,
            loss=5.0,
            ppl=148.4,
            throughput_tokens_per_sec=1000.0,
            grad_norm=1.5,
            grad_clipped=True,
            diverged=False,
        )

        d = metrics_to_dict(metrics)
        assert d["step"] == 1
        assert d["loss"] == 5.0
        assert d["ppl"] == pytest.approx(148.4, rel=1e-5)
        assert d["grad_clipped"] is True
        assert d["diverged"] is False

    def test_window_metrics_to_dict(self):
        """Test WindowMetrics conversion to dict."""
        metrics = WindowMetrics(
            step=50,
            vram_peak_mib=2048,
            divergence_rate=0.1,
            activation_norm_rms=0.5,
            attention_entropy=1.5,
            attention_entropy_normalized=0.5,
        )

        d = metrics_to_dict(metrics)
        assert d["step"] == 50
        assert d["vram_peak_mib"] == 2048
        assert d["divergence_rate"] == 0.1

    def test_run_metrics_to_dict(self):
        """Test RunMetrics conversion to dict."""
        metrics = RunMetrics(
            step=1,
            attention_degree=72,
            compute_proxy_per_token=72,
            compute_proxy_per_seq=18432,
            compute_proxy_per_step=2359296,
        )

        d = metrics_to_dict(metrics)
        assert d["step"] == 1
        assert d["attention_degree"] == 72
        assert d["compute_proxy_per_token"] == 72

    def test_eval_metrics_to_dict(self):
        """Test EvalMetrics conversion to dict."""
        metrics = EvalMetrics(
            step=1000,
            eval_ppl_512=10.5,
            eval_ppl_1024=12.3,
            eval_ppl_2048=15.2,
            eval_ppl_4096=18.9,
        )

        d = metrics_to_dict(metrics)
        assert d["step"] == 1000
        assert d["eval_ppl_512"] == 10.5


class TestMetricsLogging:
    """Test metrics logging integration."""

    def test_step_metrics_logging(self):
        """Test logging step metrics to JSONL."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            logger = JsonlLogger(metrics_path)
            tracker = MetricsTracker()

            # Record and log step metrics
            metrics = tracker.record_step_metrics(
                step=1,
                loss=5.0,
                grad_norm=1.5,
                throughput=1000.0,
            )
            logger.log({"type": "step", **metrics_to_dict(metrics)})

            # Read and verify
            line = metrics_path.read_text().strip()
            obj = json.loads(line)

            assert obj["type"] == "step"
            assert obj["step"] == 1
            assert obj["loss"] == 5.0
            assert "ppl" in obj
            assert "grad_norm" in obj
            assert "grad_clipped" in obj

    def test_window_metrics_logging(self):
        """Test logging window metrics to JSONL."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            logger = JsonlLogger(metrics_path)
            tracker = MetricsTracker()

            # Record and log window metrics
            metrics = tracker.record_window_metrics(
                step=50,
                vram_peak_mib=2048,
                activation_norm=0.5,
                attention_entropy=1.5,
                attention_entropy_normalized=0.5,
            )
            logger.log({"type": "window", **metrics_to_dict(metrics)})

            # Read and verify
            line = metrics_path.read_text().strip()
            obj = json.loads(line)

            assert obj["type"] == "window"
            assert obj["step"] == 50
            assert obj["vram_peak_mib"] == 2048

    def test_multiple_metrics_logging(self):
        """Test logging multiple metrics types."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            logger = JsonlLogger(metrics_path)
            tracker = MetricsTracker()

            # Log step metrics
            for step in range(1, 6):
                metrics = tracker.record_step_metrics(
                    step=step,
                    loss=5.0,
                    grad_norm=1.0,
                    throughput=1000.0,
                )
                logger.log({"type": "step", **metrics_to_dict(metrics)})

            # Log window metrics
            metrics = tracker.record_window_metrics(
                step=5,
                vram_peak_mib=2048,
                activation_norm=0.5,
                attention_entropy=1.5,
                attention_entropy_normalized=0.5,
            )
            logger.log({"type": "window", **metrics_to_dict(metrics)})

            # Read and verify
            lines = metrics_path.read_text().strip().split("\n")
            assert len(lines) == 6

            # Check types
            step_count = 0
            window_count = 0
            for line in lines:
                obj = json.loads(line)
                if obj["type"] == "step":
                    step_count += 1
                elif obj["type"] == "window":
                    window_count += 1

            assert step_count == 5
            assert window_count == 1
