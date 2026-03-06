from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class StepMetrics:
    """Metrics logged every training step."""

    step: int
    loss: float
    ppl: float
    throughput_tokens_per_sec: float
    grad_norm: float
    grad_clipped: bool
    diverged: bool = False
    step_time_ms: float = 0.0
    accuracy_top1: float = 0.0
    learning_rate: float = 0.0


@dataclass
class WindowMetrics:
    """Metrics logged every 50 steps (windowed over last 50 steps)."""

    step: int
    vram_peak_mib: int
    divergence_rate: float
    activation_norm_rms: float
    attention_entropy: float
    attention_entropy_normalized: float
    clip_rate: float = 0.0
    valid_neighbor_fraction: float = 0.0
    attention_mass_window_pct: float = 0.0
    attention_mass_expander_pct: float = 0.0
    attn_score_mean: float = 0.0


@dataclass
class RunMetrics:
    """Metrics logged once per run."""

    step: int
    attention_degree: int
    compute_proxy_per_token: int
    compute_proxy_per_seq: int
    compute_proxy_per_step: int
    qk_norm: bool = False
    ortho_init: bool = False
    spectral_norm: bool = False


@dataclass
class EvalMetrics:
    """Metrics logged during evaluation."""

    step: int
    eval_ppl_512: float
    eval_ppl_1024: float
    eval_ppl_2048: float
    eval_ppl_4096: float


class MetricsTracker:
    """Tracks metrics across training steps."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.diverged_steps = deque(maxlen=window_size)
        self.vram_peaks = deque(maxlen=window_size)
        self.activation_norms = deque(maxlen=window_size)
        self.attention_entropies = deque(maxlen=window_size)
        self.clipped_steps = deque(maxlen=window_size)
        self.valid_neighbor_fractions = deque(maxlen=window_size)
        self.attention_mass_window = deque(maxlen=window_size)
        self.attention_mass_expander = deque(maxlen=window_size)

    def compute_throughput(self, batch_size: int, seq_len: int, step_time_sec: float) -> float:
        """Compute throughput in tokens/sec.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            step_time_sec: Time for this step in seconds

        Returns:
            Throughput in tokens/sec (non-pad tokens)
        """
        if step_time_sec <= 0:
            return 0.0
        tokens_this_step = batch_size * seq_len
        return tokens_this_step / step_time_sec

    def check_divergence(self, loss: float, grad_norm: float) -> bool:
        """Check if training has diverged.

        Args:
            loss: Loss value
            grad_norm: Gradient norm

        Returns:
            True if diverged (NaN or Inf detected)
        """
        return (
            math.isnan(loss) or math.isinf(loss) or math.isnan(grad_norm) or math.isinf(grad_norm)
        )

    def compute_activation_norm(self, residual_output: torch.Tensor) -> float:
        """Compute RMS of residual stream activations.

        Args:
            residual_output: Residual stream output tensor [B, T, D]

        Returns:
            RMS norm of residual activations
        """
        if residual_output is None or residual_output.numel() == 0:
            return 0.0
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(residual_output**2)).item()
        return rms

    def compute_attention_entropy(
        self, attn_weights: torch.Tensor, degree: int
    ) -> tuple[float, float]:
        """Compute entropy of attention weights over sparse neighbors.

        Args:
            attn_weights: Attention weights [B, H, T, degree]
            degree: Number of attention neighbors

        Returns:
            Tuple of (entropy_raw, entropy_normalized)
        """
        # Compute entropy: -sum(p * log(p))
        # Avoid log(0) by clamping
        attn_weights_safe = torch.clamp(attn_weights, min=1e-10)
        entropy = -(attn_weights * torch.log(attn_weights_safe)).sum(dim=-1).mean().item()

        # Normalize by max entropy: log(degree)
        max_entropy = math.log(degree) if degree > 1 else 1.0
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return entropy, entropy_normalized

    def compute_top1_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute top-1 accuracy (next-token prediction).

        Args:
            logits: Model logits [B, T, vocab_size]
            targets: Target tokens [B, T]

        Returns:
            Top-1 accuracy as fraction [0, 1]
        """
        if logits.numel() == 0 or targets.numel() == 0:
            return 0.0
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float().mean().item()
        return correct

    def compute_valid_neighbor_fraction(self, attn_weights: torch.Tensor) -> float:
        """Compute effective degree / actual degree (neighbors with non-zero mass).

        Args:
            attn_weights: Sparse attention weights [B, H, T, degree]

        Returns:
            Fraction of neighbors with non-zero attention mass (effective degree / degree)
        """
        if attn_weights.numel() == 0:
            return 0.0
        degree = attn_weights.shape[-1]
        # Count neighbors with non-zero mass (invalid slots are exactly 0 after softmax)
        eff_degree = (attn_weights > 0).sum(dim=-1).float().mean().item()
        valid_neighbor_fraction = eff_degree / degree if degree > 0 else 0.0
        return valid_neighbor_fraction

    def compute_attention_mass_split(
        self, attn_weights: torch.Tensor, window_size: int
    ) -> tuple[float, float]:
        """Compute attention mass split between window and expander.

        Args:
            attn_weights: Sparse attention weights [B, H, T, degree]
            window_size: Window size

        Returns:
            Tuple of (window_mass_pct, expander_mass_pct)
        """
        if attn_weights.numel() == 0:
            return 0.0, 0.0
        # Window is first window_size positions, expander is the rest
        window_mass = attn_weights[..., :window_size].sum(dim=-1).mean().item()
        expander_mass = attn_weights[..., window_size:].sum(dim=-1).mean().item()
        total = window_mass + expander_mass
        if total > 0:
            window_pct = 100.0 * window_mass / total
            expander_pct = 100.0 * expander_mass / total
        else:
            window_pct = expander_pct = 0.0
        return window_pct, expander_pct

    def record_step_metrics(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        throughput: float,
        step_time_ms: float = 0.0,
        accuracy_top1: float = 0.0,
        learning_rate: float = 0.0,
    ) -> StepMetrics:
        """Record metrics for a single step.

        Args:
            step: Step number
            loss: Loss value (cross-entropy, which equals NLL with no label smoothing)
            grad_norm: Gradient norm (pre-clip, from clip_grad_norm_)
            throughput: Throughput in tokens/sec
            step_time_ms: Step time in milliseconds
            accuracy_top1: Top-1 accuracy (next-token prediction)
            learning_rate: Current learning rate

        Returns:
            StepMetrics object
        """
        ppl = math.exp(min(loss, 100.0))  # Clamp to avoid overflow
        grad_clipped = grad_norm > 1.0
        diverged = self.check_divergence(loss, grad_norm)

        if diverged:
            self.diverged_steps.append(1)
        else:
            self.diverged_steps.append(0)

        if grad_clipped:
            self.clipped_steps.append(1)
        else:
            self.clipped_steps.append(0)

        return StepMetrics(
            step=step,
            loss=loss,
            ppl=ppl,
            throughput_tokens_per_sec=throughput,
            grad_norm=grad_norm,
            grad_clipped=grad_clipped,
            diverged=diverged,
            step_time_ms=step_time_ms,
            accuracy_top1=accuracy_top1,
            learning_rate=learning_rate,
        )

    def record_window_metrics(
        self,
        step: int,
        vram_peak_mib: int,
        activation_norm: float,
        attention_entropy: float,
        attention_entropy_normalized: float,
        clip_rate: float = 0.0,
        valid_neighbor_fraction: float = 0.0,
        attention_mass_window_pct: float = 0.0,
        attention_mass_expander_pct: float = 0.0,
        attn_score_mean: float = 0.0,
    ) -> WindowMetrics:
        """Record windowed metrics (every 50 steps).

        Args:
            step: Step number
            vram_peak_mib: Peak VRAM in MiB over last window
            activation_norm: RMS of residual activations
            attention_entropy: Raw entropy of attention weights
            attention_entropy_normalized: Normalized entropy
            clip_rate: Fraction of steps where gradients were clipped
            valid_neighbor_fraction: Fraction of valid neighbors in sparse attention
            attention_mass_window_pct: % of attention mass on window
            attention_mass_expander_pct: % of attention mass on expander

        Returns:
            WindowMetrics object
        """
        self.vram_peaks.append(vram_peak_mib)
        self.activation_norms.append(activation_norm)
        self.attention_entropies.append(attention_entropy)
        self.valid_neighbor_fractions.append(valid_neighbor_fraction)
        self.attention_mass_window.append(attention_mass_window_pct)
        self.attention_mass_expander.append(attention_mass_expander_pct)

        # Compute divergence rate over window
        divergence_rate = (
            sum(self.diverged_steps) / len(self.diverged_steps) if self.diverged_steps else 0.0
        )

        # Compute clip rate over window
        if not clip_rate and self.clipped_steps:
            clip_rate = sum(self.clipped_steps) / len(self.clipped_steps)

        return WindowMetrics(
            step=step,
            vram_peak_mib=vram_peak_mib,
            divergence_rate=divergence_rate,
            activation_norm_rms=activation_norm,
            attention_entropy=attention_entropy,
            attention_entropy_normalized=attention_entropy_normalized,
            clip_rate=clip_rate,
            valid_neighbor_fraction=valid_neighbor_fraction,
            attention_mass_window_pct=attention_mass_window_pct,
            attention_mass_expander_pct=attention_mass_expander_pct,
            attn_score_mean=attn_score_mean,
        )

    def record_run_metrics(
        self,
        step: int,
        attention_backend: str,
        batch_size: int,
        seq_len: int,
        n_layers: int,
        n_heads: int,
        window_size: int | None = None,
        expander_degree: int | None = None,
        qk_norm: bool = False,
        ortho_init: bool = False,
        spectral_norm: bool = False,
    ) -> RunMetrics:
        """Record run-level metrics (logged once per run).

        Args:
            step: Step number
            attention_backend: Attention backend name ("dense" | "window" | "sparse")
            window_size: Attention window size
            expander_degree: Attention expander degree
            batch_size: Batch size
            seq_len: Sequence length
            n_layers: Number of layers
            n_heads: Number of heads

        Returns:
            RunMetrics object
        """
        backend = attention_backend.lower().strip()
        if backend == "dense":
            degree = seq_len
        elif backend == "window":
            resolved_window = window_size if window_size is not None else 64
            degree = min(seq_len, max(1, int(resolved_window)))
        elif backend == "sparse":
            resolved_window = window_size if window_size is not None else 64
            resolved_expander = expander_degree if expander_degree is not None else 8
            degree = min(seq_len, max(1, int(resolved_window))) + max(0, int(resolved_expander))
        else:
            degree = seq_len

        compute_proxy_per_token = degree
        compute_proxy_per_seq = seq_len * degree
        compute_proxy_per_step = batch_size * n_heads * seq_len * degree

        return RunMetrics(
            step=step,
            attention_degree=degree,
            compute_proxy_per_token=compute_proxy_per_token,
            compute_proxy_per_seq=compute_proxy_per_seq,
            compute_proxy_per_step=compute_proxy_per_step,
            qk_norm=qk_norm,
            ortho_init=ortho_init,
            spectral_norm=spectral_norm,
        )

    def record_eval_metrics(
        self,
        step: int,
        eval_ppl_512: float,
        eval_ppl_1024: float,
        eval_ppl_2048: float,
        eval_ppl_4096: float,
    ) -> EvalMetrics:
        """Record evaluation metrics at different context lengths.

        Args:
            step: Step number
            eval_ppl_512: Perplexity at 512 tokens
            eval_ppl_1024: Perplexity at 1024 tokens
            eval_ppl_2048: Perplexity at 2048 tokens
            eval_ppl_4096: Perplexity at 4096 tokens

        Returns:
            EvalMetrics object
        """
        return EvalMetrics(
            step=step,
            eval_ppl_512=eval_ppl_512,
            eval_ppl_1024=eval_ppl_1024,
            eval_ppl_2048=eval_ppl_2048,
            eval_ppl_4096=eval_ppl_4096,
        )


def metrics_to_dict(
    metrics: StepMetrics | WindowMetrics | RunMetrics | EvalMetrics,
) -> dict[str, Any]:
    """Convert metrics dataclass to dictionary, excluding None values.

    Args:
        metrics: Metrics object

    Returns:
        Dictionary representation
    """
    d = asdict(metrics)
    return {k: v for k, v in d.items() if v is not None}
