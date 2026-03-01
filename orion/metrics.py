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
    grad_norm_pre_clip: float | None = None
    diverged: bool = False


@dataclass
class WindowMetrics:
    """Metrics logged every 50 steps (windowed over last 50 steps)."""

    step: int
    vram_peak_mib: int
    divergence_rate: float
    activation_norm_rms: float
    attention_entropy: float
    attention_entropy_normalized: float


@dataclass
class RunMetrics:
    """Metrics logged once per run."""

    step: int
    attention_degree: int
    compute_proxy_per_token: int
    compute_proxy_per_step: int


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

    def compute_grad_norm(self, model: torch.nn.Module) -> tuple[float, float]:
        """Compute global gradient norm before and after clipping.

        Args:
            model: Model to compute grad norm for

        Returns:
            Tuple of (grad_norm_pre_clip, grad_norm_post_clip)
        """
        total_norm_pre = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_pre += p.grad.data.norm(2).item() ** 2
        grad_norm_pre_clip = math.sqrt(total_norm_pre)

        # Post-clip is typically the same unless you clip, so return pre for now
        # In practice, you'd track this after torch.nn.utils.clip_grad_norm_
        return grad_norm_pre_clip, grad_norm_pre_clip

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

    def compute_activation_norm(self, model: torch.nn.Module) -> float:
        """Compute RMS of residual stream activations.

        Uses the last layer's residual stream as representative.

        Args:
            model: Model to compute activation norm for

        Returns:
            RMS norm of residual activations
        """
        activation_norms = []

        # Hook to capture residual activations
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # Compute RMS: sqrt(mean(x^2))
                rms = torch.sqrt(torch.mean(output**2)).item()
                activation_norms.append(rms)

        # Register hooks on residual connections (if available)
        hooks = []
        for module in model.modules():
            if hasattr(module, "residual"):
                h = module.register_forward_hook(hook_fn)
                hooks.append(h)

        # If no residual hooks found, use output of last layer
        if not hooks:
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    h = module.register_forward_hook(hook_fn)
                    hooks.append(h)

        # Clean up hooks
        for h in hooks:
            h.remove()

        if activation_norms:
            return sum(activation_norms) / len(activation_norms)
        return 0.0

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

    def record_step_metrics(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        throughput: float,
        grad_norm_pre_clip: float | None = None,
    ) -> StepMetrics:
        """Record metrics for a single step.

        Args:
            step: Step number
            loss: Loss value
            grad_norm: Gradient norm (post-clip)
            throughput: Throughput in tokens/sec
            grad_norm_pre_clip: Gradient norm before clipping (optional)

        Returns:
            StepMetrics object
        """
        ppl = math.exp(min(loss, 100.0))  # Clamp to avoid overflow
        diverged = self.check_divergence(loss, grad_norm)

        if diverged:
            self.diverged_steps.append(1)
        else:
            self.diverged_steps.append(0)

        return StepMetrics(
            step=step,
            loss=loss,
            ppl=ppl,
            throughput_tokens_per_sec=throughput,
            grad_norm=grad_norm,
            grad_norm_pre_clip=grad_norm_pre_clip,
            diverged=diverged,
        )

    def record_window_metrics(
        self,
        step: int,
        vram_peak_mib: int,
        activation_norm: float,
        attention_entropy: float,
        attention_entropy_normalized: float,
    ) -> WindowMetrics:
        """Record windowed metrics (every 50 steps).

        Args:
            step: Step number
            vram_peak_mib: Peak VRAM in MiB over last window
            activation_norm: RMS of residual activations
            attention_entropy: Raw entropy of attention weights
            attention_entropy_normalized: Normalized entropy

        Returns:
            WindowMetrics object
        """
        self.vram_peaks.append(vram_peak_mib)
        self.activation_norms.append(activation_norm)
        self.attention_entropies.append(attention_entropy)

        # Compute divergence rate over window
        divergence_rate = (
            sum(self.diverged_steps) / len(self.diverged_steps) if self.diverged_steps else 0.0
        )

        return WindowMetrics(
            step=step,
            vram_peak_mib=vram_peak_mib,
            divergence_rate=divergence_rate,
            activation_norm_rms=activation_norm,
            attention_entropy=attention_entropy,
            attention_entropy_normalized=attention_entropy_normalized,
        )

    def record_run_metrics(
        self,
        step: int,
        window_size: int,
        expander_degree: int,
        batch_size: int,
        seq_len: int,
        n_layers: int,
        n_heads: int,
    ) -> RunMetrics:
        """Record run-level metrics (logged once per run).

        Args:
            step: Step number
            window_size: Attention window size
            expander_degree: Attention expander degree
            batch_size: Batch size
            seq_len: Sequence length
            n_layers: Number of layers
            n_heads: Number of heads

        Returns:
            RunMetrics object
        """
        degree = window_size + expander_degree
        compute_proxy_per_token = seq_len * degree
        compute_proxy_per_step = batch_size * n_heads * seq_len * degree

        return RunMetrics(
            step=step,
            attention_degree=degree,
            compute_proxy_per_token=compute_proxy_per_token,
            compute_proxy_per_step=compute_proxy_per_step,
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
