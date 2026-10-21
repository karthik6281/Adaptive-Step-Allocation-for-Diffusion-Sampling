"""Per-sample NFE constraints. Actions use 0=denoise and 1=skip."""

from __future__ import annotations

import torch

from utils.reproducibility import require_int


def validate_rollout_settings(steps, budget, mode, warmup_ratio, max_consecutive_skips):
    require_int(steps, "steps")
    if mode not in ("cap", "exact"):
        raise ValueError("budget mode must be 'cap' or 'exact'")
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must lie in [0, 1].")
    allowance = steps if budget is None else require_int(budget, "budget", maximum=steps)
    warmup = int(warmup_ratio * steps)
    remaining = steps - warmup
    if max_consecutive_skips is None:
        required = warmup + int(remaining > 0)
    else:
        require_int(max_consecutive_skips, "max_consecutive_skips")
        required = warmup + (remaining + max_consecutive_skips) // (max_consecutive_skips + 1)
    if allowance < required:
        raise ValueError(
            f"The budget must cover warmup, the terminal denoise, and the skip limit "
            f"(at least {required} denoising calls)."
        )
    return allowance, warmup


def enforce_budget(
    actions: torch.Tensor,
    used: torch.Tensor,
    budget: int,
    remaining_decisions: int,
    mode: str = "cap",
    force_denoise: torch.Tensor | None = None,
    *,
    max_consecutive_skips: int | None = None,
    consecutive_skips: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in ("cap", "exact"):
        raise ValueError("budget mode must be 'cap' or 'exact'")
    require_int(budget, "budget")
    require_int(remaining_decisions, "remaining_decisions")
    if actions.ndim != 1 or actions.shape != used.shape:
        raise ValueError("actions and used must have matching [batch] shapes")
    if not torch.all((actions == 0) | (actions == 1)):
        raise ValueError("actions must be binary")
    if not torch.all((used >= 0) & (used <= budget) & (used == used.round())):
        raise ValueError("used must contain integer counts within the budget")
    left = budget - used
    if mode == "exact" and torch.any(left > remaining_decisions):
        raise ValueError("The exact budget cannot be spent in the remaining decisions.")
    if torch.any(left < 1):
        raise ValueError("The terminal denoising call must be reserved.")

    forced_denoise = torch.zeros_like(actions, dtype=torch.bool)
    future_required = int(remaining_decisions > 1)
    if max_consecutive_skips is not None:
        require_int(max_consecutive_skips, "max_consecutive_skips")
        if (
            consecutive_skips is None
            or consecutive_skips.shape != actions.shape
            or not torch.all(
                (consecutive_skips >= 0)
                & (consecutive_skips <= max_consecutive_skips)
                & (consecutive_skips == consecutive_skips.round())
            )
        ):
            raise ValueError("consecutive_skips must contain valid per-sample integer streaks.")
        minimum = (remaining_decisions + consecutive_skips + max_consecutive_skips) // (
            max_consecutive_skips + 1
        )
        if torch.any(left < minimum):
            raise ValueError("The remaining budget cannot satisfy the maximum skip streak.")
        future_required = (remaining_decisions - 1 + max_consecutive_skips) // (
            max_consecutive_skips + 1
        )
        forced_denoise |= consecutive_skips == max_consecutive_skips
    if force_denoise is not None:
        if force_denoise.shape != actions.shape:
            raise ValueError("force_denoise must have shape [batch]")
        forced_denoise |= force_denoise.bool()
    if mode == "exact":
        forced_denoise |= left == remaining_decisions
    if remaining_decisions == 1:
        forced_denoise[:] = True
    forced_skip = left <= future_required
    if torch.any(forced_denoise & forced_skip):
        raise ValueError("A required denoising call conflicts with the remaining budget.")

    result = torch.where(forced_denoise, 0, actions)
    result = torch.where(forced_skip, 1, result)
    return result, forced_denoise | forced_skip
