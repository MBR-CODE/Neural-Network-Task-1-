"""Visualisation of depth predictions, ground truth and error maps (Matplotlib).

Only presentation lives here; no inference and no metric computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are only saved to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INVALID_COLOR = (0.55, 0.55, 0.55, 1.0)  # grey for pixels without valid depth


def robust_range(*arrays: Optional[np.ndarray], low: float = 2.0, high: float = 98.0
                 ) -> Tuple[float, float]:
    """Percentile range over the finite values of all given arrays."""
    vals = [a[np.isfinite(a)] for a in arrays if a is not None]
    vals = [v for v in vals if v.size]
    if not vals:
        return 0.0, 1.0
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1e-6
    return float(lo), float(hi)


def colorize_depth(depth: np.ndarray, cmap: str = "Spectral",
                   vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Map a depth array to an ``HxWx3`` uint8 RGB image (non-finite -> grey).

    This is for viewing only: the colours are a normalised mapping and do not
    preserve metric values (use the ``.npy`` output for numbers).
    """
    if vmin is None or vmax is None:
        lo, hi = robust_range(depth)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    rgba = matplotlib.colormaps.get_cmap(cmap)(np.nan_to_num(norm, nan=0.0))
    rgba[~np.isfinite(depth)] = INVALID_COLOR
    return (rgba[..., :3] * 255).round().astype(np.uint8)


def _masked(arr: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ma.MaskedArray:
    mask = ~np.isfinite(arr)
    if valid is not None:
        mask |= ~valid
    return np.ma.masked_array(arr, mask=mask)


def save_comparison_figure(rgb: np.ndarray, pred: np.ndarray, save_path: str | Path,
                           gt: Optional[np.ndarray] = None, valid: Optional[np.ndarray] = None,
                           title: Optional[str] = None, cmap: str = "Spectral",
                           error_cmap: str = "magma", dpi: int = 120,
                           metrics: Optional[dict] = None) -> Path:
    """Save a figure: RGB | predicted depth [| ground truth | absolute error].

    Ground-truth and error panels are only drawn when ``gt`` is given. Predicted
    and GT depth share one colour scale so they can be compared visually;
    invalid GT pixels are shown in grey and excluded from the error map.
    """
    if pred.shape != rgb.shape[:2]:
        raise ValueError(f"Prediction {pred.shape} does not match image {rgb.shape[:2]}")
    has_gt = gt is not None
    if has_gt:
        if gt.shape != pred.shape:
            raise ValueError(f"GT shape {gt.shape} does not match prediction {pred.shape}")
        valid = np.isfinite(gt) if valid is None else (valid & np.isfinite(gt))

    n = 4 if has_gt else 2
    h, w = pred.shape
    panel_w = 4.2
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_w * h / w + 0.9), squeeze=False)
    axes = axes[0]

    if has_gt:
        vmin, vmax = robust_range(pred, np.where(valid, gt, np.nan))
    else:
        vmin, vmax = robust_range(pred)

    cm = matplotlib.colormaps.get_cmap(cmap).copy()
    cm.set_bad(INVALID_COLOR)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB input")

    im = axes[1].imshow(_masked(pred), cmap=cm, vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted depth")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02, label="m")

    if has_gt:
        im = axes[2].imshow(_masked(gt, valid), cmap=cm, vmin=vmin, vmax=vmax)
        coverage = 100.0 * valid.mean()
        axes[2].set_title(f"Ground truth ({coverage:.1f}% valid)")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02, label="m")

        err = np.where(valid, np.abs(pred - gt), np.nan)
        ecm = matplotlib.colormaps.get_cmap(error_cmap).copy()
        ecm.set_bad(INVALID_COLOR)
        _, emax = robust_range(err, low=0, high=98)
        im = axes[3].imshow(_masked(err), cmap=ecm, vmin=0.0, vmax=emax)
        axes[3].set_title("|pred − GT| (valid GT only)")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02, label="m")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    sup = title or ""
    if metrics:
        sup += ("  |  " if sup else "") + "  ".join(
            f"{k}={metrics[k]:.4f}" for k in ("mae", "rmse", "abs_rel", "delta1") if k in metrics)
    if sup:
        fig.suptitle(sup, fontsize=10)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)  # release memory when processing many images
    return save_path
