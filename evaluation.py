"""Depth-estimation metrics and dataset-level evaluation.

Metrics (computed on valid ground-truth pixels only):

* MAE     = mean |d̂ - d|                      (metres)
* RMSE    = sqrt(mean (d̂ - d)²)               (metres)
* AbsRel  = mean |d̂ - d| / d                  (unitless)
* δ1      = fraction of pixels with max(d̂/d, d/d̂) < 1.25

A pixel is *valid* when its ground truth is finite, marked valid by the
loader, and inside ``[min_depth, max_depth]``. If fewer than
``min_valid_pixels`` remain, the image is reported as *not evaluated* and no
numbers are produced for it. Nothing here runs a model: predictions are read
from the saved ``.npy`` files, which keeps evaluation independent of inference.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

METRIC_NAMES = ("mae", "rmse", "abs_rel", "delta1")
DELTA1_THRESHOLD = 1.25


class InsufficientGroundTruthError(ValueError):
    """Raised when there are too few valid ground-truth pixels to evaluate."""


@dataclass
class EvalConfig:
    min_depth: float = 1e-3
    max_depth: float = 20.0
    min_valid_pixels: int = 100
    clip_predictions: bool = True       # clip d̂ to [min_depth, max_depth] before scoring
    on_size_mismatch: str = "error"     # "error" | "resize_prediction"
    median_scale_alignment: bool = False  # additionally report scale-aligned metrics (diagnostic)

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "EvalConfig":
        cfg = dict(cfg or {})
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown evaluation config keys: {sorted(unknown)}")
        obj = cls(**cfg)
        if not (0 < obj.min_depth < obj.max_depth):
            raise ValueError("evaluation requires 0 < min_depth < max_depth")
        if obj.on_size_mismatch not in ("error", "resize_prediction"):
            raise ValueError("on_size_mismatch must be 'error' or 'resize_prediction'")
        if obj.min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be >= 1")
        return obj


# --------------------------------------------------------------------------- #
# Per-image metrics
# --------------------------------------------------------------------------- #
def compute_valid_mask(gt: np.ndarray, pred: np.ndarray, min_depth: float, max_depth: float,
                       gt_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels usable for evaluation (decided by ground truth, plus finite predictions)."""
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}")
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(gt) & (gt >= min_depth) & (gt <= max_depth) & np.isfinite(pred)
    if gt_valid is not None:
        if gt_valid.shape != gt.shape:
            raise ValueError("gt_valid mask shape does not match ground truth")
        mask &= gt_valid.astype(bool)
    return mask


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                    min_valid_pixels: int = 1) -> Dict[str, float]:
    """Compute MAE, RMSE, AbsRel and δ1 over ``mask``.

    ``pred`` must be strictly positive on the masked pixels (clip beforehand).

    Raises:
        InsufficientGroundTruthError: if ``mask`` has fewer than ``min_valid_pixels`` pixels.
    """
    if not (pred.shape == gt.shape == mask.shape):
        raise ValueError(f"Shapes differ: pred {pred.shape}, gt {gt.shape}, mask {mask.shape}")
    n = int(mask.sum())
    if n < max(1, min_valid_pixels):
        raise InsufficientGroundTruthError(
            f"Only {n} valid ground-truth pixels (need >= {max(1, min_valid_pixels)})")
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    if np.any(g <= 0):
        raise ValueError("Ground truth must be > 0 on valid pixels")
    if np.any(p <= 0):
        raise ValueError("Predictions must be > 0 on valid pixels (enable clip_predictions)")
    diff = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "delta1": float(np.mean(ratio < DELTA1_THRESHOLD)),
        "n_valid": n,
    }


class MetricAccumulator:
    """Accumulates sums so dataset metrics can be pooled over *all* valid pixels."""

    def __init__(self) -> None:
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_rel = 0.0
        self.n_delta1 = 0

    def update(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
        p = pred[mask].astype(np.float64)
        g = gt[mask].astype(np.float64)
        diff = np.abs(p - g)
        self.n += p.size
        self.sum_abs += float(diff.sum())
        self.sum_sq += float((diff ** 2).sum())
        self.sum_rel += float((diff / g).sum())
        self.n_delta1 += int((np.maximum(p / g, g / p) < DELTA1_THRESHOLD).sum())

    def compute(self) -> Optional[Dict[str, float]]:
        if self.n == 0:
            return None
        return {"mae": self.sum_abs / self.n, "rmse": math.sqrt(self.sum_sq / self.n),
                "abs_rel": self.sum_rel / self.n, "delta1": self.n_delta1 / self.n, "n_valid": self.n}


def _prepare_prediction(pred: np.ndarray, gt_shape, cfg: EvalConfig) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 2:
        raise ValueError(f"Prediction must be 2-D, got {pred.shape}")
    if pred.shape != tuple(gt_shape):
        if cfg.on_size_mismatch == "resize_prediction":
            pred = cv2.resize(pred, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Prediction {pred.shape} and ground truth {tuple(gt_shape)} differ in "
                             "size. Provide GT at image resolution or set "
                             "evaluation.on_size_mismatch: resize_prediction")
    if cfg.clip_predictions:
        pred = np.clip(pred, cfg.min_depth, cfg.max_depth)
    return pred


@dataclass
class ImageEvaluation:
    name: str
    status: str                      # "evaluated" | "not_evaluated" | "error"
    reason: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    aligned_metrics: Optional[Dict[str, float]] = None
    median_scale: Optional[float] = None
    valid_fraction: Optional[float] = None
    pred_path: Optional[str] = None
    gt_path: Optional[str] = None


def evaluate_pair(pred: np.ndarray, gt: np.ndarray, gt_valid: Optional[np.ndarray],
                  cfg: EvalConfig, name: str = "") -> tuple:
    """Evaluate one prediction.

    Returns ``(ImageEvaluation, pred_prepared, mask)``; the latter two are
    ``None`` when the image could not be evaluated.
    """
    pred = _prepare_prediction(pred, gt.shape, cfg)
    mask = compute_valid_mask(gt, pred, cfg.min_depth, cfg.max_depth, gt_valid)
    frac = float(mask.mean())
    try:
        metrics = compute_metrics(pred, gt, mask, cfg.min_valid_pixels)
    except InsufficientGroundTruthError as exc:
        return ImageEvaluation(name, "not_evaluated", str(exc), valid_fraction=frac), None, None
    result = ImageEvaluation(name, "evaluated", metrics=metrics, valid_fraction=frac)
    if cfg.median_scale_alignment:
        scale = float(np.median(gt[mask]) / np.median(pred[mask]))
        result.aligned_metrics = compute_metrics(_apply_scale(pred, scale, cfg), gt, mask,
                                                 cfg.min_valid_pixels)
        result.median_scale = scale
    return result, pred, mask


def _apply_scale(pred: np.ndarray, scale: float, cfg: EvalConfig) -> np.ndarray:
    aligned = pred * scale
    return np.clip(aligned, cfg.min_depth, cfg.max_depth) if cfg.clip_predictions else aligned


# --------------------------------------------------------------------------- #
# Dataset-level evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationReport:
    config: dict
    model: Optional[dict]
    images: List[ImageEvaluation] = field(default_factory=list)
    skipped_no_ground_truth: List[str] = field(default_factory=list)
    pooled: Optional[Dict[str, float]] = None
    mean_per_image: Optional[Dict[str, float]] = None
    pooled_aligned: Optional[Dict[str, float]] = None

    @property
    def evaluated(self) -> List[ImageEvaluation]:
        return [r for r in self.images if r.status == "evaluated"]

    def to_dict(self) -> dict:
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": self.config,
            "model": self.model,
            "counts": {
                "images_with_ground_truth": len(self.images),
                "evaluated": len(self.evaluated),
                "not_evaluated": sum(r.status == "not_evaluated" for r in self.images),
                "errors": sum(r.status == "error" for r in self.images),
                "skipped_no_ground_truth": len(self.skipped_no_ground_truth),
            },
            "summary": {
                "pooled_over_pixels": self.pooled,
                "mean_over_images": self.mean_per_image,
                "pooled_over_pixels_median_scale_aligned": self.pooled_aligned,
            },
            "per_image": [r.__dict__ for r in self.images],
            "skipped_no_ground_truth": self.skipped_no_ground_truth,
        }


def evaluate_predictions(dataset, pred_dir: str | Path, cfg: EvalConfig,
                         model_info: Optional[dict] = None, visualize_dir: Optional[Path] = None,
                         cmap: str = "Spectral") -> EvaluationReport:
    """Evaluate saved ``<stem>.npy`` predictions against a ``PairedDepthDataset``."""
    pred_dir = Path(pred_dir)
    report = EvaluationReport(config=cfg.__dict__.copy(), model=model_info,
                              skipped_no_ground_truth=[p.name for p in dataset.unpaired])
    acc, acc_aligned = MetricAccumulator(), MetricAccumulator()

    for i, pair in enumerate(dataset.pairs):
        pred_path = pred_dir / f"{pair.name}.npy"
        entry_paths = dict(pred_path=str(pred_path), gt_path=str(pair.gt_path))
        try:
            if not pred_path.is_file():
                raise FileNotFoundError(f"Prediction not found: {pred_path}")
            sample = dataset[i]
            pred = np.load(str(pred_path), allow_pickle=False)
            result, pred_p, mask = evaluate_pair(pred, sample.depth, sample.valid, cfg, pair.name)
            result.pred_path, result.gt_path = entry_paths["pred_path"], entry_paths["gt_path"]
            if mask is not None:
                acc.update(pred_p, sample.depth, mask)
                if result.median_scale is not None:
                    acc_aligned.update(_apply_scale(pred_p, result.median_scale, cfg),
                                       sample.depth, mask)
            if visualize_dir is not None:
                from visualization import save_comparison_figure  # local import keeps metrics module light
                if pred_p is None:
                    pred_p = _prepare_prediction(pred, sample.depth.shape, cfg)
                if pred_p.shape == sample.rgb.shape[:2]:
                    save_comparison_figure(sample.rgb, pred_p, Path(visualize_dir) / f"{pair.name}_comparison.png",
                                           gt=sample.depth, valid=sample.valid, title=pair.name,
                                           cmap=cmap, metrics=result.metrics)
                else:
                    logger.warning("%s: GT resolution differs from RGB; skipping figure", pair.name)
        except Exception as exc:
            logger.error("%s: evaluation failed: %s", pair.name, exc)
            result = ImageEvaluation(pair.name, "error", str(exc), **entry_paths)
        if result.status == "evaluated":
            m = result.metrics
            logger.info("%s: MAE=%.4f m RMSE=%.4f m AbsRel=%.4f d1=%.4f (%d px, %.1f%% valid)",
                        pair.name, m["mae"], m["rmse"], m["abs_rel"], m["delta1"], m["n_valid"],
                        100 * result.valid_fraction)
        elif result.status == "not_evaluated":
            logger.warning("%s: not evaluated (%s)", pair.name, result.reason)
        report.images.append(result)

    report.pooled = acc.compute()
    report.pooled_aligned = acc_aligned.compute()
    ev = report.evaluated
    if ev:
        report.mean_per_image = {k: float(np.mean([r.metrics[k] for r in ev])) for k in METRIC_NAMES}
        report.mean_per_image["n_images"] = len(ev)
    return report


def write_report(report: EvaluationReport, out_dir: str | Path, stem: str = "evaluation") -> Dict[str, Path]:
    """Write ``<stem>.json``, ``<stem>_per_image.csv`` and ``<stem>_summary.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = report.to_dict()

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{stem}_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "status", "mae_m", "rmse_m", "abs_rel", "delta1", "n_valid",
                         "valid_fraction", "median_scale", "reason"])
        for r in report.images:
            m = r.metrics or {}
            writer.writerow([r.name, r.status] + [m.get(k, "") for k in METRIC_NAMES] +
                            [m.get("n_valid", ""), r.valid_fraction if r.valid_fraction is not None else "",
                             r.median_scale if r.median_scale is not None else "", r.reason or ""])

    md_path = out_dir / f"{stem}_summary.md"
    md_path.write_text(format_summary(data), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def format_summary(data: dict) -> str:
    c = data["counts"]
    lines = ["# Depth evaluation summary", "",
             f"- Created (UTC): {data['created_utc']}",
             f"- Model: {json.dumps(data['model']) if data['model'] else 'n/a'}",
             f"- Images with ground truth: {c['images_with_ground_truth']}  "
             f"(evaluated {c['evaluated']}, not evaluated {c['not_evaluated']}, errors {c['errors']})",
             f"- Images skipped (no ground truth): {c['skipped_no_ground_truth']}",
             f"- Valid depth range: [{data['config']['min_depth']}, {data['config']['max_depth']}] m", ""]
    s = data["summary"]
    if not s["pooled_over_pixels"]:
        lines += ["**No quantitative metrics: no image had sufficient valid ground truth.**", ""]
        return "\n".join(lines)
    lines += ["| Aggregation | MAE (m) | RMSE (m) | AbsRel | δ1 | N |", "|---|---|---|---|---|---|"]
    for label, key, n_key in (("Pooled over valid pixels", "pooled_over_pixels", "n_valid"),
                              ("Mean over images", "mean_over_images", "n_images"),
                              ("Pooled, median-scale aligned (diagnostic)",
                               "pooled_over_pixels_median_scale_aligned", "n_valid")):
        m = s.get(key)
        if m:
            lines.append(f"| {label} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['abs_rel']:.4f} | "
                         f"{m['delta1']:.4f} | {m[n_key]} |")
    return "\n".join(lines) + "\n"
