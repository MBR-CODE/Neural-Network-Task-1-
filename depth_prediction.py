"""Run a :class:`model.DepthEstimator` on images and save its predictions.

Outputs per image ``<stem>``:

* ``<stem>.npy``        float32 ``HxW`` depth in **metres**, same resolution and
                        pixel grid as the input image (the raw model output,
                        no normalisation or clipping).
* ``<stem>_depth.png``  colour-mapped visualisation (not metric; for viewing).

This module performs inference only; evaluation and figures live elsewhere.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from dataset_loader import list_images, load_rgb_image, validate_rgb
from model import DepthEstimator
from visualization import colorize_depth

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    name: str
    image_path: str
    npy_path: str
    png_path: Optional[str]
    height: int
    width: int
    depth_min_m: float
    depth_max_m: float
    depth_median_m: float
    inference_seconds: float


@dataclass
class BatchSummary:
    model: dict
    results: List[PredictionResult] = field(default_factory=list)
    failures: List[dict] = field(default_factory=list)


def predict_depth(estimator: DepthEstimator, rgb: np.ndarray) -> np.ndarray:
    """Predict depth for one RGB image and verify the output contract.

    Returns an ``HxW`` float32 array in metres aligned with ``rgb``.
    """
    validate_rgb(rgb)
    depth = estimator.predict(rgb)
    if depth.shape != rgb.shape[:2]:
        raise RuntimeError(f"Model returned depth of shape {depth.shape}, "
                           f"expected {rgb.shape[:2]} (must match the input image)")
    if not np.all(np.isfinite(depth)):
        raise RuntimeError("Model returned non-finite depth values")
    return depth.astype(np.float32, copy=False)


def save_depth_npy(depth: np.ndarray, path: str | Path) -> Path:
    """Save metric depth losslessly as float32 ``.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), depth.astype(np.float32, copy=False))
    return path


def save_depth_png(depth: np.ndarray, path: str | Path, cmap: str = "Spectral") -> Path:
    """Save a colour-mapped depth PNG (Unicode-safe on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = colorize_depth(depth, cmap=cmap)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Failed to encode PNG for {path}")
    buf.tofile(str(path))
    return path


def run_on_image(estimator: DepthEstimator, image_path: str | Path, out_dir: str | Path,
                 save_png: bool = True, cmap: str = "Spectral") -> PredictionResult:
    """Load one image, predict depth and write ``.npy`` (+ ``.png``)."""
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    rgb = load_rgb_image(image_path)

    t0 = time.perf_counter()
    depth = predict_depth(estimator, rgb)
    elapsed = time.perf_counter() - t0

    npy_path = save_depth_npy(depth, out_dir / f"{image_path.stem}.npy")
    png_path = save_depth_png(depth, out_dir / f"{image_path.stem}_depth.png", cmap) if save_png else None
    return PredictionResult(
        name=image_path.stem, image_path=str(image_path), npy_path=str(npy_path),
        png_path=str(png_path) if png_path else None, height=depth.shape[0], width=depth.shape[1],
        depth_min_m=float(depth.min()), depth_max_m=float(depth.max()),
        depth_median_m=float(np.median(depth)), inference_seconds=round(elapsed, 4))


def run_on_paths(estimator: DepthEstimator, image_paths: Sequence[Path], out_dir: str | Path,
                 save_png: bool = True, cmap: str = "Spectral",
                 skip_existing: bool = False) -> BatchSummary:
    """Predict depth for many images; failures are recorded and do not stop the batch."""
    out_dir = Path(out_dir)
    summary = BatchSummary(model=estimator.describe())
    total = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        path = Path(path)
        if skip_existing and (out_dir / f"{path.stem}.npy").is_file():
            logger.info("[%d/%d] %s: prediction exists, skipping", i, total, path.name)
            continue
        try:
            res = run_on_image(estimator, path, out_dir, save_png=save_png, cmap=cmap)
            summary.results.append(res)
            logger.info("[%d/%d] %s -> %dx%d, depth %.3f-%.3f m (%.2fs)", i, total, path.name,
                        res.width, res.height, res.depth_min_m, res.depth_max_m, res.inference_seconds)
        except Exception as exc:  # keep going on bad files
            logger.error("[%d/%d] %s failed: %s", i, total, path.name, exc)
            summary.failures.append({"image_path": str(path), "error": str(exc)})
    return summary


def run_on_folder(estimator: DepthEstimator, folder: str | Path, out_dir: str | Path,
                  recursive: bool = False, **kwargs) -> BatchSummary:
    paths = list_images(folder, recursive=recursive)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {folder}")
    return run_on_paths(estimator, paths, out_dir, **kwargs)


def write_manifest(summary: BatchSummary, path: str | Path) -> Path:
    """Write a JSON manifest describing the predictions that were produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": summary.model,
        "depth_unit": "meters",
        "npy_format": "float32 HxW, same resolution/pixel grid as input image",
        "results": [asdict(r) for r in summary.results],
        "failures": summary.failures,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
