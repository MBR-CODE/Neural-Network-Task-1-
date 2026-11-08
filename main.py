"""Command-line entry point for the depth-estimation pipeline.

Examples (PowerShell, from the project folder):

    python main.py check
    python main.py predict --input data\\images\\object1.jpg
    python main.py predict --input data\\images --visualize
    python main.py evaluate --images data\\images --ground-truth data\\ground_truth
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger("depth")

EXIT_OK, EXIT_ERROR, EXIT_NO_METRICS = 0, 1, 2


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("model", "runtime", "paths", "ground_truth", "evaluation", "visualization"):
        cfg.setdefault(section, {})
    cfg["_root"] = path.resolve().parent
    return cfg


def cfg_path(cfg: dict, key: str) -> Path:
    """Resolve ``paths.<key>`` relative to the config file's directory."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else cfg["_root"] / p


def cli_path(value: Optional[str], cfg: dict, key: str) -> Path:
    """Command-line paths are relative to the current directory; fall back to config."""
    return Path(value).resolve() if value else cfg_path(cfg, key)


def make_estimator(cfg: dict, args):
    from model import build_model
    device = args.device or cfg["runtime"].get("device", "auto")
    fp16 = args.fp16 or bool(cfg["runtime"].get("fp16", False))
    t0 = time.perf_counter()
    est = build_model(cfg["model"], device=device, project_root=cfg["_root"], use_fp16=fp16)
    logger.info("Loaded %s on %s in %.1fs", est.name, est.device, time.perf_counter() - t0)
    return est


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_check(cfg: dict, args) -> int:
    import numpy as np
    import torch
    from depth_prediction import predict_depth

    logger.info("Python %s | PyTorch %s | CUDA build %s", sys.version.split()[0],
                torch.__version__, torch.version.cuda)
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA not available: inference will run on CPU")
    est = make_estimator(cfg, args)
    logger.info("Model: %s", est.describe())

    rgb = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    if torch.cuda.is_available() and est.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    depth = predict_depth(est, rgb)
    t0 = time.perf_counter()
    depth = predict_depth(est, rgb)
    dt = time.perf_counter() - t0
    logger.info("Dummy inference OK: output %s %s, range %.3f-%.3f m, %.3fs/image",
                depth.shape, depth.dtype, depth.min(), depth.max(), dt)
    if est.device.type == "cuda":
        logger.info("Peak GPU memory: %.0f MiB", torch.cuda.max_memory_allocated() / 2**20)
    return EXIT_OK


def cmd_predict(cfg: dict, args) -> int:
    from dataset_loader import list_images, load_rgb_image
    from depth_prediction import run_on_paths, write_manifest

    src = Path(args.input).resolve() if args.input else cfg_path(cfg, "images")
    if src.is_file():
        paths = [src]
    elif src.is_dir():
        paths = list_images(src, recursive=args.recursive)
        if not paths:
            logger.error("No supported images found in %s", src)
            return EXIT_ERROR
    else:
        logger.error("Input not found: %s", src)
        return EXIT_ERROR

    out_dir = cli_path(args.output, cfg, "predictions")
    vis = cfg["visualization"]
    est = make_estimator(cfg, args)
    summary = run_on_paths(est, paths, out_dir, save_png=vis.get("save_depth_png", True),
                           cmap=vis.get("cmap", "Spectral"), skip_existing=args.skip_existing)
    manifest = write_manifest(summary, out_dir / "predictions_manifest.json")

    if args.visualize:
        import numpy as np
        from visualization import save_comparison_figure
        vis_dir = cli_path(args.vis_output, cfg, "visualizations")
        for r in summary.results:
            save_comparison_figure(load_rgb_image(r.image_path), np.load(r.npy_path),
                                   vis_dir / f"{r.name}_prediction.png", title=r.name,
                                   cmap=vis.get("cmap", "Spectral"))
        logger.info("Figures written to %s", vis_dir)

    logger.info("%d prediction(s) written to %s (%d failure(s)); manifest: %s",
                len(summary.results), out_dir, len(summary.failures), manifest)
    return EXIT_OK if not summary.failures else EXIT_ERROR


def cmd_evaluate(cfg: dict, args) -> int:
    from dataset_loader import DepthDecodeConfig, PairedDepthDataset
    from evaluation import EvalConfig, evaluate_predictions, write_report

    gt_cfg = dict(cfg["ground_truth"])
    if args.gt_scale is not None:
        gt_cfg["scale"] = args.gt_scale
    if args.gt_invalid is not None:
        gt_cfg["invalid_values"] = args.gt_invalid
    ev_cfg = dict(cfg["evaluation"])
    for key in ("min_depth", "max_depth"):
        if getattr(args, key) is not None:
            ev_cfg[key] = gt_cfg[key] = getattr(args, key)
    if args.median_align:
        ev_cfg["median_scale_alignment"] = True
    decode = DepthDecodeConfig.from_dict(gt_cfg)
    eval_cfg = EvalConfig.from_dict(ev_cfg)

    image_dir = cli_path(args.images, cfg, "images")
    gt_dir = cli_path(args.ground_truth, cfg, "ground_truth")
    pred_dir = cli_path(args.predictions, cfg, "predictions")
    out_dir = cli_path(args.output, cfg, "evaluations")

    dataset = PairedDepthDataset(image_dir, gt_dir, decode,
                                 gt_extensions=gt_cfg.get("extensions", (".npy", ".png")),
                                 recursive=args.recursive)
    logger.info("Found %d image(s): %d with ground truth, %d without",
                len(dataset.all_pairs), len(dataset.pairs), len(dataset.unpaired))
    if not dataset.pairs:
        logger.error("No image has a matching ground-truth file in %s; no metrics can be "
                     "computed. (Ground truth is matched by file stem.)", gt_dir)
        return EXIT_NO_METRICS

    model_info = None
    if not args.skip_inference:
        from depth_prediction import run_on_paths, write_manifest
        est = make_estimator(cfg, args)
        model_info = est.describe()
        summary = run_on_paths(est, [p.image_path for p in dataset.pairs], pred_dir,
                               save_png=cfg["visualization"].get("save_depth_png", True),
                               cmap=cfg["visualization"].get("cmap", "Spectral"),
                               skip_existing=args.reuse_predictions)
        write_manifest(summary, pred_dir / "predictions_manifest.json")
        del est  # free GPU memory before evaluation
    else:
        logger.info("Skipping inference; using existing predictions in %s", pred_dir)

    vis_dir = None if args.no_visualize else cli_path(args.vis_output, cfg, "visualizations")
    report = evaluate_predictions(dataset, pred_dir, eval_cfg, model_info=model_info,
                                  visualize_dir=vis_dir, cmap=cfg["visualization"].get("cmap", "Spectral"))
    files = write_report(report, out_dir)
    data = report.to_dict()
    c = data["counts"]
    logger.info("Evaluated %d/%d image(s) with ground truth (%d not evaluated, %d errors).",
                c["evaluated"], c["images_with_ground_truth"], c["not_evaluated"], c["errors"])
    if report.pooled is None:
        logger.warning("No quantitative metrics were produced (no sufficient valid ground truth).")
        logger.info("Report: %s", files["json"])
        return EXIT_NO_METRICS
    m = report.pooled
    logger.info("Pooled: MAE=%.4f m  RMSE=%.4f m  AbsRel=%.4f  d1=%.4f  (%d px)",
                m["mae"], m["rmse"], m["abs_rel"], m["delta1"], m["n_valid"])
    logger.info("Reports: %s | %s | %s", files["json"], files["csv"], files["markdown"])
    return EXIT_OK if c["errors"] == 0 else EXIT_ERROR


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    common.add_argument("--device", default=None, help="auto | cuda | cuda:N | cpu (overrides config)")
    common.add_argument("--fp16", action="store_true", help="fp16 autocast on CUDA")
    common.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    parser = argparse.ArgumentParser(description="AI depth estimation (Depth Anything V2 Metric-Indoor Small)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", parents=[common], help="Verify environment, checkpoint and model loading")

    p = sub.add_parser("predict", parents=[common], help="Predict depth for an image or a folder")
    p.add_argument("--input", "-i", help="Image file or folder (default: paths.images)")
    p.add_argument("--output", "-o", help="Prediction folder (default: paths.predictions)")
    p.add_argument("--recursive", action="store_true", help="Search sub-folders")
    p.add_argument("--visualize", action="store_true", help="Also save RGB|depth figures")
    p.add_argument("--vis-output", help="Figure folder (default: paths.visualizations)")
    p.add_argument("--skip-existing", action="store_true", help="Skip images whose .npy already exists")

    e = sub.add_parser("evaluate", parents=[common], help="Predict and evaluate against paired ground truth")
    e.add_argument("--images", help="RGB folder (default: paths.images)")
    e.add_argument("--ground-truth", help="GT depth folder (default: paths.ground_truth)")
    e.add_argument("--predictions", help="Prediction folder (default: paths.predictions)")
    e.add_argument("--output", "-o", help="Report folder (default: paths.evaluations)")
    e.add_argument("--vis-output", help="Figure folder (default: paths.visualizations)")
    e.add_argument("--recursive", action="store_true")
    e.add_argument("--skip-inference", action="store_true", help="Only evaluate existing .npy predictions")
    e.add_argument("--reuse-predictions", action="store_true", help="Do not recompute existing .npy files")
    e.add_argument("--no-visualize", action="store_true", help="Do not save comparison figures")
    e.add_argument("--gt-scale", type=float, help="Raw GT value -> metres multiplier (overrides config)")
    e.add_argument("--gt-invalid", type=float, nargs="*", help="Raw GT values meaning 'invalid'")
    e.add_argument("--min-depth", type=float, help="Minimum valid depth in metres")
    e.add_argument("--max-depth", type=float, help="Maximum valid depth in metres")
    e.add_argument("--median-align", action="store_true", help="Also report median-scale-aligned metrics")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        cfg = load_config(Path(args.config).resolve())
        handler = {"check": cmd_check, "predict": cmd_predict, "evaluate": cmd_evaluate}[args.command]
        return handler(cfg, args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    except Exception as exc:
        from model import ModelLoadError
        if isinstance(exc, ModelLoadError):
            logger.error("%s", exc)
            return EXIT_ERROR
        logger.exception("Unexpected error")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
