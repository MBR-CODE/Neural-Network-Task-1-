"""Depth-estimation model interface and the pretrained Depth Anything V2 backend.

Design
------
``DepthEstimator``        Abstract interface used by the rest of the pipeline:
                          ``predict(rgb_uint8_HxWx3) -> float32 HxW depth in metres``.
``TorchDepthEstimator``   Shared PyTorch logic (device placement, preprocessing,
                          ``torch.inference_mode``, optional fp16 autocast and
                          resizing the output back to the input resolution).
                          A future custom network only has to supply an
                          ``nn.Module`` whose forward returns metric depth.
``DepthAnythingV2Metric`` Wraps the *unmodified* official implementation from
                          ``third_party/Depth-Anything-V2/metric_depth``.

Models are created through :func:`build_model`, which looks up ``model.name``
in :data:`MODEL_REGISTRY`; register a new class with :func:`register_model`.
"""

from __future__ import annotations

import abc
import importlib
import logging
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_loader import IMAGENET_MEAN, IMAGENET_STD, preprocess_image, validate_rgb

logger = logging.getLogger(__name__)

# Official configuration from metric_depth/README.md and metric_depth/run.py.
DA2_MODEL_CONFIGS: Dict[str, dict] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
DA2_CHECKPOINT_URLS: Dict[str, str] = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/"
            "resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true",
}


class ModelLoadError(RuntimeError):
    """Raised when a model or its checkpoint cannot be loaded."""


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
def resolve_device(preference: str = "auto") -> torch.device:
    """Map ``"auto" | "cuda" | "cuda:N" | "cpu"`` to an available device.

    Falls back to CPU (with a warning) if CUDA is requested but unavailable.
    """
    pref = (preference or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.", RuntimeWarning)
            return torch.device("cpu")
        dev = torch.device(pref)
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise ValueError(f"{pref} requested but only {torch.cuda.device_count()} CUDA device(s) exist")
        return dev
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{preference}' (use auto, cuda, cuda:N or cpu)")


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
class DepthEstimator(abc.ABC):
    """Interface every depth model in this project must implement."""

    #: Short identifier stored in prediction metadata.
    name: str = "depth_estimator"
    #: True if outputs are absolute depth in metres (not relative/inverse depth).
    is_metric: bool = True

    def __init__(self, device: torch.device):
        self.device = device

    @abc.abstractmethod
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return an ``HxW`` float32 depth map (metres) aligned pixel-to-pixel with ``rgb``.

        Args:
            rgb: ``HxWx3`` uint8 image in RGB channel order.
        """

    def describe(self) -> dict:
        return {"name": self.name, "device": str(self.device), "is_metric": self.is_metric}


class TorchDepthEstimator(DepthEstimator):
    """Generic wrapper for an ``nn.Module`` that maps a normalised image to metric depth.

    The wrapped module receives a ``1x3xH'xW'`` float tensor (see
    :func:`dataset_loader.preprocess_image`) and must return depth in metres
    with shape ``1xH'xW'`` or ``1x1xH'xW'``. The output is bilinearly resized
    back to the input image size so it corresponds pixel-for-pixel with it.
    """

    name = "torch_module"

    def __init__(self, module: nn.Module, device: torch.device, input_size: int = 518,
                 multiple_of: int = 14, mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD, use_fp16: bool = False,
                 align_corners: bool = True):
        super().__init__(device)
        self.input_size = int(input_size)
        self.multiple_of = int(multiple_of)
        self.mean, self.std = tuple(mean), tuple(std)
        self.align_corners = align_corners
        self.use_fp16 = bool(use_fp16) and device.type == "cuda"
        if use_fp16 and device.type != "cuda":
            logger.info("fp16 autocast is only used on CUDA; running fp32 on %s.", device)
        for p in module.parameters():
            p.requires_grad_(False)
        self.module = module.to(device).eval()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        if out.ndim == 4:
            if out.shape[1] != 1:
                raise RuntimeError(f"Expected a single-channel depth output, got {tuple(out.shape)}")
            out = out[:, 0]
        if out.ndim != 3:
            raise RuntimeError(f"Expected depth output of shape BxHxW, got {tuple(out.shape)}")
        return out

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        validate_rgb(rgb)
        h, w = rgb.shape[:2]
        x = torch.from_numpy(preprocess_image(rgb, self.input_size, self.multiple_of,
                                              self.mean, self.std)).to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                depth = self._forward(x)
            # Resize in fp32 back to the original resolution (as the official infer_image does).
            depth = F.interpolate(depth[:, None].float(), size=(h, w), mode="bilinear",
                                  align_corners=self.align_corners)[0, 0]
            result = depth.cpu().numpy().astype(np.float32, copy=False)
        del x, depth
        return result

    def describe(self) -> dict:
        d = super().describe()
        d.update(input_size=self.input_size, fp16=self.use_fp16)
        return d


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: Dict[str, Callable[..., DepthEstimator]] = {}


def register_model(name: str) -> Callable[[Type[DepthEstimator]], Type[DepthEstimator]]:
    """Class decorator registering a ``DepthEstimator`` under ``name``.

    The class must provide ``from_config(model_cfg: dict, device, project_root)``.
    """
    def deco(cls: Type[DepthEstimator]) -> Type[DepthEstimator]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls.from_config  # type: ignore[attr-defined]
        return cls
    return deco


def build_model(model_cfg: dict, device: str | torch.device = "auto",
                project_root: Optional[Path] = None, use_fp16: bool = False) -> DepthEstimator:
    """Instantiate the model named ``model_cfg['name']``."""
    name = model_cfg.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}")
    dev = device if isinstance(device, torch.device) else resolve_device(device)
    root = Path(project_root) if project_root else Path(__file__).resolve().parent
    return MODEL_REGISTRY[name](model_cfg, dev, root, use_fp16=use_fp16)


# --------------------------------------------------------------------------- #
# Depth Anything V2 – Metric (Hypersim / indoor)
# --------------------------------------------------------------------------- #
def _import_official_da2(repo_path: Path):
    """Import ``DepthAnythingV2`` from the official ``metric_depth`` package.

    The repository contains two packages called ``depth_anything_v2``: the root
    one (relative depth, ReLU head) and ``metric_depth/depth_anything_v2``
    (metric depth, ``sigmoid * max_depth`` head). Loading the wrong one would
    silently fail with this checkpoint, so the import location is verified.
    """
    repo_path = repo_path.resolve()
    pkg_dir = repo_path / "depth_anything_v2"
    if not (pkg_dir / "dpt.py").is_file():
        raise ModelLoadError(
            f"Official Depth Anything V2 metric code not found at {pkg_dir}.\n"
            "Clone it with:  git clone https://github.com/DepthAnything/Depth-Anything-V2.git "
            "third_party/Depth-Anything-V2")
    existing = sys.modules.get("depth_anything_v2")
    if existing is not None:
        loaded_from = Path(getattr(existing, "__file__", "") or "").resolve().parent
        if loaded_from != pkg_dir.resolve():
            raise ModelLoadError(f"A different 'depth_anything_v2' package is already imported "
                                 f"from {loaded_from}; expected {pkg_dir}.")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    # The official DINOv2 code logs "xFormers not available" via the "dinov2" logger.
    # xFormers is optional (plain PyTorch attention is used), so silence that notice.
    dinov2_logger = logging.getLogger("dinov2")
    previous_level = dinov2_logger.level
    dinov2_logger.setLevel(logging.ERROR)
    try:
        dpt = importlib.import_module("depth_anything_v2.dpt")
    finally:
        dinov2_logger.setLevel(previous_level)
    if Path(dpt.__file__).resolve().parent != pkg_dir.resolve():
        raise ModelLoadError(f"Imported {dpt.__file__}, expected the metric_depth implementation.")
    return dpt.DepthAnythingV2


@register_model("depth_anything_v2_metric")
class DepthAnythingV2Metric(TorchDepthEstimator):
    """Pretrained Depth Anything V2 metric-depth model (official weights, inference only)."""

    name = "depth_anything_v2_metric"
    is_metric = True

    def __init__(self, encoder: str, checkpoint: Path, repo_path: Path, max_depth: float,
                 device: torch.device, input_size: int = 518, use_fp16: bool = False):
        if encoder not in DA2_MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder}'. Options: {sorted(DA2_MODEL_CONFIGS)}")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            url = DA2_CHECKPOINT_URLS.get(encoder, "see metric_depth/README.md")
            raise ModelLoadError(f"Checkpoint not found: {checkpoint}\nDownload it from:\n  {url}")

        DepthAnythingV2 = _import_official_da2(Path(repo_path))
        net = DepthAnythingV2(**{**DA2_MODEL_CONFIGS[encoder], "max_depth": float(max_depth)})
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except Exception as exc:  # corrupt / truncated / wrong file
            raise ModelLoadError(f"Could not read checkpoint {checkpoint}: {exc}") from exc
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        try:
            net.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ModelLoadError(f"Checkpoint {checkpoint.name} does not match encoder "
                                 f"'{encoder}': {exc}") from exc
        del state

        super().__init__(net, device, input_size=input_size, multiple_of=14, use_fp16=use_fp16,
                         align_corners=True)
        self.encoder = encoder
        self.max_depth = float(max_depth)
        self.checkpoint = checkpoint

    @classmethod
    def from_config(cls, model_cfg: dict, device: torch.device, project_root: Path,
                    use_fp16: bool = False) -> "DepthAnythingV2Metric":
        def resolve(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else (project_root / path)

        return cls(
            encoder=model_cfg.get("encoder", "vits"),
            checkpoint=resolve(model_cfg["checkpoint"]),
            repo_path=resolve(model_cfg["repo_path"]),
            max_depth=float(model_cfg.get("max_depth", 20.0)),
            device=device,
            input_size=int(model_cfg.get("input_size", 518)),
            use_fp16=use_fp16,
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update(encoder=self.encoder, max_depth=self.max_depth, checkpoint=self.checkpoint.name)
        return d
