"""Data loading and preprocessing for the depth-estimation pipeline.

Responsibilities
----------------
* Discover and load RGB images (common formats, Unicode-safe paths on Windows).
* Preprocess an RGB image into a network input tensor (resize to a multiple of
  the patch size, ImageNet normalisation) exactly as Depth Anything V2 expects.
* Decode ground-truth depth maps with a *configurable* encoding (scale, invalid
  sentinel values, valid range) and produce an explicit validity mask.
* Pair RGB images with ground-truth depth maps by file stem for evaluation.

Nothing in this module runs a neural network; it is shared by inference and
evaluation code and can be reused unchanged by a future custom model.
"""


#this is done by AI as i needed help with the data extraction 


from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEPTH_EXTENSIONS: Tuple[str, ...] = (".npy", ".npz", ".png", ".tif", ".tiff", ".pfm")

# ImageNet statistics used by DINOv2 / Depth Anything V2.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Low-level file I/O
# --------------------------------------------------------------------------- #
def _read_file_bytes(path: Path) -> np.ndarray:
    """Read a file into a uint8 buffer.

    ``cv2.imread`` cannot open non-ASCII paths on Windows, so files are read
    with NumPy and decoded with ``cv2.imdecode`` instead.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"File is empty: {path}")
    return data


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image as an ``HxWx3`` ``uint8`` array in **RGB** channel order.

    Grayscale images are replicated to three channels, an alpha channel is
    dropped and 16-bit images are rescaled to 8 bit.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the extension is unsupported or the file cannot be decoded.
    """
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension '{path.suffix}' for {path}. "
                         f"Supported: {', '.join(IMAGE_EXTENSIONS)}")
    image = cv2.imdecode(_read_file_bytes(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image (corrupt or unsupported): {path}")

    if image.dtype == np.uint16:
        image = (image / 257.0).round().astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"Unsupported image dtype {image.dtype} in {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image shape {image.shape} in {path}")
    return np.ascontiguousarray(image)


def list_images(folder: str | Path, recursive: bool = False,
                extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[Path]:
    """Return image files in ``folder`` sorted by path (case-insensitive extensions)."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    exts = {e.lower() for e in extensions}
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in exts)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _constrain_to_multiple(x: float, multiple: int, min_val: int) -> int:
    """Round ``x`` to a multiple of ``multiple``, never going below ``min_val``.

    Mirrors ``Resize.constrain_to_multiple_of`` in the official repository.
    """
    y = int(np.round(x / multiple) * multiple)
    if y < min_val:
        y = int(np.ceil(x / multiple) * multiple)
    return y


def compute_network_size(height: int, width: int, input_size: int = 518,
                         multiple_of: int = 14) -> Tuple[int, int]:
    """Return the ``(net_h, net_w)`` used by Depth Anything V2.

    Keeps the aspect ratio, scales so that the *shorter* side is at least
    ``input_size`` ("lower_bound" mode), and makes both sides multiples of
    the ViT patch size (14).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size {height}x{width}")
    if input_size <= 0 or input_size % multiple_of != 0:
        raise ValueError(f"input_size must be a positive multiple of {multiple_of}, got {input_size}")
    scale = max(input_size / height, input_size / width)
    net_h = _constrain_to_multiple(scale * height, multiple_of, input_size)
    net_w = _constrain_to_multiple(scale * width, multiple_of, input_size)
    return net_h, net_w


def preprocess_image(rgb: np.ndarray, input_size: int = 518, multiple_of: int = 14,
                     mean: Sequence[float] = IMAGENET_MEAN,
                     std: Sequence[float] = IMAGENET_STD) -> np.ndarray:
    """Convert an ``HxWx3`` uint8 RGB image into a ``1x3xH'xW'`` float32 array.

    The steps are numerically equivalent to ``DepthAnythingV2.image2tensor``
    (bicubic resize, ``/255``, ImageNet normalisation, CHW layout). The
    original ``(H, W)`` must be kept by the caller so the prediction can be
    resized back to the input resolution.
    """
    validate_rgb(rgb)
    h, w = rgb.shape[:2]
    net_h, net_w = compute_network_size(h, w, input_size, multiple_of)
    image = rgb.astype(np.float64) / 255.0
    image = cv2.resize(image, (net_w, net_h), interpolation=cv2.INTER_CUBIC)
    image = (image - np.asarray(mean)) / np.asarray(std)
    image = np.transpose(image, (2, 0, 1))[None].astype(np.float32)
    return np.ascontiguousarray(image)


def validate_rgb(rgb: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``rgb`` is a non-empty ``HxWx3`` uint8 array."""
    if not isinstance(rgb, np.ndarray):
        raise ValueError(f"Expected a numpy array, got {type(rgb).__name__}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got dtype {rgb.dtype}")
    if rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise ValueError(f"Image too small: {rgb.shape}")


# --------------------------------------------------------------------------- #
# Ground-truth depth decoding
# --------------------------------------------------------------------------- #
@dataclass
class DepthDecodeConfig:
    """How raw ground-truth values are converted to metres.

    ``depth_m = raw * scale`` (e.g. ``scale=0.001`` for 16-bit PNGs in mm).
    Pixels are invalid if they are non-finite, equal to one of
    ``invalid_values`` (compared on the *raw* value), non-positive, or outside
    ``[min_depth, max_depth]`` after scaling.
    """

    scale: float = 0.001
    invalid_values: List[float] = field(default_factory=lambda: [0.0])
    min_depth: float = 1e-3
    max_depth: float = 20.0
    channel: Optional[int] = None  # for multi-channel depth images, which channel holds depth
    npz_key: str = "depth"

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "DepthDecodeConfig":
        cfg = dict(cfg or {})
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(cfg) - allowed - {"extensions"}
        if unknown:
            raise ValueError(f"Unknown ground_truth config keys: {sorted(unknown)}")
        obj = cls(**{k: v for k, v in cfg.items() if k in allowed})
        obj.validate()
        return obj

    def validate(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"ground_truth.scale must be > 0, got {self.scale}")
        if not (0 <= self.min_depth < self.max_depth):
            raise ValueError(f"Need 0 <= min_depth < max_depth, got {self.min_depth}, {self.max_depth}")


def _read_raw_depth(path: Path, cfg: DepthDecodeConfig) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raw = np.load(str(path), allow_pickle=False)
    elif suffix == ".npz":
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with np.load(str(path), allow_pickle=False) as data:
            if cfg.npz_key not in data:
                raise ValueError(f"Key '{cfg.npz_key}' not found in {path} (keys: {list(data.keys())})")
            raw = data[cfg.npz_key]
    elif suffix in (".png", ".tif", ".tiff", ".pfm"):
        raw = cv2.imdecode(_read_file_bytes(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Could not decode depth image: {path}")
    else:
        raise ValueError(f"Unsupported depth file extension '{suffix}' for {path}")

    raw = np.asarray(raw)
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[..., 0]
    if raw.ndim == 3:
        if cfg.channel is None:
            raise ValueError(
                f"Depth map {path} has {raw.shape[2]} channels; set ground_truth.channel "
                f"to select the channel that stores depth (encodings differ between datasets).")
        raw = raw[..., cfg.channel]
    if raw.ndim != 2:
        raise ValueError(f"Depth map must be 2-D, got shape {raw.shape} in {path}")
    if not (np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)):
        raise ValueError(f"Unsupported depth dtype {raw.dtype} in {path}")
    return raw


def decode_depth(raw: np.ndarray, cfg: DepthDecodeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a raw depth array to metres and compute its validity mask.

    Returns:
        depth: float32 array in metres, ``NaN`` where invalid.
        valid: bool array, ``True`` where the ground truth is usable.
    """
    raw_f = raw.astype(np.float64)
    valid = np.isfinite(raw_f)
    for bad in cfg.invalid_values:
        valid &= raw_f != float(bad)
    depth = raw_f * cfg.scale
    with np.errstate(invalid="ignore"):
        valid &= (depth > 0) & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
    depth = np.where(valid, depth, np.nan).astype(np.float32)
    return depth, valid


def load_depth_map(path: str | Path, cfg: Optional[DepthDecodeConfig] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Load and decode a ground-truth depth map. See :func:`decode_depth`."""
    cfg = cfg or DepthDecodeConfig()
    raw = _read_raw_depth(Path(path), cfg)
    return decode_depth(raw, cfg)


# --------------------------------------------------------------------------- #
# Pairing images with ground truth
# --------------------------------------------------------------------------- #
def find_ground_truth(image_path: str | Path, gt_dir: str | Path,
                      extensions: Sequence[str] = DEPTH_EXTENSIONS) -> Optional[Path]:
    """Return the GT file in ``gt_dir`` sharing the image's stem, or ``None``.

    Extensions are tried in the given order; the first match wins.
    """
    stem = Path(image_path).stem
    gt_dir = Path(gt_dir)
    for ext in extensions:
        for candidate in (gt_dir / f"{stem}{ext}", gt_dir / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    return None


@dataclass
class ImageDepthPair:
    name: str
    image_path: Path
    gt_path: Optional[Path]


@dataclass
class DepthSample:
    name: str
    image_path: Path
    gt_path: Path
    rgb: np.ndarray
    depth: np.ndarray   # float32 metres, NaN where invalid
    valid: np.ndarray   # bool


class PairedDepthDataset:
    """Pairs ``images/<stem>.<ext>`` with ``ground_truth/<stem>.<ext>``.

    Images without a matching ground-truth file are kept in
    :attr:`unpaired` and are **never** evaluated. Samples are loaded lazily.
    """

    def __init__(self, image_dir: str | Path, gt_dir: str | Path,
                 decode_cfg: Optional[DepthDecodeConfig] = None,
                 gt_extensions: Sequence[str] = DEPTH_EXTENSIONS,
                 recursive: bool = False):
        self.image_dir = Path(image_dir)
        self.gt_dir = Path(gt_dir)
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"Ground-truth folder not found: {self.gt_dir}")
        self.decode_cfg = decode_cfg or DepthDecodeConfig()
        images = list_images(self.image_dir, recursive=recursive)
        self.all_pairs = [ImageDepthPair(p.stem, p, find_ground_truth(p, self.gt_dir, gt_extensions))
                          for p in images]
        self.pairs = [p for p in self.all_pairs if p.gt_path is not None]
        self.unpaired = [p for p in self.all_pairs if p.gt_path is None]
        stems = [p.name for p in self.all_pairs]
        dupes = sorted({s for s in stems if stems.count(s) > 1})
        if dupes:
            raise ValueError(f"Duplicate image stems make GT pairing ambiguous: {dupes}")
        if self.unpaired:
            logger.warning("%d image(s) have no ground truth and will not be evaluated: %s",
                           len(self.unpaired), ", ".join(p.name for p in self.unpaired[:10]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> DepthSample:
        pair = self.pairs[idx]
        rgb = load_rgb_image(pair.image_path)
        depth, valid = load_depth_map(pair.gt_path, self.decode_cfg)
        return DepthSample(pair.name, pair.image_path, pair.gt_path, rgb, depth, valid)

    def __iter__(self) -> Iterator[DepthSample]:
        for i in range(len(self)):
            yield self[i]
