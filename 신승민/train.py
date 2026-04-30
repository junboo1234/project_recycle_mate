"""
RecycleMate Center-style detector training

차건우 baseline은 YOLOv8n pretrained 모델 전체를 fine-tuning한 기준선이다.
이 파일은 pretrained/Ultralytics 모델을 쓰지 않고, RecycleMate 데이터 특성에 맞춘 Center-style detector를 처음부터 학습한다.
GPU와 autograd는 성능 실험에 필수이므로 PyTorch를 사용하되, backbone/neck/head/target/loss/eval 연결은 직접 구현한다.

이 방식을 선택한 이유:
1) 전처리 문서와 baseline 분석상 데이터는 직접촬영 중심이고, 이미지당 객체가 거의 1개이며 bbox가 크게 잡힌다.
2) 그래서 범용 YOLO처럼 많은 anchor/scale을 다루기보다, 객체 중심점 heatmap과 bbox 크기/offset을 직접 예측하는 방식이 문제 정의에 더 맞다.
3) pretrained를 쓰지 않아도 "왜 이 구조가 이 데이터에 맞는지"를 설명할 수 있고, 딥러닝 과제의 직접 모델 설계 근거가 분명하다.
대안: 성능만 최우선이면 YOLOv8s/RT-DETR fine-tuning이 유리하지만, 과목 취지상 직접 설계 모델과 비교 실험으로 두는 편이 낫다.

모델링 선택:
1) 직접 만든 Conv-BN-SiLU, depthwise separable block, SE attention, top-down fusion neck으로 feature extractor를 구성했다.
2) head는 class별 center heatmap, bbox width/height, center offset을 분리해서 예측한다.
3) 단일 또는 소수 큰 객체 탐지에 집중하므로 output stride=16을 기본으로 하고 offset regression으로 중심 좌표 오차를 보정한다.
대안: 작은 객체가 많아지면 stride=8 head를 추가한 two-scale CenterNet 구조도 해볼 가치가 있다.

하이퍼파라미터 선택:
1) baseline은 70epoch 이후 plateau였지만 from-scratch 모델은 수렴이 느리므로 epochs=180, patience=30을 기본으로 둔다.
2) AdamW, warmup, cosine decay, gradient clipping을 써서 pretrained 없이도 초반 발산을 막는다.
3) baseline에서 약했던 페트병/플라스틱 class heatmap loss를 가중해 평균 성능보다 약한 클래스 개선을 우선한다.
대안: 시간이 부족하면 --epochs 60 --patience 12 --width 0.75로 빠른 1차 실험을 먼저 돌릴 수 있다.

Loss/optimization 선택:
1) center heatmap은 CenterNet 계열 focal loss를 사용해 쉬운 배경 grid가 loss를 지배하지 않게 한다.
2) bbox는 width/height와 center offset을 SmoothL1로 학습해 큰 객체 중심 데이터에서 안정적으로 수렴하도록 했다.
3) validation loss 기반 early stopping으로 10시간 이상 불필요하게 도는 상황을 줄인다.
대안: 오답 샘플이 모이면 페트병/플라스틱 hard example oversampling을 추가하는 것도 해볼 가치가 있다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_YAML = r"C:\Users\43074\Desktop\RecycleMate\dataset-shrinked\processed\yolo_major9\metadata\dataset.yaml"
DEFAULT_SAVE_DIR = "runs/recycle_center"
DEFAULT_CLASS_NAMES = ["금속캔", "종이", "페트병", "플라스틱", "스티로폼", "비닐", "유리병", "건전지", "형광등"]
BASELINE_AP50 = {
    "금속캔": 0.9514,
    "종이": 0.9354,
    "페트병": 0.8475,
    "플라스틱": 0.8568,
    "스티로폼": 0.9677,
    "비닐": 0.9660,
    "유리병": 0.9325,
    "건전지": 0.9806,
    "형광등": 0.9856,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetInfo:
    """dataset.yaml에서 읽은 train/val/test 경로와 클래스 이름을 보관한다."""

    yaml_path: Path
    root: Path
    train_images: Path
    val_images: Path
    test_images: Path
    names: List[str]


@dataclass
class TrainConfig:
    """학습 실행에 필요한 하이퍼파라미터와 경로 설정을 한곳에 묶는다."""

    data: Path
    save_dir: Path
    epochs: int
    batch: int
    imgsz: int
    lr: float
    min_lr: float
    weight_decay: float
    warmup_epochs: int
    patience: int
    workers: int
    device: str
    seed: int
    width: float
    heatmap_weight: float
    wh_weight: float
    offset_weight: float
    grad_clip: float
    save_period: int
    output_stride: int


def parse_simple_yaml(path: Path) -> Dict[str, object]:
    """PyYAML이 없을 때 dataset.yaml의 최소 구조만 직접 파싱한다."""

    data: Dict[str, object] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].split("#", 1)[0].rstrip()
        idx += 1
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key == "names" and value == "":
            names: Dict[int, str] = {}
            while idx < len(lines) and lines[idx].startswith((" ", "\t")):
                child = lines[idx].split("#", 1)[0].strip()
                idx += 1
                if child and ":" in child:
                    child_key, child_value = child.split(":", 1)
                    names[int(child_key.strip())] = child_value.strip().strip("\"'")
            data[key] = names
        elif key == "names" and value.startswith("[") and value.endswith("]"):
            data[key] = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
        else:
            data[key] = value
    return data


def load_dataset_yaml(path_like: str | Path) -> DatasetInfo:
    """YOLO 형식 dataset.yaml을 읽어 실제 이미지 폴더 경로로 변환한다."""

    yaml_path = Path(path_like).expanduser().resolve()
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        raw = parse_simple_yaml(yaml_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid dataset yaml: {yaml_path}")

    root = Path(str(raw.get("path", yaml_path.parent)))
    root = root if root.is_absolute() else (yaml_path.parent / root)
    root = root.resolve()

    def split_path(value: object, fallback: str) -> Path:
        """상대 split 경로를 dataset root 기준 절대 경로로 바꾼다."""

        path = Path(str(value if value is not None else fallback))
        return path if path.is_absolute() else (root / path)

    names_raw = raw.get("names", DEFAULT_CLASS_NAMES)
    if isinstance(names_raw, dict):
        names = [str(names_raw[idx]) for idx in sorted(int(k) for k in names_raw.keys())]
    elif isinstance(names_raw, list):
        names = [str(item) for item in names_raw]
    else:
        names = list(DEFAULT_CLASS_NAMES)

    return DatasetInfo(
        yaml_path=yaml_path,
        root=root,
        train_images=split_path(raw.get("train"), "images/train"),
        val_images=split_path(raw.get("val"), "images/val"),
        test_images=split_path(raw.get("test", raw.get("val")), "images/test"),
        names=names,
    )


def labels_dir_for_images(image_dir: Path) -> Path:
    """images/split 경로에 대응하는 labels/split 경로를 계산한다."""

    parts = list(image_dir.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def list_yolo_samples(image_dir: Path) -> List[Tuple[Path, Path]]:
    """이미지 파일과 같은 stem의 YOLO txt 라벨 파일을 샘플 목록으로 만든다."""

    label_dir = labels_dir_for_images(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    samples = []
    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTS:
            rel = image_path.relative_to(image_dir)
            samples.append((image_path, (label_dir / rel).with_suffix(".txt")))
    if not samples:
        raise RuntimeError(f"No images found under {image_dir}")
    return samples


def read_yolo_label(path: Path, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """YOLO txt 라벨을 정규화된 xywh 박스와 클래스 배열로 읽는다."""

    boxes: List[List[float]] = []
    classes: List[int] = []
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls < 0 or cls >= num_classes:
            continue
        cx, cy, w, h = [float(x) for x in parts[1:5]]
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        if w > 1e-4 and h > 1e-4:
            boxes.append([cx, cy, w, h])
            classes.append(cls)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def pil_resample_bilinear():
    """Pillow 버전 차이를 흡수해 bilinear resize 상수를 반환한다."""

    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def letterbox(image: Image.Image, boxes: np.ndarray, img_size: int) -> Tuple[Image.Image, np.ndarray]:
    """원본 비율을 유지해 정사각 입력으로 패딩하고 bbox 좌표도 함께 보정한다."""

    width, height = image.size
    scale = min(img_size / width, img_size / height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    resized = image.resize((new_w, new_h), pil_resample_bilinear())
    canvas = Image.new("RGB", (img_size, img_size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    if len(boxes) == 0:
        return canvas, boxes.astype(np.float32)
    adjusted = boxes.copy().astype(np.float32)
    adjusted[:, 0] = (boxes[:, 0] * width * scale + pad_x) / img_size
    adjusted[:, 1] = (boxes[:, 1] * height * scale + pad_y) / img_size
    adjusted[:, 2] = boxes[:, 2] * width * scale / img_size
    adjusted[:, 3] = boxes[:, 3] * height * scale / img_size
    return canvas, np.clip(adjusted, 0.0, 1.0)


def class_loss_weights(class_names: Sequence[str]) -> torch.Tensor:
    """baseline에서 약했던 클래스를 heatmap loss에서 더 크게 반영할 가중치를 만든다."""

    scores = [BASELINE_AP50[name] for name in class_names if name in BASELINE_AP50]
    if len(scores) < max(2, len(class_names) // 2):
        return torch.ones(len(class_names), dtype=torch.float32)
    mean_ap = float(np.mean(scores))
    weights = []
    for name in class_names:
        ap = BASELINE_AP50.get(name, mean_ap)
        weight = (mean_ap / max(ap, 1e-3)) ** 1.7
        if name in {"페트병", "플라스틱"}:
            weight *= 1.10
        weights.append(float(np.clip(weight, 0.85, 1.40)))
    return torch.tensor(weights, dtype=torch.float32)


class YoloBoxDataset(Dataset):
    """YOLO 이미지/라벨 폴더를 Center-style detector 학습용 샘플로 제공한다."""

    def __init__(self, image_dir: Path, class_names: Sequence[str], img_size: int, train: bool, seed: int) -> None:
        """샘플 경로, 클래스 수, 증강 여부를 초기화한다."""

        self.samples = list_yolo_samples(image_dir)
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.img_size = int(img_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.weak_class_ids = {
            idx for idx, name in enumerate(class_names) if name in {"페트병", "플라스틱", "유리병"}
        }

    def __len__(self) -> int:
        """데이터셋 이미지 개수를 반환한다."""

        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """이미지 하나와 해당 YOLO 라벨을 tensor/box/class 형태로 반환한다."""

        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        boxes, classes = read_yolo_label(label_path, self.num_classes)
        rng = random.Random(self.seed + index * 9973 + (1 if self.train else 0))
        if self.train:
            image, boxes = self.apply_augmentation(image, boxes, classes, rng)
        image, boxes = letterbox(image, boxes, self.img_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return {
            "image": torch.from_numpy(arr).permute(2, 0, 1).contiguous(),
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "classes": torch.from_numpy(classes.astype(np.int64)),
            "image_path": str(image_path),
        }

    def apply_augmentation(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        classes: np.ndarray,
        rng: random.Random,
    ) -> Tuple[Image.Image, np.ndarray]:
        """학습 샘플에 좌우반전, 색상/밝기 변화, 약한 blur 증강을 적용한다."""

        weak_sample = any(int(cls) in self.weak_class_ids for cls in classes.tolist())
        if len(boxes) > 0 and rng.random() < 0.5:
            image = ImageOps.mirror(image)
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78 if weak_sample else 0.86, 1.24 if weak_sample else 1.16))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82 if weak_sample else 0.90, 1.22 if weak_sample else 1.14))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.84 if weak_sample else 0.92, 1.18 if weak_sample else 1.10))
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.80, 1.20))
        if rng.random() < (0.20 if weak_sample else 0.10):
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))
        return image, boxes


def collate_detection(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """크기가 다른 box 목록을 유지하면서 이미지만 batch tensor로 묶는다."""

    return {
        "images": torch.stack([item["image"] for item in batch]),  # type: ignore[index]
        "boxes": [item["boxes"] for item in batch],
        "classes": [item["classes"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


class ConvBNAct(nn.Module):
    """Conv2d, BatchNorm, SiLU를 묶은 기본 convolution 블록이다."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
        """입출력 채널과 stride/group 설정으로 convolution 블록을 만든다."""

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding=kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 feature map에 Conv-BN-SiLU를 순서대로 적용한다."""

        return self.block(x)


class SqueezeExcite(nn.Module):
    """채널별 중요도를 학습해 feature map을 재가중하는 SE attention 블록이다."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        """채널 축소/복원 1x1 convolution으로 SE 모듈을 초기화한다."""

        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """전역 평균 정보로 채널 attention 값을 만들고 입력에 곱한다."""

        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        return x * torch.sigmoid(self.fc2(scale))


class DSBlock(nn.Module):
    """Depthwise separable convolution과 SE를 결합한 경량 feature 블록이다."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """depthwise, pointwise, residual 사용 여부를 초기화한다."""

        super().__init__()
        self.depthwise = ConvBNAct(in_ch, in_ch, 3, stride, groups=in_ch)
        self.pointwise = ConvBNAct(in_ch, out_ch, 1, 1)
        self.se = SqueezeExcite(out_ch)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """depthwise-pointwise-SE를 적용하고 가능하면 residual을 더한다."""

        y = self.se(self.pointwise(self.depthwise(x)))
        return x + y if self.use_residual else y


def make_divisible(value: float, divisor: int = 8) -> int:
    """채널 수를 하드웨어 친화적인 divisor 배수로 맞춘다."""

    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class CenterRecycleNet(nn.Module):
    """직접촬영 단일 객체에 맞춘 from-scratch Center-style detector이다."""

    def __init__(self, num_classes: int, width: float = 1.0) -> None:
        """backbone, fusion neck, heatmap/wh/offset head를 구성한다."""

        super().__init__()
        c1 = make_divisible(32 * width)
        c2 = make_divisible(64 * width)
        c3 = make_divisible(128 * width)
        c4 = make_divisible(192 * width)
        hidden = make_divisible(160 * width)

        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage2 = nn.Sequential(DSBlock(c1, c2, 2), DSBlock(c2, c2, 1))
        self.stage3 = nn.Sequential(DSBlock(c2, c3, 2), DSBlock(c3, c3, 1), DSBlock(c3, c3, 1))
        self.stage4 = nn.Sequential(DSBlock(c3, c4, 2), DSBlock(c4, c4, 1), DSBlock(c4, c4, 1))
        self.reduce3 = ConvBNAct(c3, hidden, 1, 1)
        self.reduce4 = ConvBNAct(c4, hidden, 1, 1)
        self.fuse = nn.Sequential(ConvBNAct(hidden * 2, hidden, 3, 1), DSBlock(hidden, hidden, 1))
        self.heatmap_head = nn.Sequential(ConvBNAct(hidden, hidden, 3, 1), nn.Conv2d(hidden, num_classes, 1))
        self.wh_head = nn.Sequential(ConvBNAct(hidden, hidden, 3, 1), nn.Conv2d(hidden, 2, 1))
        self.offset_head = nn.Sequential(ConvBNAct(hidden, hidden, 3, 1), nn.Conv2d(hidden, 2, 1))
        self.num_classes = num_classes
        self.output_stride = 16
        self._init_heads()

    def _init_heads(self) -> None:
        """초기 학습 안정화를 위해 예측 head의 weight/bias를 초기화한다."""

        for head in [self.heatmap_head, self.wh_head, self.offset_head]:
            conv = head[-1]
            if isinstance(conv, nn.Conv2d):
                nn.init.normal_(conv.weight, mean=0.0, std=0.01)
                nn.init.constant_(conv.bias, 0.0)
        heatmap_conv = self.heatmap_head[-1]
        if isinstance(heatmap_conv, nn.Conv2d):
            nn.init.constant_(heatmap_conv.bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """이미지를 받아 class heatmap, bbox 크기, center offset 예측을 반환한다."""

        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p3_down = F.avg_pool2d(self.reduce3(p3), kernel_size=2, stride=2)
        fused = self.fuse(torch.cat([p3_down, self.reduce4(p4)], dim=1))
        return {
            "heatmap": self.heatmap_head(fused),
            "wh": self.wh_head(fused),
            "offset": self.offset_head(fused),
        }


def gaussian2d(radius: int, sigma: float) -> np.ndarray:
    """center heatmap target에 찍을 2D Gaussian kernel을 만든다."""

    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian.astype(np.float32)


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    """지정한 grid 중심에 Gaussian peak를 그려 center target을 만든다."""

    radius = max(int(radius), 0)
    gaussian = torch.from_numpy(gaussian2d(radius, sigma=max((2 * radius + 1) / 6, 1e-3))).to(heatmap.device)
    height, width = heatmap.shape
    left, right = min(cx, radius), min(width - cx - 1, radius)
    top, bottom = min(cy, radius), min(height - cy - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    patch = heatmap[cy - top : cy + bottom + 1, cx - left : cx + right + 1]
    gauss_patch = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    torch.maximum(patch, gauss_patch, out=patch)


def build_center_targets(
    boxes_list: Sequence[torch.Tensor],
    classes_list: Sequence[torch.Tensor],
    grid_h: int,
    grid_w: int,
    num_classes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """YOLO box 라벨을 heatmap, width/height, offset target으로 변환한다."""

    batch_size = len(boxes_list)
    heatmap = torch.zeros((batch_size, num_classes, grid_h, grid_w), device=device)
    wh = torch.zeros((batch_size, 2, grid_h, grid_w), device=device)
    offset = torch.zeros((batch_size, 2, grid_h, grid_w), device=device)
    pos_mask = torch.zeros((batch_size, 1, grid_h, grid_w), dtype=torch.bool, device=device)
    cls_map = torch.full((batch_size, grid_h, grid_w), -1, dtype=torch.long, device=device)

    for b_idx, (boxes, classes) in enumerate(zip(boxes_list, classes_list)):
        boxes = boxes.to(device=device, dtype=torch.float32)
        classes = classes.to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            continue
        order = torch.argsort(boxes[:, 2] * boxes[:, 3], descending=True)
        for obj_idx in order.tolist():
            cx, cy, bw, bh = boxes[obj_idx]
            cls = int(classes[obj_idx].item())
            gx_float = torch.clamp(cx * grid_w, 0, grid_w - 1e-4)
            gy_float = torch.clamp(cy * grid_h, 0, grid_h - 1e-4)
            gx = int(gx_float.floor().item())
            gy = int(gy_float.floor().item())
            radius = int(max(1, min(float(bw * grid_w), float(bh * grid_h)) * 0.30))
            draw_gaussian(heatmap[b_idx, cls], gx, gy, radius)
            if pos_mask[b_idx, 0, gy, gx]:
                continue
            pos_mask[b_idx, 0, gy, gx] = True
            cls_map[b_idx, gy, gx] = cls
            wh[b_idx, :, gy, gx] = torch.stack([bw, bh]).clamp(1e-4, 1.0)
            offset[b_idx, :, gy, gx] = torch.stack([gx_float - gx, gy_float - gy]).clamp(0.0, 1.0)
    return {"heatmap": heatmap, "wh": wh, "offset": offset, "pos_mask": pos_mask, "cls_map": cls_map}


def center_focal_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """CenterNet 방식 focal loss로 center heatmap 예측을 학습한다."""

    pred = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    pos = targets.eq(1.0).float()
    neg = targets.lt(1.0).float()
    neg_weights = (1.0 - targets).pow(4)
    pos_class_weights = class_weights.view(1, -1, 1, 1).to(device=logits.device, dtype=logits.dtype)
    pos_loss = -torch.log(pred) * (1.0 - pred).pow(2) * pos * pos_class_weights
    neg_loss = -torch.log(1.0 - pred) * pred.pow(2) * neg_weights * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


class CenterDetectionLoss(nn.Module):
    """Center heatmap loss와 bbox 크기/offset regression loss를 합산한다."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        heatmap_weight: float,
        wh_weight: float,
        offset_weight: float,
    ) -> None:
        """class 가중치와 각 loss 항의 비중을 초기화한다."""

        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.heatmap_weight = float(heatmap_weight)
        self.wh_weight = float(wh_weight)
        self.offset_weight = float(offset_weight)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        boxes_list: Sequence[torch.Tensor],
        classes_list: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """모델 출력과 box/class 라벨로 전체 학습 loss를 계산한다."""

        heatmap_logits = pred["heatmap"]
        grid_h, grid_w = heatmap_logits.shape[-2:]
        targets = build_center_targets(
            boxes_list,
            classes_list,
            grid_h,
            grid_w,
            heatmap_logits.shape[1],
            heatmap_logits.device,
        )
        hm_loss = center_focal_loss(heatmap_logits, targets["heatmap"], self.class_weights)
        pos_cells = targets["pos_mask"].squeeze(1)
        if pos_cells.any():
            pred_wh = torch.sigmoid(pred["wh"]).permute(0, 2, 3, 1)[pos_cells]
            pred_offset = torch.sigmoid(pred["offset"]).permute(0, 2, 3, 1)[pos_cells]
            target_wh = targets["wh"].permute(0, 2, 3, 1)[pos_cells]
            target_offset = targets["offset"].permute(0, 2, 3, 1)[pos_cells]
            wh_loss = F.smooth_l1_loss(pred_wh, target_wh)
            offset_loss = F.smooth_l1_loss(pred_offset, target_offset)
        else:
            wh_loss = pred["wh"].sum() * 0.0
            offset_loss = pred["offset"].sum() * 0.0
        total = self.heatmap_weight * hm_loss + self.wh_weight * wh_loss + self.offset_weight * offset_loss
        items = {
            "total": float(total.detach().cpu()),
            "heatmap": float(hm_loss.detach().cpu()),
            "wh": float(wh_loss.detach().cpu()),
            "offset": float(offset_loss.detach().cpu()),
        }
        return total, items


def set_seed(seed: int) -> None:
    """Python, NumPy, PyTorch 난수 seed를 고정해 실험 재현성을 높인다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """auto/cpu/cuda 입력을 실제 torch.device로 변환한다."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    return device


def create_loaders(info: DatasetInfo, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """train/val Dataset을 만들고 DataLoader로 감싼다."""

    train_ds = YoloBoxDataset(info.train_images, info.names, cfg.imgsz, train=True, seed=cfg.seed)
    val_ds = YoloBoxDataset(info.val_images, info.names, cfg.imgsz, train=False, seed=cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection,
        drop_last=False,
    )
    return train_loader, val_loader


def current_lr(epoch: int, cfg: TrainConfig) -> float:
    """warmup 이후 cosine decay를 적용한 현재 epoch 학습률을 계산한다."""

    if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
        return cfg.lr * float(epoch + 1) / float(cfg.warmup_epochs)
    denom = max(cfg.epochs - cfg.warmup_epochs, 1)
    progress = min(max(float(epoch - cfg.warmup_epochs) / float(denom), 0.0), 1.0)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """optimizer의 모든 parameter group 학습률을 갱신한다."""

    for group in optimizer.param_groups:
        group["lr"] = lr


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CenterDetectionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> Dict[str, float]:
    """한 epoch 동안 train 또는 validation loop를 실행하고 평균 loss를 반환한다."""

    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "heatmap": 0.0, "wh": 0.0, "offset": 0.0}
    batches = 0
    for batch in loader:
        images = batch["images"].to(device=device, dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred = model(images)
            loss, items = criterion(pred, batch["boxes"], batch["classes"])
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
        for key in totals:
            totals[key] += items[key]
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 모델 파라미터 수를 계산한다."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: Dict[str, float],
    val_loss: Dict[str, float],
    info: DatasetInfo,
    cfg: TrainConfig,
    best_val: float,
) -> None:
    """모델/optimizer 상태와 실험 설정을 checkpoint 파일로 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_type": "center_recycle_from_scratch",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "class_names": info.names,
            "num_classes": len(info.names),
            "img_size": cfg.imgsz,
            "width": cfg.width,
            "output_stride": cfg.output_stride,
            "data": str(info.yaml_path),
            "hyperparameters": {
                "epochs": cfg.epochs,
                "batch": cfg.batch,
                "lr": cfg.lr,
                "min_lr": cfg.min_lr,
                "weight_decay": cfg.weight_decay,
                "warmup_epochs": cfg.warmup_epochs,
                "patience": cfg.patience,
                "heatmap_weight": cfg.heatmap_weight,
                "wh_weight": cfg.wh_weight,
                "offset_weight": cfg.offset_weight,
                "grad_clip": cfg.grad_clip,
                "pretrained": False,
            },
        },
        path,
    )


def save_loss_curve(save_dir: Path, history: List[Dict[str, float]]) -> None:
    """loss history JSON과 train/val loss curve 이미지를 저장한다."""

    (save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in history]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [row["train_total"] for row in history], label="Train Loss")
        plt.plot(epochs, [row["val_total"] for row in history], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CenterRecycleNet Training & Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "loss_curve.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"loss curve image skipped: {exc}")


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    """CLI 인자를 TrainConfig dataclass로 파싱한다."""

    parser = argparse.ArgumentParser(description="Train RecycleMate Center-style detector from scratch")
    parser.add_argument("--data", default=DEFAULT_DATASET_YAML, help="dataset.yaml path")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--width", type=float, default=1.0, help="model width multiplier")
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--wh-weight", type=float, default=2.5)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--output-stride", type=int, default=16, choices=[16], help="Center map stride. Current model uses stride 16.")
    args = parser.parse_args(argv)
    return TrainConfig(
        data=Path(args.data),
        save_dir=Path(args.save_dir),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        width=args.width,
        heatmap_weight=args.heatmap_weight,
        wh_weight=args.wh_weight,
        offset_weight=args.offset_weight,
        grad_clip=args.grad_clip,
        save_period=args.save_period,
        output_stride=args.output_stride,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """학습 전체 파이프라인을 실행하고 best/last checkpoint를 남긴다."""

    cfg = parse_args(argv)
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    info = load_dataset_yaml(cfg.data)
    train_loader, val_loader = create_loaders(info, cfg)
    model = CenterRecycleNet(num_classes=len(info.names), width=cfg.width).to(device)
    class_weights = class_loss_weights(info.names).to(device)
    criterion = CenterDetectionLoss(class_weights, cfg.heatmap_weight, cfg.wh_weight, cfg.offset_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("=" * 70)
    print("RecycleMate Center-style Detector Training")
    print("=" * 70)
    print(f"dataset: {info.yaml_path}")
    print(f"train:   {info.train_images} ({len(train_loader.dataset)} images)")
    print(f"val:     {info.val_images} ({len(val_loader.dataset)} images)")
    print(f"classes: {len(info.names)} -> {info.names}")
    print("model:   center_recycle_from_scratch (no pretrained weights)")
    print(f"device:  {device}")
    print(f"params:  {count_parameters(model):,}")
    print(f"class heatmap weights: {[round(float(x), 3) for x in class_weights.detach().cpu()]}")
    print("=" * 70)

    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        start = time.time()
        lr_now = current_lr(epoch, cfg)
        set_optimizer_lr(optimizer, lr_now)
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, cfg.grad_clip)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, device, None, cfg.grad_clip)
        elapsed = time.time() - start
        history.append(
            {
                "epoch": epoch + 1,
                "lr": lr_now,
                "train_total": train_loss["total"],
                "train_heatmap": train_loss["heatmap"],
                "train_wh": train_loss["wh"],
                "train_offset": train_loss["offset"],
                "val_total": val_loss["total"],
                "val_heatmap": val_loss["heatmap"],
                "val_wh": val_loss["wh"],
                "val_offset": val_loss["offset"],
                "seconds": elapsed,
            }
        )
        print(
            f"Epoch [{epoch + 1:3d}/{cfg.epochs}] "
            f"train={train_loss['total']:.4f} "
            f"val={val_loss['total']:.4f} "
            f"(hm={val_loss['heatmap']:.4f}, wh={val_loss['wh']:.4f}, off={val_loss['offset']:.4f}) "
            f"lr={lr_now:.6f} time={elapsed:.1f}s"
        )

        improved = val_loss["total"] < best_val - 1e-5
        if improved:
            best_val = val_loss["total"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(cfg.save_dir / "best.pt", model, optimizer, epoch, train_loss, val_loss, info, cfg, best_val)
            print(f"  >>> best.pt saved (val_loss={best_val:.4f})")
        else:
            stale_epochs += 1

        if cfg.save_period > 0 and (epoch + 1) % cfg.save_period == 0:
            save_checkpoint(
                cfg.save_dir / f"epoch_{epoch + 1}.pt",
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                info,
                cfg,
                best_val,
            )
        save_checkpoint(cfg.save_dir / "last.pt", model, optimizer, epoch, train_loss, val_loss, info, cfg, best_val)
        save_loss_curve(cfg.save_dir, history)

        if stale_epochs >= cfg.patience:
            print(f"Early stopping: no val improvement for {cfg.patience} epochs.")
            break

    print("=" * 70)
    print(f"training finished. best_epoch={best_epoch + 1}, best_val_loss={best_val:.4f}")
    print(f"saved to: {cfg.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    freeze_support()
    main()
