"""
RecycleMate Center-style detector evaluation

평가 방법 선택:
1) baseline의 mAP@0.5만으로는 느슨한 IoU 기준 성능만 보이므로, mAP@0.5와 mAP@0.5:0.95를 함께 계산한다.
2) 페트병/플라스틱처럼 헷갈리는 클래스가 핵심 병목이므로, background 행/열을 포함한 confusion matrix를 저장한다.
3) 단일 이미지 latency, batch throughput, NMS 포함 시간을 함께 측정해 정확도와 실시간성을 분리해 분석한다.
대안: 발표 전에는 실제 카메라 프레임 100장 정도를 따로 모아 field-test confusion matrix를 추가하는 것도 해볼 가치가 있다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train import (  # noqa: E402
    DEFAULT_DATASET_YAML,
    DEFAULT_SAVE_DIR,
    CenterRecycleNet,
    YoloBoxDataset,
    choose_device,
    collate_detection,
    load_dataset_yaml,
)


DEFAULT_CHECKPOINT = str(Path(DEFAULT_SAVE_DIR) / "best.pt")


def xywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    """정규화된 xywh box를 xyxy box로 변환한다."""

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=1).clip(0.0, 1.0)


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """두 xyxy box 집합 사이의 IoU 행렬을 계산한다."""

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    tl = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    br = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0.0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0.0, None)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-9)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """같은 클래스 예측 box들에 대해 confidence 기준 NMS를 수행한다."""

    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou_np(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return keep


def decode_center_predictions(
    pred: Dict[str, torch.Tensor],
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
) -> List[np.ndarray]:
    """Return per-image detections as [x1,y1,x2,y2,score,class] in normalized coordinates."""

    heatmap = torch.sigmoid(pred["heatmap"])
    wh = torch.sigmoid(pred["wh"])
    offset = torch.sigmoid(pred["offset"])
    pooled = torch.nn.functional.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep_peaks = heatmap.eq(pooled)
    heatmap = heatmap * keep_peaks
    batch, num_classes, grid_h, grid_w = heatmap.shape
    outputs: List[np.ndarray] = []

    for b_idx in range(batch):
        scores, flat_indices = torch.topk(heatmap[b_idx].reshape(-1), k=min(max_det * 4, heatmap[b_idx].numel()))
        det_rows: List[List[float]] = []
        for score, flat_idx in zip(scores, flat_indices):
            score_value = float(score.item())
            if score_value < conf_thresh:
                continue
            flat = int(flat_idx.item())
            cls_id = flat // (grid_h * grid_w)
            rem = flat % (grid_h * grid_w)
            y = rem // grid_w
            x = rem % grid_w
            off_x = float(offset[b_idx, 0, y, x].item())
            off_y = float(offset[b_idx, 1, y, x].item())
            bw = float(wh[b_idx, 0, y, x].item())
            bh = float(wh[b_idx, 1, y, x].item())
            cx = (x + off_x) / grid_w
            cy = (y + off_y) / grid_h
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(1.0, cx + bw / 2)
            y2 = min(1.0, cy + bh / 2)
            if x2 > x1 and y2 > y1:
                det_rows.append([x1, y1, x2, y2, score_value, float(cls_id)])

        if not det_rows:
            outputs.append(np.zeros((0, 6), dtype=np.float32))
            continue
        det = np.asarray(det_rows, dtype=np.float32)
        keep_all: List[int] = []
        for cls in np.unique(det[:, 5].astype(np.int64)):
            idxs = np.where(det[:, 5].astype(np.int64) == cls)[0]
            keep_rel = nms_numpy(det[idxs, :4], det[idxs, 4], iou_thresh)
            keep_all.extend([int(idxs[k]) for k in keep_rel])
        keep_all = sorted(keep_all, key=lambda idx: float(det[idx, 4]), reverse=True)[:max_det]
        outputs.append(det[keep_all].astype(np.float32))
    return outputs


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolation 방식으로 AP 값을 계산한다."""

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        valid = mpre[mrec >= threshold]
        ap += (valid.max() if valid.size else 0.0) / 101.0
    return float(ap)


def evaluate_class_ap(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    num_classes: int,
    iou_threshold: float,
) -> Tuple[List[float], List[float], List[float]]:
    """클래스별 AP, precision, recall을 지정 IoU threshold에서 계산한다."""

    ap_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []

    for cls_id in range(num_classes):
        all_tp: List[int] = []
        all_scores: List[float] = []
        total_gt = 0
        for det, target in zip(predictions, targets):
            gt_mask = target[:, 0].astype(np.int64) == cls_id if len(target) else np.zeros((0,), dtype=bool)
            gt_boxes = xywh_to_xyxy_np(target[gt_mask, 1:5]) if len(target) else np.zeros((0, 4), dtype=np.float32)
            total_gt += len(gt_boxes)
            pred_mask = det[:, 5].astype(np.int64) == cls_id if len(det) else np.zeros((0,), dtype=bool)
            cls_det = det[pred_mask] if len(det) else np.zeros((0, 6), dtype=np.float32)
            if len(cls_det) == 0:
                continue
            cls_det = cls_det[cls_det[:, 4].argsort()[::-1]]
            matched: set[int] = set()
            for row in cls_det:
                all_scores.append(float(row[4]))
                if len(gt_boxes) == 0:
                    all_tp.append(0)
                    continue
                ious = box_iou_np(row[:4][None, :], gt_boxes)[0]
                best_idx = int(np.argmax(ious))
                if float(ious[best_idx]) >= iou_threshold and best_idx not in matched:
                    all_tp.append(1)
                    matched.add(best_idx)
                else:
                    all_tp.append(0)
        if total_gt == 0 or not all_scores:
            ap_list.append(0.0)
            precision_list.append(0.0)
            recall_list.append(0.0)
            continue
        order = np.argsort(-np.asarray(all_scores, dtype=np.float32))
        tp = np.asarray(all_tp, dtype=np.float32)[order]
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(total_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap_list.append(compute_ap(recall, precision))
        precision_list.append(float(precision[-1]) if precision.size else 0.0)
        recall_list.append(float(recall[-1]) if recall.size else 0.0)
    return ap_list, precision_list, recall_list


def evaluate_map_suite(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray], num_classes: int) -> Dict[str, object]:
    """mAP@0.5와 COCO식 mAP@0.5:0.95를 함께 계산한다."""

    ap50, precision, recall = evaluate_class_ap(predictions, targets, num_classes, 0.50)
    thresholds = np.arange(0.50, 0.96, 0.05)
    ap_by_threshold = []
    for threshold in thresholds:
        ap_thr, _, _ = evaluate_class_ap(predictions, targets, num_classes, float(threshold))
        ap_by_threshold.append(ap_thr)
    ap_matrix = np.asarray(ap_by_threshold, dtype=np.float32)
    return {
        "mAP@0.5": float(np.mean(ap50)) if ap50 else 0.0,
        "mAP@0.5:0.95": float(np.mean(ap_matrix)) if ap_matrix.size else 0.0,
        "ap50": ap50,
        "ap5095": ap_matrix.mean(axis=0).tolist() if ap_matrix.size else [0.0] * num_classes,
        "precision": precision,
        "recall": recall,
    }


def confusion_matrix(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray], num_classes: int, iou_threshold: float) -> np.ndarray:
    """background 행/열을 포함한 detection confusion matrix를 만든다."""

    bg = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for det, target in zip(predictions, targets):
        gt_classes = target[:, 0].astype(np.int64) if len(target) else np.zeros((0,), dtype=np.int64)
        gt_boxes = xywh_to_xyxy_np(target[:, 1:5]) if len(target) else np.zeros((0, 4), dtype=np.float32)
        pred_classes = det[:, 5].astype(np.int64) if len(det) else np.zeros((0,), dtype=np.int64)
        pred_boxes = det[:, :4] if len(det) else np.zeros((0, 4), dtype=np.float32)
        pred_scores = det[:, 4] if len(det) else np.zeros((0,), dtype=np.float32)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        if len(gt_boxes) and len(pred_boxes):
            pairs = []
            ious = box_iou_np(pred_boxes, gt_boxes)
            for pred_idx in range(len(pred_boxes)):
                for gt_idx in range(len(gt_boxes)):
                    if ious[pred_idx, gt_idx] >= iou_threshold:
                        pairs.append((float(pred_scores[pred_idx]), float(ious[pred_idx, gt_idx]), pred_idx, gt_idx))
            pairs.sort(reverse=True)
            for _, _, pred_idx, gt_idx in pairs:
                if pred_idx in matched_pred or gt_idx in matched_gt:
                    continue
                matrix[int(gt_classes[gt_idx]), int(pred_classes[pred_idx])] += 1
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        for gt_idx, true_cls in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                matrix[int(true_cls), bg] += 1
        for pred_idx, pred_cls in enumerate(pred_classes):
            if pred_idx not in matched_pred:
                matrix[bg, int(pred_cls)] += 1
    return matrix


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """test DataLoader를 순회하며 예측 결과와 정답 라벨을 수집한다."""

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(device=device, dtype=torch.float32)
            detections = decode_center_predictions(model(images), conf_thresh, iou_thresh, max_det)
            predictions.extend(detections)
            for boxes, classes in zip(batch["boxes"], batch["classes"]):
                if len(boxes) == 0:
                    targets.append(np.zeros((0, 5), dtype=np.float32))
                else:
                    targets.append(
                        np.concatenate(
                            [classes.numpy().astype(np.float32)[:, None], boxes.numpy().astype(np.float32)],
                            axis=1,
                        )
                    )
            if (batch_idx + 1) % 50 == 0:
                print(f"  processed {batch_idx + 1}/{len(loader)} batches")
    return predictions, targets


def synchronize_if_cuda(device: torch.device) -> None:
    """CUDA timing 측정 전후에 GPU 작업을 동기화한다."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def speed_test(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int,
    batch_size: int,
    conf_thresh: float,
    iou_thresh: float,
) -> Dict[str, float]:
    """dummy 입력으로 단일/배치 추론 latency와 throughput을 측정한다."""

    model.eval()
    warmups = 10 if device.type == "cuda" else 2
    runs = 100 if device.type == "cuda" else 10
    batch_runs = 30 if device.type == "cuda" else 5
    with torch.no_grad():
        dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
        for _ in range(warmups):
            _ = decode_center_predictions(model(dummy), conf_thresh, iou_thresh, max_det=100)
        latencies = []
        for _ in range(runs):
            dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
            synchronize_if_cuda(device)
            start = time.perf_counter()
            _ = decode_center_predictions(model(dummy), conf_thresh, iou_thresh, max_det=100)
            synchronize_if_cuda(device)
            latencies.append((time.perf_counter() - start) * 1000.0)
        batch_latencies = []
        for _ in range(batch_runs):
            dummy_batch = torch.randn(batch_size, 3, imgsz, imgsz, device=device)
            synchronize_if_cuda(device)
            start = time.perf_counter()
            _ = decode_center_predictions(model(dummy_batch), conf_thresh, iou_thresh, max_det=100)
            synchronize_if_cuda(device)
            batch_latencies.append((time.perf_counter() - start) * 1000.0)
    single_mean = float(np.mean(latencies))
    single_std = float(np.std(latencies))
    batch_mean = float(np.mean(batch_latencies))
    return {
        "single_latency_ms": single_mean,
        "single_latency_std_ms": single_std,
        "single_fps": 1000.0 / max(single_mean, 1e-9),
        "batch_size": float(batch_size),
        "batch_latency_ms": batch_mean,
        "batch_throughput_img_s": batch_size / max(batch_mean / 1000.0, 1e-9),
        "runs": float(runs),
    }


def save_confusion_outputs(save_dir: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    """confusion matrix를 CSV와 PNG 이미지로 저장한다."""

    labels = list(class_names) + ["background"]
    with (save_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size = max(7, len(labels) * 0.75)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (IoU >= 0.5)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(save_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"confusion matrix image skipped: {exc}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """평가 스크립트의 CLI 인자를 파싱한다."""

    parser = argparse.ArgumentParser(description="Evaluate RecycleMate Center-style detector")
    parser.add_argument("--data", default=DEFAULT_DATASET_YAML, help="dataset.yaml path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--save-dir", default=str(Path(DEFAULT_SAVE_DIR) / "eval"))
    parser.add_argument("--imgsz", type=int, default=0, help="0 means use checkpoint img_size")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=float, default=0.0, help="0 means use checkpoint width")
    parser.add_argument("--conf-thresh", type=float, default=0.20)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """checkpoint를 로드하고 정확도/속도 평가 결과를 저장한다."""

    args = parse_args(argv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    info = load_dataset_yaml(args.data)
    class_names = list(checkpoint.get("class_names", info.names))
    imgsz = int(args.imgsz or checkpoint.get("img_size", 640))
    width = float(args.width or checkpoint.get("width", 1.0))

    model = CenterRecycleNet(num_classes=len(class_names), width=width).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    test_dataset = YoloBoxDataset(info.test_images, class_names, imgsz, train=False, seed=20260430)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection,
        drop_last=False,
    )

    print("=" * 70)
    print("RecycleMate Center-style Detector Evaluation")
    print("=" * 70)
    print(f"checkpoint: {args.checkpoint}")
    print("model:      center_recycle_from_scratch")
    print(f"dataset:    {info.yaml_path}")
    print(f"test:       {info.test_images} ({len(test_dataset)} images)")
    print(f"device:     {device}")
    print(f"imgsz:      {imgsz}")
    print("=" * 70)

    predictions, targets = collect_predictions(model, test_loader, device, args.conf_thresh, args.iou_thresh, args.max_det)
    metric_pack = evaluate_map_suite(predictions, targets, len(class_names))
    matrix = confusion_matrix(predictions, targets, len(class_names), iou_threshold=0.50)
    speed = speed_test(model, device, imgsz, args.batch, args.conf_thresh, args.iou_thresh)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "AP@0.5": round(float(metric_pack["ap50"][idx]), 4),  # type: ignore[index]
            "AP@0.5:0.95": round(float(metric_pack["ap5095"][idx]), 4),  # type: ignore[index]
            "Precision": round(float(metric_pack["precision"][idx]), 4),  # type: ignore[index]
            "Recall": round(float(metric_pack["recall"][idx]), 4),  # type: ignore[index]
        }

    save_confusion_outputs(save_dir, matrix, class_names)
    results = {
        "model": "center_recycle_from_scratch",
        "pretrained": False,
        "checkpoint": str(args.checkpoint),
        "dataset": str(info.yaml_path),
        "test_images": str(info.test_images),
        "img_size": imgsz,
        "width": width,
        "conf_thresh": args.conf_thresh,
        "iou_thresh": args.iou_thresh,
        "accuracy": {
            "mAP@0.5": round(float(metric_pack["mAP@0.5"]), 4),
            "mAP@0.5:0.95": round(float(metric_pack["mAP@0.5:0.95"]), 4),
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_labels": list(class_names) + ["background"],
        },
        "speed": {key: round(value, 3) for key, value in speed.items()},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "checkpoint_val_loss": checkpoint.get("val_loss"),
    }
    result_path = save_dir / "eval_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mAP@0.5:      {results['accuracy']['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['accuracy']['mAP@0.5:0.95']:.4f}")
    print(f"single latency: {results['speed']['single_latency_ms']:.2f} ms")
    print(f"batch throughput: {results['speed']['batch_throughput_img_s']:.1f} img/s")
    print(f"results saved: {result_path}")


if __name__ == "__main__":
    freeze_support()
    main()
