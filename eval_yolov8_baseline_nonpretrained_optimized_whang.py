"""
YOLOv8 Baseline 평가 - 정확도(mAP) & 속도(FPS, Latency)
세종대학교 Recycle Mate 프로젝트
"""

import os
import time
import json
import torch
import numpy as np
from multiprocessing import freeze_support
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import box_iou

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_YAML = r"C:\Users\43074\Desktop\RecycleMate\dataset-shrinked\processed\yolo_major9\metadata\dataset.yaml"
CHECKPOINT = r"runs\baseline\best.pt"
MODEL_NAME = "yolov8n.yaml"
IMG_SIZE = 640
BATCH_SIZE = 32
# [최적화]0.25->0.40으로 변경.저사양 기기일수록 딥러닝 모델 연산 후의 후처리(NMS, 박스 겹침 제거)에서 병목 현상이 발생합니다. 임계값을 0.40으로 높이면 '자신 없는' 예측값들을 초기 단계에서 아예 버리게 되어 CPU 연산 부하가 크게 줄어듭니다. 재활용 안내 시 확실한 것만 대답하는 것이 서비스 품질에도 좋음
CONF_THRESH = 0.40
IOU_THRESH = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "runs/baseline/eval"

CLASS_NAMES = [
    "금속캔",
    "종이",
    "페트병",
    "플라스틱",
    "스티로폼",
    "비닐",
    "유리병",
    "건전지",
    "형광등",
]
NC = len(CLASS_NAMES)


def extract_preds(preds):
    """모델 출력에서 NMS 입력 형태로 변환"""
    if isinstance(preds, dict):
        boxes = preds["boxes"]
        scores = preds["scores"]
        return torch.cat([boxes, scores], dim=-1).permute(0, 2, 1)
    if isinstance(preds, (list, tuple)):
        # 첫 번째 원소가 텐서면 그걸 사용
        for p in preds:
            if isinstance(p, torch.Tensor) and p.dim() == 3:
                return p
        return preds[0]
    return preds


def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(preds, conf_thresh=0.25, iou_thresh=0.45):
    output = []
    if preds.dim() == 3:
        preds = preds.transpose(1, 2)

    for pred in preds:
        boxes = pred[:, :4]
        cls_scores = pred[:, 4:]
        max_scores, max_cls = cls_scores.max(dim=1)
        mask = max_scores > conf_thresh
        boxes = boxes[mask]
        scores = max_scores[mask]
        classes = max_cls[mask]

        if len(boxes) == 0:
            output.append(torch.zeros((0, 6), device=pred.device))
            continue

        boxes_xyxy = xywh2xyxy(boxes)
        keep_all = []

        for cls_id in classes.unique():
            cls_mask = classes == cls_id
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores_f = scores[cls_mask]
            order = cls_scores_f.argsort(descending=True)
            keep = []

            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break
                ious = box_iou(cls_boxes[i : i + 1], cls_boxes[order[1:]])[0]
                remaining = (ious <= iou_thresh).nonzero(as_tuple=True)[0] + 1
                order = order[remaining]

            for k in keep:
                keep_all.append(
                    torch.cat(
                        [
                            cls_boxes[k],
                            cls_scores_f[k : k + 1],
                            cls_id.unsqueeze(0).float(),
                        ]
                    )
                )

        if keep_all:
            output.append(torch.stack(keep_all))
        else:
            output.append(torch.zeros((0, 6), device=pred.device))

    return output


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_rec = mpre[mrec >= t]
        if len(prec_at_rec) > 0:
            ap += prec_at_rec.max() / 101
    return ap


def evaluate_map(all_preds, all_targets, nc, iou_threshold=0.5):
    ap_per_class = []
    precision_per_class = []
    recall_per_class = []

    for cls_id in range(nc):
        all_tp = []
        all_conf = []
        n_gt = 0

        for preds, targets in zip(all_preds, all_targets):
            gt_mask = targets[:, 0] == cls_id
            gt_boxes = targets[gt_mask]
            n_gt += len(gt_boxes)

            if len(preds) == 0:
                continue
            pred_mask = preds[:, 5] == cls_id
            cls_preds = preds[pred_mask]

            if len(cls_preds) == 0 or len(gt_boxes) == 0:
                all_tp.extend([0] * len(cls_preds))
                all_conf.extend(
                    cls_preds[:, 4].cpu().tolist() if len(cls_preds) > 0 else []
                )
                continue

            gt_xyxy = xywh2xyxy(gt_boxes[:, 1:5] * IMG_SIZE)
            matched = set()
            order = cls_preds[:, 4].argsort(descending=True)
            cls_preds = cls_preds[order]

            for pred in cls_preds:
                all_conf.append(pred[4].item())
                if len(gt_xyxy) == 0:
                    all_tp.append(0)
                    continue
                ious = box_iou(pred[:4].unsqueeze(0), gt_xyxy)[0]
                best_iou, best_idx = (
                    ious.max(0) if ious.dim() == 1 else (ious.max(), ious.argmax())
                )
                best_idx = best_idx.item()
                if best_iou >= iou_threshold and best_idx not in matched:
                    all_tp.append(1)
                    matched.add(best_idx)
                else:
                    all_tp.append(0)

        if n_gt == 0 or len(all_conf) == 0:
            ap_per_class.append(0.0)
            precision_per_class.append(0.0)
            recall_per_class.append(0.0)
            continue

        indices = np.argsort(-np.array(all_conf))
        tp = np.array(all_tp)[indices]
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(1 - tp)
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recall, precision)
        ap_per_class.append(ap)
        precision_per_class.append(precision[-1] if len(precision) > 0 else 0)
        recall_per_class.append(recall[-1] if len(recall) > 0 else 0)

    return ap_per_class, precision_per_class, recall_per_class


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 모델 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("=" * 60)
    print("1. 모델 로드")
    print("=" * 60)

    yolo = YOLO(MODEL_NAME)
    model = yolo.model.to(DEVICE)

    hyp = model.args if hasattr(model, "args") else {}
    if isinstance(hyp, dict):
        model.args = SimpleNamespace(**{"box": 7.5, "cls": 0.5, "dfl": 1.5, **hyp})

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    # [최적화]model.fuse()를 추가함으로써딥러닝 모델의 핵심인 Convolution 레이어와 Batch Normalization 레이어를 수학적으로 하나의 레이어로 병합(Fuse)합니다. 결과값은 동일하지만 연산 단계가 줄어들어 추론 속도(FPS)가 즉각적으로 상승
    model.fuse()
    model.eval()

    print(f"   체크포인트: {CHECKPOINT}")
    print(f"   학습 epoch: {ckpt.get('epoch', '?')}")
    print(f"   학습 val_loss: {ckpt.get('val_loss', '?'):.4f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Test 데이터로더
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("2. Test 데이터 로드")
    print("=" * 60)

    cfg = get_cfg(
        overrides={
            "data": DATASET_YAML,
            "imgsz": IMG_SIZE,
            "batch": BATCH_SIZE,
            "task": "detect",
            "mode": "val",
        }
    )
    data_info = check_det_dataset(cfg.data)
    test_path = data_info.get("test", data_info["val"])
    print(f"   Test 경로: {test_path}")

    test_dataset = build_yolo_dataset(
        cfg, img_path=test_path, batch=BATCH_SIZE, data=data_info, mode="val"
    )
    test_loader = build_dataloader(test_dataset, batch=BATCH_SIZE, workers=4, rank=-1)
    print(f"   Test 배치 수: {len(test_loader)}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. 정확도 평가 (mAP)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("3. 정확도 평가 (mAP@0.5)")
    print("=" * 60)

    all_preds = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["img"].to(DEVICE).float() / 255.0
            preds = model(images)
            preds = extract_preds(preds)
            detections = non_max_suppression(preds, CONF_THRESH, IOU_THRESH)

            batch_idx = batch["batch_idx"]
            cls_targets = batch["cls"]
            bboxes = batch["bboxes"]

            for j in range(images.shape[0]):
                mask = batch_idx == j
                if mask.sum() > 0:
                    targets_j = torch.cat([cls_targets[mask], bboxes[mask]], dim=1)
                else:
                    targets_j = torch.zeros((0, 5))
                all_targets.append(targets_j.to(DEVICE))
                all_preds.append(
                    detections[j]
                    if j < len(detections)
                    else torch.zeros((0, 6), device=DEVICE)
                )

            if (i + 1) % 50 == 0:
                print(f"   처리 중... {i+1}/{len(test_loader)} 배치")

    ap_list, prec_list, rec_list = evaluate_map(
        all_preds, all_targets, NC, iou_threshold=0.5
    )
    mAP50 = np.mean(ap_list)

    print(f"\n{'클래스':<12} {'AP@0.5':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 42)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<12} {ap_list[i]:>8.4f} {prec_list[i]:>10.4f} {rec_list[i]:>8.4f}"
        )
    print("-" * 42)
    print(f"{'mAP@0.5':<12} {mAP50:>8.4f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 속도 평가 (Latency & FPS)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("4. 속도 평가 (Latency & FPS)")
    print("=" * 60)

    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for _ in range(10):
        _ = model(dummy)

    NUM_RUNS = 100
    latencies = []
    torch.cuda.synchronize()

    for _ in range(NUM_RUNS):
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        torch.cuda.synchronize()
        start = time.perf_counter()
        preds = model(dummy)
        preds = extract_preds(preds)
        detections = non_max_suppression(preds, CONF_THRESH, IOU_THRESH)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency

    print(
        f"   Inference + NMS 평균 Latency: {avg_latency:.2f} +/- {std_latency:.2f} ms"
    )
    print(f"   FPS: {fps:.1f}")

    BATCH_RUNS = 30
    batch_latencies = []
    torch.cuda.synchronize()

    for _ in range(BATCH_RUNS):
        dummy_batch = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        torch.cuda.synchronize()
        start = time.perf_counter()
        preds = model(dummy_batch)
        preds = extract_preds(preds)
        detections = non_max_suppression(preds, CONF_THRESH, IOU_THRESH)
        torch.cuda.synchronize()
        end = time.perf_counter()
        batch_latencies.append((end - start) * 1000)

    avg_batch = np.mean(batch_latencies)
    batch_throughput = BATCH_SIZE / (avg_batch / 1000)

    print(f"   배치({BATCH_SIZE}) 평균 Latency: {avg_batch:.2f} ms")
    print(f"   배치 Throughput: {batch_throughput:.1f} img/s")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 결과 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("5. 결과 저장")
    print("=" * 60)

    results = {
        "model": "yolov8n",
        "checkpoint": CHECKPOINT,
        "dataset": DATASET_YAML,
        "img_size": IMG_SIZE,
        "conf_thresh": CONF_THRESH,
        "iou_thresh": IOU_THRESH,
        "accuracy": {
            "mAP@0.5": round(mAP50, 4),
            "per_class": {
                name: {
                    "AP@0.5": round(ap_list[i], 4),
                    "Precision": round(prec_list[i], 4),
                    "Recall": round(rec_list[i], 4),
                }
                for i, name in enumerate(CLASS_NAMES)
            },
        },
        "speed": {
            "single_latency_ms": round(avg_latency, 2),
            "single_latency_std_ms": round(std_latency, 2),
            "single_fps": round(fps, 1),
            "batch_size": BATCH_SIZE,
            "batch_latency_ms": round(avg_batch, 2),
            "batch_throughput_img_s": round(batch_throughput, 1),
        },
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    result_path = os.path.join(SAVE_DIR, "eval_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"   결과 저장: {result_path}")
    print("\n평가 완료!")


if __name__ == "__main__":
    freeze_support()
    main()
