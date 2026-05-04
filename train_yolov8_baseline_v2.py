"""
YOLOv8 Baseline Training v2
세종대학교 Recycle Mate 프로젝트


v1 -> v2 변경점
1. val루프 BatchNorm 토글 버그 수정

    - 변경 전
    with torch.no_grad():
        model.train()
        preds = model(images)
        model.eval()
    
    - 변경 후
    model.train()
    with torch.no_grad():
        preds = model(images)
    
    - 변경이유
        v8DetectionLoss는 train모드의 raw feature map을 입력으로 요구함.
        기존 코드는 no_grad 블록 안에서 model.train() -> model.eval()을 토글해 BatchNorm의
        running mean/variance가 val 데이터로 오염됨. 오염된 BatchNorm 통계는 이후 train 루프에도 영향을 줌
        => 학습 전체의 신뢰성 하락
    
    - 미수정 시
        val_loss 수치가 부정확해지고 그 수치 기준으로 best.pt가 선택됨
        => 실제로 가장 좋은 가중치가 저장되지 않을 수 있음

2. Warmup + Cosine 스케줄러 추가

    - 변경 전
    CosineAnnealingLR(optimizer, T_max=EPOCHS)

    - 변경 후
    LambdaLR(optimizer, lr_lambda)
        => epoch 0~2 -> LR 0 -> 1e-3 선형 증가
           epoch 3~99 -> Cosine decay
    
    - 변경 이유
        pretrained 모델에 LR=1e-3을 warmup 없이 바로 적용하면 초반 gradient가 크게 튀어 학습 불안정
        pretrained fine-tuning에서 warmup은 표준 관행임

    - 미수정 시
        초반 몇 epoch에서 loss가 튀거나 발산 가능
        좋은 pretrained 가중치가 초반에 망가질 수 있음

3. workers 수 조정

    - 변경 전
    worker = 4

    - 변경 후
    worker = 8

    - 변경 이유
        원본 이미지가 장당 평균 3.6MB라 worker=4로는 CPU 데이터 로딩이 GPU 연산을 못따라가 GPU 사용률이 15~39%에 불과했음
    
    - 미수정 시
        GPU가 데이터 대기하느라 놀게됨. 학습 속도 저하

4. 이미지 사전 리사이즈 (image_resize.py)

    - 변경 내용
    image_resize.py로 36000장을 640x640으로 미리 리사이즈 후 yolo_major9_640 경로 사용

    - 변경 이유
        worker=8로 올려도 여전히 GPU 사용량이 낮았음.
        근본적인 원인은 매 에포크마다 3.6MB 이미지를 읽고 리사이즈하는 CPU 병목이었기 때문
    
    - 미수정 시
        GPU가 데이터 기다리느라 놀게됨 -> 학습 속도 저하
"""

import os
import math
import time
import torch
import torch.nn as nn
from types import SimpleNamespace
from multiprocessing import freeze_support
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import v8DetectionLoss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_YAML = r"C:\Users\43074\Desktop\RecycleMate\dataset-shrinked\processed\yolo_major9\metadata\dataset.yaml"
MODEL_NAME   = "yolov8n.pt"
EPOCHS       = 100
WARMUP_EPOCHS = 3        # [추가] pretrained fine-tuning 표준 warmup
BATCH_SIZE   = 32
IMG_SIZE     = 640
LR           = 1e-3      # warmup이 0에서 시작하므로 v1과 동일하게 유지
WEIGHT_DECAY = 5e-4
SAVE_DIR     = "runs/baseline_v2"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Warmup (선형) + Cosine decay 복합 스케줄러
    - epoch 0 ~ warmup_epochs-1 : LR 0 → LR (선형 증가)
    - epoch warmup_epochs ~ total_epochs : Cosine decay
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 0에서 1까지 선형 증가 (warmup_epochs=3이면 0/3, 1/3, 2/3, 3/3)
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


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
        model.args = SimpleNamespace(**{
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            **hyp,
        })

    for param in model.parameters():
        param.requires_grad = True

    NC = 9
    print(f"   클래스 수: {NC}")
    print(f"   디바이스: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"   학습 가능 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 데이터 로더
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("2. 데이터셋 로드")
    print("=" * 60)

    cfg = get_cfg(overrides={
        "data":  DATASET_YAML,
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "task":  "detect",
        "mode":  "train",
    })

    data_info = check_det_dataset(cfg.data)
    print(f"   Train: {data_info['train']}")
    print(f"   Val:   {data_info['val']}")

    train_dataset = build_yolo_dataset(
        cfg, img_path=data_info["train"], batch=BATCH_SIZE, data=data_info, mode="train"
    )
    train_loader = build_dataloader(train_dataset, batch=BATCH_SIZE, workers=8, rank=-1)

    val_dataset = build_yolo_dataset(
        cfg, img_path=data_info["val"], batch=BATCH_SIZE, data=data_info, mode="val"
    )
    val_loader = build_dataloader(val_dataset, batch=BATCH_SIZE, workers=8, rank=-1)

    print(f"   Train 배치 수: {len(train_loader)}")
    print(f"   Val 배치 수:   {len(val_loader)}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Loss / Optimizer / Scheduler
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("3. Loss / Optimizer / Scheduler 설정")
    print("=" * 60)

    criterion = v8DetectionLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # [수정] Warmup + Cosine 스케줄러
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    print(f"   Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
    print(f"   Scheduler: Warmup({WARMUP_EPOCHS} epoch) + CosineDecay")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 학습 루프
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("4. 학습 시작")
    print("=" * 60)

    best_val_loss  = float("inf")
    train_losses   = []
    val_losses     = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        running_loss = 0.0
        num_batches  = 0

        for batch in train_loader:
            images = batch["img"].to(DEVICE).float() / 255.0
            preds  = model(images)
            loss, _ = criterion(preds, batch)

            if loss.dim() > 0:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches  += 1

        scheduler.step()
        train_loss = running_loss / max(num_batches, 1)
        train_losses.append(train_loss)

        # ── Validation ──
        # [수정] no_grad 안에서 model.train() 고정
        # 이유: v8DetectionLoss는 train 모드의 raw feature map을 입력으로 요구함
        #       기존처럼 model.eval() ↔ model.train()을 토글하면
        #       BatchNorm의 running statistics가 오염되어 val_loss 수치가 부정확해짐
        val_running_loss = 0.0
        val_batches      = 0

        model.train()  # val 동안 train 모드 고정
        with torch.no_grad():
            for batch in val_loader:
                images = batch["img"].to(DEVICE).float() / 255.0
                preds  = model(images)
                loss, _ = criterion(preds, batch)
                if loss.dim() > 0:
                    loss = loss.mean()
                val_running_loss += loss.item()
                val_batches      += 1

        val_loss = val_running_loss / max(val_batches, 1)
        val_losses.append(val_loss)

        elapsed = time.time() - epoch_start
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"lr: {lr_now:.6f} | "
            f"time: {elapsed:.1f}s"
        )

        # ── Best 저장 ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, "best.pt")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "train_loss":           train_loss,
            }, save_path)
            print(f"   >>> Best 모델 저장 (val_loss: {val_loss:.4f})")

        # ── 주기적 체크포인트 ──
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "train_loss":           train_loss,
            }, ckpt_path)
            print(f"   >>> 체크포인트 저장: {ckpt_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 최종 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    final_path = os.path.join(SAVE_DIR, "last.pt")
    torch.save({
        "epoch":                EPOCHS - 1,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_losses[-1],
        "train_loss":           train_losses[-1],
    }, final_path)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"   Best val_loss: {best_val_loss:.4f}")
    print(f"   저장 경로: {SAVE_DIR}")
    print("=" * 60)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. 학습 곡선 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        axes[0].plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
        axes[0].plot(range(1, EPOCHS + 1), val_losses,   label="Val Loss")
        axes[0].axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--", alpha=0.5, label=f"Warmup end ({WARMUP_EPOCHS})")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # LR schedule 시각화
        dummy_opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=LR)
        dummy_sch = build_scheduler(dummy_opt, WARMUP_EPOCHS, EPOCHS)
        lrs = []
        for _ in range(EPOCHS):
            lrs.append(dummy_opt.param_groups[0]["lr"])
            dummy_sch.step()
        axes[1].plot(range(1, EPOCHS + 1), lrs, color="orange")
        axes[1].axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--", alpha=0.5, label=f"Warmup end ({WARMUP_EPOCHS})")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("LR Schedule (Warmup + Cosine)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=150)
        plt.close()
        print("loss_curve.png 저장 완료")
    except ImportError:
        print("matplotlib 미설치 - 학습 곡선 저장 생략")


if __name__ == "__main__":
    freeze_support()
    main()