"""
YOLOv8 Baseline Training - 순수 PyTorch 학습 루프
세종대학교 Recycle Mate 프로젝트
"""

import os
import time
import torch
import torch.nn as nn
from types import SimpleNamespace
from multiprocessing import freeze_support
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import v8DetectionLoss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_YAML = r"C:\Users\43074\Desktop\RecycleMate\dataset-shrinked\processed\yolo_major9\metadata\dataset.yaml"
MODEL_NAME = "yolov8n.yaml"      # nano 모델 (베이스라인) -> 빈 구조로 변경 
EPOCHS = 100
# yolo 계열에서 가장 일반적인 기본값
BATCH_SIZE = 32
# 안정적인 기본값 = 16이지만, 그래픽카드에 여유가 있어 32로 상향조정.
IMG_SIZE = 640
# yolov8의 기본 입력 크기. 모델 구조 자체가 640을 기준으로 설정되어있음.
LR = 1e-3
# AdamW의 일반적인 시작 학습률. pretrained 모델을 미세조정할때 1e-3 ~ 1e-4범위가 표준적. 베이스라인이니 중간값
WEIGHT_DECAY = 5e-4
#Yolo 공식 학습에서 쓰이는 값. 과적합 방지용. detection 분야에서 관행적으로 쓰이는 수치
SAVE_DIR = "runs/baseline"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



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
    
    # model = torch.compile(model, mode="reduce-overhead") #C++ 로우레벨 코드로 컴파일
    # 하이퍼파라미터를 SimpleNamespace로 설정
    hyp = model.args if hasattr(model, 'args') else {}
    if isinstance(hyp, dict):
        model.args = SimpleNamespace(**{
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            **hyp,
        })

    # 모든 파라미터 gradient 활성화
    for param in model.parameters():
        param.requires_grad = True

    NC = 9
    print(f"   클래스 수: {NC}")
    print(f"   디바이스: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"   학습 가능 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 데이터 로더 구성
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("2. 데이터셋 로드")
    print("=" * 60)

    cfg = get_cfg(overrides={
        "data": DATASET_YAML,
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "task": "detect",
        "mode": "train",
    })

    data_info = check_det_dataset(cfg.data)
    print(f"   Train: {data_info['train']}")
    print(f"   Val:   {data_info['val']}")

    train_dataset = build_yolo_dataset(
        cfg, img_path=data_info["train"], batch=BATCH_SIZE, data=data_info, mode="train"
    )
    train_loader = build_dataloader(
        train_dataset, 
        batch=BATCH_SIZE, 
        workers=4, 
        rank=-1,
        pin_memory=True,          # CPU 메모리를 고정하여 GPU로의 전송 속도 극대화
        # persistent_workers=True   # 스레드 재사용
    )
    val_dataset = build_yolo_dataset(
        cfg, img_path=data_info["val"], batch=BATCH_SIZE, data=data_info, mode="val"
    )
    val_loader = build_dataloader(val_dataset, batch=BATCH_SIZE, workers=4, rank=-1)

    print(f"   Train 배치 수: {len(train_loader)}")
    print(f"   Val 배치 수:   {len(val_loader)}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Loss, Optimizer, Scheduler
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("3. Loss / Optimizer / Scheduler 설정")
    print("=" * 60)

    criterion = v8DetectionLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    # fused=True 옵션은 AdamW의 연산을 통합하여 GPU에서 더 빠르게 실행되도록 최적화.학습 속도 향상
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"   Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
    print(f"   Scheduler: CosineAnnealing (T_max={EPOCHS})")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 학습 루프
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("4. 학습 시작")
    print("=" * 60)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["img"].to(DEVICE).float() / 255.0

            # forward - training 모드에서 raw predictions 반환
            preds = model(images)

            # preds가 tuple이면 학습용 feature maps
            loss, loss_items = criterion(preds, batch)

            # loss가 스칼라가 아닐 경우 처리
            if loss.dim() > 0:
                loss = loss.mean()

            # backward
            optimizer.zero_grad(set_to_none=True) # 0으로 초기화 대신 참조 링크를 끊는 방식
            
            loss.backward()

            # gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss = running_loss / max(num_batches, 1)
        train_losses.append(train_loss)

        # ── Validation ──
        model.eval()
        val_running_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["img"].to(DEVICE).float() / 255.0

                # eval 모드에서도 training forward가 필요 -> 임시로 train 모드
                model.train()
                preds = model(images)
                model.eval()

                loss, loss_items = criterion(preds, batch)
                if loss.dim() > 0:
                    loss = loss.mean()
                val_running_loss += loss.item()
                val_batches += 1

        val_loss = val_running_loss / max(val_batches, 1)
        val_losses.append(val_loss)

        elapsed = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"lr: {lr_now:.6f} | "
            f"time: {elapsed:.1f}s"
        )

        # ── Best 모델 저장 ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, save_path)
            print(f"   >>> Best 모델 저장 (val_loss: {val_loss:.4f})")

        # ── 주기적 체크포인트 ──
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, ckpt_path)
            print(f"   >>> 체크포인트 저장: {ckpt_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 최종 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    final_path = os.path.join(SAVE_DIR, "last.pt")
    torch.save({
        "epoch": EPOCHS - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_losses[-1],
        "train_loss": train_losses[-1],
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

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
        plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=150)
        plt.close()
        print("loss_curve.png 저장 완료")
    except ImportError:
        print("matplotlib 미설치 - 학습 곡선 저장 생략")


if __name__ == "__main__":
    freeze_support()
    main()


