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
MODEL_NAME = "yolov8n.pt"  # nano 모델 (베이스라인)
EPOCHS = 100
# yolo 계열에서 가장 일반적인 기본값
BATCH_SIZE = 32
# 안정적인 기본값 = 16이지만, 그래픽카드에 여유가 있어 32로 상향조정.
IMG_SIZE = 640
# yolov8의 기본 입력 크기. 모델 구조 자체가 640을 기준으로 설정되어있음.
LR = 1e-2
# [최적화]AdamW를 SGD로 바꿔서
# AdamW의 일반적인 시작 학습률. pretrained 모델을 미세조정할때 1e-3 ~ 1e-4범위가 표준적. 베이스라인이니 중간값
WEIGHT_DECAY = 5e-4
# Yolo 공식 학습에서 쓰이는 값. 과적합 방지용. detection 분야에서 관행적으로 쓰이는 수치
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

    # 하이퍼파라미터를 SimpleNamespace로 설정
    hyp = model.args if hasattr(model, "args") else {}
    if isinstance(hyp, dict):
        # [최적화]box": 7.5 -> 5.0, "cls": 0.5 -> 2.0
        # 모델이 물체의 위치(box)는 찾았으나, 둘 다 투명해서 종류(cls)를 헷갈려 한다는 뜻임. 그래서 분류 오차(cls)에 대한 페널티(가중치)를 4배 높여, 투명 재질 간의 미세한 차이를 더 강하게 학습하도록 강제 .
        model.args = SimpleNamespace(
            **{
                "box": 5.0,
                "cls": 2.0,
                "dfl": 1.5,
                # [최적화]Focal Loss는 모델이 '이미 잘 맞추는 건전지/형광등'은 무시하고, '계속 틀리는 페트병/플라스틱'에만 집중해서 공부하게 만드는 파라미터
                "fl_gamma": 1.5,
                **hyp,
            }
        )

    # 모든 파라미터 gradient 활성화
    for param in model.parameters():
        param.requires_grad = True

    NC = 9
    print(f"   클래스 수: {NC}")
    print(f"   디바이스: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(
        f"   학습 가능 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 데이터 로더 구성
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("2. 데이터셋 로드")
    print("=" * 60)

    cfg = get_cfg(
        overrides={
            "data": DATASET_YAML,
            "imgsz": IMG_SIZE,
            "batch": BATCH_SIZE,
            "task": "detect",
            "mode": "train",
            # 이미지 4장을 하나로 합치는 Mosaic 기법은 배경을 복잡하게 만들어, 모델이 배경에 의존하지 않고 페트병 본연의 형태를 학습하게 만듦.
            "mosaic": 1.0,  # [최적화]새로 추가한 부분 (100% 모자이크 적용)
            "mixup": 0.1,  # [최적화]새로 추가한 부분 (10% 믹스업 적용)
            # 투명한 페트병이나 비닐은 실내 조명(형광등 빛 반사, 그림자)에 따라 색이나 밝기가 계속 변함. 모델이 이런 조명 변화에 속지 않도록 훈련시켜야 함.
            "hsv_h": 0.015,  # [최적화] 색상 무작위 변경
            "hsv_s": 0.7,  # [죄적화] 채도 무작위 변경
            "hsv_v": 0.4,  # [최적화] 명도(밝기) 무작위 변경
        }
    )

    data_info = check_det_dataset(cfg.data)
    print(f"   Train: {data_info['train']}")
    print(f"   Val:   {data_info['val']}")

    train_dataset = build_yolo_dataset(
        cfg, img_path=data_info["train"], batch=BATCH_SIZE, data=data_info, mode="train"
    )
    train_loader = build_dataloader(train_dataset, batch=BATCH_SIZE, workers=4, rank=-1)

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
    # [최적화]AdamW는 초기 학습이 빠르지만, 100 Epoch 이상 충분히 학습할 때는 SGD + Momentum 조합이 모델의 일반화 성능(새로운 이미지를 맞추는 능력)을 더 높게 끌어올려 최종 mAP 확보에 유리하다는 것이 YOLO에서 정석적.
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.937, weight_decay=5e-4
    )
    # -----------------[최적화]--------------------------

    #  Scheduler 설계 (Warmup + Cosine Annealing)
    WARMUP_EPOCHS = 3  # 처음 3 에포크 동안 Warmup 진행

    # [Step 1] Linear Warmup: 시작 학습률을 목표 학습률(1e-2)의 1% 수준(0.01)에서 100%(1.0)까지 서서히 올림
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )

    # [Step 2] Cosine Annealing: 남은 에포크 동안 부드럽게 학습률을 감소시킴
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(EPOCHS - WARMUP_EPOCHS),  # 전체 에포크에서 웜업 기간을 뺀 만큼만 진행
    )

    # [Step 3] SequentialLR: 두 스케줄러를 하나로 이어 붙임
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[
            WARMUP_EPOCHS
        ],  # WARMUP_EPOCHS(3)에 도달하면 main_scheduler로 바통 터치
    )

    # -----------------[최적화]--------------------------

    # [최적화로 인한 수정]최적화 함수를 AdamW에서 SGD로 변경했기 때문에
    print(f"   Optimizer: SGD (lr={LR}, wd={WEIGHT_DECAY}, momentum=0.937)")
    # [최적화로 인한 수정]웜업 3 에포크를 넣었고, T_max는 97(EPOCHS - WARMUP_EPOCHS)로 돌아가고 있dma
    print(
        f"   Scheduler: Linear Warmup ({WARMUP_EPOCHS} epochs) + CosineAnnealing (T_max={EPOCHS - WARMUP_EPOCHS})"
    )
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
            optimizer.zero_grad()

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
                # [밑의 두개 최적화 이유]with torch.no_grad():는 가중치(Weight)의 기울기 업데이트만 막을 뿐, BatchNorm 레이어의 평균/분산 통계치(Running Mean/Var)가 갱신되는 것은 막지 못함.
                # 이 코드를 그대로 돌리면 모델이 검증(Validation) 데이터를 볼 때마다 BatchNorm 통계치가 오염되어, 결국 최종 정확도(mAP)가 심각하게 떨어지게 됨.
                # [최적화] BatchNorm 통계치 업데이트 강제 차단 (오염 방지)
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.track_running_stats = False
                # ----------------------------------------
                # eval 모드에서도 training forward가 필요 -> 임시로 train 모드
                model.train()
                preds = model(images)
                model.eval()
                # [최적화] BatchNorm 통계치 업데이트 다시 복구
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.track_running_stats = True
                # ----------------------------------------
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                },
                save_path,
            )
            print(f"   >>> Best 모델 저장 (val_loss: {val_loss:.4f})")

        # ── 주기적 체크포인트 ──
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                },
                ckpt_path,
            )
            print(f"   >>> 체크포인트 저장: {ckpt_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 최종 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    final_path = os.path.join(SAVE_DIR, "last.pt")
    torch.save(
        {
            "epoch": EPOCHS - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_losses[-1],
            "train_loss": train_losses[-1],
        },
        final_path,
    )

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
