"""
YOLOv8 Baseline Training v3 - nc=9 교체 수정
세종대학교 Recycle Mate 프로젝트

v2 -> v3 변경점
1. 모델 nc 교체 수정

    - 변경 전
    yolo = YOLO("yolov8n.pt")
    model = yolo.model.to(DEVICE)

    - 변경 후
    model = DetectionModel("yolov8n.yaml", nc=9)
    # pretrained backbone/neck 가중치만 이식
    # head(model.22.*) 는 nc=9로 랜덤 초기화

    - 변경 이유
        YOLO("yolov8n.pt")는 COCO pretrained 80클래스 구조를 그대로 가져옴.
        nc 교체 코드가 없어서 v1, v2 모두 nc=80으로 학습됨.
        데이터셋 라벨은 0~8(9개)인데 모델은 0~79(80개)로 출력하니
        loss 계산 자체가 엉뚱하게 됨.
        eval mAP도 클래스 ID가 우연히 겹쳐서 나온 신뢰할 수 없는 수치였음
        => v1, v2의 모든 학습 결과 무효

    - 미수정 시
        모델이 9클래스가 아닌 80클래스로 학습됨
        => eval mAP가 무의미한 수치가 됨


"""

import os
import math
import time
import torch
from types import SimpleNamespace
from multiprocessing import freeze_support
from torch.optim.lr_scheduler import LambdaLR
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import v8DetectionLoss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_YAML  = r"C:\Users\43074\Desktop\RecycleMate\dataset-shrinked\processed\yolo_major9_640\metadata\dataset.yaml"
MODEL_NAME    = "yolov8n.pt"
EPOCHS        = 100
WARMUP_EPOCHS = 3
BATCH_SIZE    = 32
IMG_SIZE      = 640
LR            = 1e-3
WEIGHT_DECAY  = 5e-4
SAVE_DIR      = "runs/baseline_v3"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NC            = 9


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def load_model_with_nc(model_name, nc, device):
    """
    DetectionModel("yolov8n.yaml", nc=9) 로 9클래스 구조 생성 후
    pretrained backbone/neck 가중치만 이식
    """
    # Step 1: nc=9 구조 생성
    model = DetectionModel("yolov8n.yaml", nc=nc)

    # Step 2: pretrained 가중치 로드
    yolo_pt = YOLO(model_name)
    state_pt = yolo_pt.model.state_dict()

    # Step 3: head(model.22.*) 제외, shape 맞는 것만 이식
    state_new = model.state_dict()
    transfer = {
        k: v for k, v in state_pt.items()
        if k in state_new
        and not k.startswith("model.22.")
        and state_new[k].shape == v.shape
    }
    state_new.update(transfer)
    model.load_state_dict(state_new)

    print(f"   가중치 이식: {len(transfer)}/{len(state_new)} 레이어 (head 제외 backbone/neck)")
    print(f"   head nc: {model.model[-1].nc}")

    return model.to(device)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 모델 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("=" * 60)
    print("1. 모델 로드")
    print("=" * 60)

    model = load_model_with_nc(MODEL_NAME, NC, DEVICE)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    for param in model.parameters():
        param.requires_grad = True

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
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    print(f"   Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
    print(f"   Scheduler: Warmup({WARMUP_EPOCHS} epoch) + CosineDecay")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 학습 루프
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 60)
    print("4. 학습 시작")
    print("=" * 60)

    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

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
        model.train()
        with torch.no_grad():
            val_running_loss = 0.0
            val_batches      = 0

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
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "train_loss":           train_loss,
                "nc":                   NC,
            }, os.path.join(SAVE_DIR, "best.pt"))
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
                "nc":                   NC,
            }, ckpt_path)
            print(f"   >>> 체크포인트 저장: {ckpt_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 최종 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    torch.save({
        "epoch":                EPOCHS - 1,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_losses[-1],
        "train_loss":           train_losses[-1],
        "nc":                   NC,
    }, os.path.join(SAVE_DIR, "last.pt"))

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

        axes[0].plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
        axes[0].plot(range(1, EPOCHS + 1), val_losses,   label="Val Loss")
        axes[0].axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--",
                        alpha=0.5, label=f"Warmup end ({WARMUP_EPOCHS})")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        dummy_opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=LR)
        dummy_sch = build_scheduler(dummy_opt, WARMUP_EPOCHS, EPOCHS)
        lrs = []
        for _ in range(EPOCHS):
            lrs.append(dummy_opt.param_groups[0]["lr"])
            dummy_sch.step()
        axes[1].plot(range(1, EPOCHS + 1), lrs, color="orange")
        axes[1].axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--",
                        alpha=0.5, label=f"Warmup end ({WARMUP_EPOCHS})")
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
