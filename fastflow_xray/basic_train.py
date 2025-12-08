from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import AUROC, F1Score, F1AdaptiveThreshold, Evaluator

from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype
import torch


# ================================
# 0. 하이퍼파라미터
# ================================
IMAGE_SIZE       = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE  = 8
NUM_WORKERS      = 4
MAX_EPOCHS       = 20

ROOT = Path("../final_dataset")

TRAIN_NORMAL_DIR  = "train/normal_5000"   # 네 폴더 구조에 맞게
TEST_ABNORMAL_DIR = "test/abnormal_card"
TEST_NORMAL_DIR   = "test/normal_3000"

EXPERIMENT_NAME = "fastflow_xray_card"


def main():
    # ---------------------------
    # 1. 전처리: 모델 쪽 pre_processor 로만 사용
    # ---------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),   # [0,1]
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.25, 0.25, 0.25]),
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------
    # 2. 데이터 모듈 (★ task / pre_processor 같은 건 안 넣는다)
    # ---------------------------
    datamodule = Folder(
        name="chest_xray_fastflow",
        root=str(ROOT),
        normal_dir=TRAIN_NORMAL_DIR,
        abnormal_dir=TEST_ABNORMAL_DIR,
        normal_test_dir=TEST_NORMAL_DIR,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # ---------------------------
    # 3. evaluator: 이미지 레벨 metric만!
    #    → gt_mask 안 쓰게 해서 에러 차단
    # ---------------------------
    image_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
    ]
    evaluator = Evaluator(
        val_metrics=image_metrics,
        test_metrics=image_metrics,
    )

    # ---------------------------
    # 4. FastFlow 모델
    # ---------------------------
    model = Fastflow(
        backbone="resnet18",
        pre_trained=True,
        flow_steps=8,
        conv3x3_only=False,
        hidden_ratio=1.0,
        pre_processor=pre_processor,
        evaluator=evaluator,  # ★ 우리가 만든 evaluator 사용
    )

    # ---------------------------
    # 5. Engine
    # ---------------------------
    engine = Engine(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,  # GPU 번호
        default_root_dir=f"results/{EXPERIMENT_NAME}",
        enable_progress_bar=True,
    )

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
