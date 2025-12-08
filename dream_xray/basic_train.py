# train_draem.py

from anomalib.data import Folder
from anomalib.models import Draem
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor

from anomalib.callbacks import ModelCheckpoint   # ★ anomalib 전용 체크포인트
from lightning.pytorch.callbacks import EarlyStopping

from torchvision.transforms.v2 import Compose, Resize, ToDtype
import torch


# ================================
# 0. 하이퍼파라미터 모음
# ================================
IMAGE_SIZE       = 256          # 입력 이미지 크기 (정사각형 기준)
TRAIN_BATCH_SIZE = 8            # 학습 배치
EVAL_BATCH_SIZE  = 8            # 평가 배치
NUM_WORKERS      = 4            # DataLoader worker 수
MAX_EPOCHS       = 20           # DRAEM은 여러 epoch 학습 필요

DATA_ROOT        = "../final_dataset"
TRAIN_NORMAL_DIR = "train/normal_7000"
TEST_ABNORM_DIR  = "test/abnormal_card"
TEST_NORMAL_DIR  = "test/normal_3000"

EXPERIMENT_NAME  = "draem_xray_card"         # results/ 아래 폴더 이름


def main():
    # ---------------------------------------------------
    # 1. 전처리: DRAEM은 Normalize 쓰지 말고 0~1 스케일만
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),  # [0,1] 로 스케일만
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. 데이터 모듈
    #    - train: 정상만 사용
    #    - test : 정상 + Cardiomegaly 비정상
    # ---------------------------------------------------
    datamodule = Folder(
        name="chest_xray_draem",
        root=DATA_ROOT,
        normal_dir=TRAIN_NORMAL_DIR,
        abnormal_dir=TEST_ABNORM_DIR,
        normal_test_dir=TEST_NORMAL_DIR,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    datamodule.setup()

    # ---------------------------------------------------
    # 3. DRAEM 모델 정의
    # ---------------------------------------------------
    model = Draem(
        pre_processor=pre_processor,
        # lr=1e-4,                 # 안정적
        # hidden_dim=128,          # 표현력 증가
        # anomaly_source="perlin", # X-ray에 잘 맞음
        # beta=0.3,                # segmentation loss 조금 더 반영
        # weight_decay=1e-5,
    )


    # ---------------------------------------------------
    # 5. Engine: 학습/테스트
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        default_root_dir=f"results/{EXPERIMENT_NAME}",
        enable_progress_bar=True,
    )

    print(f"=== DRAEM 학습 시작 (epochs={MAX_EPOCHS}, image={IMAGE_SIZE}) ===")
    engine.fit(model=model, datamodule=datamodule)

    print("=== DRAEM 테스트 시작 ===")
    results = engine.test(model=model, datamodule=datamodule)
    print(results)

    print(f"=== 완료! 체크포인트와 로그는 results/{EXPERIMENT_NAME} 아래에 저장됨 ===")
    print(f"    → 베스트 ckpt: results/{EXPERIMENT_NAME}/checkpoints/best-image-auroc.ckpt")


if __name__ == "__main__":
    main()
