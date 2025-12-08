# train_revdist.py

from anomalib.data import Folder
from anomalib.models import ReverseDistillation
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor

from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype
import torch


# ================================
# 0. 하이퍼파라미터 모음
# ================================
IMAGE_SIZE       = 256   # 입력 이미지 크기
TRAIN_BATCH_SIZE = 8     # 학습 배치
EVAL_BATCH_SIZE  = 8     # 평가 배치
NUM_WORKERS      = 4     # DataLoader worker 수
MAX_EPOCHS       = 20    # epoch 수

DATA_ROOT         = "../final_dataset"
TRAIN_NORMAL_DIR = "train/normal_5000"
TEST_ABNORM_DIR  = "test/abnormal_card"
TEST_NORMAL_DIR  = "test/normal_3000"
EXPERIMENT_NAME = "revdist_xray_card"        # results/ 아래 폴더 이름


def main():
    # ---------------------------------------------------
    # 1. 전처리: PatchCore랑 맞춰서 Normalize 사용
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),  # [0,1] 스케일
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25],
        ),
    ])

    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. 데이터 모듈: Folder 기반 (너가 쓰던 방식 그대로)
    # ---------------------------------------------------
    datamodule = Folder(
        name="chest_xray_revdist",
        root=DATA_ROOT,
        normal_dir=TRAIN_NORMAL_DIR,
        abnormal_dir=TEST_ABNORM_DIR,
        normal_test_dir=TEST_NORMAL_DIR,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        # pre_processor=pre_processor,
    )

    # ---------------------------------------------------
    # 3. 모델 정의: Reverse Distillation
    #    - backbone / layers는 PatchCore 쪽이랑 비슷하게
    # ---------------------------------------------------
    model = ReverseDistillation(
        backbone="wide_resnet50_2",          # 성능 괜찮은 백본
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        pre_processor=pre_processor,
        # feature_maps 통합 방식이나 temperature 같은 것
        # 더 튜닝하고 싶으면 여기 인자들 추가하면 됨.
    )

    # ---------------------------------------------------
    # 4. Engine 설정 & 학습 / 테스트
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,                         # GPU 번호 맞게 수정
        default_root_dir=f"results/{EXPERIMENT_NAME}",
        enable_progress_bar=True,
    )

    # 학습
    engine.fit(model=model, datamodule=datamodule)

    # 테스트 (AUROC / AUPR / F1 + 히트맵 저장)
    engine.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
