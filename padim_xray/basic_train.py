# train_padim.py

from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import torch


# ================================
# 0. 하이퍼파라미터 한 눈에 모아두기
# ================================
IMAGE_SIZE = 256            # 입력 이미지 크기 (정사각형)
TRAIN_BS   = 4              # 학습 배치 사이즈
EVAL_BS    = 4              # 평가 배치 사이즈
BACKBONE   = "resnet18"     # 백본: resnet18, resnet34, resnet50 등
N_FEATURES = 200            # PaDiM feature 압축 차원 (너무 크게 하면 메모리↑)

DATA_ROOT        = "../final_dataset"
TRAIN_NORMAL_DIR = "train/normal_5000"
TEST_ABNORM_DIR  = "test/abnormal_card"
TEST_NORMAL_DIR  = "test/normal_3000"

EXPERIMENT_NAME  = f"padim_xray_{BACKBONE}_{IMAGE_SIZE}_{BACKBONE}_{N_FEATURES}"  # results/ 아래 폴더 이름


def main():
    # ---------------------------------------------------
    # 1. 전처리: X-ray용 리사이즈 + 정규화
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),  # [0,1] 스케일
        # Normalize(
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.25, 0.25, 0.25],
        # ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. 데이터 모듈: Cardiomegaly 전용 abnormal 세팅
    # ---------------------------------------------------
    datamodule = Folder(
        name="chest_xray_padim",
        root=DATA_ROOT,
        normal_dir=TRAIN_NORMAL_DIR,   # 정상 학습
        abnormal_dir=TEST_ABNORM_DIR,  # Cardiomegaly 비정상
        normal_test_dir=TEST_NORMAL_DIR,
        train_batch_size=TRAIN_BS,
        eval_batch_size=EVAL_BS,
        num_workers=2,
    )

    datamodule.setup()

    # ---------------------------------------------------
    # 3. PaDiM 모델 정의
    #    BACKBONE + N_FEATURES 조합
    # ---------------------------------------------------
    model = Padim(
        backbone=BACKBONE,
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        n_features=N_FEATURES,
        pre_processor=pre_processor,
    )

    # ---------------------------------------------------
    # 4. Engine: 학습/테스트
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=1,                # PaDiM은 epoch=1이면 충분
        accelerator="gpu",
        devices=1,
        default_root_dir=f"results/{EXPERIMENT_NAME}",
        enable_progress_bar=True,
    )

    print(f"=== PaDiM 학습 시작 ({BACKBONE}, {IMAGE_SIZE}x{IMAGE_SIZE}) ===")
    engine.fit(model=model, datamodule=datamodule)

    print("=== PaDiM 테스트 시작 ===")
    results = engine.test(model=model, datamodule=datamodule)
    print(results)

    print(f"=== 완료! 체크포인트와 로그는 results/{EXPERIMENT_NAME} 아래에 저장됨 ===")


if __name__ == "__main__":
    main()
