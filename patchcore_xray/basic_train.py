# train_patchcore.py

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import torch


# ================================
# 0. 하이퍼파라미터 한 눈에 모아두기
# ================================
IMAGE_SIZE = 224                  # PatchCore 기본: 224 or 256
TRAIN_BS   = 2                    # GPU 메모리 따라 조절
EVAL_BS    = 2
BACKBONE   = "resnet34"           # resnet18 / resnet34 / wide_resnet50_2 등
CORESET_RATIO = 0.02              # 0.001~0.1 사이 추천
NUM_NEIGHBORS = 9                 # k-NN 개수 (PatchCore 기본=9)

DATA_ROOT        = "../final_dataset"
TRAIN_NORMAL_DIR = "train/normal_5000"
TEST_ABNORM_DIR  = "test/abnormal_card"
TEST_NORMAL_DIR  = "test/normal_3000"

EXPERIMENT_NAME = f"patchcore_{BACKBONE}_{IMAGE_SIZE}_{BACKBONE}_{CORESET_RATIO}"


def main():

    # ---------------------------------------------------
    # 1. 전처리: PatchCore는 Normalize 반드시 필요
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25],
        ),
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. 데이터 모듈
    # ---------------------------------------------------
    datamodule = Folder(
        name="chest_xray_patchcore",
        root=DATA_ROOT,
        normal_dir=TRAIN_NORMAL_DIR,
        abnormal_dir=TEST_ABNORM_DIR,
        normal_test_dir=TEST_NORMAL_DIR,
        train_batch_size=TRAIN_BS,
        eval_batch_size=EVAL_BS,
        num_workers=2,
    )
    datamodule.setup()

    # ---------------------------------------------------
    # 3. PatchCore 모델 정의
    # ---------------------------------------------------
    model = Patchcore(
        backbone=BACKBONE,
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=CORESET_RATIO,
        num_neighbors=NUM_NEIGHBORS,
        pre_processor=pre_processor,
    )

    # ---------------------------------------------------
    # 4. Engine: 학습/테스트 오케스트레이터
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        default_root_dir=f"results/{EXPERIMENT_NAME}",
        enable_progress_bar=True,
    )

    print(f"=== PatchCore 학습 시작 ({BACKBONE}, img={IMAGE_SIZE}) ===")
    engine.fit(model=model, datamodule=datamodule)

    print("=== PatchCore 테스트 시작 ===")
    results = engine.test(model=model, datamodule=datamodule)
    print(results)

    print(f"=== 완료! results/{EXPERIMENT_NAME} 폴더에 저장됨 ===")


if __name__ == "__main__":
    main()
