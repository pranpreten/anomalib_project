# train_padim.py

from anomalib.data import Folder
from anomalib.models import Padim       # ★ Patchcore → Padim
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize, CenterCrop, ToDtype
import torch


def main():
    # ---------------------------------------------------
    # 1. 데이터 모듈: 우리 X-ray 폴더 구조 그대로 사용
    # ---------------------------------------------------
    transform = Compose([
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),  # 0~1 스케일
        Normalize(mean=[0.5, 0.5, 0.5],
              std=[0.25, 0.25, 0.25]),
    ])
    pre_processor = PreProcessor(transform=transform)

    datamodule = Folder(
        name="chest_xray_padim",
        root="../final_dataset",
        normal_dir="train/normal_patch",
        abnormal_dir="test/abnormal_card",
        normal_test_dir="test/normal_patch",
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=2,
    )

    datamodule.setup()

    # ---------------------------------------------------
    # 2. PaDiM 모델 정의
    # ---------------------------------------------------
    model = Padim(
        backbone="wide_resnet50_2",              # 논문에서 많이 쓰는 백본
        layers=["layer1", "layer2", "layer3"],   # PaDiM 기본 세팅
        pre_trained=True,
        n_features=100,                          # wide_resnet50_2 권장값
        pre_processor=pre_processor,
    )

    # ---------------------------------------------------
    # 3. Engine: 학습/테스트
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=1,          # PaDiM도 epoch=1이면 충분
        accelerator="gpu",
        devices=1,
        default_root_dir="results/padim_xray",   # 저장 위치만 이름 바꿈
        enable_progress_bar=True,
    )

    print("=== PaDiM 학습 시작 ===")
    engine.fit(model=model, datamodule=datamodule)

    print("=== PaDiM 테스트 시작 ===")
    results = engine.test(model=model, datamodule=datamodule)
    print(results)

    print("=== 완료! 체크포인트와 로그는 results/padim_xray 아래에 저장됨 ===")


if __name__ == "__main__":
    main()
