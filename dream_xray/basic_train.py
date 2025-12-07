# train_draem.py

from anomalib.data import Folder
from anomalib.models import Draem          # ★ PaDiM → DRAEM
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor

from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype
import torch


def main():
    # ---------------------------------------------------
    # 1. 데이터 모듈: X-ray 폴더 구조 그대로 사용
    #    - train: normal 만 사용 (DRAEM은 정상만으로 학습)
    #    - test : normal / abnormal 둘 다 사용해서 평가
    # ---------------------------------------------------
    transform = Compose([
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),  # 0~1 스케일
    ])
    pre_processor = PreProcessor(transform=transform)

    datamodule = Folder(
        name="chest_xray_draem",
        root="../final_dataset",
        normal_dir="train/normal_patch",      # 학습에 쓰는 정상 이미지
        abnormal_dir="test/abnormal_patch",   # 평가용 비정상
        normal_test_dir="test/normal_patch",  # 평가용 정상
        train_batch_size=4,                   # DRAEM은 보통 배치 조금 키워도 됨 (GPU 보고 조절)
        eval_batch_size=4,
        num_workers=4,
    )

    datamodule.setup()

    # ---------------------------------------------------
    # 2. DRAEM 모델 정의
    #    - anomalib의 Draem은 기본 설정으로도 돌아가게 설계되어 있음
    #    - pre_processor만 우리가 만든 transform으로 붙여줌
    #    - synthetic anomaly(가짜 이상) 생성은 모델 내부 기본 설정 사용
    # ---------------------------------------------------
    model = Draem(
        pre_processor=pre_processor,
        # encoder, decoder 같은 세부 옵션은 기본값 그대로 두고
        # 먼저 한 번 돌려본 뒤 필요하면 나중에 튜닝
    )

    # ---------------------------------------------------
    # 3. Engine: 학습/테스트 오케스트레이터
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=20,                        # DRAEM은 reconstruction 기반이라 epoch 1은 너무 짧음
        accelerator="gpu",
        devices=1,
        default_root_dir="results/draem_xray",
        enable_progress_bar=True,
    )

    print("=== DRAEM 학습 시작 ===")
    engine.fit(model=model, datamodule=datamodule)

    print("=== DRAEM 테스트 시작 ===")
    results = engine.test(model=model, datamodule=datamodule)
    print(results)

    print("=== 완료! 체크포인트와 로그는 results/draem_xray 아래에 저장됨 ===")


if __name__ == "__main__":
    main()
