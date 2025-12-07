# train_patchcore.py

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize, CenterCrop,ToDtype
import torch



def main():
    # ---------------------------------------------------
    # 1. 데이터 모듈: 우리 X-ray 폴더 구조 연결
    # ---------------------------------------------------

    transform = Compose([
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.25, 0.25, 0.25]),
    ])
    pre_processor = PreProcessor(transform=transform)
    datamodule = Folder(
        name="chest_xray_patchcore",
        root="../final_dataset",                # dataset/ 아래에 train/test 폴더 있는 구조
        normal_dir="train/normal_patch",
        abnormal_dir="test/abnormal_card",
        normal_test_dir="test/normal_patch",
        train_batch_size=2,    
        eval_batch_size=2,     
        num_workers=2,    
    )


    # (선택) 명시적으로 셋업
    datamodule.setup()

    # ---------------------------------------------------
    # 3. PatchCore 모델 정의 (기본 하이퍼파라로 시작)
    # ---------------------------------------------------
    model = Patchcore(
        backbone="wide_resnet50_2",      
        # backbone="resnet34",      
        layers=("layer2", "layer3"),     # 논문 기본 조합
        pre_trained=True,                # ImageNet 사전학습 사용
        coreset_sampling_ratio=0.01,     # 메모리 뱅크 샘플링 비율 (처음은 0.1로)
        num_neighbors=9,                 # 기본 9-NN
        pre_processor=pre_processor

    )

    # ---------------------------------------------------
    # 4. Engine: 학습/테스트 오케스트레이터
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        default_root_dir="results/patchcore_xray",
        enable_progress_bar=True,
    )
    # ---------------------------------------------------
    # 5. 학습 + 테스트
    # ---------------------------------------------------
    print("=== PatchCore 학습 시작 ===")
    engine.fit(model=model, datamodule=datamodule)

    print("=== PatchCore 테스트 시작 ===")
    results = engine.test(model=model, datamodule=datamodule)
    print(results)

    print("=== 완료! 체크포인트와 로그는 results/patchcore_xray 아래에 저장됨 ===")


if __name__ == "__main__":
    main()
