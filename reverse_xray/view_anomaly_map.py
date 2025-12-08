# infer_revdist_one_image.py

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import ReverseDistillation
from anomalib.pre_processing import PreProcessor

from anomalib.visualization import visualize_anomaly_map
from anomalib.visualization.image.functional import visualize_pred_mask

from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype
import matplotlib.pyplot as plt
import torch

# ================================
# 0. 학습 때와 동일한 하이퍼파라미터
# ================================
IMAGE_SIZE = 256  # train_revdist.py 와 동일

# anomalib 기본 저장 규칙 쓰면 대략 이런 구조일 것임.
# 실제 폴더 구조 확인해서 경로 정확히 맞춰줘.
CKPT_PATH = "../model_ckpt/reverse_model.ckpt"  # 네가 옮겨둔 ckpt 경로

# 테스트할 이미지 한 장
IMAGE_PATH = "../inf_dataset/normal/00005515_002.png"
# IMAGE_PATH = "../inf_dataset/abnormal/00030039_004.png"


def main():
    # ---------------------------------------------------
    # 1. 전처리: train_revdist.py 와 동일
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),  # [0,1]
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25],
        ),
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. Reverse Distillation 모델 (학습 때와 동일)
    #    backbone / layers 바꾸면 여기랑 train 둘 다 같이 바꿔야 함
    # ---------------------------------------------------
    model = ReverseDistillation(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        pre_processor=pre_processor,
    )

    # ---------------------------------------------------
    # 3. Engine: 추론용
    # ---------------------------------------------------
    engine = Engine(
        accelerator="gpu",  # 노트북 CPU면 "cpu"
        devices=1,
    )

    # ---------------------------------------------------
    # 4. PredictDataset: 한 장짜리 이미지
    # ---------------------------------------------------
    dataset = PredictDataset(
        path=IMAGE_PATH,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    # ---------------------------------------------------
    # 5. Get predictions (ckpt 로드)
    # ---------------------------------------------------
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
    )

    # test 결과에서 나온 best_threshold 로 갈아끼우면 됨
    BEST_THRESHOLD = 0.5  # placeholder

    # ---------------------------------------------------
    # 6. 결과 출력 + 어노멀리 맵 시각화
    # ---------------------------------------------------
    if predictions is not None:
        for prediction in predictions:
            image_path  = prediction.image_path
            anomaly_map = prediction.anomaly_map      # (H, W) heatmap
            pred_label  = int(prediction.pred_label)  # 0/1
            pred_score  = float(prediction.pred_score)
            mask        = prediction.pred_mask        # (H, W) binary mask

            custom_label = 1 if pred_score > BEST_THRESHOLD else 0

            print("==================================")
            print(f"Image Path      : {image_path}")
            print(f"Anomaly Score   : {pred_score:.4f}")
            print(f"RevDist Label   : {pred_label}")
            print(f"My Label(τ={BEST_THRESHOLD}) : {custom_label}")
            print("==================================")

            # 1) anomaly heatmap
            vis = visualize_anomaly_map(anomaly_map)

            # 2) threshold 적용 mask overlay
            pred_vis = visualize_pred_mask(
                mask,
                mode="fill",
                color=(255, 0, 0),
                alpha=0.5,
            )

            # 3) 시각화
            plt.figure(figsize=(6, 6))
            plt.imshow(vis)
            plt.imshow(pred_vis)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
