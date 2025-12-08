# infer_patchcore_one_image.py

# 1. Import required modules
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.pre_processing import PreProcessor
import matplotlib.pyplot as plt
import cv2
import numpy as np
from anomalib.visualization import visualize_anomaly_map
from anomalib.visualization.image.functional import visualize_gt_mask, visualize_pred_mask

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import torch

# ================================
# 0. 학습 때와 동일한 하이퍼파라미터
# ================================
IMAGE_SIZE    = 224                  # train_patchcore.py 와 동일
BACKBONE      = "resnet34"           # resnet18 / resnet34 / wide_resnet50_2 등
CORESET_RATIO = 0.02                 # 0.001~0.1 사이 추천
NUM_NEIGHBORS = 9                    # k-NN 개수 (PatchCore 기본=9)

CKPT_PATH = "../model_ckpt/patchcore_model.ckpt"  # 네가 옮겨둔 ckpt 경로
# IMAGE_PATH = "../inf_dataset/normal/00005515_002.png"  # 한 장 테스트용 이미지
IMAGE_PATH = "../inf_dataset/abnormal/00030039_004.png"  # 한 장 테스트용 이미지


def main():
    # ---------------------------------------------------
    # 1. 전처리: 학습 때와 동일하게
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
    # 2. PatchCore 모델 정의 (학습 때와 동일!)
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
    # 3. Engine: 추론용 (윈도우면 CPU로 두는 거 추천)
    # ---------------------------------------------------
    engine = Engine(
        accelerator="gpu",  # GPU 있으면 "gpu", devices=[0] 이런 식으로 가도 됨
        devices=1,
    )

    # ---------------------------------------------------
    # 4. PredictDataset: 한 장짜리 이미지
    # ---------------------------------------------------
    dataset = PredictDataset(
        path=IMAGE_PATH,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),  # train과 동일하게 224x224
    )

    # ---------------------------------------------------
    # 5. Get predictions
    # ---------------------------------------------------
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
    )

    # ---------------------------------------------------
    # 6. 결과 출력
    # ---------------------------------------------------
    BEST_THRESHOLD = 0.4786  # 네가 test 단계에서 찾은 베스트 threshold

    if predictions is not None:
        for prediction in predictions:
            image_path  = prediction.image_path
            anomaly_map = prediction.anomaly_map
            pred_label  = int(prediction.pred_label)    # anomalib 내부 기준 (기본 threshold 0.5)
            pred_score  = float(prediction.pred_score)  # anomaly score
            mask = prediction.pred_mask

            # ➤ 내 custom threshold 적용
            custom_label = 1 if pred_score > BEST_THRESHOLD else 0

            vis = visualize_anomaly_map(anomaly_map)

            # 2) threshold 적용 후의 mask 시각화(빨간색 영역)
            pred_vis = visualize_pred_mask(
                mask,
                mode="fill",
                color=(255, 0, 0),
                alpha=0.5,
            )

            # 3) plt 로 바로 overlay
            plt.figure(figsize=(6, 6))
            plt.imshow(vis)       # anomaly heatmap
            plt.imshow(pred_vis)  # mask overlay
            plt.axis("off")
            plt.show()

            print("==================================")
            print(f"Image Path     : {image_path}")
            print(f"Anomaly Score  : {pred_score:.4f}")
            print(f"Anomalib Label : {pred_label}")
            print(f"My Label(τ={BEST_THRESHOLD}) : {custom_label}")
            print("==================================")

if __name__ == "__main__":
    main()
