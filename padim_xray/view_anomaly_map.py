# infer_padim_one_image.py

# 1. Import required modules
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor

import matplotlib.pyplot as plt
import cv2
import numpy as np

from anomalib.visualization import visualize_anomaly_map
from anomalib.visualization.image.functional import visualize_pred_mask

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import torch

# ================================
# 0. 학습 때와 동일한 하이퍼파라미터
# ================================
IMAGE_SIZE  = 256                # train_padim.py 의 IMAGE_SIZE
BACKBONE    = "resnet50"          # train_padim.py 의 BACKBONE
N_FEATURES  = 120                 # train_padim.py 의 N_FEATURES


CKPT_PATH = "../model_ckpt/padim_model.ckpt"  # 네가 옮겨둔 ckpt 경로
# ↑ 여기 경로는 네 결과 폴더 구조에 맞게 수정해줘.
#   (예: results/{EXPERIMENT_NAME}/chest_xray_padim/weights/last.ckpt 이런 식일 수도 있음)

# 테스트할 이미지 한 장
IMAGE_PATH = "../inf_dataset/normal/00026048_000.png"
# IMAGE_PATH = "../inf_dataset/abnormal/00030039_004.png"


def main():
    # ---------------------------------------------------
    # 1. 전처리: 학습 때와 동일하게
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),  # [0, 1]
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. PaDiM 모델 정의 (학습 때와 동일!)
    # ---------------------------------------------------
    model = Padim(
        backbone=BACKBONE,
        layers=["layer1", "layer2", "layer3"],  # train_padim.py 와 동일
        pre_trained=True,
        n_features=N_FEATURES,
        pre_processor=pre_processor,
    )

    # ---------------------------------------------------
    # 3. Engine: 추론용
    # ---------------------------------------------------
    engine = Engine(
        accelerator="gpu",  # CPU로 돌리고 싶으면 "cpu" 로 바꾸면 됨
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
    # 5. Get predictions
    # ---------------------------------------------------
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
    )

# ---------------------------------------------------
    # 6. 결과 출력 + 어노멀리 맵 시각화
    # ---------------------------------------------------
    if predictions is not None:
        for prediction in predictions:
            image_path  = prediction.image_path
            anomaly_map = prediction.anomaly_map
            pred_label  = int(prediction.pred_label)
            pred_score  = float(prediction.pred_score)
            mask        = prediction.pred_mask


            print("==================================")
            print(f"Image Path     : {image_path}")
            print(f"Anomaly Score  : {pred_score:.4f}")
            print(f"PaDiM Label    : {pred_label}")
            print("==================================")

            # 1) anomaly heatmap
            vis = visualize_anomaly_map(anomaly_map)

            # 2) threshold 적용 후 mask 시각화
            pred_vis = visualize_pred_mask(
                mask,
                mode="fill",
                color=(255, 0, 0),
                alpha=0.5,
            )

            # 3) overlay 해서 보기
            plt.figure(figsize=(6, 6))
            plt.imshow(vis)
            plt.imshow(pred_vis)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
