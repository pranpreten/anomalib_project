# infer_ganomaly_one_image.py

from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Ganomaly
from anomalib.pre_processing import PreProcessor

from anomalib.visualization import visualize_anomaly_map
from anomalib.visualization.image.functional import visualize_pred_mask

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import matplotlib.pyplot as plt
import torch


# ================================
# 0. 학습 때와 동일한 하이퍼파라미터
# ================================
IMAGE_SIZE = 256          # train_ganomaly.py 와 동일
TRAIN_BS   = 8            # batch_size 하이퍼파라미터 (ckpt랑 맞추기용)
N_FEATURES = 64
LATENT_DIM = 100

W_ADV = 1
W_CON = 50
W_ENC = 1

# ✔ ckpt 경로: 실제 results 폴더 구조 보고 필요하면 수정
# 보통 이런 구조일 확률이 높다:
# results/ganomaly_256/chest_xray_ganomaly/weights/last.ckpt
CKPT_PATH = "../model_ckpt/ganomaly_model.ckpt"  # 네가 옮겨둔 ckpt 경로

# 테스트할 이미지 한 장
# IMAGE_PATH = "../inf_dataset/normal/00005515_002.png"
IMAGE_PATH = "../inf_dataset/abnormal/00030039_004.png"


def main():
    # ---------------------------------------------------
    # 1. 전처리: train_ganomaly.py 와 완-전 동일
    #    [-1, 1] 범위 맞추는 Normalize(mean=0.5, std=0.5)
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. GANomaly 모델 정의 (train 코드랑 세팅 맞추기)
    # ---------------------------------------------------
    model = Ganomaly(
        batch_size=TRAIN_BS,
        n_features=N_FEATURES,
        latent_vec_size=LATENT_DIM,
        wadv=W_ADV,
        wcon=W_CON,
        wenc=W_ENC,
        pre_processor=pre_processor,
    )

    # ---------------------------------------------------
    # 3. Engine: 추론용
    # ---------------------------------------------------
    engine = Engine(
        accelerator="gpu",   # 윈도우 노트북이면 "cpu" 로 바꿔도 됨
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
    # 5. 예측 (ckpt 로딩 포함)
    # ---------------------------------------------------
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
    )

    # test 로그에서 나온 best_threshold 로 교체
    BEST_THRESHOLD = 0.5  # placeholder 숫자, 나중에 너 값으로 바꿔

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
            print(f"Image Path        : {image_path}")
            print(f"Anomaly Score     : {pred_score:.4f}")
            print(f"GANomaly Label    : {pred_label}")
            print(f"My Label(τ={BEST_THRESHOLD}) : {custom_label}")
            print("==================================")

            # 1) heatmap
            vis = visualize_anomaly_map(anomaly_map)

            # 2) thresh 적용된 mask overlay
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
