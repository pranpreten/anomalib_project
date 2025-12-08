# infer_fastflow_one_image.py

from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.pre_processing import PreProcessor

from anomalib.visualization import visualize_anomaly_map
from anomalib.visualization.image.functional import visualize_pred_mask

from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype
import matplotlib.pyplot as plt
import torch


# ================================
# 0. 학습 때와 동일한 하이퍼파라미터
# ================================
IMAGE_SIZE = 256  # train_fastflow.py 와 동일

# 실제 저장된 ckpt 경로에 맞게 수정
# 보통 구조가 이런 느낌일 가능성이 큼:
# results/fastflow_xray_card/chest_xray_fastflow/weights/last.ckpt
CKPT_PATH = "../model_ckpt/fastflow_model.ckpt"  # 네가 옮겨둔 ckpt 경로

# 테스트할 이미지 경로
# IMAGE_PATH = "../inf_dataset/normal/00005515_002.png"
IMAGE_PATH = "../inf_dataset/abnormal/00030039_004.png"


def main():
    # ---------------------------------------------------
    # 1. 전처리: train_fastflow.py 와 완전히 동일
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),   # [0,1]
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25],
        ),
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. FastFlow 모델 정의 (학습 때와 동일!)
    # ---------------------------------------------------
    model = Fastflow(
        backbone="resnet18",
        pre_trained=True,
        flow_steps=8,
        conv3x3_only=False,
        hidden_ratio=1.0,
        pre_processor=pre_processor,
        # evaluator는 ckpt 안에 저장된 걸 쓰고 싶으면 생략 가능
        # 여기서는 train_fastflow.py 와 동일하게 evaluator 안 넘겨도 됨
    )

    # ---------------------------------------------------
    # 3. Engine: 추론용
    # ---------------------------------------------------
    engine = Engine(
        accelerator="gpu",   # CPU면 "cpu"
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
    # 5. 예측 + ckpt 로드
    # ---------------------------------------------------
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
    )

    # test 로그에서 나온 best_threshold 로 바꿔 넣어
    BEST_THRESHOLD = 0.5  # placeholder

    # ---------------------------------------------------
    # 6. 결과 출력 + 어노멀리 맵 시각화
    # ---------------------------------------------------
    if predictions is not None:
        for prediction in predictions:
            image_path  = prediction.image_path
            anomaly_map = prediction.anomaly_map      # (H, W)
            pred_label  = int(prediction.pred_label)  # 0/1
            pred_score  = float(prediction.pred_score)
            mask        = prediction.pred_mask        # (H, W) binary mask

            custom_label = 1 if pred_score > BEST_THRESHOLD else 0

            print("==================================")
            print(f"Image Path        : {image_path}")
            print(f"Anomaly Score     : {pred_score:.4f}")
            print(f"FastFlow Label    : {pred_label}")
            print(f"My Label(τ={BEST_THRESHOLD}) : {custom_label}")
            print("==================================")

            # 1) heatmap
            vis = visualize_anomaly_map(anomaly_map)

            # 2) threshold 적용된 mask overlay
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
