# infer_draem_one_image.py

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Draem
from anomalib.pre_processing import PreProcessor

from anomalib.visualization import visualize_anomaly_map
from anomalib.visualization.image.functional import visualize_pred_mask

from torchvision.transforms.v2 import Compose, Resize, ToDtype
import matplotlib.pyplot as plt
import torch

# ================================
# 0. 학습 때와 동일한 하이퍼파라미터
# ================================
IMAGE_SIZE = 256   # train_draem.py 와 동일

# train_draem.py 마지막에 프린트한 경로
CKPT_PATH = "../model_ckpt/deram_model.ckpt"  # 네가 옮겨둔 ckpt 경로

# 테스트할 이미지 한 장
IMAGE_PATH = "../inf_dataset/normal/00005515_002.png"
# IMAGE_PATH = "../inf_dataset/abnormal/00030039_004.png"


def main():
    # ---------------------------------------------------
    # 1. 전처리: train_draem.py 와 동일
    #    - Resize(256, 256)
    #    - ToDtype(float32, scale=True) 만 사용
    # ---------------------------------------------------
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToDtype(torch.float32, scale=True),  # [0,1] 로 스케일
    ])
    pre_processor = PreProcessor(transform=transform)

    # ---------------------------------------------------
    # 2. DRAEM 모델 정의 (학습 때와 동일!)
    #    train_draem.py 에서 주석 풀고 쓴 인자 있으면
    #    여기에도 똑같이 넣어줘야 함.
    # ---------------------------------------------------
    model = Draem(
        pre_processor=pre_processor,
        # lr=1e-4,
        # hidden_dim=128,
        # anomaly_source="perlin",
        # beta=0.3,
        # weight_decay=1e-5,
    )

    # ---------------------------------------------------
    # 3. Engine: 추론용
    # ---------------------------------------------------
    engine = Engine(
        accelerator="gpu",  # 윈도우 노트북이면 "cpu" 로 바꿔도 됨
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
    # 5. Get predictions (ckpt 로 weight 로딩)
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
            anomaly_map = prediction.anomaly_map      # (H, W) heatmap
            pred_label  = int(prediction.pred_label)  # 0/1
            pred_score  = float(prediction.pred_score)
            mask        = prediction.pred_mask        # (H, W) binary mask


            print("==================================")
            print(f"Image Path      : {image_path}")
            print(f"Anomaly Score   : {pred_score:.4f}")
            print(f"DRAEM Label     : {pred_label}")
            print("==================================")

            # 1) anomaly heatmap
            vis = visualize_anomaly_map(anomaly_map)

            # 2) threshold 적용 후 mask overlay
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
