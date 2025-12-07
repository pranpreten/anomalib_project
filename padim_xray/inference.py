# eval_padim_images_export.py

import torch
import numpy as np

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize

from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score


# ==============================
# 1. 모델 & 엔진 준비
# ==============================
transform = Compose([
    Resize((256, 256)),
    ToDtype(torch.float32, scale=True),  # 0~1 스케일
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])
pre_processor = PreProcessor(transform=transform)

model = Padim(
    backbone="wide_resnet50_2",          # ★ 학습 때 쓴 것과 같게
    layers=["layer1", "layer2", "layer3"],
    pre_trained=True,
    n_features=100,
    pre_processor=pre_processor,
)

engine = Engine()

CKPT_PATH = "/home/euclidsoft/pattern/padim_xray/results/padim_xray/Padim/chest_xray_padim/v0/weights/lightning/model.ckpt"

NORMAL_DIR = "/home/euclidsoft/pattern/final_dataset/test/normal_patch/"
ABNORMAL_DIR = "/home/euclidsoft/pattern/final_dataset/test/abnormal_patch/"


# ==============================
# 2. 한 폴더(정상 or 비정상)를 전부 추론
# ==============================
def run_split(path, label):
    """
    path 폴더 안의 모든 이미지를 PredictDataset으로 돌려서
    anomaly score와 라벨(0/1)을 반환
    """
    dataset = PredictDataset(
        path=path,               # ★ 여기: 파일이 아니라 '폴더' 경로
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
    )

    scores = []
    labels = []

    for pred in predictions:
        scores.append(float(pred.pred_score))
        labels.append(label)

    return np.array(scores), np.array(labels)


# ==============================
# 3. F1 기준 최적 threshold 찾기
# ==============================
def find_best_f1_threshold(y_true, scores):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    idx = np.argmax(f1s)
    return thr[idx], f1s[idx]


# ==============================
# 4. 메인
# ==============================
def main():
    # normal = 0, abnormal = 1
    normal_scores, normal_labels = run_split(NORMAL_DIR, label=0)
    abnormal_scores, abnormal_labels = run_split(ABNORMAL_DIR, label=1)

    all_scores = np.concatenate([normal_scores, abnormal_scores])
    all_labels = np.concatenate([normal_labels, abnormal_labels])

    # 평균 anomaly score
    print("=== 클래스별 평균 anomaly score ===")
    print(f"Normal   mean: {normal_scores.mean():.4f}")
    print(f"Abnormal mean: {abnormal_scores.mean():.4f}")

    # AUROC
    auroc = roc_auc_score(all_labels, all_scores)

    # F1 기준 최적 threshold
    best_thr, best_f1 = find_best_f1_threshold(all_labels, all_scores)

    # 그 threshold에서 라벨 만들고 F1 다시 계산(검증)
    y_pred = (all_scores >= best_thr).astype(int)
    f1_at_thr = f1_score(all_labels, y_pred)

    print("\n=== 성능 지표 ===")
    print(f"Image-level AUROC        : {auroc:.4f}")
    print(f"Best F1 (PR-curve 기준)  : {best_f1:.4f}")
    print(f"F1 at threshold={best_thr:.4f}: {f1_at_thr:.4f}")


if __name__ == "__main__":
    main()
