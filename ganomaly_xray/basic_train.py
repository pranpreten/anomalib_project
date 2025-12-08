# train_ganomaly.py

from anomalib.data import Folder
from anomalib.models import Ganomaly
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import torch


# ================================
# 0. 하이퍼파라미터 한 눈에 모아두기
# ================================
IMAGE_SIZE = 256          # GANomaly는 보통 128/256 많이 씀
TRAIN_BS   = 8            # GPU 보고 줄이거나 늘리기 (2,4,8...)
EVAL_BS    = 8
N_FEATURES = 64           # 기본값
LATENT_DIM = 100          # 기본값

# loss weight (논문/기본값)
W_ADV = 1
W_CON = 50
W_ENC = 1

DATA_ROOT        = "../final_dataset"
TRAIN_NORMAL_DIR = "train/normal_5000"
TEST_ABNORM_DIR  = "test/abnormal_card"
TEST_NORMAL_DIR  = "test/normal_3000"

EXPERIMENT_NAME = f"ganomaly_{IMAGE_SIZE}"


def main():

    # ---------------------------------------------------
    # 1. 전처리
    #    GANomaly는 DCGAN 스타일이라 [-1,1] 범위가 무난 → mean=0.5, std=0.5
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
    # 2. 데이터 모듈
    #    Folder 구조는 PatchCore 때랑 동일하게 사용 가능
    # ---------------------------------------------------
    datamodule = Folder(
        name="chest_xray_ganomaly",
        root=DATA_ROOT,
        normal_dir=TRAIN_NORMAL_DIR,   # train: 정상만 사용
        abnormal_dir=TEST_ABNORM_DIR,  # test: 비정상
        normal_test_dir=TEST_NORMAL_DIR,
        train_batch_size=TRAIN_BS,
        eval_batch_size=EVAL_BS,
        num_workers=2,
    )
    datamodule.setup()

    # ---------------------------------------------------
    # 3. GANomaly 모델 정의
    #    문서 기준 기본 파라미터 그대로 두고 몇 개만 노출
    # ---------------------------------------------------
    model = Ganomaly(
        batch_size=TRAIN_BS,
        n_features=N_FEATURES,
        latent_vec_size=LATENT_DIM,
        wadv=W_ADV,
        wcon=W_CON,
        wenc=W_ENC,
        pre_processor=pre_processor,
        # post_processor / evaluator / visualizer 는 기본값 True 유지
    )

    # ---------------------------------------------------
    # 4. Engine: 학습/테스트 오케스트레이터
    #    GAN 기반이라 PatchCore보다 에폭을 충분히 돌려야 함
    # ---------------------------------------------------
    engine = Engine(
        max_epochs=50,  # GPU/시간 보고 20~100 사이에서 조절
        accelerator="gpu",
        devices=1,      # 하나의 GPU 사용 (여러 개면 [0,1] 이런 식)
        default_root_dir=f"results/{EXPERIMENT_NAME}",
        enable_progress_bar=True,
    )

    print(f"=== GANomaly 학습 시작 (img={IMAGE_SIZE}, bs={TRAIN_BS}) ===")
    engine.fit(model=model, datamodule=datamodule)

    print("=== GANomaly 테스트 시작 ===")
    results = engine.test(model=model, datamodule=datamodule)
    print(results)

    print(f"=== 완료! results/{EXPERIMENT_NAME} 폴더에 저장됨 ===")


if __name__ == "__main__":
    main()
