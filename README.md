# 흉부 X-ray 이상탐지 모델 실험

흉부 X-ray에서 정상 영상과 다른 패턴을 찾기 위해 여러 이상탐지 모델을 실험한 프로젝트다. 데이터 탐색, 관심 영역 전처리, 정상 영상 표본 구성, 모델별 학습·추론 및 이상 영역 시각화 코드를 포함한다.

## 실험 구성

- 정상·비정상 분포와 연령·성별 데이터 탐색
- 영상 ROI 자르기와 학습·평가 표본 분리
- PatchCore, PaDiM, FastFlow, GANomaly, DRAEM, Reverse Distillation 실험 폴더
- 이상 점수 및 anomaly map 확인

정상 영상으로 기준 분포를 학습하고 비정상 영상에 대한 반응을 확인하는 이상탐지 실습이다. 질환을 확정하는 임상 진단 시스템은 아니다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `eda_xray_data.py` | 메타데이터 탐색 및 정상·비정상 분리 |
| `age_distribution.py`, `gender_distribution.py` | 연령·성별 분포 확인 |
| `image_preprocessing_roi.py` | ROI 영상 생성 |
| `roi_to_patchcore.py` | 정상 학습·평가 영상 표본 구성 |
| `patchcore_xray/`, `padim_xray/` | 대표 이상탐지 모델 실험 |
| `fastflow_xray/`, `ganomaly_xray/`, `dream_xray/`, `reverse_xray/` | 다른 모델의 학습·시각화 코드 |
| `example_code/` | 데이터셋·학습·추론 예제 |

## 데이터 준비

영상 데이터와 학습 가중치는 GitHub 저장소에 포함되어 있지 않다. 메타데이터 처리 코드는 `archive/Data_Entry_2017.csv`를 읽는다. 해당 Chest X-ray 데이터와 영상 파일을 별도로 준비한 뒤 정상·비정상 분리 및 ROI 처리를 수행해야 한다.

PatchCore 스크립트가 기대하는 구조는 다음과 같다.

```text
final_dataset/
  train/normal_5000/
  test/normal_3000/
  test/abnormal_card/
```

파일 복사·표본 생성 스크립트의 원본 경로와 표본 수를 먼저 확인한다. `roi_to_patchcore.py`의 비정상 영상 복사 부분은 주석 처리되어 있어 비정상 폴더가 자동으로 준비된다고 가정하면 안 된다.

## 실행 방법

필요한 주요 패키지는 PyTorch, torchvision, Anomalib, pandas, OpenCV, tqdm, matplotlib이다. 고정된 환경 명세가 없으므로 `anomalib.engine.Engine`, `anomalib.pre_processing.PreProcessor`, `torchvision.transforms.v2`를 지원하는 호환 버전으로 환경을 구성한다.

데이터와 환경을 준비한 뒤 PatchCore 폴더에서 실행한다.

```bash
cd patchcore_xray
python basic_train.py
```

현재 코드는 `accelerator="gpu"`로 설정되어 있다. 데이터 경로는 실행 위치 기준 `../final_dataset`이며, CPU 실험이나 다른 폴더 구조를 사용하려면 설정 변경이 필요하다.

## 결과 확인

학습 후 평가 결과가 출력되고 `results/` 아래에 실험 산출물이 생성된다. 각 모델의 `view_anomaly_map.py`는 내부에 지정한 체크포인트·영상 경로를 맞춘 뒤 사용한다. 비교 시 영상 분할, 정상 표본 수, 해상도와 평가 조건을 동일하게 맞춰야 한다. 저장소 코드만으로 모든 모델의 성능 우열이나 진단 정확도를 단정하지 않는다.

---

[GitHub 저장소](https://github.com/pranpreten/anomalib_project) · [문서 작성 기준 코드](https://github.com/pranpreten/anomalib_project/tree/8ad1655aeb3ecf27073ef70686cdf8229d47bfc2)
