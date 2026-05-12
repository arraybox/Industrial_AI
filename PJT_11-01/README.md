# PJT_11-01

PyTorch를 이용한 선형회귀와 이미지 분류 실습 폴더입니다. 기초 회귀 모델을 직접 학습하는 노트북, CIFAR-10 기반 AlexNet 전이학습 노트북, 단일 이미지 추론 노트북, Raspberry Pi 카메라 기반 실시간 분류 스크립트가 함께 들어 있습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `선형회귀.ipynb` | PyTorch Tensor와 SGD로 단순 선형회귀를 직접 학습 |
| `CNN_Classification_training.ipynb` | 사전학습 AlexNet을 CIFAR-10 10개 클래스로 재학습 |
| `CNN_Image_Classification.ipynb` | 사전학습 AlexNet으로 `dog.jpg` 단일 이미지 추론 |
| `rpi_pytorch_imageclassification.py` | Raspberry Pi `picamera2` 카메라 입력을 MobileNetV2로 실시간 분류 |
| `dog.jpg` | 단일 이미지 분류 예제 입력 |

## 주요 학습 내용

1. 선형회귀
   - `weight`, `bias`를 직접 학습 가능한 텐서로 정의합니다.
   - MSE 비용 함수를 계산하고 `optim.SGD`로 10000 epoch 학습합니다.
   - 학습된 직선을 Matplotlib으로 실제 데이터와 함께 시각화합니다.

2. AlexNet 전이학습
   - `torchvision.datasets.CIFAR10`을 다운로드합니다.
   - 입력 이미지를 224 크기로 변환하고 정규화합니다.
   - 사전학습 AlexNet의 마지막 classifier 레이어를 10개 클래스 출력으로 교체합니다.
   - CrossEntropyLoss와 Adam optimizer로 학습 후 테스트 정확도를 출력합니다.

3. 단일 이미지 추론
   - ImageNet 사전학습 AlexNet을 평가 모드로 사용합니다.
   - `dog.jpg`를 Resize, CenterCrop, Normalize한 뒤 예측 클래스를 출력합니다.
   - ImageNet 클래스 이름 파일을 GitHub에서 내려받아 결과 라벨을 표시합니다.

4. Raspberry Pi 실시간 분류
   - MobileNetV2 사전학습 가중치를 사용합니다.
   - `Picamera2` 프레임을 PIL 이미지로 변환해 전처리합니다.
   - OpenCV 화면에 예측 라벨과 확률을 표시합니다.

## 실행 방법

노트북은 Jupyter 또는 VS Code에서 셀 단위로 실행합니다.

```bash
jupyter notebook
```

Raspberry Pi 스크립트는 카메라가 연결된 환경에서 실행합니다.

```bash
python rpi_pytorch_imageclassification.py
```

필요 라이브러리는 `torch`, `torchvision`, `matplotlib`, `Pillow`, `opencv-python`이며, Raspberry Pi 실습에는 `picamera2`가 추가로 필요합니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


