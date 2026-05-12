# PJT_21-03

OpenCV 기반 이미지 필터링과 주파수 영역 필터링을 학습하는 폴더입니다. 공간 영역에서는 Unsharp Mask, Sobel, Gabor, Threshold, Opening/Closing을 다루고, 주파수 영역에서는 DFT와 저역통과 필터를 실습합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `chapter4.py` | 공간 필터링과 DFT 기반 주파수 필터링 종합 실습 |
| `example3.py` | Gaussian 커널을 변형해 Unsharp Mask를 직접 적용하는 예제 |
| `Lena.png` | 실습용 입력 이미지 |

## `chapter4.py` 주요 내용

1. Unsharp Mask
   - Gaussian Blur를 만든 뒤 원본과 가중 합성해 영상을 선명하게 만듭니다.

2. Sobel Filter
   - X/Y 방향 기울기를 계산하고 magnitude로 edge 강도를 구합니다.

3. Gabor Filter
   - 특정 방향과 주파수 성분에 반응하는 Gabor 커널을 적용합니다.

4. Threshold Trackbar
   - Sobel 결과와 Gabor 결과의 차이를 계산합니다.
   - OpenCV trackbar로 threshold 값을 바꾸며 이진 결과를 확인합니다.

5. Morphology
   - threshold 결과에 Opening과 Closing을 적용해 작은 잡음과 빈틈을 보정합니다.

6. Frequency-based Filtering
   - DFT로 영상을 주파수 영역으로 변환합니다.
   - magnitude spectrum을 시각화합니다.
   - 원형/사각형 low-pass mask를 적용한 뒤 inverse DFT로 복원합니다.

## `example3.py` 주요 내용

- `cv2.getGaussianKernel()`로 2차원 Gaussian 커널을 만듭니다.
- 중앙값을 조정해 sharpening kernel 형태로 바꿉니다.
- `cv2.filter2D()`로 원본 이미지에 필터를 적용합니다.

## 실행 방법

```bash
python chapter4.py
python example3.py
```

필요 라이브러리는 `opencv-python`, `numpy`, `matplotlib`, `scipy`입니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


