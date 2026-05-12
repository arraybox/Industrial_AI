# PJT_21-02

OpenCV 영상 처리 기초를 학습하는 폴더입니다. 컬러 영상 읽기, 흑백 변환, 히스토그램 평활화, 감마 보정, HSV 색공간 변환, 공간 필터링을 각각 예제 파일과 종합 실습 파일에서 확인할 수 있습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `chapter3.py` | 컬러/흑백 변환, 히스토그램 평활화, 감마 보정, HSV 채널 분리, 필터링 종합 실습 |
| `example1.py` | 이미지 dtype/shape 확인, BGR/Gray/HSV 변환 기초 |
| `example2.py` | grayscale histogram 계산과 histogram equalization 비교 |
| `example3.py` | 노이즈 추가 후 Gaussian, Median, Bilateral 필터 비교 |
| `Lena.png` | 실습용 입력 이미지 |

## `chapter3.py` 주요 내용

1. `cv2.imread()`로 컬러 이미지 읽기
2. `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`로 흑백 변환
3. `cv2.equalizeHist()`로 히스토그램 평활화
4. `gamma = 2.2` 값을 사용한 감마 보정
5. HSV 색공간 변환 후 H, S, V 채널 정규화
6. H 채널에 Median Filter 적용
7. S 채널에 Gaussian Filter 적용
8. V 채널에 Bilateral Filter 적용

## 예제별 포인트

- `example1.py`: OpenCV 이미지가 기본적으로 BGR 순서라는 점과 float 변환 후 표시 방식을 확인합니다.
- `example2.py`: NumPy histogram과 Matplotlib을 이용해 평활화 전후 픽셀 분포를 비교합니다.
- `example3.py`: 노이즈가 섞인 영상에서 필터별 보존/평활화 특성을 비교합니다.

## 실행 방법

```bash
python chapter3.py
python example1.py
python example2.py
python example3.py
```

필요 라이브러리는 `opencv-python`, `numpy`, `matplotlib`입니다. `cv2.imshow()`를 사용하므로 GUI가 가능한 환경에서 실행해야 합니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


