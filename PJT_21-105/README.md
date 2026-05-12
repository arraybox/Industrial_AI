# PJT_21-105

Contour, Connected Component, Distance Transform을 학습하는 OpenCV 실습 폴더입니다. 흑백 입력 이미지 `BnW.png`를 대상으로 객체 외곽선, 내부 구멍, 연결 요소, 거리 변환을 확인합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `example.py` | `RETR_CCOMP` 계층 구조를 이용해 external/internal contour를 분리하는 기본 예제 |
| `chapter5.py` | Otsu 이진화, contour 분리, connected component, distance transform 종합 실습 |
| `BnW.png` | 입력 이진 이미지 |

## `chapter5.py` 주요 내용

1. Otsu Thresholding
   - `cv2.threshold(..., cv2.THRESH_BINARY + cv2.THRESH_OTSU)`로 자동 임계값 이진화를 수행합니다.

2. External/Internal Contour
   - `cv2.findContours(..., cv2.RETR_CCOMP, ...)`로 계층 정보를 얻습니다.
   - hierarchy의 parent 값이 `-1`이면 외부 contour, 그렇지 않으면 내부 contour로 분류합니다.

3. Connected Component
   - 연결된 객체 영역을 라벨링합니다.
   - 스페이스 키를 누를 때마다 랜덤 5개 컴포넌트를 색상으로 표시하는 구조입니다.

4. Distance Transform
   - `cv2.distanceTransform()`으로 배경에서 객체 내부까지의 거리를 계산합니다.
   - 거리 값 분포를 시각화해 객체 중심부가 더 큰 값을 갖는 것을 확인합니다.

## 실행 방법

```bash
python example.py
python chapter5.py
```

필요 라이브러리는 `opencv-python`, `numpy`, `matplotlib`입니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


