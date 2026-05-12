# PJT_21-108

OpenCV의 기하학적 영상 변환과 프레임 간 특징점 추적을 실습하는 폴더입니다. Affine/Perspective Transform을 이용한 이미지 워핑 예제와 Lucas-Kanade Optical Flow 기반 keypoint tracking 예제가 포함되어 있습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `example1.py` | 마우스로 선택한 점을 기준으로 affine transform, inverse affine transform, rotation, perspective transform을 수행 |
| `example2.py` | `traffic.mp4` 영상에서 Lucas-Kanade 알고리즘으로 프레임 간 특징점을 추적 |
| `circlesgrid.png` | `example1.py`에서 사용하는 워핑 실습용 이미지 |
| `traffic.mp4` | `example2.py`에서 사용하는 교통 영상 |

## `example1.py` 주요 내용

1. 마우스 클릭으로 변환 기준점 선택
   - affine transform은 3점을 선택합니다.
   - perspective transform은 4점을 선택합니다.

2. Affine Transform
   - `cv2.getAffineTransform()`으로 변환 행렬을 계산합니다.
   - `cv2.warpAffine()`으로 이미지를 240x240 크기로 보정합니다.

3. Inverse Affine Transform
   - `cv2.invertAffineTransform()`으로 역변환 행렬을 구합니다.
   - 보정된 이미지를 다시 원래 좌표계로 되돌리는 과정을 확인합니다.

4. Rotation
   - 선택한 첫 번째 점을 중심으로 `cv2.getRotationMatrix2D()`를 사용해 이미지를 회전합니다.

5. Perspective Transform
   - `cv2.getPerspectiveTransform()`과 `cv2.warpPerspective()`로 원근 왜곡을 보정합니다.

## `example2.py` 주요 내용

1. 영상 입력
   - `traffic.mp4`를 `example2.py`와 같은 폴더에서 읽습니다.

2. 특징점 검출
   - 첫 프레임에서 `cv2.goodFeaturesToTrack()`으로 추적할 특징점을 검출합니다.

3. Lucas-Kanade Optical Flow
   - `cv2.calcOpticalFlowPyrLK()`로 이전 프레임의 특징점이 현재 프레임에서 어디로 이동했는지 계산합니다.
   - 추적된 점을 초록색 원으로 표시합니다.

4. 키 입력
   - `ESC`: 종료
   - `c`: 추적점을 초기화하고 다시 검출

## 실행 방법

```bash
python example1.py
python example2.py
```

필요 라이브러리는 `opencv-python`, `numpy`입니다. 두 예제 모두 OpenCV 창을 사용하므로 GUI가 가능한 환경에서 실행해야 합니다.

## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |
