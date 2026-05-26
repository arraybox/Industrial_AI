# PJT_21-110

OpenCV의 `cv2.triangulatePoints()`를 사용해 두 카메라 투영 좌표로부터 3차원 점을 복원하는 triangulation 실습 폴더입니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `example1.py` | 두 카메라 투영 행렬과 2D 대응점을 이용해 3D 점을 삼각측량으로 복원 |
| `example2.py` | stereo calibration 결과와 좌/우 이미지 쌍을 이용해 stereo rectification 수행 |
| `example3.py` | stereo calibration 결과의 좌/우 코너로 Fundamental/Essential matrix 재계산 |
| `example4.py` | Essential matrix를 두 회전 후보와 이동 방향으로 분해 |

## 주요 내용

1. 카메라 투영 행렬 정의
   - `P1`은 기준 카메라 투영 행렬입니다.
   - `P2`는 x축 방향으로 baseline이 있는 두 번째 카메라 투영 행렬입니다.

2. 3D 점 생성
   - 난수로 5개의 homogeneous 3D 점을 생성합니다.
   - 마지막 행은 homogeneous 좌표이므로 1로 설정합니다.

3. 2D 투영점 생성
   - `P1 @ points3d`, `P2 @ points3d`로 각 카메라 영상 평면에 투영합니다.
   - 투영 후 z 값으로 나누어 normalized image coordinate를 만듭니다.
   - 실제 측정 오차를 흉내 내기 위해 작은 Gaussian noise를 추가합니다.

4. Triangulation
   - `cv2.triangulatePoints(P1, P2, points1, points2)`로 3D 점을 복원합니다.
   - 결과를 homogeneous 좌표에서 일반 3D 좌표로 변환합니다.
   - 원본 3D 점과 복원된 3D 점을 출력해 비교합니다.

## `example2.py` 주요 내용

1. Stereo calibration 결과 로드
   - 기본적으로 `PJT_21-110/case1/stereo.npy`를 읽습니다.
   - 해당 파일이 없으면 `PJT_21-110/stereo.npy`도 fallback으로 확인합니다.

2. 좌/우 이미지 입력
   - 기본 입력은 `PJT_21-110/case1/left14.png`, `PJT_21-110/case1/right14.png`입니다.
   - 해당 파일이 없으면 `PJT_21-110/left.png`, `PJT_21-110/right.png`도 fallback으로 확인합니다.

3. Stereo Rectification
   - `cv2.stereoRectify()`로 좌/우 보정 회전 행렬과 투영 행렬을 계산합니다.
   - `cv2.initUndistortRectifyMap()`으로 remap table을 만듭니다.
   - `cv2.remap()`으로 원본 좌/우 이미지를 rectified 이미지로 변환합니다.

4. 결과 시각화
   - Matplotlib subplot으로 원본 좌/우 이미지와 rectified 좌/우 이미지를 한 화면에서 비교합니다.
   - 결과 화면을 `stereo_rectification_result.png`로 저장한 뒤 표시합니다.

## `example3.py` 주요 내용

1. Stereo calibration 결과 로드
   - 기본적으로 `PJT_21-110/case1/stereo.npy`를 읽습니다.
   - 해당 파일이 없으면 `PJT_21-110/stereo.npy`도 fallback으로 확인합니다.

2. 좌/우 코너 데이터 준비
   - `stereo.npy`에 저장된 `left_pts`, `right_pts`를 하나의 배열로 결합합니다.
   - `cv2.undistortPoints()`로 좌/우 이미지의 왜곡을 제거합니다.

3. Fundamental Matrix 계산
   - `cv2.findFundamentalMat(..., cv2.FM_LMEDS)`로 robust하게 Fundamental matrix를 추정합니다.

4. Essential Matrix 계산
   - `E = Kr.T @ F @ Kl` 공식으로 Essential matrix를 계산합니다.
   - 계산 결과와 `cv2.stereoCalibrate()`가 저장한 `E`, `F`를 함께 출력해 비교합니다.

## `example4.py` 주요 내용

1. Essential matrix 로드
   - 기본적으로 `PJT_21-110/case1/stereo.npy`를 읽습니다.
   - 해당 파일이 없으면 `PJT_21-110/stereo.npy`도 fallback으로 확인합니다.

2. Essential matrix 분해
   - `cv2.decomposeEssentialMat(E)`를 사용합니다.
   - 가능한 회전 행렬 후보 `R1`, `R2`와 이동 방향 `T`를 계산합니다.

3. 결과 출력
   - `Rotation 1`, `Rotation 2`, `Translation`을 콘솔에 출력합니다.

## 실행 방법

```bash
python example1.py
python example2.py
python example3.py
python example4.py
```

`example2.py`를 실행하려면 다음 파일이 필요합니다.

```text
PJT_21-110/
├── case1/
│   ├── stereo.npy
│   ├── left14.png
│   └── right14.png
├── left.png
└── right.png
```

필요 라이브러리는 `opencv-python`, `numpy`, `matplotlib`입니다.

## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |
