# PJT_21-109

OpenCV 체스보드 패턴을 이용해 pinhole camera calibration을 수행하는 실습 폴더입니다. 여러 장의 체스보드 이미지를 읽고, 코너를 검출한 뒤 카메라 내부 파라미터와 왜곡 계수를 계산합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `example1.py` | 체스보드 코너 검출, sub-pixel 보정, `cv2.calibrateCamera()` 기반 카메라 캘리브레이션 |
| `example2.py` | fisheye 체스보드 코너 검출, sub-pixel 보정, `cv2.fisheye.calibrate()` 기반 어안 카메라 캘리브레이션 |
| `example3.py` | 좌/우 체스보드 이미지 쌍을 이용한 `cv2.stereoCalibrate()` 기반 스테레오 리그 캘리브레이션 |
| `example4.py` | 캘리브레이션 결과를 이용해 체스보드 코너를 undistort하고 다시 project하는 예제 |
| `pinhole_calib/` | 캘리브레이션 입력 이미지 폴더. `img`로 시작하는 이미지 파일 필요 |
| `fisheyes/` | 어안 캘리브레이션 입력 이미지 폴더. `Fisheye1`로 시작하는 이미지 파일 필요 |
| `stereo/case1/` | 스테레오 캘리브레이션 입력 이미지 폴더. `left*.png`, `right*.png` 쌍 필요 |
| `camera_mat.npy` | 실행 후 저장되는 카메라 행렬 |
| `dist_coefs.npy` | 실행 후 저장되는 왜곡 계수 |
| `fisheye_camera_mat.npy` | `example2.py` 실행 후 저장되는 어안 카메라 행렬 |
| `fisheye_dist_coefs.npy` | `example2.py` 실행 후 저장되는 어안 왜곡 계수 |
| `stereo.npy` | `example3.py` 실행 후 저장되는 스테레오 캘리브레이션 결과 |

## 주요 내용

1. 이미지 입력
   - `PJT_21-109/pinhole_calib` 폴더에서 `img`로 시작하는 이미지 파일을 읽습니다.
   - 지원 확장자는 `.jpg`, `.jpeg`, `.png`, `.bmp`입니다.

2. 체스보드 코너 검출
   - 패턴 크기는 `(10, 7)`입니다.
   - `cv2.findChessboardCorners()`로 내부 코너를 검출합니다.
   - 검출 결과를 `cv2.drawChessboardCorners()`로 화면에 표시합니다.

3. 샘플 선택
   - 코너가 검출된 이미지에서는 창이 대기 상태가 됩니다.
   - `s` 키를 누르면 해당 프레임을 캘리브레이션 샘플로 저장합니다.
   - `ESC` 키를 누르면 이미지 순회를 종료합니다.

4. 코너 정밀화와 캘리브레이션
   - `cv2.cornerSubPix()`로 코너 좌표를 sub-pixel 단위로 보정합니다.
   - `cv2.calibrateCamera()`로 RMS error, camera matrix, distortion coefficients를 계산합니다.

5. 결과 저장
   - `camera_mat.npy`에 카메라 행렬을 저장합니다.
   - `dist_coefs.npy`에 왜곡 계수를 저장합니다.

## `example2.py` 주요 내용

1. 이미지 입력
   - `PJT_21-109/fisheyes` 폴더에서 `Fisheye1`로 시작하는 이미지 파일을 읽습니다.
   - 지원 확장자는 `.jpg`, `.jpeg`, `.png`, `.bmp`입니다.

2. 체스보드 코너 검출
   - fisheye 캘리브레이션 패턴 크기는 `(8, 6)`입니다.
   - `cv2.findChessboardCorners()`로 코너를 검출하고 화면에 표시합니다.

3. 샘플 선택
   - 코너가 검출된 이미지에서 `s` 키를 누르면 샘플로 저장합니다.
   - `ESC` 키를 누르면 이미지 순회를 종료합니다.

4. Fisheye 캘리브레이션
   - `cv2.cornerSubPix()`로 코너를 정밀화합니다.
   - `cv2.fisheye.calibrate()`로 RMS error, fisheye camera matrix, fisheye distortion coefficients를 계산합니다.

5. 결과 저장
   - `fisheye_camera_mat.npy`에 어안 카메라 행렬을 저장합니다.
   - `fisheye_dist_coefs.npy`에 어안 왜곡 계수를 저장합니다.

## `example3.py` 주요 내용

1. 이미지 입력
   - `PJT_21-109/stereo/case1` 폴더에서 `left*.png`, `right*.png` 파일을 읽습니다.
   - 좌/우 이미지 개수가 같아야 하며, 정렬된 순서대로 한 쌍으로 처리합니다.

2. 체스보드 코너 검출
   - 스테레오 캘리브레이션 패턴 크기는 `(9, 6)`입니다.
   - 좌/우 이미지 모두에서 `cv2.findChessboardCorners()`가 성공한 쌍만 사용합니다.

3. 코너 정밀화
   - `cv2.cornerSubPix()`로 좌/우 코너 좌표를 sub-pixel 단위로 보정합니다.

4. Stereo Calibration
   - `cv2.stereoCalibrate()`로 좌/우 카메라 행렬, 왜곡 계수, 회전 행렬, 이동 벡터, Essential/Fundamental matrix를 계산합니다.

5. 결과 저장
   - `stereo.npy`에 `K1`, `D1`, `K2`, `D2`, `R`, `T`, `E`, `F`, 이미지 크기, 검출 코너를 저장합니다.

## `example4.py` 주요 내용

1. 캘리브레이션 결과 로드
   - `camera_mat.npy`, `dist_coefs.npy`를 읽습니다.
   - 기본 위치는 `PJT_21-109` 폴더이며, 없으면 `pinhole_calib` 폴더도 확인합니다.

2. 체스보드 코너 검출
   - `pinhole_calib/img_00.png`에서 `(10, 7)` 패턴의 체스보드 코너를 검출합니다.
   - `cv2.cornerSubPix()`로 코너 위치를 정밀화합니다.

3. Undistort와 Project
   - `cv2.undistortPoints()`로 왜곡이 제거된 정규화 좌표를 계산합니다.
   - `cv2.projectPoints()`로 다시 이미지 좌표에 투영합니다.
   - 원본 검출 코너와 투영된 코너를 색상으로 구분해 표시합니다.

4. Reprojection
   - 왜곡 계수를 다시 적용한 재투영 결과를 별도 창에 표시합니다.

## 실행 방법

```bash
python example1.py
python example2.py
python example3.py
python example4.py
```

실행 전 폴더 구조는 다음과 같이 준비합니다.

```text
PJT_21-109/
├── example1.py
├── example2.py
└── pinhole_calib/
    ├── img01.jpg
    ├── img02.jpg
    └── ...
└── fisheyes/
    ├── Fisheye1_01.jpg
    ├── Fisheye1_02.jpg
    └── ...
└── stereo/
    └── case1/
        ├── left01.png
        ├── right01.png
        ├── left02.png
        ├── right02.png
        └── ...
```

필요 라이브러리는 `opencv-python`, `numpy`입니다. `cv2.imshow()`를 사용하므로 GUI가 가능한 환경에서 실행해야 합니다.

## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |
