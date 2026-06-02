# PJT_21-111

OpenCV HOG Descriptor와 기본 SVM 사람 검출기를 이용해 이미지에서 보행자를 검출하는 실습 폴더입니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `example1.py` | `cv2.HOGDescriptor()`와 기본 people detector를 사용해 사람 위치를 사각형으로 표시 |
| `example2.py` | Haar/LBP cascade classifier를 이용해 영상 또는 웹캠에서 얼굴 검출 |
| `example3.py` | ArUco marker 생성, blur 적용, marker 검출 및 시각화 |
| `people.jpg` | `example1.py`에서 사용하는 입력 이미지 |
| `faces.mp4` | `example2.py`의 Haar cascade 예제에서 사용하는 선택 입력 영상 |
| `haarcascade_frontalface_default.xml` | Haar cascade 입력 파일. 없으면 OpenCV 내장 haarcascade 경로를 fallback으로 사용 |
| `lbpcascade_frontalface.xml` | LBP cascade 입력 파일. `example2.py`에서 필요 |

## 주요 내용

1. 이미지 입력
   - `PJT_21-111/people.jpg`를 읽습니다.
   - 이미지가 없으면 필요한 경로를 안내하는 오류를 출력합니다.

2. HOG 사람 검출기 설정
   - `cv2.HOGDescriptor()`로 HOG descriptor를 생성합니다.
   - `cv2.HOGDescriptor_getDefaultPeopleDetector()`를 SVM detector로 설정합니다.

3. 사람 검출
   - `hog.detectMultiScale(image)`로 사람 후보 영역과 score를 계산합니다.
   - 검출된 영역을 초록색 사각형으로 표시합니다.

4. 결과 시각화
   - Matplotlib으로 원본 이미지와 검출 결과 이미지를 나란히 표시합니다.
   - OpenCV의 BGR 이미지를 Matplotlib 표시용 RGB 순서로 바꿔 출력합니다.

## `example2.py` 주요 내용

1. 얼굴 검출 함수
   - `detect_faces(video_file, detector, win_title)` 함수로 영상 또는 카메라 프레임을 읽습니다.
   - 프레임을 grayscale로 변환한 뒤 `detector.detectMultiScale(gray, 1.3, 5)`로 얼굴을 검출합니다.

2. Haar cascade 검출
   - `haarcascade_frontalface_default.xml`을 우선 현재 폴더에서 찾습니다.
   - 파일이 없으면 OpenCV가 제공하는 기본 haarcascade 경로를 사용합니다.
   - `faces.mp4` 또는 `Faces.mp4`가 있으면 해당 영상을 사용하고, 없으면 웹캠 `0`을 사용합니다.

3. LBP cascade 검출
   - `lbpcascade_frontalface.xml`을 현재 폴더에서 로드합니다.
   - Haar cascade와 같은 영상 소스를 사용합니다.

4. 결과 표시
   - 검출된 얼굴 영역에 초록색 사각형과 `Face` 라벨을 표시합니다.
   - `ESC` 키를 누르면 검출 창을 종료합니다.

## `example3.py` 주요 내용

1. ArUco dictionary 생성
   - `aruco.DICT_6X6_250` 사전을 사용합니다.
   - marker ID `2`, `76`, `42`, `123`을 700x700 흰 배경 이미지에 배치합니다.

2. Marker 생성
   - OpenCV 버전에 따라 `aruco.drawMarker()` 또는 `aruco.generateImageMarker()`를 사용합니다.
   - 생성한 marker에 Gaussian Blur를 적용해 검출 예제를 구성합니다.

3. Marker 검출
   - OpenCV 새 API에서는 `aruco.ArucoDetector`를 사용합니다.
   - 구버전 API에서는 `aruco.detectMarkers()`를 fallback으로 사용합니다.

4. 결과 표시
   - 생성된 marker 이미지를 먼저 표시합니다.
   - 검출된 marker의 테두리와 ID를 `aruco.drawDetectedMarkers()`로 표시합니다.

## 실행 방법

```bash
python example1.py
python example2.py
python example3.py
```

필요 라이브러리는 `opencv-python`, `matplotlib`, `numpy`입니다. ArUco 모듈이 없는 OpenCV 환경에서는 `opencv-contrib-python`이 필요할 수 있습니다.

## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |
