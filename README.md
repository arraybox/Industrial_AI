# Industrial_AI

충북대학교 산업인공지능학과 학습 과정에서 작성한 실습 코드와 프로젝트 산출물을 정리한 저장소입니다. 머신러닝 기초, 딥러닝, Flask 웹 실습, OpenCV 영상처리, 데이터 분석 프로젝트를 폴더별로 분리해 관리합니다.

## 작성자

| 항목 | 내용 |
|---|---|
| 이름 | 이일주 |
| 학번 | 2025254015 |
| 소속 | 충북대학교 산업인공지능학과 |
| Git 작성자 | arraybox |
| 이메일 | arraybox@chungbuk.ac.kr |

## 저장소 목적

이 저장소는 산업 현장에서 활용할 수 있는 인공지능과 컴퓨터비전 기술을 학습하고 실습한 기록입니다. Python 기반 데이터 처리, 수치 연산, 머신러닝 모델 평가, 딥러닝 이미지 분류, 웹 애플리케이션 기초, OpenCV 영상처리, 지역 데이터 분석 프로젝트를 포함합니다.

주요 학습 분야는 다음과 같습니다.

- NumPy와 Python 기반 데이터 처리
- 정규화, PCA, 회귀 평가, 정규화 회귀 등 머신러닝 기초
- PyTorch와 TensorFlow/Keras 기반 딥러닝 실습
- Flask 기반 웹 라우팅과 로그인 처리
- OpenCV 기반 영상처리, 세그멘테이션, 특징 검출, 영상 변환
- 충청북도 토지 이용 및 인구/인프라 데이터 분석

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `numpy_student` | NumPy 배열 생성, reshape, 연산, stack, copy/view 학습 노트북 |
| `Apprentice-1` | 정규화와 표준화 비교 실습 |
| `Apprentice-2` | PCA 원리와 scikit-learn/직접 구현 비교 |
| `Apprentice-3` | 선형회귀 모델의 R2 결정계수 계산 실습 |
| `Apprentice-4` | 선형회귀, Ridge, Lasso 정규화 비교 |
| `PJT_11-01` | PyTorch 선형회귀, AlexNet 이미지 분류, Raspberry Pi 카메라 추론 |
| `PJT_11-02` | Flask 로그인 폼 기초 |
| `PJT_11-03` | Flask 라우팅, 동적 URL, 404 처리 |
| `PJT_11-04` | 공기질/전력 시계열 데이터 전처리와 예측 모델 |
| `PJT_21-01` | OpenCV 이미지 표시, 마우스 드로잉, 키보드 이벤트 |
| `PJT_21-02` | 컬러 변환, 히스토그램, 감마 보정, 공간 필터 |
| `PJT_21-03` | 영상 필터링, 모폴로지, 주파수 영역 필터링 |
| `PJT_21-103` | OpenCV 3주차 통합 예제 |
| `PJT_21-105` | Contour, Connected Component, Distance Transform |
| `PJT_21-106` | K-means, Watershed, GrabCut 기반 세그멘테이션 |
| `PJT_21-107` | Harris, FAST, GFTT, SIFT 특징 검출 |
| `PJT_21-108` | Affine/Perspective Transform 이미지 워핑, Lucas-Kanade 특징점 추적, 파노라마 생성 |
| `PJT_21-109` | 체스보드 패턴 기반 pinhole/fisheye/stereo camera calibration |
| `PJT_21-110` | 3D triangulation과 stereo rectification |
| `image_labeling_tool` | 웹 기반 이미지 알파벳 라벨링 도구 |
| `CBLand` | 충청북도 토지 이용, 인구, 인프라 상관분석 프로젝트 |

## 최근 추가/수정 내용

| 폴더 | 파일 | 내용 |
|---|---|---|
| `PJT_21-108` | `example1.py`, `circlesgrid.png` | 선택점 기반 affine/perspective image warping 실습 |
| `PJT_21-108` | `example2.py`, `traffic.mp4` | Lucas-Kanade Optical Flow 기반 프레임 간 keypoint tracking 실습 |
| `PJT_21-108` | `example3.py`, `0.jpg`, `1.jpg` | OpenCV Stitcher와 특징점 매칭 fallback 기반 panorama stitching 실습 |
| `PJT_21-109` | `example1.py`, `example2.py`, `example3.py`, `example4.py` | 체스보드 코너 검출, pinhole/fisheye/stereo 캘리브레이션, 왜곡/보정 좌표 변환 실습 |
| `PJT_21-110` | `example1.py`, `example2.py`, `example3.py`, `example4.py` | 3D 점 복원, stereo rectification, Fundamental/Essential matrix 계산과 Essential matrix 분해 실습 |

## 실행 환경

폴더마다 필요한 라이브러리가 다릅니다. 공통적으로 Python, Jupyter Notebook, NumPy, Matplotlib이 자주 사용되며, 프로젝트 성격에 따라 OpenCV, Flask, PyTorch, TensorFlow, scikit-learn, pandas, seaborn, folium 등이 필요합니다.

OpenCV 예제는 `cv2.imshow()` 기반이라 데스크톱 GUI 환경에서 실행하는 것이 좋습니다. 딥러닝 노트북은 데이터셋 다운로드나 사전학습 가중치 다운로드가 포함될 수 있어 인터넷 연결이 필요할 수 있습니다.
