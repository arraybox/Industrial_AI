# PJT_21-103

지능화캡스톤프로젝트 3주차 OpenCV 영상처리 통합 예제 폴더입니다. `day3_all_examples.py` 한 파일에 이미지/비디오/카메라 입력, 색공간 변환, 색상 추출, 히스토그램, 대비 조정, 필터링 예제가 함수 단위로 정리되어 있습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `day3_all_examples.py` | 3주차 OpenCV 예제를 하나로 모은 Python 스크립트 |
| `README.md` | 실행 방법과 예제 설명 |

## 예제 함수 구성

| 함수 | 설명 |
|---|---|
| `ex1_read_image()` | 이미지 파일 읽기와 화면 표시 |
| `ex2_read_video()` | 동영상 파일 읽기 |
| `ex3_read_camera()` | 카메라 입력 읽기 |
| `ex4_split_bgr()` | BGR 채널 분리 |
| `ex5_cvtcolor_hsv()` | BGR에서 HSV 색공간 변환 |
| `ex6_extract_color_rgb()` | RGB 기준 색상 영역 추출 |
| `ex7_extract_color_hsv()` | HSV 범위 기반 색상 영역 추출 |
| `ex8_histogram_grayscale()` | grayscale histogram 계산 |
| `ex9_adjust_contrast()` | 영상 대비 조정 |
| `ex10_equalize_hist()` | 히스토그램 평활화 |
| `ex11_normalize_hist()` | 히스토그램 정규화 |
| `ex12_backprojection()` | Histogram Backprojection |
| `ex13_image_filtering()` | 평균, Gaussian, Median 등 필터 적용 |

## 필요한 예제 파일

스크립트 주석에는 `Lenna.png`, `Candies.png`, `Hawkes.jpg`, `desert.jpg`, `test_video.mp4` 등의 외부 예제 파일이 필요하다고 안내되어 있습니다. 해당 파일들은 `day3_all_examples.py`와 같은 폴더에 두고 실행하는 구조입니다.

## 실행 방법

```bash
pip install opencv-python matplotlib numpy
python day3_all_examples.py
```

스크립트 하단에서 호출할 예제 함수를 선택해 실행하는 방식으로 사용하면 됩니다. 카메라/동영상 예제는 장치 연결 또는 영상 파일 경로가 필요합니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


