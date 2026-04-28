# 지능화캡스톤프로젝트 - 3주차: OpenCV 기반 영상처리

충북대학교 산업인공지능연구센터 | OpenCV 기반 영상처리 심화 및 실습

---

## 📁 파일 구성

```
├── day3_all_examples.py          # 3주차 전체 예제 통합 파일
├── 지능화캡스톤프로젝트(3주차).pdf  # 강의 자료
└── README.md
```

---

## ⚙️ 환경 설정

### 라이브러리 설치

```bash
pip install opencv-python
pip install matplotlib
pip install numpy
```

### 필요 이미지/영상 파일 다운로드

예제 실행 전 아래 파일들을 `day3_all_examples.py`와 **같은 폴더**에 저장하세요.

| 파일명 | 다운로드 링크 |
|--------|--------------|
| `Lenna.png` | https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png |
| `Candies.png` | https://www.charlezz.com/wordpress/wp-content/uploads/2021/04/www.charlezz.com-opencv-candies.png |
| `Hawkes.jpg` | https://blog.kakaocdn.net/dn/ZVlAe/btrr1mWG2SK/VsCRrXfpvOZsuD1EKYyHu1/Hawkes.jpg |
| `desert.jpg` | https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Rub_al_Khali_002.JPG/640px-Rub_al_Khali_002.JPG |
| `test_video.mp4` | https://www.kaggle.com/datasets/dpamgautam/video-file-for-lane-detection-project |

---

## ▶️ 실행 방법

```bash
python day3_all_examples.py
```

실행하면 예제 목록이 출력되고, 번호를 입력하면 해당 예제가 실행됩니다.

```
==================================================
  3주차 OpenCV 예제 목록
==================================================
   1. 이미지 파일 읽고 출력
   2. 동영상 파일 읽고 출력
   3. 카메라 영상 읽고 출력
   4. BGR 채널 분리
   5. HSV 색공간 변환 후 채널 분리
   6. RGB 기반 빨간색 영역 검출
   7. HSV 기반 빨간색 캔디 추출
   8. 그레이스케일 히스토그램
   9. 명암비 조작
  10. 히스토그램 평활화 (equalizeHist)
  11. 히스토그램 스트래칭 (normalize)
  12. 히스토그램 역투영
  13. 이미지 필터링 (평균/샤프닝/라플라시안)
==================================================
실행할 예제 번호를 입력하세요 (종료: q):
```

---

## 📋 예제 목록

### 영상 입출력

| # | 함수명 | 설명 | 필요 파일 |
|---|--------|------|-----------|
| 1 | `ex1_read_image` | 이미지 파일 읽고 출력 | Lenna.png |
| 2 | `ex2_read_video` | 동영상 파일 읽고 출력 | test_video.mp4 |
| 3 | `ex3_read_camera` | 카메라 영상 읽고 출력 | 웹캠 |

### 색공간 처리

| # | 함수명 | 설명 | 필요 파일 |
|---|--------|------|-----------|
| 4 | `ex4_split_bgr` | BGR 채널 분리 | Lenna.png |
| 5 | `ex5_cvtcolor_hsv` | HSV 색공간 변환 후 채널 분리 | Lenna.png |
| 6 | `ex6_extract_color_rgb` | RGB 기반 빨간색 영역 검출 | Candies.png |
| 7 | `ex7_extract_color_hsv` | HSV 기반 빨간색 캔디 추출 | Candies.png |

### 히스토그램

| # | 함수명 | 설명 | 필요 파일 |
|---|--------|------|-----------|
| 8 | `ex8_histogram_grayscale` | 그레이스케일 히스토그램 시각화 | Lenna.png |
| 9 | `ex9_adjust_contrast` | 명암비 조작 | Lenna.png |
| 10 | `ex10_equalize_hist` | 히스토그램 평활화 (equalizeHist) | Hawkes.jpg |
| 11 | `ex11_normalize_hist` | 히스토그램 스트래칭 (normalize) | Hawkes.jpg |
| 12 | `ex12_backprojection` | 히스토그램 역투영 | desert.jpg |

### 이미지 필터링

| # | 함수명 | 설명 | 필요 파일 |
|---|--------|------|-----------|
| 13 | `ex13_image_filtering` | 평균값/샤프닝/라플라시안 필터 적용 | Lenna.png |

---

## 📚 주요 OpenCV API

| 함수 | 설명 |
|------|------|
| `cv2.imread(path, flag)` | 이미지 파일 읽기 |
| `cv2.imshow(title, image)` | 이미지 윈도우 출력 |
| `cv2.VideoCapture(src)` | 동영상/카메라 스트림 열기 |
| `cv2.cvtColor(image, code)` | 색공간 변환 (BGR↔HSV 등) |
| `cv2.split(image)` | 다채널 이미지 채널 분리 |
| `cv2.inRange(src, lower, upper)` | 특정 값 범위 마스크 생성 |
| `cv2.bitwise_and/or(src1, src2)` | 비트 연산 |
| `cv2.calcHist(...)` | 히스토그램 계산 |
| `cv2.equalizeHist(image)` | 히스토그램 평활화 |
| `cv2.normalize(...)` | 히스토그램 정규화 |
| `cv2.calcBackProject(...)` | 히스토그램 역투영 |
| `cv2.filter2D(src, ddepth, kernel)` | 사용자 정의 2D 필터 적용 |

---

## 🔗 참고 자료

- [OpenCV 공식 홈페이지](https://opencv.org/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [OpenCV Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [LearnOpenCV](https://learnopencv.com/getting-started-with-opencv/)
- [OpenCV 한글 강좌](https://076923.github.io/posts/Python-opencv-1/)
- [OpenCV Korea](https://cafe.naver.com/opencv)
