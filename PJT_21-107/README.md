# PJT_21-107

특징 검출(Feature Detection)을 학습하는 OpenCV 실습 폴더입니다. Harris Corner, FAST, Good Features To Track, SIFT를 사용해 이미지의 코너와 특징점을 검출하고, 원본 이미지와 회전 이미지에서 특징점 수와 위치를 비교합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `example.py` | Harris Corner와 FAST 코너 검출 기본 예제 |
| `example2.py` | FAST keypoint를 `drawKeypoints`, `drawMatches`로 시각화하는 예제 |
| `example3.py` | SIFT 특징점을 Lena 원본/회전 이미지에서 검출해 비교 |
| `chapter7.py` | FAST, Harris, GFTT, SIFT 4가지 특징 검출기를 한 번에 비교 |
| `scenetext01.jpg` | `example.py`, `example2.py` 입력 이미지 |
| `Lena.png` | `chapter7.py`, `example3.py` 입력 이미지 |
| `Lena_rotated.png` | 회전 비교용 입력 이미지 |
| `Lena_rotate.png` | 회전 이미지 관련 파일 |

## `chapter7.py` 주요 내용

1. FAST
   - `cv2.FastFeatureDetector_create()`로 빠른 코너 검출을 수행합니다.
   - threshold와 nonmax suppression 설정에 따라 검출량이 달라집니다.

2. Harris Corner
   - `cv2.cornerHarris()` 결과에서 threshold 이상인 지점을 코너로 표시합니다.

3. Good Features To Track
   - `cv2.goodFeaturesToTrack()`으로 추적에 적합한 코너 후보를 찾습니다.

4. SIFT
   - `cv2.SIFT_create()`로 scale/rotation 변화에 비교적 강한 특징점을 검출합니다.

5. 원본/회전 이미지 비교
   - `detect_all_features()`에서 네 종류의 특징점을 검출합니다.
   - `draw_all_features()`로 결과를 한 이미지에 표시합니다.
   - `count_features()`로 검출된 특징점 개수를 출력합니다.

## 실행 방법

```bash
python example.py
python example2.py
python example3.py
python chapter7.py
```

필요 라이브러리는 `opencv-python`, `numpy`입니다. SIFT는 OpenCV 버전에 따라 `opencv-contrib-python`이 필요할 수 있습니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


