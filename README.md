# Chapter 7 - 특징 검출 (Feature Detection)

## 파일 구성

| 파일 | 설명 |
|------|------|
| `example.py` | Harris Corner, FAST 코너 검출 예제 |
| `example2.py` | FAST 키포인트 시각화 및 drawMatches 예제 |
| `chapter7.py` | FAST, Harris, GFTT, SIFT 4가지 특징 검출 비교 |

## 이미지 파일

| 파일 | 설명 |
|------|------|
| `scenetext01.jpg` | example, example2에서 사용하는 입력 영상 |
| `Lena.png` | chapter7에서 사용하는 영상(1) |
| `Lena_rotate.png` | Lena.png를 270도 회전한 영상(2) |

## 실행 방법

```bash
cd src/chapter7
python example.py
python example2.py
python chapter7.py
```

## 각 파일 상세 설명

### example.py

- **Harris Corner Detector**: 코너를 검출하고 빨간색으로 표시, 정규화된 코너 맵과 좌우로 비교 출력
- **FAST Corner Detector**: threshold 30으로 코너를 검출하여 초록색 원으로 표시
- **FAST (NonmaxSuppression OFF)**: 비최대 억제를 끄고 더 많은 코너점을 검출

### example2.py

- **drawKeypoints**: FAST로 검출한 키포인트를 보라색으로 표시
- **drawKeypoints (RICH)**: 키포인트의 크기와 방향을 포함하여 초록색으로 표시
- **drawMatches**: 자기 자신과의 매칭 결과를 Rich 모드로 시각화

### chapter7.py

이미지(1) `Lena.png`와 이미지(2) `Lena_rotate.png`에서 4가지 특징을 검출하고 비교:

| 특징 검출기 | 표시 모양 | 색상 |
|---|---|---|
| FAST | 원 (circle) | 초록색 |
| Harris Corner | 사각형 (rectangle) | 빨간색 |
| GoodFeaturesToTrack | 삼각형 (triangle) | 노란색 |
| SIFT | 다이아몬드 (diamond) | 분홍색 |

두 영상을 좌우로 합쳐 한 창에 출력하며, 각 이미지 좌측 하단에 반투명 범례를 표시합니다.

## 필요 라이브러리

```bash
pip install opencv-python numpy
```
