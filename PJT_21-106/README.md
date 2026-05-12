# PJT_21-106

이미지 세그멘테이션을 학습하는 OpenCV 실습 폴더입니다. K-means 색상 클러스터링, 좌표 정보를 포함한 클러스터링, Watershed, GrabCut을 비교합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `example.py` | Lab 색공간에서 K-means 클러스터링 기반 이미지 분할 |
| `example1.py` | 마우스 시드 입력 기반 Watershed 인터랙티브 세그멘테이션 |
| `example2.py` | 사각형 지정과 마스크 보정을 사용하는 GrabCut 예제 |
| `chapter6.py` | RGB K-means, RGB+XY K-means, GrabCut을 포함한 종합 과제 |
| `Lena.png` | 테스트 이미지 |

## `chapter6.py` 주요 내용

1. K-means with RGB
   - 픽셀의 RGB 값만 사용해 8개 클래스로 클러스터링합니다.
   - 색이 비슷한 영역끼리 묶이지만 공간적 연속성은 강하게 반영되지 않습니다.

2. K-means with RGB+XY
   - RGB 값에 픽셀 좌표 X, Y를 추가합니다.
   - 색상과 위치를 함께 고려해 더 연속적인 분할 결과를 만듭니다.

3. GrabCut Interactive
   - 사용자가 마우스로 전경을 포함하는 사각형을 지정합니다.
   - 초기 GrabCut 결과를 확인한 뒤 브러시로 전경/배경 마스크를 보정할 수 있습니다.

## 인터랙티브 조작

GrabCut 예제에서 사용하는 기본 흐름은 다음과 같습니다.

| 키/동작 | 설명 |
|---|---|
| 마우스 드래그 | 초기 전경 포함 사각형 지정 |
| `0` | 배경 브러시 |
| `1` | 전경 브러시 |
| `n` | 수정된 마스크로 GrabCut 재실행 |
| `r` | 리셋 |
| `ESC` | 종료 |

Watershed 예제는 마우스로 시드를 칠하고 키 입력으로 현재 seed label을 바꿔 분할 결과를 갱신하는 방식입니다.

## 실행 방법

```bash
python example.py
python example1.py
python example2.py
python chapter6.py
```

필요 라이브러리는 `opencv-python`, `numpy`, `matplotlib`입니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


