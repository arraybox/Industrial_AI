# Chapter 3 - 영상 처리 기초

## 파일 구성

| 파일 | 설명 |
|------|------|
| `chpter3.py` | 영상 처리 실습 (아래 과제 내용) |
| `example.py` | 컬러/흑백/HSV 변환 예제 |
| `Lena.png` | 실습용 Lena 이미지 |

## chpter3.py 과제 내용

1. **컬러 영상 읽기** - Lena 이미지를 컬러로 읽고 출력
2. **흑백 변환** - `cv2.cvtColor(BGR2GRAY)`로 변환 후 출력
3. **Histogram Equalization** - `cv2.equalizeHist()`로 히스토그램 평활화 적용
4. **Gamma Correction** - gamma=2.2로 감마 보정 적용
5. **HSV 변환** - HSV 컬러 스페이스로 변환, H/S/V 각 채널을 0~255로 정규화하여 출력
6. **Median Filter** - H 채널에 미디언 필터 적용 (ksize=5)
7. **Gaussian Filter** - S 채널에 가우시안 필터 적용 (5x5)
8. **Bilateral Filter** - V 채널에 바이래터럴 필터 적용 (d=9, sigmaColor=75, sigmaSpace=75)

## 실행 방법

```bash
python src/chapter3/chpter3.py
```

## 사용 라이브러리

- OpenCV (`cv2`)
- NumPy (`numpy`)
