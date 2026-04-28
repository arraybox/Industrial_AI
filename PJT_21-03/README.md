# Chapter 4 - Image Filtering & Frequency-based Filtering

## 파일 구성

| 파일 | 설명 |
|------|------|
| `chapter4.py` | 영상 필터링 실습 (아래 과제 내용) |
| `example3.py` | Unsharp Mask 예제 (Gaussian 커널 직접 구성) |
| `Lena.png` | 실습용 Lena 이미지 |

## chapter4.py 과제 내용

### Part 1 - Image Filtering

1. **Unsharp Mask** - Gaussian Blur 후 원본과 가중 합산으로 선명화 적용
2. **Sobel Filter** - (1) 결과에 X/Y 방향 Sobel 필터 적용 후 magnitude 계산
3. **Gabor Filter** - (1) 결과에 Gabor 커널(θ=45°, σ=5, λ=10) 적용
4. **Threshold Trackbar** - Sobel과 Gabor 차이를 트랙바로 threshold 조절하며 출력
5. **Opening / Closing** - (4) 결과에 모폴로지 연산(Opening, Closing) 적용

### Part 2 - Frequency-based Filtering

1. **DFT** - 영상에 이산 푸리에 변환 적용 후 Magnitude Spectrum 출력
2. **원형 필터** - 중심으로부터 원 모양(반지름 30) Low-Pass 필터링
3. **사각형 필터** - 중심으로부터 사각형 모양(30×30) Low-Pass 필터링

## 실행 방법

```bash
python src/chapter4/chapter4.py
python src/chapter4/example3.py
```

## 사용 라이브러리

- OpenCV (`cv2`)
- NumPy (`numpy`)
- SciPy (`scipy`) - example3.py
- Matplotlib (`matplotlib`) - example3.py
