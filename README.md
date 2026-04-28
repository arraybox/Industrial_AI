# Chapter 5 - Contour, Connected Component, Distance Transform

## 파일 구성

| 파일 | 설명 |
|------|------|
| `example.py` | External/Internal Contour 기본 예제 |
| `chapter5.py` | Otsu, Contour, Connected Component, Distance Transform 종합 과제 |
| `BnW.png` | 입력 이미지 |

## chapter5.py 주요 내용

1. **Otsu Thresholding** - 자동 임계값을 이용한 이진화
2. **External/Internal Contour** - `RETR_CCOMP` 계층 구조 기반 외부/내부 윤곽선 분류
3. **Connected Component** - 스페이스 키를 누를 때마다 랜덤 5개 컴포넌트를 랜덤 색상으로 표시 (ESC로 종료)
4. **Distance Transform** - L2 거리 변환 계산 및 시각화

## 실행 방법

```bash
python chapter5.py
```
