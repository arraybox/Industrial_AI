# Apprentice-1

정규화(Normalization)와 표준화(Standardization)의 차이를 비교하는 실습 폴더입니다. 키와 몸무게 배열 데이터를 사용해 서로 다른 스케일의 데이터를 같은 기준으로 변환하고, 산점도로 변환 전후를 확인합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `lec03_3_normalization_for_student.ipynb` | Min-Max Scaling과 Z-score Standardization을 직접 구현하고 시각화하는 노트북 |

## 주요 학습 내용

1. 원본 데이터 확인
   - `weight`, `height` 배열을 생성합니다.
   - 서로 단위와 범위가 다른 두 데이터를 산점도로 비교합니다.

2. 정규화
   - `min_max_scaling(data)` 함수를 직접 구현합니다.
   - 공식은 `(data - min) / (max - min)`입니다.
   - 모든 값이 같은 배열일 때 0으로 나누는 상황을 피하도록 예외 처리가 포함되어 있습니다.

3. 표준화
   - `standardization(data)` 함수를 직접 구현합니다.
   - 공식은 `(data - mean) / std`입니다.
   - 표준편차가 0인 경우를 처리합니다.

4. 결과 비교
   - 원본, 정규화 결과, 표준화 결과를 각각 시각화합니다.
   - 정규화는 값을 0~1 범위로 맞추고, 표준화는 평균 0과 표준편차 1 기준으로 재배치한다는 차이를 확인합니다.

## 실행 방법

```bash
jupyter notebook lec03_3_normalization_for_student.ipynb
```

필요 라이브러리는 `numpy`, `matplotlib`입니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


