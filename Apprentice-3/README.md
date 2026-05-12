# Apprentice-3

선형회귀 결과를 평가하는 R2 결정계수 실습 폴더입니다. 간단한 1차원 입력 데이터에 선형회귀 모델을 학습하고, scikit-learn의 `r2_score`와 직접 구현한 R2 계산 결과를 비교합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `lec07_1_r2_for_student.ipynb` | 선형회귀 학습, 예측 시각화, R2 직접 계산 실습 노트북 |

## 주요 학습 내용

1. 데이터 생성과 시각화: 10개의 입력 `x`와 출력 `y` 데이터를 산점도로 확인합니다.
2. 선형회귀 모델 학습: `sklearn.linear_model.LinearRegression`으로 예측선을 학습합니다.
3. R2 직접 구현: `my_r2_score(y_true, y_pred)`에서 전체 제곱합(SST)과 잔차 제곱합(SSR)을 계산합니다.
4. 라이브러리 결과 비교: `sklearn.metrics.r2_score`와 직접 구현 결과를 비교합니다.

## 실행 방법

```bash
jupyter notebook lec07_1_r2_for_student.ipynb
```

필요 라이브러리는 `numpy`, `matplotlib`, `scikit-learn`입니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


