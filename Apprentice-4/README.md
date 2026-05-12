# Apprentice-4

회귀 모델의 정규화(Regularization)를 비교하는 실습 폴더입니다. 노이즈가 포함된 합성 데이터를 만들고, 일반 선형회귀(MLE), Ridge(L2), Lasso(L1)를 시각적으로 비교합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `lec07_regularization_v2_for_student.ipynb` | 다항 특성과 정규화 회귀를 비교하는 노트북 |

## 주요 학습 내용

1. 합성 데이터 생성: 비선형 ground truth와 노이즈가 섞인 관측 데이터를 만듭니다.
2. MLE 기반 선형회귀: `LinearRegression`으로 정규화가 없는 회귀선을 학습합니다.
3. Ridge 회귀: `PolynomialFeatures`, `StandardScaler`, `Ridge`를 `Pipeline`으로 구성해 L2 패널티 효과를 확인합니다.
4. Lasso 회귀: `Lasso`와 다항 특성을 결합해 L1 패널티와 특성 선택 효과를 확인합니다.
5. 모델 비교: ground truth, MLE, Ridge, Lasso 예측선을 한 그래프에서 비교합니다.

## 실행 방법

```bash
jupyter notebook lec07_regularization_v2_for_student.ipynb
```

필요 라이브러리는 `numpy`, `matplotlib`, `scikit-learn`입니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


