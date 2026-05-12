# PJT_11-04

시계열 데이터를 전처리하고 딥러닝 모델로 예측하는 실습 폴더입니다. 공기질 UCI 데이터 전처리 및 센서 변수 시각화, LSTM/GRU/1D CNN 비교, 전력 수요 예측 모델 예제가 포함되어 있습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `AirQualityUCI_Preprocessing_Visualization_Updated.ipynb` | AirQualityUCI 데이터 전처리, 상관관계 분석, 시계열 예측 모델 비교 |
| `power_forecasting_models.ipynb` | 전력 수요 예측을 위한 LSTM, GRU, Transformer 통합 예제 |

## AirQualityUCI 노트북 내용

1. 데이터 로딩
   - `AirQualityUCI.csv`를 `;` 구분자와 `,` 소수점 형식으로 읽습니다.
   - 빈 열과 결측 행을 제거합니다.

2. 날짜/시간 전처리
   - `Date`, `Time` 컬럼을 합쳐 `Datetime` 인덱스를 만듭니다.
   - 숫자형 변환 후 결측치 표기값 `-200`을 제거합니다.

3. 시각화
   - `C6H6(GT)`, `CO(GT)`, `NOx(GT)` 등 주요 오염물질 흐름을 그립니다.
   - 주요 센서 변수와 실제 오염물질 변수 간 상관관계 heatmap을 생성합니다.

4. 예측 모델
   - LSTM, GRU, 1D CNN 모델을 TensorFlow/Keras로 정의합니다.
   - 입력 변수 조합을 high-correlation, low-correlation으로 나눠 모델 성능을 비교합니다.
   - MSE 기준으로 결과를 막대그래프로 표시합니다.

## power_forecasting 노트북 내용

- 전력 수요 데이터를 로딩하고 정규화합니다.
- sliding window 방식으로 시계열 입력 시퀀스를 만듭니다.
- LSTM, GRU, Transformer 기반 예측 모델을 구성합니다.
- 예측 결과를 MSE로 평가하고 시각화합니다.

## 실행 방법

```bash
jupyter notebook
```

필요 라이브러리는 `pandas`, `matplotlib`, `seaborn`, `numpy`, `tensorflow`, `scikit-learn`입니다. 데이터 CSV 파일은 노트북에서 지정한 이름과 같은 위치에 있어야 합니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


