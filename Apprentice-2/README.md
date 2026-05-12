# Apprentice-2

PCA(Principal Component Analysis, 주성분 분석)를 학습하는 실습 폴더입니다. scikit-learn의 PCA 결과를 확인한 뒤, 데이터 중심화, 공분산 행렬, 고유값/고유벡터 계산을 통해 PCA의 내부 동작을 직접 따라갑니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `lec04_pca_v2_for_student.ipynb` | PCA를 라이브러리 방식과 직접 구현 방식으로 비교하는 노트북 |

## 주요 학습 내용

1. 데이터 준비
   - 키와 몸무게 형태의 2차원 데이터를 사용합니다.
   - 랜덤 클러스터와 상관관계가 있는 2차원 분포도 생성합니다.

2. scikit-learn PCA
   - `sklearn.decomposition.PCA`로 주성분 축을 계산합니다.
   - `components_`, `explained_variance_ratio_`를 출력해 분산 설명력을 확인합니다.

3. PCA 직접 구현
   - 데이터 중심화
   - 공분산 행렬 계산
   - 고유값과 고유벡터 계산
   - 주성분 선택
   - 저차원 표현과 원본 복원

## 실행 방법

```bash
jupyter notebook lec04_pca_v2_for_student.ipynb
```

필요 라이브러리는 `numpy`, `matplotlib`, `scikit-learn`입니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


