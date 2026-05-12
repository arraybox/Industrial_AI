# CBLand

충청북도 시군구별 토지 이용, 인구, 도로/공장 등 인프라 데이터를 이용해 지역 변화와 상관관계를 분석하는 프로젝트 폴더입니다. 2017년부터 2025년까지의 지적통계 원자료를 정리하고, PCA, K-means 군집화, 변화율 분석, 상관분석, 지도 시각화, 발표자료 생성을 수행합니다.

## 분석 목적

이 프로젝트는 충청북도 14개 시군구의 토지 이용 변화를 정량화하고, 인구 변화와 물리적 인프라 사이의 관계를 설명하는 것을 목표로 합니다. 주요 관심사는 공장용지, 대지, 도로, 임야, 농경지 비율이 지역 성장 또는 감소와 어떤 관계를 갖는지 확인하는 것입니다.

## 주요 데이터 폴더

| 폴더 | 설명 |
|---|---|
| `original_data` | 연도별 충청북도 지적통계 원자료 CSV/XLSX |
| `data` | 정리된 연도별 데이터, 변화율, PCA, 군집, 상관/회귀 결과 CSV |
| `result` | 초기 분석 결과, HTML 지도, heatmap, PCA 이미지, 보고서 |
| `result_v2` | 시계열 분석, 군집 통계, folium 지도, 시각화 이미지 |
| `result_v3` | 최종 분석 산출물, PCA/군집/상관/지도/발표자료 |

## 주요 스크립트

| 파일 | 설명 |
|---|---|
| `v2_data_prep.py` | 연도별 원자료를 읽고 지역명 정리, 인구 데이터 병합, 시계열 데이터 생성 |
| `v2_analysis_viz.py` | 변화율 분석, 지역 분류, 추세 그래프와 scatter plot 생성 |
| `v3_deep_analysis.py` | 지역별 상세 지표와 고급 시각화 생성 |
| `v4_final_classification.py` | 지역 군집 통계 산출과 군집 시각화 |
| `v5_comprehensive_viz.py` | PCA, heatmap, donut chart, folium 지도 등 종합 시각화 생성 |
| `chungbuk_analysis.py` | 도넛 차트, PCA plot, 변화율 heatmap 등 기본 분석 함수 모음 |
| `chungbuk_final_analysis.py` | 최종 프로젝트용 통합 분석 코드. PCA, K-means, 연도별 추세, 청주 4개구, 인구/도로 상관분석 포함 |
| `final_analysis_execution.py` | 인구/토지 데이터를 병합하고 상관분석 및 folium 지도 생성 |
| `new_analysis.py` | 정제/병합, 분석, HTML 지도 생성을 하나로 묶은 분석 스크립트 |
| `pca_export.py` | 토지 이용 비율 컬럼으로 PCA 좌표를 계산해 CSV로 저장 |
| `update_pptx.py` | 기존 PPT에 분석 이미지와 텍스트 슬라이드를 추가 |
| `inspect_headers.py`, `inspect_headers_utf8.py`, `inspect_raw_names.py` | 원자료 컬럼명과 지역명 인코딩/정합성 확인 |
| `check_data_integrity.py` | 최종 분석 CSV의 지역 수와 데이터 무결성 확인 |

## 핵심 분석 흐름

1. 데이터 준비
   - 연도별 지적통계 파일을 읽습니다.
   - 행정구역명을 표준화합니다.
   - 인구/세대수 데이터와 토지 이용 데이터를 병합합니다.

2. 지표 계산
   - 임야, 농경지, 대지, 공장용지, 도로 등 주요 토지 이용 면적과 비율을 계산합니다.
   - 2017~2025 변화율과 연도별 추세를 산출합니다.

3. PCA와 군집화
   - 토지 이용 비율을 표준화한 뒤 PCA로 2차원 축을 계산합니다.
   - K-means로 지역 유형을 분류하고 군집별 특성을 비교합니다.

4. 상관분석
   - 인구 변화율과 도로율, 공장용지 비율, 임야 비율 등 사이의 상관관계를 계산합니다.
   - 상관행렬과 유의성 결과를 CSV/시각화 파일로 저장합니다.

5. 시각화와 보고
   - heatmap, trend plot, scatter matrix, donut chart, folium HTML 지도, PPT 산출물을 생성합니다.

## 주요 결과 파일

| 파일/폴더 | 설명 |
|---|---|
| `result_v3/analysis_summary.csv` | 최종 분석 요약 |
| `result_v3/classification_basis.txt` | 지역 유형 분류 기준 설명 |
| `result_v3/13_comprehensive_map.html` | 종합 folium 지도 |
| `result_v3/02_PCA_classification_basis.png` | PCA 기반 분류 근거 시각화 |
| `result_v3/10_correlation_matrix.csv` | 인구/토지 이용 상관행렬 |
| `result_v3/산업 빅데이터 분석 실제 프로젝트결과서-충청북도 지적통계 토지 이용 현황 분석(최종)_v2.pptx` | 최종 발표자료 |
| `result_v2/Chungbuk_Comprehensive_Report.md` | v2 종합 분석 보고서 |

## 실행 순서 예시

데이터 준비부터 분석 산출물 생성을 새로 수행하려면 아래 흐름으로 실행합니다.

```bash
python v2_data_prep.py
python v2_analysis_viz.py
python v3_deep_analysis.py
python v4_final_classification.py
python v5_comprehensive_viz.py
python chungbuk_final_analysis.py
```

일부 스크립트에는 `D:\GITHUB\CBLand` 기준의 절대 경로가 들어 있습니다. 현재 저장소 위치에서 재실행하려면 스크립트 상단의 `DATA_DIR`, `RESULT_DIR`, `POP_FILE`, `LAND_FILE`, `OUTPUT_DIR` 경로를 현재 폴더 기준으로 수정해야 합니다.

## 필요 라이브러리

```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium python-pptx openpyxl
```

Windows 환경에서는 한글 그래프 표시를 위해 Malgun Gothic 폰트를 사용하도록 설정되어 있습니다.
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


