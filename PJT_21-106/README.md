# Chapter 6 - Image Segmentation

## 파일 구성

| 파일 | 설명 |
|------|------|
| `example.py` | K-means 클러스터링 기반 이미지 세그멘테이션 (Lab 색공간) |
| `example1.py` | Watershed 알고리즘 기반 인터랙티브 세그멘테이션 |
| `example2.py` | GrabCut 단계별 인터랙티브 세그멘테이션 |
| `chapter6.py` | 과제 코드 (K-means 비교 + GrabCut) |
| `Lena.png` | 테스트 이미지 |

## chapter6.py 상세

### (1) K-means Clustering 비교

- **(R, G, B)** : 색상 정보만으로 클러스터링
- **(R, G, B, X, Y)** : 색상 + 픽셀 좌표를 결합하여 클러스터링
- 두 결과를 원본과 함께 비교 출력
- (R, G, B, X, Y)는 공간적으로 인접한 픽셀이 같은 클러스터에 속할 확률이 높아져 더 연속적인 세그멘테이션 결과를 보여줌

### (2) GrabCut Interactive

사용자 입력을 통해 반복적으로 세그멘테이션을 수정할 수 있는 인터랙티브 프로그램

**사용법:**
1. 마우스 드래그로 전경 영역을 포함하는 사각형을 그림
2. 초기 GrabCut 결과 확인
3. 키 입력으로 마스크 수정:
   - `0` : 배경(BGD) 브러시
   - `1` : 전경(FGD) 브러시
   - `n` : 수정된 마스크로 GrabCut 재실행
   - `r` : 리셋
   - `ESC` : 종료

## example1.py - Watershed

마우스로 시드를 칠한 뒤 Watershed 알고리즘으로 세그멘테이션 수행

**사용법:**
- 마우스 드래그 : 시드 영역 칠하기
- 숫자 `1`~`9` : 시드 번호(클래스) 전환
- `c` : 리셋
- `ESC` : 종료

## example2.py - GrabCut 단계별 실행

RECT 모드로 초기 세그멘테이션 후, MASK 모드로 사용자가 직접 보정하는 2단계 GrabCut

**실행 흐름:**
1. 마우스 드래그로 전경 영역 사각형 선택 → `a` 키로 확정
2. 초기 GrabCut 결과 표시 (배경 영역 어둡게 처리)
3. 아무 키 → 마스크 수정 단계 진입
   - 마우스로 브러시 칠하기
   - `l` : 배경/전경 브러시 전환
   - `a` : 확정 → GrabCut 재실행 (MASK 모드)
4. 최종 결과 표시

## 실행 환경

```bash
pip install opencv-python numpy matplotlib
```
