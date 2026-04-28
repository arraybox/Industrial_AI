# indCV - OpenCV 학습 프로젝트

OpenCV를 활용한 컴퓨터 비전 학습 프로젝트입니다.

## 환경 설정

```bash
conda activate indCV
pip install opencv-python
```

- Python 3.14
- OpenCV 4.13

## 프로젝트 구조

```
src/
└── chapter2/
    ├── chapter2.py    # 이미지 표시 및 마우스/키보드 입력 처리
    └── img/
        └── Lena.png
```

## Chapter 2 - 이미지 기본 조작

마우스와 키보드를 이용하여 이미지 위에 도형을 그리는 프로그램입니다.

### 실행

```bash
cd src/chapter2
python chapter2.py
```

### 키보드 조작

| 키 | 동작 |
|---|---|
| `r` | 사각형(Rectangle) 모드 |
| `l` | 직선(Line) 모드 |
| `a` | 화살표 직선(Arrowed Line) 모드 |
| `w` | 현재 이미지를 `lena_draw.png`로 저장 |
| `c` | 모든 도형 지우고 원본 복원 |
| `ESC` | 프로그램 종료 |

### 마우스 조작

- 왼쪽 버튼 드래그로 선택한 모드의 도형을 그립니다.
