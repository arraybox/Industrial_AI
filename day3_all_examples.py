"""
지능화캡스톤프로젝트 3주차 - OpenCV 기반 영상처리 예제 모음
모든 예제를 하나의 파일로 통합

필요 파일:
  - Lenna.png       : https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png
  - Candies.png     : https://www.charlezz.com/wordpress/wp-content/uploads/2021/04/www.charlezz.com-opencv-candies.png
  - Hawkes.jpg      : https://blog.kakaocdn.net/dn/ZVlAe/btrr1mWG2SK/VsCRrXfpvOZsuD1EKYyHu1/Hawkes.jpg
  - desert.jpg      : https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Rub_al_Khali_002.JPG/640px-Rub_al_Khali_002.JPG
  - test_video.mp4  : https://www.kaggle.com/datasets/dpamgautam/video-file-for-lane-detection-project

사용법:
  python day3_all_examples.py
  → 실행 후 예제 번호를 입력하면 해당 예제가 실행됩니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Ex1. 이미지 파일 읽고 출력 (ex1_opencv_readImg.py)
# ============================================================
def ex1_read_image():
    """Lenna.png 이미지를 읽고 화면에 출력"""
    image = cv2.imread('Lenna.png')

    if image is None:
        print("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
    else:
        cv2.imshow('Lenna', image)      # 'Lenna'는 윈도우 이름
        cv2.waitKey(0)                  # 키 입력을 대기
        cv2.destroyAllWindows()         # 모든 창 닫기


# ============================================================
# Ex2. 동영상 파일 읽고 출력 (ex2_opencv_readVideo.py)
# ============================================================
def ex2_read_video():
    """test_video.mp4 동영상을 읽고 프레임 단위로 출력"""
    video_path = "test_video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"동영상 파일을 열 수 없습니다: {video_path}")
        return

    while True:
        ret, frame = cap.read()         # 동영상 프레임 읽기
        if not ret:
            print("동영상 재생이 끝났습니다.")
            break

        cv2.imshow('Video Playback', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()                       # 비디오 캡처 객체 닫기
    cv2.destroyAllWindows()


# ============================================================
# Ex3. 카메라 영상 읽고 출력 (ex3_opencv_readCam.py)
# ============================================================
def ex3_read_camera():
    """노트북/웹캠으로부터 영상을 읽고 화면에 출력"""
    cap = cv2.VideoCapture(0)           # 기본 카메라: ID 0

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()         # 비디오 프레임 읽기
        if not ret:
            print("프레임을 읽을 수 없습니다. 카메라를 확인하세요.")
            break

        cv2.imshow('Camera', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Ex4. BGR 채널 분리 (ex4_opencv_split.py)
# ============================================================
def ex4_split_bgr():
    """Lenna.png 이미지를 읽고 Blue/Green/Red 채널을 분리하여 출력"""
    image = cv2.imread('Lenna.png')

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # Blue, Green, Red 성분 분리 (OpenCV는 BGR 순서)
    blue, green, red = cv2.split(image)

    print("Blue Component:")
    print(blue)
    print("\nGreen Component:")
    print(green)
    print("\nRed Component:")
    print(red)

    cv2.imshow('Original Image', image)
    cv2.imshow('Blue Component', blue)
    cv2.imshow('Green Component', green)
    cv2.imshow('Red Component', red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex5. HSV 색공간 변환 후 채널 분리 (ex5_opencv_cvtColor(HSV).py)
# ============================================================
def ex5_cvtcolor_hsv():
    """Lenna.png 이미지를 HSV 색공간으로 변환 후 H/S/V 채널 분리"""
    image = cv2.imread('Lenna.png')

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # BGR → HSV 색공간 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV 성분 분리
    h, s, v = cv2.split(hsv_image)

    print("Hue Component:")
    print(h)
    print("\nSaturation Component:")
    print(s)
    print("\nValue Component:")
    print(v)

    cv2.imshow('Original Image', image)
    cv2.imshow('Hue', h)
    cv2.imshow('Saturation', s)
    cv2.imshow('Value', v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex6. RGB 기반 색상 영역 검출 (ex6_opencv_extractColor(RGB).py)
# ============================================================
def ex6_extract_color_rgb():
    """Candies.png 이미지에서 Red 성분이 50 이상인 영역만 추출 (BGR 색공간)"""
    image = cv2.imread('Candies.png')

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # BGR 채널 분리
    blue, green, red = cv2.split(image)

    # Red 성분이 50 이상인 영역(마스크) 생성
    red_mask = cv2.inRange(red, 50, 255)

    # 마스크를 적용하여 원래 이미지의 해당 영역 추출
    filtered_image = cv2.bitwise_and(image, image, mask=red_mask)

    cv2.imshow('Original Image', image)
    cv2.imshow('Red >= 50 Mask', red_mask)
    cv2.imshow('Filtered Image (RGB)', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex7. HSV 기반 빨간색 캔디 추출 (ex7_opencv_extractColor(HSV).py)
# ============================================================
def ex7_extract_color_hsv():
    """Candies.png 이미지에서 HSV 색공간을 활용해 빨간색 캔디만 추출
    OpenCV에서 빨간색 Hue 범위: (0~10) 과 (170~180)
    """
    image = cv2.imread('Candies.png')

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # BGR → HSV 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 첫 번째 빨간색 범위 (Hue: 0~10)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    # 두 번째 빨간색 범위 (Hue: 170~180)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 각 범위에 대해 마스크 생성 후 합치기
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 마스크를 원본 이미지에 적용
    red_candies = cv2.bitwise_and(image, image, mask=red_mask)

    cv2.imshow('Original Image', image)
    cv2.imshow('Red Mask (HSV)', red_mask)
    cv2.imshow('Red Candies', red_candies)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex8. 그레이스케일 히스토그램 (ex7_opencv_histogram(grayscale).py)
# ============================================================
def ex8_histogram_grayscale():
    """Lenna.png를 그레이스케일로 읽고 히스토그램 출력"""
    image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # 히스토그램 계산
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    cv2.imshow('Grayscale Image', image)

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex9. 명암비 조작 (ex8_opencv_adjustWBR.py)
# ============================================================
def ex9_adjust_contrast():
    """Lenna.png 흑백 이미지의 명암비 조절
    수식: dst = saturate((1+alpha)*src - alpha*128)
    """
    image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    alpha = 1.0
    func = (1 + alpha) * image - (alpha * 128)
    dst = np.clip(func, 0, 255).astype(np.uint8)

    cv2.imshow('Original', image)
    cv2.imshow(f'Contrast Adjusted (alpha={alpha})', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex10. 히스토그램 평활화 (ex8_equalizeHist.py)
# ============================================================
def ex10_equalize_hist():
    """Hawkes.jpg 이미지에 히스토그램 평활화(equalizeHist) 적용"""
    image = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    dst = cv2.equalizeHist(image)

    cv2.imshow('Original', image)
    cv2.imshow('Equalized Histogram', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex11. 히스토그램 스트래칭 / 정규화 (ex8_opencv_normalizeHist.py)
# ============================================================
def ex11_normalize_hist():
    """Hawkes.jpg 이미지에 히스토그램 스트래칭(normalize) 적용"""
    image = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    dst = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow('Original', image)
    cv2.imshow('Normalized Histogram', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex12. 히스토그램 역투영 (ex8_opencv_backproj.py)
# ============================================================
def ex12_backprojection():
    """desert.jpg 이미지에서 사용자가 ROI를 선택하면 해당 색상 분포를 역투영으로 검출"""
    src = cv2.imread('desert.jpg', cv2.IMREAD_COLOR)

    if src is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # 사용자가 마우스로 ROI(관심 영역) 선택
    x, y, w, h = cv2.selectROI(src)

    # YCrCb 색공간으로 변환
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    roi = src_ycrcb[y:y+h, x:x+w]

    # ROI 히스토그램 계산
    hist = cv2.calcHist([roi], [1, 2], None, [128, 128], [0, 256, 0, 256])

    # 역투영 계산
    backproj = cv2.calcBackProject([src_ycrcb], [1, 2], hist, [0, 256, 0, 256], 1)

    # 역투영 마스크로 원본에서 해당 영역 추출
    dst = cv2.copyTo(src, backproj)

    cv2.imshow('Original', src)
    cv2.imshow('BackProjection Mask', backproj)
    cv2.imshow('BackProjection Result', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Ex13. 이미지 필터링 (ex9_opencv_filtering.py)
# ============================================================
def apply_filter(image, kernel, filter_name):
    """2D 필터 적용 후 결과 출력 및 저장"""
    filtered_image = cv2.filter2D(image, -1, kernel)
    cv2.imshow(f"{filter_name}", filtered_image)
    cv2.imwrite(f"{filter_name}.jpg", filtered_image)
    return filtered_image


def ex13_image_filtering():
    """Lenna.png 이미지에 평균값/샤프닝/라플라시안 필터 적용"""
    image = cv2.imread('Lenna.png')

    if image is None:
        print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # 1. 평균값 필터 (3x3) - 노이즈 제거, 블러
    average_filter = np.ones((3, 3), np.float32) / 9.0

    # 2. 샤프닝 필터 (3x3) - 경계 강조
    sharpening_filter = np.array([[0, -1,  0],
                                   [-1,  5, -1],
                                   [0, -1,  0]], np.float32)

    # 3. 라플라시안 필터 (3x3) - 엣지(경계선) 검출
    laplacian_filter = np.array([[0,  1,  0],
                                  [1, -4,  1],
                                  [0,  1,  0]], np.float32)

    cv2.imshow('Original Image', image)
    apply_filter(image, average_filter, "Average Filter")
    apply_filter(image, sharpening_filter, "Sharpening Filter")
    apply_filter(image, laplacian_filter, "Laplacian Filter")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 메인 - 예제 선택 실행
# ============================================================
EXAMPLES = {
    '1':  ('이미지 파일 읽고 출력',              ex1_read_image),
    '2':  ('동영상 파일 읽고 출력',              ex2_read_video),
    '3':  ('카메라 영상 읽고 출력',              ex3_read_camera),
    '4':  ('BGR 채널 분리',                     ex4_split_bgr),
    '5':  ('HSV 색공간 변환 후 채널 분리',        ex5_cvtcolor_hsv),
    '6':  ('RGB 기반 빨간색 영역 검출',           ex6_extract_color_rgb),
    '7':  ('HSV 기반 빨간색 캔디 추출',           ex7_extract_color_hsv),
    '8':  ('그레이스케일 히스토그램',              ex8_histogram_grayscale),
    '9':  ('명암비 조작',                        ex9_adjust_contrast),
    '10': ('히스토그램 평활화 (equalizeHist)',    ex10_equalize_hist),
    '11': ('히스토그램 스트래칭 (normalize)',     ex11_normalize_hist),
    '12': ('히스토그램 역투영',                   ex12_backprojection),
    '13': ('이미지 필터링 (평균/샤프닝/라플라시안)', ex13_image_filtering),
}

if __name__ == "__main__":
    print("=" * 50)
    print("  3주차 OpenCV 예제 목록")
    print("=" * 50)
    for key, (desc, _) in EXAMPLES.items():
        print(f"  {key:>2}. {desc}")
    print("=" * 50)

    choice = input("실행할 예제 번호를 입력하세요 (종료: q): ").strip()

    if choice == 'q':
        print("종료합니다.")
    elif choice in EXAMPLES:
        desc, func = EXAMPLES[choice]
        print(f"\n[Ex{choice}] {desc} 실행 중...\n")
        func()
    else:
        print("올바른 번호를 입력하세요.")
