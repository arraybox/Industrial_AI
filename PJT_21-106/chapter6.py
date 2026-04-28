import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# (1) K-means: (R,G,B) vs (R,G,B,X,Y) Image Segmentation 비교
# =============================================================

image = cv2.imread('./Lena.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
h, w, _ = image_rgb.shape

num_classes = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)

# --- (R, G, B) 만 사용 ---
data_rgb = image_rgb.reshape((-1, 3))
_, labels_rgb, centers_rgb = cv2.kmeans(
    data_rgb, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)
seg_rgb = centers_rgb[labels_rgb.flatten()].reshape(image_rgb.shape)
seg_rgb = np.clip(seg_rgb, 0, 255).astype(np.uint8)

# --- (R, G, B, X, Y) 사용 ---
# X, Y 좌표 생성 후 RGB와 결합
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
coords = np.stack([xx, yy], axis=-1).reshape((-1, 2)).astype(np.float32)

# 좌표를 RGB 스케일(0~255)에 맞게 정규화
coords_scaled = coords.copy()
coords_scaled[:, 0] = coords[:, 0] / w * 255  # X
coords_scaled[:, 1] = coords[:, 1] / h * 255  # Y

data_rgbxy = np.hstack([data_rgb, coords_scaled])
_, labels_rgbxy, centers_rgbxy = cv2.kmeans(
    data_rgbxy, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)
# 복원 시 RGB 채널만 사용
seg_rgbxy = centers_rgbxy[labels_rgbxy.flatten()][:, :3].reshape(image_rgb.shape)
seg_rgbxy = np.clip(seg_rgbxy, 0, 255).astype(np.uint8)

# --- 결과 비교 출력 ---
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image_rgb.astype(np.uint8))
plt.title('Original')
plt.axis('off')

plt.subplot(132)
plt.imshow(seg_rgb)
plt.title('K-means (R, G, B)')
plt.axis('off')

plt.subplot(133)
plt.imshow(seg_rgbxy)
plt.title('K-means (R, G, B, X, Y)')
plt.axis('off')

plt.tight_layout()
plt.show()


# =============================================================
# (2) GrabCut - 사용자 입력으로 반복 수정 가능한 Segmentation
# =============================================================
# 사용법:
#   1. 먼저 마우스 드래그로 전경 영역을 포함하는 사각형을 그립니다.
#   2. GrabCut 초기 결과가 표시됩니다.
#   3. 이후 키 입력으로 마스크를 수정합니다:
#      - '0' : 확실한 배경(BGD) 브러시
#      - '1' : 확실한 전경(FGD) 브러시
#      - 'n' : 수정된 마스크로 GrabCut 재실행
#      - 'r' : 리셋
#      - ESC : 종료
# =============================================================

def grabcut_interactive():
    img = cv2.imread('./Lena.png')
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return

    img_copy = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (0, 0, 1, 1)
    drawing = False
    rect_done = False
    mode = cv2.GC_BGD  # 기본 브러시: 배경
    ix, iy = 0, 0

    def show_result():
        # 마스크에서 전경(확실+추정)만 추출
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        result = cv2.bitwise_and(img, img, mask=mask2)
        cv2.imshow('Segmentation Result', result)

    def on_mouse(event, x, y, flags, param):
        nonlocal ix, iy, drawing, rect_done, rect, img_copy
        nonlocal mask, bgd_model, fgd_model

        if not rect_done:
            # --- 사각형 그리기 단계 ---
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                img_copy = img.copy()
                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Input', img_copy)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                rect_done = True
                rect = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Input', img_copy)

                # 초기 GrabCut 실행 (RECT 모드)
                cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                show_result()
        else:
            # --- 마스크 수정 단계 ---
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                # 브러시로 마스크 수정
                cv2.circle(img_copy, (x, y), 3, (0, 0, 255) if mode == cv2.GC_FGD else (255, 0, 0), -1)
                cv2.circle(mask, (x, y), 3, mode, -1)
                cv2.imshow('Input', img_copy)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse)
    cv2.imshow('Input', img)

    print("=== GrabCut Interactive ===")
    print("1) 마우스 드래그로 전경 영역 사각형을 그리세요.")
    print("2) 사각형 후 키 입력:")
    print("   '0' -> 배경 브러시 | '1' -> 전경 브러시")
    print("   'n' -> GrabCut 재실행 | 'r' -> 리셋 | ESC -> 종료")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('0'):
            mode = cv2.GC_BGD
            print("브러시 모드: 배경 (BGD)")
        elif key == ord('1'):
            mode = cv2.GC_FGD
            print("브러시 모드: 전경 (FGD)")
        elif key == ord('n'):
            # 수정된 마스크로 GrabCut 재실행
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            show_result()
            print("GrabCut 재실행 완료")
        elif key == ord('r'):
            # 리셋
            img_copy = img.copy()
            mask[:] = 0
            bgd_model[:] = 0
            fgd_model[:] = 0
            rect_done = False
            cv2.imshow('Input', img_copy)
            cv2.destroyWindow('Segmentation Result')
            print("리셋 완료. 사각형을 다시 그리세요.")

    cv2.destroyAllWindows()


grabcut_interactive()
