import cv2
import numpy as np

# ── 영상 로드 ──
img1 = cv2.imread('./Lena.png', cv2.IMREAD_COLOR)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 영상(2): 270도 회전된 Lena
img2 = cv2.imread('./Lena_rotated.png', cv2.IMREAD_COLOR)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def detect_all_features(gray):
    """FAST, Harris, GoodFeaturesToTrack, SIFT 특징점을 모두 검출"""
    # 1) FAST
    fast = cv2.FastFeatureDetector_create(30, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    kp_fast = fast.detect(gray)

    # 2) Harris Corner
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    pts_harris = np.argwhere(harris > 0.01 * harris.max())  # (row, col)

    # 3) Good Features to Track
    gftt = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)

    # 4) SIFT
    sift = cv2.SIFT_create()
    kp_sift = sift.detect(gray)

    return kp_fast, pts_harris, gftt, kp_sift


def draw_all_features(img, kp_fast, pts_harris, gftt, kp_sift):
    """각 특징을 서로 다른 모양과 색상으로 표시"""
    result = img.copy()

    # FAST: 초록색 원 (circle)
    for kp in kp_fast:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(result, (x, y), 4, (0, 255, 0), 1)

    # Harris: 빨간색 사각형 (rectangle)
    for pt in pts_harris:
        y, x = pt[0], pt[1]
        cv2.rectangle(result, (x - 3, y - 3), (x + 3, y + 3), (0, 0, 255), 1)

    # Good Features to Track: 노란색 삼각형 (triangle)
    if gftt is not None:
        for corner in gftt:
            x, y = int(corner[0][0]), int(corner[0][1])
            pts = np.array([[x, y - 5], [x - 5, y + 5], [x + 5, y + 5]], np.int32)
            cv2.polylines(result, [pts], True, (0, 255, 255), 1)

    # SIFT: 분홍색 다이아몬드 (diamond)
    for kp in kp_sift:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        pts = np.array([[x, y - 5], [x + 5, y], [x, y + 5], [x - 5, y]], np.int32)
        cv2.polylines(result, [pts], True, (255, 0, 255), 1)

    # 범례를 이미지 좌측 하단에 표시 (반투명 검정 박스)
    h = result.shape[0]
    overlay = result.copy()
    cv2.rectangle(overlay, (5, h - 95), (230, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
    cv2.putText(result, "FAST - Green Circle", (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result, "Harris - Red Rect", (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(result, "GFTT - Yellow Triangle", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(result, "SIFT - Pink Diamond", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return result


def count_features(kp_fast, pts_harris, gftt, kp_sift):
    """각 특징 검출 개수를 출력"""
    n_gftt = len(gftt) if gftt is not None else 0
    print(f"  FAST:               {len(kp_fast)}")
    print(f"  Harris Corner:      {len(pts_harris)}")
    print(f"  GoodFeaturesToTrack: {n_gftt}")
    print(f"  SIFT:               {len(kp_sift)}")


# ── 영상(1) 특징 검출 및 표시 ──
kp_fast1, pts_harris1, gftt1, kp_sift1 = detect_all_features(gray1)
result1 = draw_all_features(img1, kp_fast1, pts_harris1, gftt1, kp_sift1)

print("[영상(1) 특징 검출 결과]")
count_features(kp_fast1, pts_harris1, gftt1, kp_sift1)

# ── 영상(2) 특징 검출 및 표시 ──
kp_fast2, pts_harris2, gftt2, kp_sift2 = detect_all_features(gray2)
result2 = draw_all_features(img2, kp_fast2, pts_harris2, gftt2, kp_sift2)

print("\n[영상(2) 특징 검출 결과 - 이동+회전 적용]")
count_features(kp_fast2, pts_harris2, gftt2, kp_sift2)

# ── 결과 출력 (좌우로 합쳐서 한 창에 표시) ──
combined = np.hstack((result1, result2))
cv2.imshow('Image 1 (left) vs Image 2 - rotated 270 (right)', combined)
cv2.waitKey()
cv2.destroyAllWindows()
