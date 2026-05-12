import os
from pathlib import Path

import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))

image_paths = [
    os.path.join(script_dir, name)
    for name in sorted(os.listdir(script_dir))
    if name.lower().endswith((".jpg", ".jpeg", ".png"))
    and Path(name).stem.isdigit()
]

if len(image_paths) < 2:
    raise FileNotFoundError(
        "At least two panorama images are required. "
        f"Place images such as 0.jpg and 1.jpg in: {script_dir}"
    )

images = []
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    images.append(image)


def stitch_two_images_with_features(left_img, right_img):
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(2000)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        detector = cv2.ORB_create(5000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    kp_left, des_left = detector.detectAndCompute(gray_left, None)
    kp_right, des_right = detector.detectAndCompute(gray_right, None)

    if des_left is None or des_right is None:
        return None

    matches = matcher.knnMatch(des_right, des_left, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 4:
        return None

    src_pts = np.float32([kp_right[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_left[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if homography is None:
        return None

    h_left, w_left = left_img.shape[:2]
    h_right, w_right = right_img.shape[:2]

    left_corners = np.float32(
        [[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]]
    ).reshape(-1, 1, 2)
    right_corners = np.float32(
        [[0, 0], [0, h_right], [w_right, h_right], [w_right, 0]]
    ).reshape(-1, 1, 2)
    warped_right_corners = cv2.perspectiveTransform(right_corners, homography)

    all_corners = np.concatenate((left_corners, warped_right_corners), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]],
        dtype=np.float64,
    )
    output_size = (x_max - x_min, y_max - y_min)

    panorama = cv2.warpPerspective(right_img, translation @ homography, output_size)
    panorama[-y_min:h_left - y_min, -x_min:w_left - x_min] = left_img
    return panorama


if hasattr(cv2, "Stitcher_create"):
    stitcher = cv2.Stitcher_create()
else:
    stitcher = cv2.createStitcher()

ret, pano = stitcher.stitch(images)

if ret != cv2.Stitcher_OK and len(images) == 2:
    print(f"OpenCV Stitcher failed with status code {ret}. Trying feature fallback.")
    pano = stitch_two_images_with_features(images[0], images[1])
    ret = cv2.Stitcher_OK if pano is not None else ret

if ret == cv2.Stitcher_OK:
    output_path = os.path.join(script_dir, "panorama_result.jpg")
    cv2.imwrite(output_path, pano)
    print(f"Panorama saved to: {output_path}")
    pano = cv2.resize(pano, dsize=(0, 0), fx=0.2, fy=0.2)
    cv2.imshow("panorama", pano)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print(f"Error during stitching. Stitcher status code: {ret}")
