import glob
import os

import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
stereo_dir = os.path.join(script_dir, "stereo", "case1")

PATTERN_SIZE = (9, 6)

left_imgs = sorted(glob.glob(os.path.join(stereo_dir, "left*.png")))
right_imgs = sorted(glob.glob(os.path.join(stereo_dir, "right*.png")))

if not left_imgs or not right_imgs:
    raise FileNotFoundError(
        "Stereo calibration images not found. "
        f"Place left*.png and right*.png in: {stereo_dir}"
    )

if len(left_imgs) != len(right_imgs):
    raise RuntimeError(
        "The number of left and right stereo images must match. "
        f"left={len(left_imgs)}, right={len(right_imgs)}"
    )

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-3,
)

left_pts = []
right_pts = []
img_size = None

for left_img_path, right_img_path in zip(left_imgs, right_imgs):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        print(f"Cannot read pair: {left_img_path}, {right_img_path}")
        continue

    if left_img.shape != right_img.shape:
        print(f"Image size mismatch: {left_img_path}, {right_img_path}")
        continue

    if img_size is None:
        img_size = (left_img.shape[1], left_img.shape[0])

    res_left, corners_left = cv2.findChessboardCorners(left_img, PATTERN_SIZE)
    res_right, corners_right = cv2.findChessboardCorners(right_img, PATTERN_SIZE)

    if not (res_left and res_right):
        print(f"Chessboard not found in pair: {left_img_path}, {right_img_path}")
        continue

    corners_left = cv2.cornerSubPix(
        left_img,
        corners_left,
        (10, 10),
        (-1, -1),
        criteria,
    )
    corners_right = cv2.cornerSubPix(
        right_img,
        corners_right,
        (10, 10),
        (-1, -1),
        criteria,
    )

    left_pts.append(corners_left)
    right_pts.append(corners_right)

if not left_pts:
    raise RuntimeError("No valid stereo chessboard pairs were detected.")

pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points = [pattern_points] * len(left_pts)

err, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    pattern_points,
    left_pts,
    right_pts,
    None,
    None,
    None,
    None,
    img_size,
    flags=0,
)

print("Stereo calibration RMS error:")
print(err)
print("Left camera:")
print(K1)
print("Left camera distortion:")
print(D1)
print("Right camera:")
print(K2)
print("Right camera distortion:")
print(D2)
print("Rotation matrix:")
print(R)
print("Translation:")
print(T)

stereo_path = os.path.join(script_dir, "stereo.npy")
np.save(
    stereo_path,
    {
        "K1": K1,
        "D1": D1,
        "K2": K2,
        "D2": D2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "img_size": img_size,
        "left_pts": left_pts,
        "right_pts": right_pts,
    },
)

print(f"Stereo calibration saved to: {stereo_path}")
