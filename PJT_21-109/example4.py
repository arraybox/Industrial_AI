import os

import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
pinhole_dir = os.path.join(script_dir, "pinhole_calib")

camera_matrix_path = os.path.join(script_dir, "camera_mat.npy")
dist_coefs_path = os.path.join(script_dir, "dist_coefs.npy")
image_path = os.path.join(pinhole_dir, "img_00.png")

if not os.path.exists(camera_matrix_path):
    camera_matrix_path = os.path.join(pinhole_dir, "camera_mat.npy")

if not os.path.exists(dist_coefs_path):
    dist_coefs_path = os.path.join(pinhole_dir, "dist_coefs.npy")

if not os.path.exists(camera_matrix_path):
    raise FileNotFoundError(
        "camera_mat.npy not found. Run example1.py first or place it in PJT_21-109."
    )

if not os.path.exists(dist_coefs_path):
    raise FileNotFoundError(
        "dist_coefs.npy not found. Run example1.py first or place it in PJT_21-109."
    )

camera_matrix = np.load(camera_matrix_path)
dist_coefs = np.load(dist_coefs_path)

img = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Cannot read input image: {image_path}")

pattern_size = (10, 7)
res, corners = cv2.findChessboardCorners(img, pattern_size)
if not res:
    raise RuntimeError(f"Chessboard corners were not found in: {image_path}")

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-3,
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerSubPix(
    gray,
    corners,
    (10, 10),
    (-1, -1),
    criteria,
)

h_corners = cv2.undistortPoints(corners, camera_matrix, dist_coefs)
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]

img_pts, _ = cv2.projectPoints(
    h_corners,
    (0, 0, 0),
    (0, 0, 0),
    camera_matrix,
    None,
)

undistorted_view = np.copy(img)
for c in corners:
    cv2.circle(undistorted_view, tuple(c[0].astype(int)), 10, (0, 255, 0), 2)

for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(undistorted_view, tuple(c.astype(int)), 5, (0, 0, 255), 2)

cv2.imshow("undistorted corners", undistorted_view)
cv2.waitKey()
cv2.destroyAllWindows()

reprojected_img_pts, _ = cv2.projectPoints(
    h_corners,
    (0, 0, 0),
    (0, 0, 0),
    camera_matrix,
    dist_coefs,
)

reprojected_view = np.copy(img)
for c in reprojected_img_pts.squeeze().astype(np.float32):
    cv2.circle(reprojected_view, tuple(c.astype(int)), 2, (255, 255, 0), 2)

cv2.imshow("reprojected corners", reprojected_view)
cv2.waitKey()
cv2.destroyAllWindows()
