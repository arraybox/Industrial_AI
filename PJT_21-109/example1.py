import os

import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(script_dir, "pinhole_calib")

pattern_size = (10, 7)
samples = []

if not os.path.isdir(calib_dir):
    raise FileNotFoundError(
        "Calibration image folder not found. "
        f"Place chessboard images in: {calib_dir}"
    )

file_list = os.listdir(calib_dir)
img_file_list = sorted(
    file for file in file_list
    if file.startswith("img") and file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
)

if not img_file_list:
    raise FileNotFoundError(
        "No calibration images found. "
        "Image filenames should start with 'img' in the pinhole_calib folder."
    )

for filename in img_file_list:
    frame = cv2.imread(os.path.join(calib_dir, filename), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Cannot read image: {filename}")
        continue

    res, corners = cv2.findChessboardCorners(frame, pattern_size)

    img_show = np.copy(frame)
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(
        img_show,
        "Samples captured: %d" % len(samples),
        (0, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    cv2.imshow("chessboard", img_show)

    wait_time = 0 if res else 30
    key = cv2.waitKey(wait_time)

    if key == ord("s") and res:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append((gray_frame, corners))
    elif key == 27:
        break

cv2.destroyAllWindows()

if not samples:
    raise RuntimeError("No chessboard samples were captured. Press 's' on detected frames.")

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-3,
)

refined_samples = []
for img, corners in samples:
    refined_corners = cv2.cornerSubPix(
        img,
        corners,
        (10, 10),
        (-1, -1),
        criteria,
    )
    refined_samples.append((img, refined_corners))

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

images, corners = zip(*refined_samples)
pattern_points = [pattern_points] * len(corners)

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
    pattern_points,
    corners,
    images[0].shape,
    None,
    None,
)

camera_mat_path = os.path.join(script_dir, "camera_mat.npy")
dist_coefs_path = os.path.join(script_dir, "dist_coefs.npy")

np.save(camera_mat_path, camera_matrix)
np.save(dist_coefs_path, dist_coefs)

print("RMS:", rms)
print("Camera matrix:")
print(np.load(camera_mat_path))
print("Distortion coefficients:")
print(np.load(dist_coefs_path))
