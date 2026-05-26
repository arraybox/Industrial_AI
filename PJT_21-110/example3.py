import os

import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
stereo_path = os.path.join(script_dir, "case1", "stereo.npy")

if not os.path.exists(stereo_path):
    fallback_stereo_path = os.path.join(script_dir, "stereo.npy")
    if os.path.exists(fallback_stereo_path):
        stereo_path = fallback_stereo_path
    else:
        raise FileNotFoundError(
            "stereo.npy not found. Place it in PJT_21-110/case1 or PJT_21-110."
        )

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
data = np.load(stereo_path).item()
np.load = np_load_old

Kl = data["K1"] if "K1" in data else data["Kl"]
Kr = data["K2"] if "K2" in data else data["Kr"]
Dl = data["D1"] if "D1" in data else data["Dl"]
Dr = data["D2"] if "D2" in data else data["Dr"]
left_pts = data["left_pts"]
right_pts = data["right_pts"]
E_from_stereo = data["E"]
F_from_stereo = data["F"]

left_pts = np.vstack(left_pts)
right_pts = np.vstack(right_pts)

left_pts = cv2.undistortPoints(left_pts, Kl, Dl, P=Kl)
right_pts = cv2.undistortPoints(right_pts, Kr, Dr, P=Kr)

F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)
if F is None:
    raise RuntimeError("Failed to estimate the fundamental matrix.")

E = Kr.T @ F @ Kl

print("Fundamental matrix:")
print(F)
print("Essential matrix:")
print(E)
print("Fundamental matrix from stereoCalibrate:")
print(F_from_stereo)
print("Essential matrix from stereoCalibrate:")
print(E_from_stereo)
