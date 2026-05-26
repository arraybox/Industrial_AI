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

data = np.load(stereo_path, allow_pickle=True).item()
E = data["E"]

R1, R2, T = cv2.decomposeEssentialMat(E)

print("Rotation 1:")
print(R1)
print("Rotation 2:")
print(R2)
print("Translation:")
print(T)
