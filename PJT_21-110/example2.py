import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({"font.size": 20})

script_dir = os.path.dirname(os.path.abspath(__file__))
stereo_dir = os.path.join(script_dir, "case1")
stereo_path = os.path.join(stereo_dir, "stereo.npy")
left_img_path = os.path.join(stereo_dir, "left14.png")
right_img_path = os.path.join(stereo_dir, "right14.png")

if not os.path.exists(stereo_path):
    fallback_stereo_path = os.path.join(script_dir, "stereo.npy")
    if os.path.exists(fallback_stereo_path):
        stereo_path = fallback_stereo_path
    else:
        raise FileNotFoundError(
            "stereo.npy not found. Place stereo.npy in PJT_21-110/case1 "
            "or in the PJT_21-110 folder. Expected path: "
            f"stereo.npy in: {stereo_dir}"
        )

if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
    fallback_left_img_path = os.path.join(script_dir, "left.png")
    fallback_right_img_path = os.path.join(script_dir, "right.png")
    if os.path.exists(fallback_left_img_path) and os.path.exists(fallback_right_img_path):
        left_img_path = fallback_left_img_path
        right_img_path = fallback_right_img_path
    else:
        raise FileNotFoundError(
            "Stereo image pair not found. Place left14.png and right14.png in "
            f"{stereo_dir}, or left.png and right.png in {script_dir}."
        )

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
data = np.load(stereo_path).item()
np.load = np_load_old

Kl = data["K1"] if "K1" in data else data["Kl"]
Dl = data["D1"] if "D1" in data else data["Dl"]
Kr = data["K2"] if "K2" in data else data["Kr"]
Dr = data["D2"] if "D2" in data else data["Dr"]
R = data["R"]
T = data["T"]
img_size = tuple(data["img_size"])

left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

if left_img is None or right_img is None:
    raise FileNotFoundError("Failed to read left14.png or right14.png.")

R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
    Kl,
    Dl,
    Kr,
    Dr,
    img_size,
    R,
    T,
)

xmap1, ymap1 = cv2.initUndistortRectifyMap(
    Kl,
    Dl,
    R1,
    Kl,
    img_size,
    cv2.CV_32FC1,
)
xmap2, ymap2 = cv2.initUndistortRectifyMap(
    Kr,
    Dr,
    R2,
    Kr,
    img_size,
    cv2.CV_32FC1,
)

left_img_rectified = cv2.remap(left_img, xmap1, ymap1, cv2.INTER_LINEAR)
right_img_rectified = cv2.remap(right_img, xmap2, ymap2, cv2.INTER_LINEAR)

plt.figure(0, figsize=(12, 10))
plt.subplot(221)
plt.title("left original")
plt.imshow(left_img, cmap="gray")
plt.subplot(222)
plt.title("right original")
plt.imshow(right_img, cmap="gray")
plt.subplot(223)
plt.title("left rectified")
plt.imshow(left_img_rectified, cmap="gray")
plt.subplot(224)
plt.title("right rectified")
plt.imshow(right_img_rectified, cmap="gray")
plt.tight_layout()
output_path = os.path.join(script_dir, "stereo_rectification_result.png")
plt.savefig(output_path)
print(f"Stereo rectification result saved to: {output_path}")
plt.show()
