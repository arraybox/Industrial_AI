import cv2
import numpy as np

# ============================================================
# Part 1 : Image Filtering
# ============================================================

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(script_dir, "Lena.png"))
if img is None:
    raise FileNotFoundError("Lena.png not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (1) Unsharp Mask
blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

cv2.imshow("Original", gray)
cv2.imshow("(1) Unsharp Mask", unsharp)

# (2) Sobel filter on unsharp result
sobel_x = cv2.Sobel(unsharp, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(unsharp, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

cv2.imshow("(2) Sobel Filter", sobel)

# (3) Gabor filter on unsharp result
gabor_kernel = cv2.getGaborKernel(
    ksize=(21, 21), sigma=5, theta=np.pi / 4,
    lambd=10, gamma=0.5, psi=0
)
gabor = cv2.filter2D(unsharp, cv2.CV_8U, gabor_kernel)

cv2.imshow("(3) Gabor Filter", gabor)

# (4) Difference of Sobel and Gabor with threshold trackbar
diff = cv2.absdiff(sobel, gabor)

def on_threshold(val):
    _, thresh_img = cv2.threshold(diff, val, 255, cv2.THRESH_BINARY)

    # Opening and Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("(4) Diff Threshold", thresh_img)
    cv2.imshow("(5) Opening", opened)
    cv2.imshow("(5) Closing", closed)

cv2.namedWindow("(4) Diff Threshold")
cv2.createTrackbar("Threshold", "(4) Diff Threshold", 30, 255, on_threshold)
on_threshold(30)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ============================================================
# Part 2 : Frequency-based Filtering (DFT)
# ============================================================

img2 = cv2.imread(os.path.join(script_dir, "Lena.png"), cv2.IMREAD_GRAYSCALE)

# DFT
dft = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Magnitude spectrum for display
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = 20 * np.log(magnitude + 1)
magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255).astype(np.uint8)

cv2.imshow("Original (gray)", img2)
cv2.imshow("DFT Magnitude Spectrum", magnitude_spectrum)

rows, cols = img2.shape
crow, ccol = rows // 2, cols // 2

# --- Circular (Low-Pass) Filter ---
radius = 30
circle_mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(circle_mask[:, :, 0], (ccol, crow), radius, 1, -1)
circle_mask[:, :, 1] = circle_mask[:, :, 0]

filtered_circle = dft_shift * circle_mask
f_ishift_c = np.fft.ifftshift(filtered_circle)
img_circle = cv2.idft(f_ishift_c)
img_circle = cv2.magnitude(img_circle[:, :, 0], img_circle[:, :, 1])
img_circle = np.clip(img_circle, 0, 255).astype(np.uint8)

cv2.imshow("Circle Filter (Low-Pass)", img_circle)

# --- Rectangular (Low-Pass) Filter ---
rect_w, rect_h = 30, 30
rect_mask = np.zeros((rows, cols, 2), np.uint8)
rect_mask[crow - rect_h:crow + rect_h, ccol - rect_w:ccol + rect_w] = 1

filtered_rect = dft_shift * rect_mask
f_ishift_r = np.fft.ifftshift(filtered_rect)
img_rect = cv2.idft(f_ishift_r)
img_rect = cv2.magnitude(img_rect[:, :, 0], img_rect[:, :, 1])
img_rect = np.clip(img_rect, 0, 255).astype(np.uint8)

cv2.imshow("Rectangle Filter (Low-Pass)", img_rect)

cv2.waitKey(0)
cv2.destroyAllWindows()
