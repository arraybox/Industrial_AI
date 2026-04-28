import cv2
import numpy as np
import os

# 1. Lena 이미지를 컬러영상으로 읽고 출력
img_path = os.path.join(os.path.dirname(__file__), "Lena.png")
img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
cv2.imshow("Color", img_color)
cv2.waitKey(0)

# 2. 흑백으로 변환하고 출력
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", img_gray)
cv2.waitKey(0)

# 3. Histogram Equalization 적용 및 출력
img_hist_eq = cv2.equalizeHist(img_gray)
cv2.imshow("Histogram Equalization", img_hist_eq)
cv2.waitKey(0)

# 4. Gamma Correction 적용 및 출력
gamma = 2.2
img_normalized = img_gray / 255.0
img_gamma = np.power(img_normalized, gamma)
img_gamma = np.uint8(img_gamma * 255)
cv2.imshow("Gamma Correction", img_gamma)
cv2.waitKey(0)

# 5. HSV 컬러 스페이스로 변환 및 각 채널 출력 (0~255 정규화)
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)

h_normalized = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
s_normalized = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
v_normalized = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("H Channel", h_normalized)
cv2.imshow("S Channel", s_normalized)
cv2.imshow("V Channel", v_normalized)
cv2.waitKey(0)

# 6. H 채널에 Median Filter 적용 및 출력
h_median = cv2.medianBlur(h, 5)
cv2.imshow("H - Median Filter", h_median)
cv2.waitKey(0)

# 7. S 채널에 Gaussian Filter 적용 및 출력
s_gaussian = cv2.GaussianBlur(s, (5, 5), 0)
cv2.imshow("S - Gaussian Filter", s_gaussian)
cv2.waitKey(0)

# 8. V 채널에 Bilateral Filter 적용 및 출력
v_bilateral = cv2.bilateralFilter(v, 9, 75, 75)
cv2.imshow("V - Bilateral Filter", v_bilateral)
cv2.waitKey(0)

cv2.destroyAllWindows()
