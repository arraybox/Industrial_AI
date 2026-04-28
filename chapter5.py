import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

dir_path = os.path.dirname(__file__)
image = cv2.imread(os.path.join(dir_path, 'BnW.png'), 0)

# (1) Otsu thresholding
_, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# External / Internal Contour
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

image_external = np.zeros(otsu.shape, otsu.dtype)
image_internal = np.zeros(otsu.shape, otsu.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)
    else:
        cv2.drawContours(image_internal, contours, i, 255, -1)

plt.figure(figsize=(12, 3))
plt.subplot(141)
plt.axis('off')
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(142)
plt.axis('off')
plt.title('Otsu')
plt.imshow(otsu, cmap='gray')
plt.subplot(143)
plt.axis('off')
plt.title('External')
plt.imshow(image_external, cmap='gray')
plt.subplot(144)
plt.axis('off')
plt.title('Internal')
plt.imshow(image_internal, cmap='gray')
plt.tight_layout()
plt.show()

# Connected Component - 스페이스 누를 때마다 랜덤 5개 component 표시
num_labels, labels = cv2.connectedComponents(otsu)
print(f'Total components: {num_labels - 1} (excluding background)')

while True:
    canvas = np.zeros((*otsu.shape, 3), dtype=np.uint8)
    all_labels = list(range(1, num_labels))
    chosen = random.sample(all_labels, min(5, len(all_labels)))
    for lbl in chosen:
        color = [random.randint(50, 255) for _ in range(3)]
        canvas[labels == lbl] = color

    cv2.imshow('Connected Components (Space: refresh, ESC: exit)', canvas)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        continue
cv2.destroyAllWindows()

# Distance Transform
dist = cv2.distanceTransform(otsu, cv2.DIST_L2, 5)
dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.axis('off')
plt.title('Otsu')
plt.imshow(otsu, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('Distance Transform')
plt.imshow(dist, cmap='jet')
plt.colorbar(fraction=0.046)
plt.subplot(133)
plt.axis('off')
plt.title('Distance (normalized)')
plt.imshow(dist_normalized, cmap='gray')
plt.tight_layout()
plt.show()
