import cv2
import numpy as np
import os

img_path = os.path.join(os.path.dirname(__file__), 'Lena.png')
image = cv2.imread(img_path).astype(np.float32) / 255
print('Shape:', image.shape)
print('Data type:' , image.dtype)
cv2.imshow('original image', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale')
print('Shape:', gray.dtype)
cv2.imshow('gray-scale image', gray)

from_hsv = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shap:', from_hsv.dtype)
cv2.imshow('from_hsv', from_hsv)
cv2.waitKey()
cv2.destroyAllWindows()