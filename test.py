
import cv2
import numpy as np
from skimage.transform import radon

img = cv2.imread('circle.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
