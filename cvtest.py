import cv2
import numpy as np

filename = "crossfar.png"  # Change this to your image path
source = cv2.imread(filename)
if source is None:
    print("Cannot read image:", filename)
    exit()

# Crop the photo removing the outside 15% on all sides
height, width, _ = source.shape
left = int(width * 0.15)
right = int(width * 0.85)
top = int(height * 0.15)
bottom = int(height * 0.85)
source = source[top:bottom, left:right]

kernel_size = 9
kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

destination = cv2.filter2D(source, -1, kernel)

# Convert image to gray and blur it
srcGray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
srcGray = cv2.blur(srcGray, (3, 3))

# Histogram equalization
srcGray = cv2.equalizeHist(srcGray)

# Denoise
_, srcGray = cv2.threshold(srcGray, 25, 255, cv2.THRESH_BINARY_INV)

morphingMatrix = np.ones((1, 1), np.uint8)
srcGray = cv2.morphologyEx(srcGray, cv2.MORPH_OPEN, morphingMatrix)

# Image denoising
srcGray = cv2.fastNlMeansDenoising(srcGray, None, 30, 7, 21)

# Dilation
dilation_size = 5
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
srcGray = cv2.dilate(srcGray, element1)

# Find contours
contours, hierarchy = cv2.findContours(srcGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

drawing = np.zeros_like(source)
for contour in contours:
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.drawContours(drawing, [contour], 0, color, 2)

cv2.imwrite("output1.jpg", drawing)

