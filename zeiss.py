import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

# opening image in RGB format
img = cv2.imread('Fundus.png')
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])

# The two arrays. Will be parsed by Adi's tesseract script
full_array = [[0, 0, 0, 22, 24, 23, 22, 0, 0], [0, 0, 26, 28, 28, 26, 27, 31, 0], [0, 20, 24, 31, 32, 29, 30, 28, 28], [21, 22, 25, 30, 32, 32, 29, 27, 28], [28, 28, 28, 32, 33, 34, 34, 0, 31], [0, 28, 30, 35, 33, 34, 33, 31, 29], [0, 0, 29, 33, 32, 34, 32, 30, 0], [0, 0, 0, 29, 32, 33, 33, 0, 0]]
difference_array = [[0, 0, 0, -5, -3, -4, -4, 0, 0], [0, 0, -2, -1, -1, -3, -2, 3, 0], [0, -9, -6, 0, 0, -2, 0, -1, 0], [-6, -8, -6, -2, -1, 0, -2, 0, -1], [1, -1, -3, 0, 0, 1, 2, 0, 1], [0, -1, -1, 3, 1, 2, 2, 0, 0], [0, 0, 0, 2, 1, 3, 1, 0, 0], [0, 0, 0, 0, 3, 4, 3, 0, 0]]

full_array = np.array(full_array)
difference_array = np.array(difference_array)


# calculate image center
center_x = int(img.shape[0]/2)
center_y = int(img.shape[1]/2)

# splice the image
blind_spot_detector = img.copy()
blind_spot_detector = blind_spot_detector[center_x:, center_y:]

# calculate blind spot
blind_spot = (0, 0)
highest_rms = 0
for i in range(blind_spot_detector.shape[0]):
    for j in range(blind_spot_detector.shape[1]):
        (r, g, b) = blind_spot_detector[i][j]
        rms = math.sqrt(math.pow(r, 2)+math.pow(g, 2)+math.pow(b, 2))
        if rms > highest_rms:
            highest_rms = rms
            blind_spot = (i, j)

# defining increments
x_increments = blind_spot[1]/5
y_increments = blind_spot_detector.shape[0]/8

# get image for mapping numbers
num_image = img.copy()
font = cv2.FONT_HERSHEY_SIMPLEX

# Starting value of Y
y = center_y - (len(full_array)/2)*y_increments*2 - y_increments
for i in range(len(full_array)):
    x = center_x - (len(full_array[i])/2)*x_increments*2 + 2*x_increments
    for j in range(len(full_array[i])):
        if full_array[i][j]:
            cv2.putText(num_image, str(full_array[i][j]),(int(x),int(y)), font, 1,(0,0,255),2)
        x += (x_increments * 2)
    y += (y_increments*2)

# plt.imshow(num_image) # comment this out on the server

# normalizing the difference array
g = (difference_array/ max(difference_array.min(), difference_array.max(), key=abs))
g = g.clip(max=0)
g = ((g+1)*255).astype(int)
cv2.imwrite('difference_picture.png', g)
g = cv2.imread('difference_picture.png')
for i in range(g.shape[0]):
    for j in range(g.shape[1]):
        g[i][j] = np.array([g[i][j][0], g[i][j][1], 255])

# resising the array
h = y_increments*2 * len(g)
w = x_increments*2 * len(g[0])
g = cv2.resize(g, (int(w), int(h)))

# save the image
cv2.imwrite('difference_picture.png', g)

f, axarr = plt.subplots(2,2)
f.set_size_inches(12, 12)
axarr[0][0].set_title('General retina image')
axarr[0][1].set_title('Data from Humphrey Field Analyser')
axarr[1][0].set_title('Data mapped onto retina')
axarr[1][1].set_title('Numeric mapping')
axarr[1][0].imshow(g, alpha=1)
axarr[1][0].imshow(img, alpha=0.5)
axarr[0][1].imshow(g)
axarr[0][0].imshow(img)
axarr[1][1].imshow(num_image)
print("Image is saved in combined.png")

f.savefig('combined.png',bbox_inches='tight')