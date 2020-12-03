import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [-1, -2, -3],
              [-4, -5, -6],
              [-7, -8, -9]])

print(a)
b = a.reshape((-1, 1))
print(b)

x = np.arange(a.shape[0])
y = np.arange(a.shape[1])
X, Y = np.meshgrid(x, y)
print(X)
print(Y)

index_x = X.T.reshape((-1, 1))
index_y = Y.T.reshape((-1, 1))
print(index_x)
print(index_y)

new = np.hstack((index_x, index_y, b))
print(new)

import matplotlib.pyplot as plt
import cv2
target = cv2.imread("B_ML.png")

target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

kernel_noise = np.ones((16, 16), dtype='uint8')
base_img_de_noise = cv2.morphologyEx(target_gray, cv2.MORPH_CLOSE, kernel_noise)
base_img_de_noise = cv2.morphologyEx(base_img_de_noise, cv2.MORPH_CLOSE, kernel_noise)
ret2, th2 = cv2.threshold(base_img_de_noise, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(th2, cmap='gray')
plt.show()
