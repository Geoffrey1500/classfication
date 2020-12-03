import numpy as np
import cv2
from math import cos,sin,radians
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


target = cv2.imread("B_MU_pic.png")
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

kernel_noise = np.ones((16, 16), dtype='uint8')
base_img_de_noise = cv2.morphologyEx(target_gray, cv2.MORPH_CLOSE, kernel_noise)

ret2, th2 = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print(th2.shape)

th2_re = th2.reshape((-1, 1))
x = np.arange(th2.shape[0])
y = np.arange(th2.shape[1])
X, Y = np.meshgrid(x, y)

index_x = X.T.reshape((-1, 1))
index_y = Y.T.reshape((-1, 1))

new_index = np.hstack((index_x, index_y, th2_re))
print(new_index)
print(target.shape, "aaaaa")

print(new_index[:, -1] == 0)

updated = np.delete(new_index, new_index[:, -1] == 0, axis=0)
print(updated)

updated_2 = np.delete(updated, -1, axis=1)
print(updated_2)
# new_index = np.uint8(new_index)

k_mean_sk = KMeans(n_clusters=2, random_state=0).fit(updated_2)
print(k_mean_sk.cluster_centers_, "最后的希望")
result = k_mean_sk.cluster_centers_
result2 = np.abs(result[0] - result[1])
thetattt = np.arctan(result2[0]/result2[1])*180/np.pi
print(result2)
print(thetattt)

updated = np.delete(new_index, new_index[:, -1] == 255, axis=0)
print(updated)

updated_2 = np.delete(updated, -1, axis=1)
print(updated_2)
# new_index = np.uint8(new_index)

k_mean_sk = KMeans(n_clusters=2, random_state=0).fit(updated_2)
print(k_mean_sk.cluster_centers_, "最后的希望")
result = k_mean_sk.cluster_centers_
result2 = np.abs(result[0] - result[1])
thetattt = np.arctan(result2[0]/result2[1])*180/np.pi
print(result2)
print(thetattt)


# Z = target.reshape((-1, 3))
# Z = np.float32(updated_2)
# #
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 2
# ret, label, center = cv2.kmeans(Z, K, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
# print(center)
# #
# center = np.uint8(center)
# res = center[label.flatten()]
# print(res)

plt.imshow(th2, cmap='gray')
plt.show()
# res2 = res.reshape((th2.shape))

# # 提取边缘
# edges = cv2.Canny(th2, 200, 250, apertureSize=3)
# plt.imshow(edges)
# plt.show()
# print(int(0.25*np.sqrt(edges.shape[0]**2 + edges.shape[1]**2)), '阈值')
# # 提取直线
# lines = cv2.HoughLines(edges, 1, 1*np.pi/180, 80)
# print(lines)
#
# for line in lines:
#     for rho, theta in line:
#         print(rho, theta)
#         a = np.cos(theta)
#         b = np.sin(theta)
#         print(theta/np.pi*180)
#         x0 = a*rho
#         y0 = b*rho
#         x_check = x0 + np.tan(theta) + 10
#         y_check = y0 + np.tan(theta) + 10
#         print(th2[int(x_check), int(y_check)])
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         #把直线显示在图片上
#         cv2.line(th2,(x1,y1),(x2,y2),(0,0,255),2)
#
# # 显示原图和处理后的图像
# cv2.imshow("org", target)
# cv2.imshow("processed", th2)
#
# cv2.waitKey(0)
#
#
#
# # result
# # kmean: k=3, cost=-668
