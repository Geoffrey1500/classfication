import numpy as np
import cv2
from math import cos, sin, radians
import matplotlib.pyplot as plt


#读取目标图片
target = cv2.imread("B_MU_pic.png")
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

x_, y_ = int(th2.shape[0]/16), int(th2.shape[1]/16)
part_1 = np.zeros((x_, y_))
part_2 = np.ones((x_, y_))*255
template_up = np.hstack((part_1, part_2))
template_low = np.hstack((part_2, part_1))

template_new = np.vstack((template_up, template_low))
template_new = template_new.astype('uint8')
print(part_1, template_new.shape)
print(target_gray.shape)

# fig = plt.figure()
# plt.imshow(th2, cmap='gray')
# plt.show()

original_x = th2.shape[0]
original_y = th2.shape[1]

side_x = template_new.shape[0]
side_y = template_new.shape[1]


def getRotationMatrix2D(theta, cx=0, cy=0):
    # 角度值转换为弧度值
    # 因为图像的左上角是原点 需要×-1
    theta = radians(-1 * theta)

    M = np.float32([
        [cos(theta), -sin(theta), (1-cos(theta))*cx + sin(theta)*cy],
        [sin(theta), cos(theta), -sin(theta)*cx + (1-cos(theta))*cy]])
    return M

# 求得图片中心点， 作为旋转的轴心
cx = int(side_y / 2)
cy = int(side_x / 2)

# 进行2D 仿射变换
# 围绕原点 逆时针旋转30度
print(original_x, original_y)
print(side_x, side_y)

circle = np.zeros(template_new.shape, dtype="uint8")
mask_ = cv2.circle(circle, (int(side_y / 2), int(side_x / 2)), int((side_x + side_y)/2), 255, -1)


print(mask_.shape, 'mask 的形状')

M = getRotationMatrix2D(-40, cx=cx, cy=cy)
rotated_30 = cv2.warpAffine(template_new, M, (template_new.shape[1], template_new.shape[0]))

result = cv2.matchTemplate(th2, rotated_30, cv2.TM_SQDIFF)
print(result, '未正则化结果')
print(cv2.minMaxLoc(result))
cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(min_val, max_val, min_loc, max_loc)
print(np.array(list(min_loc)) + np.array([side_y/2, side_x/2]))

# fig = plt.figure()
# plt.imshow(mask_, cmap='gray')
fig2 = plt.figure()
plt.imshow(rotated_30, cmap='gray')
plt.show()

strmin_val = str(min_val)

cv2.rectangle(target, min_loc, (min_loc[0]+side_y,min_loc[1]+side_x),(0,0,225),2)
cv2.imshow("MatchResult----MatchingValue="+strmin_val,target)
cv2.waitKey()
cv2.destroyAllWindows()

# data_container = np.zeros((original_x-side_x + 1, original_y-side_y + 1))

# for i in range(target_gray.shape[0]):
#     i_end = i+side_x
#     for j in range(target_gray.shape[1]):
#         j_end = j+side_y
#         if i_end <= original_x and j_end <= original_y:
#             # print("hello world")
#             cut_ = th2[i:i+side_x, j:j+side_y]
#             con_sum_ = np.sum(template_new * cut_)
#             # print(con_sum_)
#
#             data_container[i, j] = con_sum_
#
# loc = np.where(data_container == np.max(data_container))
# print(loc)
# print(loc[-1] + np.array([side_x/2, side_y/2]))


# a_ = th2[250:250+template_new.shape[0], 200:200+template_new.shape[1]]
# print(a_)
# sum_ = np.sum(template_new * a_)
# print(np.sum(sum_))
#
# b_ = th2[100:100+template_new.shape[0], 100:100+template_new.shape[1]]
# print(b_)
# sum_b = np.sum(template_new * b_)
# print(sum_b)

