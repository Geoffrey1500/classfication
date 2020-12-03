import numpy as np
import cv2
from math import cos, sin, radians
import matplotlib.pyplot as plt


#读取目标图片
target = cv2.imread("zehao.png")
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
ret2, target_gray = cv2.threshold(target_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

x_, y_ = int(target_gray.shape[0] / 8), int(target_gray.shape[1] / 8)
x_ = min(x_, y_)
y_ = min(x_, y_)
part_1 = np.zeros((x_, y_))
part_2 = np.ones((x_, y_))*255
template_up = np.hstack((part_1, part_2))
template_low = np.hstack((part_2, part_1))

template_new = np.vstack((template_up, template_low))
template_new = template_new.astype('uint8')

fig1 = plt.figure()
plt.axis('equal')
plt.imshow(target_gray)

print(target_gray.shape, '导入后照片形状')

cx = template_new.shape[1]
cy = template_new.shape[0]

min_val_initial = 10**10000

for ang_ in np.linspace(0, 180, 181, endpoint=True):
    rotation_matrix_ = cv2.getRotationMatrix2D((int(cx / 2), int(cy / 2)), ang_, 2)
    rotated_ = cv2.warpAffine(template_new, rotation_matrix_, (cx, cy))
    ret_rotation, rotated_ = cv2.threshold(rotated_, 150, 255, cv2.THRESH_BINARY)

    result = cv2.matchTemplate(target_gray, rotated_, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if min_val < min_val_initial:
        min_val_initial = min_val
        left_corner = min_loc
        rotated_draw = rotated_
        center_loc = np.array(list(min_loc)) + np.array([cx / 2, cy / 2])
        circle_center = (int(min_loc[0]+cx/2), int(min_loc[1]+cy/2))
        up_corner = (min_loc[0] + cx, min_loc[1] + cy)

# M = cv2.getRotationMatrix2D((int(cx/2), int(cy/2)), 145, 1.7)
# rotated_30 = cv2.warpAffine(template_new, M, (cx, cy))
# ret, rotated_30 = cv2.threshold(rotated_30, 150, 255, cv2.THRESH_BINARY)
#
# result = cv2.matchTemplate(target_gray, rotated_30, cv2.TM_SQDIFF)
# print(cv2.minMaxLoc(result))
# cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# print(min_val, max_val, min_loc, max_loc)
# print(np.array(list(min_loc)) + np.array([cx/2, cy/2]))

# fig = plt.figure()
# plt.imshow(mask_, cmap='gray')
fig2 = plt.figure()
plt.axis('equal')
plt.imshow(rotated_draw, cmap='gray')
plt.show()

strmin_val = str(min_val_initial)

cv2.rectangle(target, left_corner, up_corner, (0, 0, 225), 2)
cv2.circle(target, circle_center, 2, (0, 0, 225), 3)
print(circle_center, "角点")
cv2.imshow("MatchResult----MatchingValue="+strmin_val, target)
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

