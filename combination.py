from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, radians
import cv2
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


def rotation(normal_vector_, support_vector):
    a_ = np.array(normal_vector_)
    b_ = np.array(support_vector)
    theta_ = np.arccos(np.dot(a_, b_)/(np.linalg.norm(a_) * np.linalg.norm(b_)))

    rotation_axis = np.cross(a_, b_)

    q_angle = np.array([np.cos(theta_/2), np.sin(theta_/2), np.sin(theta_/2), np.sin(theta_/2)])
    q_vector = np.hstack((np.array([1]), rotation_axis))
    q = q_vector*q_angle
    q_1 = np.hstack((np.array([1]), -rotation_axis))*q_angle

    return q, q_1


def quaternion_mal(q_a, q_b):
    s = q_a[0] * q_b[0] - q_a[1] * q_b[1] - q_a[2] * q_b[2] - q_a[3] * q_b[3]
    x = q_a[0] * q_b[1] + q_a[1] * q_b[0] + q_a[2] * q_b[3] - q_a[3] * q_b[2]
    y = q_a[0] * q_b[2] - q_a[1] * q_b[3] + q_a[2] * q_b[0] + q_a[3] * q_b[1]
    z = q_a[0] * q_b[3] + q_a[1] * q_b[2] - q_a[2] * q_b[1] + q_a[3] * q_b[0]

    return np.array([s, x, y, z])


def fit_plane(input_data, dis_sigma=0.05, depth_approx=1, loop_time=2):
    (m_, n_) = input_data.shape
    j_count = 0
    inner_total_pre = 0
    best_param = [0, 0, 0]
    row_rand_array = np.arange(m_)

    a_, b_ = [-3, 2, 1], [1, 1, 1]

    q_before_, q_after_ = rotation(a_, b_)
    in_data_tra = np.hstack((np.zeros((input_data.shape[0], 1)), input_data))
    in_data_rota = quaternion_mal(q_before_, quaternion_mal(in_data_tra.T, q_after_))
    input_data = np.delete(in_data_rota.T, 0, axis=1)

    while j_count <= loop_time:
        i_ = 0

        while i_ <= int(m_/500):
            index_ = np.random.choice(row_rand_array, 3, replace=False)
            picked_points = input_data[index_]

            param_ = np.linalg.solve(picked_points, -depth_approx*np.ones(picked_points.shape[0]))

            points_dis = np.abs(np.dot(input_data, param_) + depth_approx)/np.sqrt(np.sum(param_**2))
            total = np.sum(points_dis <= dis_sigma)

            if total > inner_total_pre:
                inner_total_pre = total
                best_param = param_
            i_ += 1
        print("百分比", inner_total_pre/len(input_data))
        j_count += 1

    q_before_pa, q_after_pa = rotation(b_, a_)
    best_param_be = np.hstack((np.zeros((1, 1)), best_param.reshape([1, -1])))
    para_rota = quaternion_mal(q_before_pa, quaternion_mal(best_param_be.T, q_after_pa))
    best_param_ = np.delete(para_rota.T, 0, axis=1).flatten()

    best_param_ = best_param_ / np.sqrt(np.sum(best_param_ ** 2, axis=0))

    print("Calculated normal vector is: " + str(best_param_))

    return best_param_


def data_preprocess(file_name):
    data_raw_ = np.loadtxt(file_name)
    m_, n_ = data_raw_.shape
    coordinate_data_ = data_raw_[:, 0:3]

    if n_ == 6:
        color_data_ = data_raw_[:, 3::]
        color_in_gray_ = np.dot(color_data_, np.array([[299], [587], [114]]))/1000
        color_in_gray_ = color_in_gray_.astype(np.int)
        scale_k_ = 255 / (np.max(color_in_gray_) - np.min(color_in_gray_))
        color_in_gray_ = 0 + scale_k_ * (color_in_gray_ - np.min(color_in_gray_))
    else:
        color_data_ = data_raw_[:, 3:4]
        scale_k_ = 255/(np.max(color_data_) - np.min(color_data_))
        color_in_gray_ = 0 + scale_k_*(color_data_ - np.min(color_data_))

    color_in_gray_ = color_in_gray_.astype('uint8')

    return coordinate_data_, color_in_gray_


def corner_detector(src_gray_, val):
    maxCorners = max(val, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.99

    circle = np.zeros(src_gray_.shape, dtype="uint8")
    central_x, central_y = src_gray_.shape
    mask_ = cv2.circle(circle, (int(central_x/2), int(central_y/2)), int((central_x + central_y)/8), 255, -1)
    blockSize = int((central_x + central_y) / 16)

    # Copy the source image
    copy = np.copy(src_gray_)

    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(copy, maxCorners, qualityLevel, minDistance=None, mask=mask_, blockSize=blockSize)
    if corners.shape[0] > 1:
        print('**URGENT** Number of corners detected is larger than 1:', corners.shape[0], "You should have a check!!!")

    # Set the needed parameters to find the refined corners
    winSize = (3, 3)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 50, 0.001)
    # Calculate the refined corner locations
    corners = cv2.cornerSubPix(copy, corners, winSize, zeroZone, criteria)
    # Write them down
    # for i in range(corners.shape[0]):
    #     print(" -- Refined Corner [", i, "]  (", corners[i, 0, 0], ",", corners[i, 0, 1], ")")
    #     print(corners[0], '所有的角点')

    return corners[0]


def pattern_corner_detector(src_gray_):
    print(src_gray_.shape, '照片形状')
    x_, y_ = int(src_gray_.shape[0] / 16), int(src_gray_.shape[1] / 16)
    part_1 = np.zeros((x_, y_))
    part_2 = np.ones((x_, y_)) * 255
    template_up = np.hstack((part_1, part_2))
    template_low = np.hstack((part_2, part_1))

    template_ = np.vstack((template_up, template_low))
    template_ = template_.astype('uint8')

    cx = template_.shape[1]
    cy = template_.shape[0]

    print(cx, cy, 'template shape')

    rotation_matrix_ = cv2.getRotationMatrix2D((int(cx/2), int(cy/2)), -35, 1)
    rotated_ = cv2.warpAffine(template_, rotation_matrix_, (cx, cy))
    ret_rotation, rotated_ = cv2.threshold(rotated_, 150, 255, cv2.THRESH_BINARY)

    result = cv2.matchTemplate(src_gray_, rotated_, cv2.TM_SQDIFF)
    # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # print(min_val, max_val, min_loc, max_loc)
    # print(np.array(list(min_loc)) + np.array([cx / 2, cy / 2]))
    center_loc = np.array(list(min_loc)) + np.array([cx / 2, cy / 2])
    return center_loc


file_name = 'B_ML'

cordi, color = data_preprocess(file_name + '.txt')

vector = fit_plane(cordi, dis_sigma=0.008)

a, b = list(vector), [1, 0, 0]
q_before, q_after = rotation(a, b)
data_tra = np.hstack((np.zeros((cordi.shape[0], 1)), cordi))
data_rota = quaternion_mal(q_before, quaternion_mal(data_tra.T, q_after))
data_final = np.delete(data_rota.T, 0, axis=1)

print(np.std(data_final[:, 0]), np.std(data_final[:, 1]), np.std(data_final[:, 2]))
print(np.mean(data_final[:, 0]), "我是平均值")

x_n = np.max(data_final[:, 1]) - np.min(data_final[:, 1])
y_n = np.max(data_final[:, 2]) - np.min(data_final[:, 2])

scale_x_y = y_n / x_n

x_number = np.sqrt(len(data_final)/scale_x_y)
y_number = scale_x_y*np.sqrt(len(data_final)/scale_x_y)
print(x_number, y_number, x_number*y_number)

spacing_x, spacing_y = x_n/x_number, y_n/y_number
spacing_avg = np.average([spacing_x, spacing_y])/8
print(spacing_x, spacing_y, spacing_avg, "x 和 y 的尺度")

cut_data_ = data_final[:, 1::]
data_helper = np.array([[np.min(data_final[:, 1]), np.min(data_final[:, 2])]])
print(data_helper, 'I am helping you')
new_cor = np.round((cut_data_ - data_helper)/spacing_avg).astype(int)


pixel_x, pixel_y = np.max(new_cor[:, 0]), np.max(new_cor[:, 1])
print(pixel_x, pixel_y)
print(len(new_cor), pixel_x*pixel_y)

base_img = np.zeros((int(pixel_x) + 1, int(pixel_y) + 1))
print(base_img.shape)

for i in range(len(new_cor)):
    img_index_x, img_index_y = new_cor[i, 0], new_cor[i, 1]
    base_img[img_index_x, img_index_y] = color[i, 0]

kernel_noise = np.ones((1, 1), dtype='uint8')
base_img = base_img.astype(np.uint8)
base_img_de_noise = cv2.morphologyEx(base_img, cv2.MORPH_CLOSE, kernel_noise)
# base_img_de_noise = cv2.morphologyEx(base_img_de_noise, cv2.MORPH_CLOSE, kernel_noise)

cv2.imwrite(file_name + '.png', base_img)


corner_detected = pattern_corner_detector(base_img_de_noise)
print(corner_detected, '角点坐标')

# plt.imshow(base_img, cmap='gray')
# fig = plt.figure()
# plt.imshow(base_img_de_noise, cmap='gray')
# plt.show()

# cv2.imshow('image', base_img_de_noise)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite('B_MU_pic.png', base_img_de_noise)

corner_pixel_cor = np.array([corner_detected[1], corner_detected[0]])
back_cor = corner_pixel_cor*spacing_avg + data_helper
print(back_cor)
back_cor_full = np.insert(back_cor, 0, np.mean(data_final[:, 0]))

cordi_before = np.array([list(back_cor_full)])
# print(cordi_before, "before rotation")

b_2, a_2 = list(vector), [1, 0, 0]
q_before_2, q_after_2 = rotation(a_2, b_2)
data_tra_2 = np.hstack((np.zeros((cordi_before.shape[0], 1)), cordi_before))
data_rota_2 = quaternion_mal(q_before_2, quaternion_mal(data_tra_2.T, q_after_2))
data_final_2 = np.delete(data_rota_2.T, 0, axis=1)

print(data_final_2, "after_rotation")
print([-4.977487, 1.802372, -1.798017], 'Ground Truth')
print([-6.083892, 0.139264, -1.839561], "Ground Truth")

