import numpy as np

corner_pixel_cor = np.array([109.80508, 101.805664])
spacing = np.array([0.001149567934554709, 0.001149567934554709])

help_x_y = np.array([0.03818093, 5.93416889])

back_cor = corner_pixel_cor*spacing + help_x_y
print(back_cor)
back_cor_full = np.insert(back_cor, 0, -1.9438325453827976)
print(back_cor_full, '转换前中心点坐标')


def rotation(normal_vector_, support_vector):
    a = np.array(normal_vector_)
    b = np.array(support_vector)
    theta_ = np.arccos(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)))

    rotation_axis = np.cross(a, b)

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


vector = [0.01722587, 0.00309399, 0.99984684]
print(np.array(vector), "比一下")


cordi = np.array([list(back_cor_full)])
print(cordi, "before rotation")

b, a = vector, [1, 0, 0]
q_before, q_after = rotation(a, b)
data_tra = np.hstack((np.zeros((cordi.shape[0], 1)), cordi))
data_rota = quaternion_mal(q_before, quaternion_mal(data_tra.T, q_after))
data_final = np.delete(data_rota.T, 0, axis=1)

print(data_final, "after_rotation")
