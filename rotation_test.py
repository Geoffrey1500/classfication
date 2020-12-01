import numpy as np

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


cordi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

a, b = [0, 0, 1], [1, 0, 0]
q_before, q_after = rotation(a, b)
data_tra = np.hstack((np.zeros((cordi.shape[0], 1)), cordi))
data_rota = quaternion_mal(q_before, quaternion_mal(data_tra.T, q_after))
data_final = np.delete(data_rota.T, 0, axis=1)

print(data_final)
