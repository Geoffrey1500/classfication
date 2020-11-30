from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

all_data = np.loadtxt('111.txt')
print(all_data)
print(all_data.shape)

cordi = all_data[:, 0:3]
color = all_data[:, 3::]
# print(color)

points = Points(cordi)

plane = Plane.best_fit(points)
print(type(plane.point), plane.point)

point_projected = plane.project_point(cordi[0])
print(cordi[0])
print(len(cordi))

cordi_after = np.zeros_like(cordi)

for i in range(len(cordi)):
    point_tem = plane.project_point(cordi[i])
    cordi_after[i] = np.array(point_tem)

# print(cordi_after)

points_after = Points(cordi_after)

# print(points)
# print(type(points))

plot_3d(
    points_after.plotter(c='k', s=5, depthshade=False),
    plane.plotter(alpha=0.2, lims_x=(-1, 1), lims_y=(-1, 1)),
)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(cordi[:, 0], cordi[:, 1], cordi[:, 2])
plt.show()


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


def fit_plane(input_data, dis_sigma=1.5, depth_approx=1000, loop_time=2):
    (m, n) = input_data.shape
    j_count = 0
    inner_total_pre = 0
    best_param = [0, 0, 0]
    row_rand_array = np.arange(m)

    a_, b_ = [2, 1, 3], [1, 1, 1]

    q_before_, q_after_ = rotation(a_, b_)
    in_data_tra = np.hstack((np.zeros((input_data.shape[0], 1)), input_data))
    in_data_rota = quaternion_mal(q_before_, quaternion_mal(in_data_tra.T, q_after_))
    input_data = np.delete(in_data_rota.T, 0, axis=1)

    while j_count <= loop_time:
        i_ = 0

        while i_ <= m:
            index_ = np.random.choice(row_rand_array, 3, replace=False)
            picked_points = input_data[index_]

            param_ = np.linalg.solve(picked_points, -depth_approx*np.ones(picked_points.shape[0]))

            points_dis = np.abs(np.dot(input_data, param_) + depth_approx)/np.sqrt(np.sum(param_**2))
            total = np.sum(points_dis <= dis_sigma)

            if total > inner_total_pre:
                inner_total_pre = total
                best_param = param_
            i_ += 1

        j_count += 1

    q_before_pa, q_after_pa = rotation(b_, a_)
    best_param_be = np.hstack((np.zeros((1, 1)), best_param.reshape([1, -1])))
    para_rota = quaternion_mal(q_before_pa, quaternion_mal(best_param_be.T, q_after_pa))
    best_param_ = np.delete(para_rota.T, 0, axis=1).flatten()

    print("Calculated normal vector is: " + str(best_param_))

    return best_param_


def sampling_point(data_):
    (m, n) = data_.shape

    row_rand_array = np.arange(m)
    index_ = np.random.choice(row_rand_array, int(m*0.05), replace=False)

    sampling_set = data_[index_]

    return sampling_set
