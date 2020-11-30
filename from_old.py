import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats as sts


def import_data(name_):
    data_ = np.loadtxt("./data/" + name_ + ".txt", dtype=np.float32)*1000
    return data_


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


def loa_predict(data_input, normal_vector, measure_time=2, d_0=3.5, divergence=0.00023,):
    points_cor = sampling_point(data_input)
    normal_vector_re = normal_vector.reshape(-1, 1)
    distance_1 = np.linalg.norm(points_cor, axis=1, keepdims=True)
    distance_ = np.linalg.norm(points_cor, axis=1, keepdims=True) + (d_0 / 2) / np.tan(divergence / 2)
    incidence_angle_ = np.arccos(
        np.dot(points_cor, normal_vector_re) / (np.linalg.norm(points_cor, axis=1, keepdims=True) *
                                                np.linalg.norm(normal_vector)))

    for i in np.arange(len(incidence_angle_)):
        if incidence_angle_[i] > np.pi / 2:
            incidence_angle_[i] = np.pi - incidence_angle_[i]

    s_s = distance_1 * np.cos(incidence_angle_)
    print(np.average(s_s))

    d_1 = 2 * distance_ * (np.sin(divergence) * np.cos(incidence_angle_)) / (
                np.cos(2 * incidence_angle_) + np.cos(divergence))
    d_2 = np.sqrt(1 - (np.sin(incidence_angle_) / np.cos(divergence / 2)) ** 2) * d_1

    d_set = np.hstack((d_1, d_2))
    d_major = d_set.max(axis=1) / 2
    d_minor = d_set.min(axis=1) / 2
    print(np.average(d_major))

    r_equal = np.sqrt(d_major * d_minor)
    r_equal_re = r_equal.reshape(-1, 1)
    # std_pre = 0.5 * r_equal_re * np.cos(incidence_angle_) / (np.sqrt(measure_time)*np.sin(incidence_angle_))
    std_pre = 0.25*np.cos(incidence_angle_) * (np.sqrt((s_s * np.tan(incidence_angle_) + r_equal_re) ** 2 + s_s ** 2) -
                   np.sqrt((s_s * np.tan(incidence_angle_) - r_equal_re) ** 2 + s_s ** 2))/np.sqrt(measure_time)

    std_pre_aver = np.average(std_pre)

    print("Predict LoA is: " + str(std_pre_aver) + " mm")
    print(np.average(incidence_angle_/np.pi*180))

    return std_pre


def lod_prediction(data_, param_, angular_resolution, depth_approx=1000):
    angular_resolution = angular_resolution*np.pi/180

    sampling_points_cor = sampling_point(data_)

    x_range = np.max(data_[:, 1]) - np.min(data_[:, 1])
    y_range = np.max(data_[:, 0]) - np.min(data_[:, 0])
    print('x_range is {:.2f} mm, y_range is {:.2f} mm'.format(x_range, y_range))
    approximate_length = np.sqrt(x_range**2 + y_range**2)

    dis_to_surface = abs(depth_approx)/np.sqrt(np.sum(param_**2))
    alpha_cos = dis_to_surface/np.sqrt(np.sum(sampling_points_cor**2, axis=1))
    spacing_hor = np.average(dis_to_surface*angular_resolution/alpha_cos**2)

    pre_number_of_scan_lines = approximate_length/spacing_hor
    min_size = data_.shape[0]/pre_number_of_scan_lines*0.55

    print("Predict LoD is: {:.2f} mm".format(spacing_hor))
    print("Predict number of scan lines is: {:.0f}".format(np.ceil(pre_number_of_scan_lines)))

    return spacing_hor, pre_number_of_scan_lines, min_size


def loa_analysis(data_, param_, depth_approx=1000):
    dis_info = (np.dot(data_, param_) + depth_approx)/np.sqrt(np.sum(param_**2))
    print("Measured LoA is: " + str(np.std(dis_info)) + " mm")

    return dis_info


def dis_visualize(name_, dis_):
    sns.distplot(dis_, bins=20, hist=True, kde=True, norm_hist=False,
                 rug=True, vertical=False,
                 color='b', label=name_, axlabel='Distance to surface')

    plt.legend()
    plt.show()


def lod_analysis(in_data, size=20):
    in_data_pd = pd.DataFrame(in_data, columns=['x', 'y', 'z'])

    plt.figure()
    sns.scatterplot(in_data_pd['x'], in_data_pd['y'])
    plt.figure()
    sns.scatterplot(in_data_pd['x'], in_data_pd['z'])

    in_data_pd_2 = in_data_pd.drop(['y', 'z'], axis=1)
    in_data_pd_2 = in_data_pd_2[['x']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    in_data_cluster = hdbscan.HDBSCAN(min_cluster_size=size).fit(in_data_pd_2)
    cluster_index = in_data_cluster.labels_
    print("Measured number of scan lines is: " + str(max(cluster_index) + 1))

    scan_line_set = []
    for i in np.arange(max(cluster_index) + 1):
        scan_line_cor = in_data[cluster_index == i]
        x = np.average(scan_line_cor[:, 0])
        y = np.average(scan_line_cor[:, 1])
        z = np.average(scan_line_cor[:, 2])
        scan_line_set.append([x, y, z])

    scan_line_set = np.array(scan_line_set)
    spacing_set = scan_line_set[np.argsort(scan_line_set[:, 0])]

    spacing_a = np.delete(spacing_set, -1, axis=0)
    spacing_b = np.delete(spacing_set, 0, axis=0)

    spacing_combine = (spacing_a - spacing_b) ** 2
    spacing_result = np.sqrt(spacing_combine.sum(axis=1))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(scan_line_cor[:, 0], scan_line_cor[:, 1], scan_line_cor[:, 2], c='r')
    ax.scatter(in_data[:, 0], in_data[:, 1], in_data[:, 2])

    print("Measured LoD is: " + str(np.average(spacing_result)) + ' mm')

    plt.show()

    return np.average(spacing_result)


def lod_analysis_2(in_data, param_, support_axis='x', size=20):
    if support_axis == 'x':
        support_ = np.array([1, 0, 0])
        drop_ = ['z']
        column_ = ['y']
        indexx_ = 0
    elif support_axis == 'y':
        support_ = np.array([0, 1, 0])
        drop_ = ['z']
        column_ = ['x']
        indexx_ = 1
    else:
        support_ = np.array([0, 0, 1])
        drop_ = ['y']
        column_ = ['x']
        indexx_ = 1

    q_before_, q_after_ = rotation(param_, support_)
    in_data_tra = np.hstack((np.zeros((in_data.shape[0], 1)), in_data))
    in_data_rota = quaternion_mal(q_before_, quaternion_mal(in_data_tra.T, q_after_))
    in_data_cut = np.delete(in_data_rota.T, 0, axis=1)

    in_data_pd = pd.DataFrame(in_data_cut, columns=['x', 'y', 'z'])

    plt.figure()
    sns.scatterplot(in_data_pd[column_[0]], in_data_pd[drop_[0]])
    plt.figure()
    sns.scatterplot(in_data_pd[column_[0]], in_data_pd[support_axis])

    in_data_pd_2 = in_data_pd.drop([drop_[0]], axis=1)
    # print(in_data_pd_2, size)
    in_data_pd_2 = in_data_pd_2[[column_[0]]].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    in_data_cluster = hdbscan.HDBSCAN(min_cluster_size=size).fit(in_data_pd_2)
    cluster_index = in_data_cluster.labels_
    print("Measured number of scan lines is: {:.0f}".format(max(cluster_index)+1))

    scan_line_set = []
    for i in np.arange(max(cluster_index) + 1):
        scan_line_cor = in_data[cluster_index == i]
        x = np.average(scan_line_cor[:, 0])
        y = np.average(scan_line_cor[:, 1])
        z = np.average(scan_line_cor[:, 2])
        scan_line_set.append([x, y, z])

    scan_line_set = np.array(scan_line_set)
    # print(scan_line_set)
    spacing_set = scan_line_set[np.argsort(scan_line_set[:, indexx_])]

    spacing_a = np.delete(spacing_set, -1, axis=0)
    spacing_b = np.delete(spacing_set, 0, axis=0)

    spacing_combine = (spacing_a - spacing_b) ** 2
    spacing_result = np.sqrt(spacing_combine.sum(axis=1))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(scan_line_cor[:, 0], scan_line_cor[:, 1], scan_line_cor[:, 2], c='r')
    ax.scatter(in_data[:, 0], in_data[:, 1], in_data[:, 2])

    print("Measured LoD is: {:.2f} mm".format(np.average(spacing_result)))

    plt.show()

    return spacing_result


angular_resolution_set = [0, 0.286, 0.143, 0.072, 0.036, 0.018, 0.009, 0.004]
angular_index = [0, 1, 2, 3, 4, 5, 6, 7]
location_index = [0, 1, 2, 3, 4]
chosen = 3
print("The selected resolution is {} degree".format(angular_resolution_set[chosen]))

test_data = import_data('BCT' + "_" + '2_2')

cut_data_de = np.delete(test_data, -1, axis=1)
depth_about = np.average(np.sqrt(np.sum(cut_data_de ** 2, axis=1)))
print("The approximated scanning depth is {:.2f} mm".format(depth_about))

param = fit_plane(test_data, depth_approx=depth_about)
lod_pre = lod_prediction(test_data, param, angular_resolution_set[chosen], depth_approx=depth_about)

spacing_set = lod_analysis_2(test_data, param, support_axis='y', size=int(lod_pre[-1]))
# np.save("BCT_2_2.npy", spacing_set)
# print(spacing_set.shape)
# df_spacing = pd.DataFrame(spacing_set)
# print(df_spacing.head(10))
# box = plt.figure()
# sns.boxplot(data=df_spacing)
# plt.show()
