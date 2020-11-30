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
