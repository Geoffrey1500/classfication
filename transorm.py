from skspatial.transformation import transform_coordinates

points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
vectors_basis = [[1, 0, 0], [-1, 1, 0]]

a = transform_coordinates(points, [0, 0, 0], vectors_basis)
print(a)
