import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# Define the two surface functions
def surface1(x, y):
    return 2 * x**2 + 2 * y**2

def surface2(x, y):
    return 2 * np.exp(-x**2 - y**2)


x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
x, y = np.meshgrid(x, y)


z1 = surface1(x, y)
z2 = surface2(x, y)


points_surface1 = np.c_[x.ravel(), y.ravel(), z1.ravel()]
points_surface2 = np.c_[x.ravel(), y.ravel(), z2.ravel()]


points = np.vstack((points_surface1, points_surface2))

tri = Delaunay(points[:, :2]) 


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(x, y, z1, alpha=0.5, color='blue', edgecolor='none')


ax.plot_surface(x, y, z2, alpha=0.5, color='green', edgecolor='none')


ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='green', s=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Closed Surface Point Cloud and Delaunay Triangulation')

plt.show()
