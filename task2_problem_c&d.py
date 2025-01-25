#!/usr/bin/env python

from collections import Counter
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def surface1(x, y):
	return 2*x**2 + 2*y**2

def surface2(x, y):
	return 2*np.exp(-x**2 - y**2)

x = np.linspace(-1, 1, 51)
y = np.linspace(-1, 1, 51)
z = np.linspace(0, 2, 21)
x, y, z = np.meshgrid(x, y, z)
indices = np.logical_and(surface1(x, y) <= z, z <= surface2(x, y))
x = x[indices]
y = y[indices]
z = z[indices]
tetrahedra = Delaunay(np.c_[x, y, z]).simplices

face_counter = Counter()
for vertices in tetrahedra:
	i1, i2, i3, i4 = sorted(vertices)
	face_counter.update([(i1, i2, i3), (i1, i2, i4), (i1, i3, i4), (i2, i3, i4)])
boundary_faces = [face for face, count in face_counter.items() if count == 1]

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(x, y, z, triangles=boundary_faces, cmap=cm.viridis)
plt.show()