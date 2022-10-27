import pymeshlab
from scipy.optimize import least_squares, minimize
import numpy as np
from scipy.spatial.distance import cdist

# USEFUL LINKS:
# https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
# https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
# https://stackoverflow.com/questions/31667070/max-distance-between-2-points-in-a-data-set-and-identifying-the-points

plot = True

ms = pymeshlab.MeshSet()
# ms.load_new_mesh('talartrochlea.stl')
ms.load_new_mesh('epicondyle_r.stl')
m = ms.current_mesh()
points = m.vertex_matrix()

row,_ = np.shape(points)

endPoint10 = np.min(points,axis=0)
endPoint20 = np.max(points,axis=0)
radius0 = np.linalg.norm(endPoint20-endPoint10, ord=2)# diameter/2

init = np.hstack((endPoint10, endPoint20, radius0)) # [1,7]
# lb = np.ones(7) * -np.inf #  no need to define bounds
# ub = np.ones(7) * +np.inf


def projectPointOnLine(a, b, p):
    ap = p - a
    ab = b - a
    return a + np.dot(ap, ab) / np.dot(ab, ab) * ab

def cylinderResiduals(parameters):
    p1 = parameters[0:3] # first 3D point
    p2 = parameters[3:6] # second 3D point
    radius = parameters[6]
    v1 = p1-p2 # 1D (vector) axis of the cylinder
    v2 = points-p1 # 2D vector between points and the cylinder axis
    v3 = np.cross(v1,v2) #
    dist = np.linalg.norm(v3, axis=1)/ np.linalg.norm(v1) # perpendicular distance to axis
    return np.sum((dist-radius)**2) # sum of squared diferences
    # return dist-radius

out = minimize(cylinderResiduals, init)['x'] #
# out = least_squares(cylinderResiduals, init)['x'] # , ftol=1e-8, method='trf', jac='3-point'

p1 = out[:3]
p2 = out[3:6]
radius = out[6]

# project all points to the cylinder axis
points2 = np.empty_like(points)
for i in range(points.shape[0]):
    points2[i,:] = projectPointOnLine(p1, p2, points[i,:])

# using distance matrix to find the end points (slow solution)
# a = np.where(squareform(pdist(points2)) == squareform(pdist(points2)).max())[0]
a = np.where(cdist(points2,points2) == cdist(points2,points2).max())[0]
endPoint1 = points2[a[0],:]
endPoint2 = points2[a[1],:]
center = np.mean((endPoint1,endPoint2), axis=0)


# dist = np.empty(row)
# for i in range(row):
#     dist[i] = np.linalg.norm(center-points2[i,:])
# endPoint = points2[np.argmax(dist),:]

# v = p1-p2 # 1D (vector) axis of the cylinder
# u = v/np.linalg.norm(v)
# R.from_rotvec(u)



# with open('nodes.txt', mode='w') as f:
#     f.write(f'center,{center[0]},{center[1]},{center[2]}\n')
#     f.write(f'endPoint1,{endPoint1[0]},{endPoint1[1]},{endPoint1[2]}\n')
#     f.write(f'endPoint2,{endPoint2[0]},{endPoint2[1]},{endPoint2[2]}\n')

if plot:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('TKAgg')
    from skspatial.objects import Cylinder, Point, Points

    cylinder = Cylinder(endPoint1, endPoint2-endPoint1, radius)
    center = np.round(center, 3)
    radius = np.round(radius, 3)
    endPoint1 = np.round(endPoint1, 3)
    endPoint2 = np.round(endPoint2, 3)

    plt.close('all')
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(projection='3d')
    # ax.view_init=(60, 60)
    Points(points).plot_3d(ax, c='b', s=1, depthshade=False)
    # Points(points2).plot_3d(ax, c='r', s=1, depthshade=False)

    Point(center).plot_3d(ax, s=100)
    # Point(endPoint1).plot_3d(ax, s=100)
    # Point(endPoint2).plot_3d(ax, s=100)

    cylinder.plot_3d(ax, alpha=0.2)
    # cylinder.point.plot_3d(ax, s=100)
    plt.title(f'Cylinder\ncenter: {center[0]}, {center[1]}, {center[2]}\nendpoint: {endPoint1[0]}, {endPoint1[1]}, {endPoint1[2]}\nradius: {radius}')
    plt.savefig('fittedCylinder.png')
    plt.show(block=False)