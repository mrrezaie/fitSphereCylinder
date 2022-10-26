import pymeshlab
from scipy.optimize import least_squares
import numpy as np

# USEFUL HINTS:
# https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
# https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points

plot = True

ms = pymeshlab.MeshSet()
ms.load_new_mesh('epicondyle_r.stl')
m = ms.current_mesh()
points = m.vertex_matrix()

endPointMin = np.min(points,axis=0)
endPointMax = np.max(points,axis=0)
center0 = np.mean((endPointMax,endPointMin),axis=0)
radius0 = np.linalg.norm(endPointMax-endPointMin, ord=2) * 0.5 # diameter/2
init = np.hstack((endPointMin, endPointMax, radius0)) # [1,7]
lb = np.ones(7) * -np.inf
ub = np.ones(7) * +np.inf

def cylinderResiduals(parameters):
    endPoint1 = parameters[0:3]
    endPoint2 = parameters[3:6]
    radius = parameters[6]

    v1 = endPoint1-endPoint2 # (vector) axis of the cylinder
    v2 = points-endPoint1 # vector between points and the cylinder axis
    v3 = np.cross(v1,v2) # perpendicular vector
    closetstDist = np.linalg.norm(v3, axis=1)/ np.linalg.norm(v1)
    return closetstDist - radius

out = least_squares(cylinderResiduals, init, bounds=(lb,ub), method='trf', jac='3-point')['x']

radius = out[-1]

def projectPointOnLine(a, b, p):
    ap = p - a
    ab = b - a
    return a + np.dot(ap, ab) / np.dot(ab, ab) * ab

center = projectPointOnLine(out[:3], out[3:6], center0)
p1 = projectPointOnLine(out[:3], out[3:6], endPointMin)
p2 = projectPointOnLine(out[:3], out[3:6], endPointMax)

with open('nodes.txt', mode='w') as f:
    f.write(f'center,{center[0]},{center[1]},{center[2]}\n')
    f.write(f'endPoint1,{p1[0]},{p1[1]},{p1[2]}\n')
    f.write(f'endPoint2,{p2[0]},{p2[1]},{p2[2]}\n')

if plot:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('TKAgg')
    from skspatial.objects import Cylinder, Point, Points

    cylinder = Cylinder(p1, p2-p1, radius)
    point = Point(center)
    pointData = Points(points)
    center = np.round(center, 3)
    radius = np.round(radius, 3)
    p2     = np.round(p2    , 3)

    plt.close('all')
    fig = plt.figure(figsize=(), tight_layout=True)
    ax = fig.add_subplot(projection='3d')
    pointData.plot_3d(ax, c='b', s=1, depthshade=False)
    point.plot_3d(ax, s=100)
    cylinder.plot_3d(ax, alpha=0.2)
    cylinder.point.plot_3d(ax, s=100)
    plt.title(f'Cylinder\ncenter: {center[0]}, {center[1]}, {center[2]}\npoint: : {p2[0]}, {p2[1]}, {p2[2]}\nradius: {radius}')
    plt.savefig('fittedCylinder.png')
    plt.show(block=False)