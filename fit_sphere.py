import pymeshlab
from scipy.optimize import least_squares
import numpy as np

# USEFUL HINTS:
# https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#specify-a-sphere-by-multiple-control-points

plotting = True

ms = pymeshlab.MeshSet()
ms.load_new_mesh('femur_head_r.stl')
m = ms.current_mesh()
points = m.vertex_matrix()

# initial guesses
center0 = np.mean(points, axis=0)
radius0 = np.linalg.norm(np.min(points,axis=0)-np.max(points,axis=0), ord=2) * 0.5 # diameter/2
init = np.hstack((center0, radius0)) # [1,4]
lb = np.ones(4) * -np.inf
ub = np.ones(4) * +np.inf

def calcSphereResiduals(parameters):
	values = parameters[:3]
	radius = parameters[3]
	dist = np.linalg.norm(points-values, ord=2, axis=1) # distance between points and estimated center
	return dist - radius	

out = least_squares(calcSphereResiduals, init, bounds=(lb,ub), method='trf', jac='3-point')['x']
center = out[:3]
radius = out[3]
print('center:', center)
print('radius:', radius)

if plotting:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('TKAgg')
    from skspatial.objects import Sphere, Points

    sphere = Sphere(center, radius)
    pointData = Points(points)
    center = np.round(center, 3)
    radius = np.round(radius, 3)

    plt.close('all')
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(projection='3d')
    pointData.plot_3d(ax, c='b', s=1, depthshade=False)
    sphere.plot_3d(ax, alpha=0.2)
	sphere.point.plot_3d(ax, s=0.1)
    sphere.point.plot_3d(ax, s=100)
    plt.title(f'Sphere\ncenter: {center[0]}, {center[1]}, {center[2]}\nradius: {radius}')
    plt.savefig('fittedSphere.png')
    plt.show(block=False)