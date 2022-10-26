import pymeshlab
from scipy.optimize import least_squares
import numpy as np

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
    perp = np.cross(v1,v2)
    closetstDist = np.linalg.norm(perp, axis=1)/ np.linalg.norm(v1)
    return closetstDist - radius

out = least_squares(cylinderResiduals, init, bounds=(lb,ub), method='trf', jac='3-point')['x']

radius = out[-1]

def projectPointOnLine(a, b, p):
    ap = p - a
    ab = b - a
    return a + np.dot(ap, ab) / np.dot(ab, ab) * ab

center = projectPointOnLine(out[:3], out[3:6], center0)
p4 = projectPointOnLine(out[:3], out[3:6], endPointMin)
p5 = projectPointOnLine(out[:3], out[3:6], endPointMax)

with open('nodes.txt', mode='w') as f:
    # f.write(f'p1,{out[0]},{out[1]},{out[2]}\n')
    # f.write(f'p2,{out[3]},{out[4]},{out[5]}\n')
    f.write(f'center,{center[0]},{center[1]},{center[2]}\n')
    f.write(f'endPoint1,{p4[0]},{p4[1]},{p4[2]}\n')
    f.write(f'endPoint2,{p5[0]},{p5[1]},{p5[2]}\n')

if plot:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('TKAgg')
    from skspatial.objects import Cylinder, Point, Points

    cylinder = Cylinder(p4, p5-p4, radius)
    point = Point(center)
    pointData = Points(points)

    plt.close('all')
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(projection='3d')
    pointData.plot_3d(ax, c='b', s=1, depthshade=False)
    point.plot_3d(ax, s=100)
    cylinder.plot_3d(ax, alpha=0.2)
    cylinder.point.plot_3d(ax, s=100)
    plt.title(f'Cylinder\ncenter: {round(center[0],3)}, {round(center[1],3)}, {round(center[2],3)}\npoint: : {round(p5[0],3)}, {round(p5[1],3)}, {round(p5[2],3)}\nradius: {round(radius,3)}')
    plt.savefig('fittedCylinder.png')
    plt.show(block=False)
# %%





# def cylinderFitting(xyz,p,th):

#     """
#     This is a fitting for a vertical cylinder fitting
#     Reference:
#     http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf

#     xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
#     p is initial values of the parameter;
#     p[0] = Xc, x coordinate of the cylinder centre
#     P[1] = Yc, y coordinate of the cylinder centre
#     P[2] = alpha, rotation angle (radian) about the x-axis
#     P[3] = beta, rotation angle (radian) about the y-axis
#     P[4] = r, radius of the cylinder
#     th, threshold for the convergence of the least squares

#     """   
#     x = xyz[:,0]
#     y = xyz[:,1]
#     z = xyz[:,2]

#     fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
#     errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

#     est_p , success = leastsq(errfunc, p, args=(x, y, z), maxfev=10000)

#     return est_p

# np.set_printoptions(suppress=True) 
# p = np.array([0,0,0,0,0.3])
# est_p =  cylinderFitting(xyz,p,0.00001)








# import numpy as np
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.linalg import norm

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# origin = np.array([0, 0, 0])
# #axis and radius
# p0 = np.array([1, 3, 2])
# p1 = np.array([8, 5, 9])
# R = 2.5

# #vector in direction of axis
# v = p1 - p0
# #find magnitude of vector
# mag = norm(v)
# #unit vector in direction of axis
# v = v / mag
# #make some vector not in the same direction as v
# not_v = np.array([1, 0, 0])
# if (v == not_v).all():
#     not_v = np.array([0, 1, 0])
# #make vector perpendicular to v
# n1 = np.cross(v, not_v)
# #normalize n1
# n1 /= norm(n1)
# #make unit vector perpendicular to v and n1
# n2 = np.cross(v, n1)
# #surface ranges over t from 0 to length of axis and 0 to 2*pi
# t = np.linspace(0, mag, 100)
# theta = np.linspace(0, 2 * np.pi, 100)
# #use meshgrid to make 2d arrays
# t, theta = np.meshgrid(t, theta)
# #generate coordinates for surface
# X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
# ax.plot_surface(X, Y, Z)
# #plot axis
# ax.plot(*zip(p0, p1), color = 'red')
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_zlim(0, 10)
# plt.show()