
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points, snap
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline, splev, splrep, LSQUnivariateSpline
from scipy.optimize import curve_fit
from scipy.spatial import Voronoi, voronoi_plot_2d
from utils import strictly_decreasing, strictly_increasing
from utils import non_decreasing, non_increasing, monotonic

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
import math



point_l13 = (1.19,-0.3)
point_c13 = (1.29, -0.3)
point_r13 = (1.39, -0.3)

point_l12 = (1.2,-0.2)
point_c12 = (1.3, -0.2)
point_r12 = (1.4, -0.2)

point_l11 = (1.21,-0.1)
point_c11 = (1.31,0.0)
point_r11 = (1.41,0.1)

point_l21 = (1.6,0.7)
point_c21= (1.6,0.8)
point_r21 = (1.6,0.9)

point_l22 = (1.7,0.7)
point_c22= (1.7,0.8)
point_r22 = (1.7,0.9)

point_l23 = (1.72,0.7)
point_c23= (1.72,0.8)
point_r23 = (1.72,0.9)


# point_l31 = (1.1,1.05)
# point_c31= (1.2,1.04)
# point_r31 = (1.3,1.03)

# point_l32 = (1.1,1.15)
# point_c32= (1.2,1.14)
# point_r32 = (1.3,1.13)



points_final = [point_l11, point_c11, point_r11,point_l21, point_c21, point_r21]
points = [point_l13, point_c13, point_r13 ,point_l12, point_c12, point_r12, point_l11, point_c11, point_r11, point_l21, point_c21, point_r21, point_l22, point_c22, point_r22, point_l23, point_c23, point_r23]
polygon = Polygon([(1,0), (1.5,0), (1.51,0.5), (1.59,0.5), (1.58,1), (1,1)])

polygon_corrected = polygon.exterior.coords[:]
[polygon_corrected.append(point) for point in points_final]


listx = [point[0] for point in polygon_corrected]
listy = [point[1] for point in polygon_corrected]

start_point = listx[0], listy[0]
sorted_points = []

while len(start_point)>0:
    sorted_points.append(start_point)
    x1, y1 = start_point
    dists = {(x2, y2): np.sqrt((x1-x2)**2 + (y1-y2)**2) for x2, y2 in zip(listx, listy)}
    dists = sorted(dists.items(), key=lambda item: item[1])
    for dist in dists:
        if dist[0] not in sorted_points: 
            start_point = dist[0]
            break
        if dist == dists[-1]:
            start_point = ()
            break

# vor = Voronoi(polygon_corrected)
# voronoi_plot_2d(vor)
polygon_corrected = Polygon(sorted_points)
# polygon_corrected = Polygon(MultiPoint(polygon_corrected).convex_hull)

for point in points:
    point = Point(point)
    plt.scatter(*point.xy, color='b')
    #print(point.intersects(polygon_corrected))


point1 = (1.4,0.6)
point2 = (1.75,0.7)
point3 = (2.0,0.72)
#plt.scatter(point2[0], point2[1])
#plt.scatter(point1[0], point1[1])
#plt.scatter(point3[0], point3[1])
# print(point1.within(polygon_corrected))


guiding_points = [point_c13, point_c12, point_c11, point1, point_c21, point_c22, point_c23]
guiding_points = [point_c13, point_c12, point_c11, point_c21, point_c22, point_c23]
# guiding_points = [point_c12, point_c11, point_c21, point_c22]

print(monotonic(guiding_points), strictly_decreasing(guiding_points), strictly_increasing(guiding_points))

connection_line = LineString(guiding_points)

x_data = []
y_data = []

for point in guiding_points:
    x_data.append(point[0])
    y_data.append(point[1])

x_data = np.array(x_data)
y_data = np.array(y_data)

# order = np.argsort(x_data)
# x = x_data[order]
# y = y_data[order]
#y = y + 0.2 * np.random.normal(size=len(x))

# t = [1.5]
# spl = LSQUnivariateSpline(x_data, y_data, t)


def interpolation(x, y):

    points = np.array([x, y]).T

    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]

    alpha = np.linspace(0, 1, 75)

    interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
    interpolated_points = interpolator(alpha)

    out_x = interpolated_points.T[0]
    out_y = interpolated_points.T[1]

    return out_x, out_y

xt, yt = interpolation(x_data, y_data)

plt.plot(xt, yt)

# def fit_func(x, a, b, c):
#     return a*x**2 + b*x + c

# params, cov = curve_fit(fit_func, x, y)

# xnew = np.linspace(np.min(x_data), np.max(x_data), 100)
# ynew = spl(xnew)
#plt.plot(xnew, fit_func(xnew, *params))
# plt.plot(xnew, ynew)

# print(xnew, ynew)

# # creating the spline object
# spline = splrep(x_data,y_data,k=2, s=0.03)

# x_range = np.linspace(np.min(x_data), np.max(x_data), 100)

# new_spline = splev(x_range, spline)
# plt.plot(x_range, new_spline)

            
plt.plot(*polygon.exterior.xy, color='b', alpha=0.2)
#plt.plot(*polygon_corrected.exterior.xy, color='b')
#plt.plot(*connection_line.xy)

plt.show()

        