import pyXLIGHT
from airfoilprep_free import cst_to_coordinates
import os
import numpy as np
CST = [-0.17200255338600826, -0.13744743777735921, -0.24288986290945222, 0.15085289615063024, 0.20650016452789369, 0.35540642522188848, 0.32797634888819488, 0.2592276816645861]

[x, y] = cst_to_coordinates(CST)
# basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
# airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates.dat'
# coord_file = open(airfoil_shape_file, 'w')
# print >> coord_file, 'CST'
# for i in range(len(x)):
#     print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])
# coord_file.close()
airfoil_shape_file = None
Re = 1e6
airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
airfoil.re = Re
airfoil.mach = 0.00
airfoil.iter = 100

alphas = np.linspace(-15,15,100)
import time
cl = np.zeros(len(alphas))
cd = np.zeros(len(alphas))
cm = np.zeros(len(alphas))
time1 = np.zeros(len(alphas))
to_delete = np.zeros(0)
for j in range(len(alphas)):
    time0 = time.time()
    cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(alphas[j])
    time1[j] = time.time() - time0

airfoil.re = 5e5
airfoil.mach = 0.00
airfoil.iter = 100

alphas = np.linspace(-15,15,100)
import time
cl1 = np.zeros(len(alphas))
cd1 = np.zeros(len(alphas))
cm1 = np.zeros(len(alphas))
time2 = np.zeros(len(alphas))
to_delete = np.zeros(0)
for j in range(len(alphas)):
    time0 = time.time()
    cl1[j], cd1[j], cm1[j], lexitflag = airfoil.solveAlpha(alphas[j])
    time2[j] = time.time() - time0

import matplotlib.pylab as plt
plt.figure()
plt.plot(alphas, cl, label='RE=1e5')
plt.plot(alphas, cl1, label='RE=1e6')
plt.legend(loc='best')

plt.figure()
plt.plot(alphas, cd, label='RE=1e5')
plt.plot(alphas, cd1, label='RE=1e6')
plt.legend(loc='best')

plt.figure()
plt.plot(alphas, time1, label='RE=1e5')
plt.plot(alphas, time2, label='RE=1e6')
plt.legend(loc='best')
plt.show()

# print "CL", cl
# print "CD", cd
# print "TIME", time1
# print