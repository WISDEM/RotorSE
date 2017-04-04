import numpy as np
import matplotlib.pyplot as plt

# Inputs from bryce_test
rhub = 1.5375
rtip = 63.0375

# Calculated in Call_FAST
rnodes = np.array([  2.90417499,   5.63750005,   9.05415008,  12.98332496,  16.3999999,   19.9875,
  24.08750005,  28.18749995,  32.2875,      36.38750005,  39.56499995,
  41.71749995,  45.71499995,  51.59167488,  56.03334987,  58.93749995,
  61.67082501]);

# r_aero from bryce_test
rnodes = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
    0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
    0.97777724])*rtip

# Calculated in Call_FAST
drnodes = np.array([ 2.73334997,  2.73330016,  4.0999999,   3.75834987,  3.075,       4.10000021,
  4.0999999,   4.0999999,   4.10000021,  4.0999999,   2.2549999,   2.0500001,
  5.9449999,   5.80834997,  3.075,       2.73330016,  2.73334997])

boundary = np.zeros([16,1])
for i in range(0,16):
    boundary[i] = rhub + np.sum(drnodes[0:i+1])

# changed_rnodes = np.array([3.34632353, 6.96397059, 10.58161765,
#                            14.19926471, 17.81691176, 21.43455882, 25.05220588, 28.66985294, 32.2875,
#                            35.90514706, 39.52279412, 43.14044118, 46.75808824, 50.37573529,
#                            53.99338235, 57.61102941, 61.22867647])
#
# changed_drnodes = np.array([ 3.61764706,  3.61764706,  3.61764706,  3.61764706,  3.61764704,  3.61764708,
#   3.61764704,  3.61764708,  3.61764704,  3.61764708,  3.61764704,  3.61764708,
#   3.61764704,  3.61764706,  3.61764706,  3.61764706,  3.61764706])

# changed_boundary = np.zeros([16,1])
# for i in range(0,16):
#     changed_boundary[i] = rhub + np.sum(changed_drnodes[0:i+1])

#plot blade tip deflection
plt.figure()
plt.plot(rhub,1,'x')
plt.plot(rtip,1,'.')
n, = plt.plot(rnodes,np.ones(np.size(rnodes)),'*', label= 'node position')
b, = plt.plot(boundary,np.ones(np.size(boundary)),'|', label = 'calculated boundary')
plt.legend()
plt.ylim([0.95,1.05])

plt.show()

# RNodes comparison
# plt.figure()
# plt.plot(rhub,1,'x')
# plt.plot(rtip,1,'.')
# plt.plot(rhub,0.95,'x')
# plt.plot(rtip,0.95,'.')
# n, = plt.plot(rnodes,np.ones(np.size(rnodes)),'*', label= 'node position')
# b, = plt.plot(boundary,np.ones(np.size(boundary)),'|', label = 'calculated boundary')
# nc, = plt.plot(changed_rnodes,0.95*np.ones(np.size(rnodes)),'*', label= 'changed node position')
# bc, = plt.plot(changed_boundary,0.95*np.ones(np.size(boundary)),'|', label = 'changed calculated boundary')
#
# plt.legend()
# plt.ylim([0.90,1.05])
#
# plt.show()