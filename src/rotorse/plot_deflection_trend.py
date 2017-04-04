import numpy as np
import matplotlib.pyplot as plt

windspeed = np.linspace(8,16,9)

deflection = [0.06314, 0.09494,0.1219,0.1477,0.1765,0.2065,0.2344,0.2653,0.2953]
deflection = np.array(deflection)

cycle_time = [4.31607-3.20243,4.09799-2.99712,3.85162-2.74714,3.64891-2.55223,4.5739-3.47348,
              3.27251-2.17461,3.1-2.02422,3.69769-2.69917,5.69646-4.84798]
cycle_time = np.array(cycle_time)

#plot blade tip deflection
plt.figure()
plt.plot(windspeed,deflection, '*--')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Blade Tip Deflection (m)')
plt.title('Deflection Trend')
plt.show()

plt.figure()
plt.plot(windspeed,cycle_time, '*--')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Cycle Time (s)')
plt.title('Frequency Trend')
plt.show()
