import commonse
from math import pi
import numpy as np
# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi
TURBULENCE_CLASS = commonse.enum.Enum('A B C')
TURBINE_CLASS = commonse.enum.Enum('I II III')
DRIVETRAIN_TYPE = commonse.enum.Enum('geared single_stage multi_drive pm_direct_drive')
r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
	           0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
	           0.97777724])  # (Array): new aerodynamic grid on unit radius
r_str = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
	          0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
	          0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
	          0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
	          0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
	          1.0])  # (Array): initial structural grid on unit radius
 
