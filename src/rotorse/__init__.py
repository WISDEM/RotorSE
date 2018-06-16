import commonse
from math import pi
import numpy as np
# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi
TURBULENCE_CLASS = commonse.enum.Enum('A B C')
TURBINE_CLASS = commonse.enum.Enum('I II III')
DRIVETRAIN_TYPE = commonse.enum.Enum('geared single_stage multi_drive pm_direct_drive')
REFERENCE_TURBINE = commonse.enum.Enum('5MW 10MW')
from reference_turbine import NAERO, NSTR
