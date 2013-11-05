#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from scipy import interpolate
from math import pi
from openmdao.main.api import VariableTree, Component, Assembly, Driver
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Str, List, Slot
from openmdao.util.decorators import add_delegate
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasobjective import HasObjectives
# from openmdao.lib.drivers.api import SLSQPdriver
# from pyopt_driver.pyopt_driver import pyOptDriver
from zope.interface import implements, Interface

from wisdem.common.utilities import RPM2RS, RS2RPM
from external.ccblade import CCAirfoil, CCBlade as CCBlade_PY


# -----------------
#  Helper Classes
# -----------------

class ProbDist(Interface):

    def CDF(x):
        """cumulative distribution function"""
        pass

    def PDF(x):
        """probability distribution function"""
        pass


class Weibull:
    implements(ProbDist)

    def __init__(self, A, k):
        """Weibull probability distribution

        Arguments
        ---------
        A : float (m/s)
            scale factor
        k : float
            shape or form factor

        """
        self.A = A
        self.k = k


    def CDF(self, U):
        return 1.0 - np.exp(-(U/self.A)**self.k)

    def PDF(self, U):
        k = self.k
        A = self.A
        return k/A*(U/A)**(k-1)*np.exp(-(U/A)**k)



class Rayleigh:
    implements(ProbDist)

    def __init__(self, Ubar):
        """Rayleigh probability distribution

        Arguments
        ---------
        Ubar : float (m/s)
            mean wind speed of distribution

        """
        A = 2.0*Ubar/pi**0.5
        k = 2
        self.prob = Weibull(A, k)

        # self.Ubar = Ubar
        # self.sigma = Ubar*(2/pi)**0.5


    def CDF(self, U):
        return self.prob.CDF(U)
        # A = 2.0*self.Ubar/pi**0.5
        # k = 2
        # return 1.0 - np.exp(-(U/A)**k)

    def PDF(self, U):
        return self.prob.PDF(U)
        # return U/self.sigma**2 * np.exp(-U**2/2/self.sigma**2)



# TODO: move to CommonSE
def run_workflow(driver, x):
    """manually run a workflow"""

    driver.set_parameters(x)
    driver.run_iteration()
    return driver.eval_objectives()


# TODO: delete once OpenMDAO adds this capability in stable branch
@add_delegate(HasParameters, HasObjectives)
class TempDriver(Driver):

    def add_parameters(self, name, len, assembly):

        assembly.set(name, np.zeros(len))

        for i in range(len):
            self.add_parameter(name + '[%s]' % i, low=-1e9, high=1e9)


    def add_objectives(self, name, len):

        for i in range(len):
            self.add_objective(name + '[%s]' % i)



# -----------------
#  Variable Trees
# -----------------


class MachineTypeBase(VariableTree):
    """not meant to be instantiated directly"""

    Vin = Float(units='m/s', desc='cut-in wind speed')
    Vout = Float(units='m/s', desc='cut-out wind speed')
    ratedPower = Float(units='W', desc='rated power')


class FixedSpeedFixedPitch(MachineTypeBase):

    Omega = Float(units='rpm', desc='fixed rotor rotation speed')
    pitch = Float(units='deg', desc='fixed blade pitch setting')

    varSpeed = False
    varPitch = False


class FixedSpeedVarPitch(MachineTypeBase):

    Omega = Float(units='rpm', desc='fixed rotor rotation speed')

    varSpeed = False
    varPitch = True


class VarSpeedFixedPitch(MachineTypeBase):

    minOmega = Float(units='deg', desc='minimum allowed rotor rotation speed')
    maxOmega = Float(units='deg', desc='maximum allowed rotor rotation speed')
    pitch = Float(units='deg', desc='fixed blade pitch setting')

    varSpeed = True
    varPitch = False


class VarSpeedVarPitch(MachineTypeBase):

    minOmega = Float(units='deg', desc='minimum allowed rotor rotation speed')
    maxOmega = Float(units='deg', desc='maximum allowed rotor rotation speed')

    varSpeed = True
    varPitch = True



# -----------------
#  Base Components
# -----------------


class RotorAeroAnalysis(Component):
    """base component for wind speed/direction"""

    # in
    Uhub = Float(iotype='in', units='m/s', desc='hub height wind speed')
    Omega = Float(iotype='in', units='rpm', desc='rotor rotation speed')
    pitch = Float(iotype='in', units='deg', desc='blade pitch setting')
    azimuth = Float(iotype='in', units='deg', desc='azimuth location (0 vertical, right hand rule when looking downwind)')

    # out
    P = Float(iotype='out', units='W', desc='corresponding power for wind speed (power curve)')



# -----------------------
#  Subclassed Components
# -----------------------


class CCBlade(RotorAeroAnalysis):
    """blade element momentum code"""

    # variables
    r = Array(iotype='in', units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
    chord = Array(iotype='in', units='m', desc='chord length at each section')
    theta = Array(iotype='in', units='deg', desc='twist angle at each section (positive decreases angle of attack)')
    Rhub = Float(iotype='in', units='m', desc='hub radius')
    Rtip = Float(iotype='in', units='m', desc='tip radius')
    hubheight = Float(iotype='in', units='m')

    # parameters
    airfoil_files = List(Str, iotype='in', desc='names of airfoil file')
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg')
    tilt = Float(0.0, iotype='in', desc='shaft tilt', units='deg')
    yaw = Float(0.0, iotype='in', desc='yaw error', units='deg')
    B = Int(3, iotype='in', desc='number of blades')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    mu = Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air')
    shearExp = Float(0.2, iotype='in', desc='shear exponent')
    nSector = Int(4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power')


    def execute(self):

        # airfoil files
        n = len(self.airfoil_files)
        af = [0]*n
        afinit = CCAirfoil.initFromAerodynFile
        for i in range(n):
            af[i] = afinit(self.airfoil_files[i])

        ccblade = CCBlade_PY(self.r, self.chord, self.theta, af, self.Rhub, self.Rtip, self.B,
            self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubheight,
            self.nSector, derivatives=True)

        # Np, Tp = ccblade.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        CP, CT, CQ, dCP_ds, dCT_ds, dCQ_ds, dCP_dv, dCT_dv, dCQ_dv = ccblade.evaluate([self.Uhub], [self.Omega], [self.pitch], coefficient=True)

        self.P = CP[0]

        # self.J = np.array([dP_dv[0, 0, :], dP_dv[0, 1, :], dP_dv[0, 2, :], dP_ds[0, :5]])
        self.J = np.array([np.concatenate([dCP_dv[0, 1, :], dCP_dv[0, 2, :]])])


    def provideJ(self):
        """Jacobian"""

        # inputs = ('r', 'chord', 'theta', 'precone', 'tilt', 'hubheight', 'Rhub', 'Rtip')
        inputs = ('chord', 'theta')
        outputs = ('P')

        return inputs, outputs, self.J



# -----------------
#  Components
# -----------------


class SetupPowerCurve(Component):
    """setup conditions to evaluate aerodynamics code running power curve"""

    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')
    R = Float(iotype='in', units='m', desc='rotor radius')
    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    pitch_opt = Float(iotype='in', units='deg', desc='set pitch angle')
    npts = Int(30, iotype='in', desc='number of points to run aero analysis at to create tsr-cp curve')

    tsr = Array(iotype='out', desc='array of tip-speed ratios to run RotorAeroAnalysis at')
    Uinf = Array(iotype='out', units='m/s', desc='array of freestream velocities to run RotorAeroAnalysis at')
    Omega = Array(iotype='out', units='rpm', desc='array of rotation speeds to run RotorAeroAnalysis at')
    pitch = Array(iotype='out', units='deg', desc='array of pitch angles to run RotorAeroAnalysis at')


    def execute(self):

        R = self.R
        mt = self.machineType
        npts = self.npts

        if mt.varSpeed:

            # at Vin
            tsr_low_Vin = mt.minOmega*RPM2RS*R/mt.Vin
            tsr_high_Vin = mt.maxOmega*RPM2RS*R/mt.Vin

            tsr_max = min(max(self.tsr_opt, tsr_low_Vin), tsr_high_Vin)

            # at Vout
            tsr_low_Vout = mt.minOmega*RPM2RS*R/mt.Vout
            tsr_high_Vout = mt.maxOmega*RPM2RS*R/mt.Vout

            tsr_min = max(min(self.tsr_opt, tsr_high_Vout), tsr_low_Vout)

            # a nominal rotation speed to use for this tip speed ratio (affects Reynolds number)
            omegaNom = 0.5*(mt.maxOmega + mt.minOmega)

        else:
            tsr_max = mt.Omega*RPM2RS*R/mt.Vin
            tsr_min = mt.Omega*RPM2RS*R/mt.Vout

            omegaNom = mt.Omega

        # if abs(tsr_min - tsr_max) < 1e-3:
        #     npts = 1
        # else:
        #     npts = self.npts

        # compute nominal power curve
        self.tsr = np.linspace(tsr_min, tsr_max, npts)
        self.Omega = omegaNom*np.ones(npts)
        self.pitch = self.pitch_opt*np.ones(npts)
        self.Uinf = self.Omega*RPM2RS*R/self.tsr



class AeroPowerCurve(Component):
    """create aerodynamic power curve (no drivetrain losses, no regulation)"""

    tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
    cp_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')
    R = Float(iotype='in', units='m', desc='rotor radius')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')

    npts = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')

    V = Array(iotype='out', units='m/s', desc='aerodynamic power curve (wind speeds)')
    P = Array(iotype='out', units='W', desc='aerodynamic power curve (corresponding power)')


    def execute(self):

        R = self.R
        mt = self.machineType

        # evaluate from cut-in to cut-out
        V = np.linspace(mt.Vin, mt.Vout, self.npts)

        # get corresponding tip speed ratios
        if mt.varSpeed:
            tsr = self.tsr_opt*np.ones(len(V))
            min_tsr = mt.minOmega*RPM2RS*R/V
            max_tsr = mt.maxOmega*RPM2RS*R/V
            tsr = np.minimum(np.maximum(tsr, min_tsr), max_tsr)

        else:
            tsr = mt.Omega*RPM2RS*R/V

        # evalaute nondimensional power curve using a spline
        spline = interpolate.interp1d(self.tsr_array, self.cp_array, kind='cubic')
        cp = spline(tsr)

        # convert to dimensional form
        self.V = V
        self.P = cp * 0.5*self.rho*V**3 * pi*self.R**2



class RegulatedPowerCurve(Component):
    """power curve after drivetrain efficiency losses and control regulation"""

    Vaero = Array(iotype='in', units='m/s', desc='aerodynamic power curve (wind speeds)')
    Paero = Array(iotype='in', units='W', desc='aerodynamic power curve (corresponding power)')
    eff = Array(iotype='in', desc='drivetrain efficiency at each point in the power-curve')
    ratedPower = Float(iotype='in', units='W', desc='rated power')
    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')

    ratedSpeed = Float(iotype='out', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
    V = Array(iotype='out', units='m/s', desc='power curve (wind speed)')
    P = Array(iotype='out', units='m', desc='power curve (power)')

    def execute(self):

        # rename
        mt = self.machineType

        # apply drivetrain efficiency
        self.V = self.Vaero
        self.P = self.Paero * self.eff

        # find rated speed
        idx = np.argmax(self.P)
        if (self.P[idx] <= self.ratedPower):  # check if rated power not reached
            self.ratedSpeed = self.V[idx]  # speed at maximum power generation for this case
        else:
            self.ratedSpeed = np.interp(self.ratedPower, self.P[:idx], self.V[:idx])

        # apply control regulation
        if mt.varSpeed or mt.varPitch:
            self.P[self.V > self.ratedSpeed] = self.ratedPower




class AEP(Component):
    """annual energy production"""

    V = Array(iotype='in', units='m/s', desc='power curve (wind speed)')
    P = Array(iotype='in', units='m', desc='power curve (power)')
    probDist = Slot(ProbDist, iotype='in', desc='wind speed probability distribution')
    lossFactor = Float(iotype='in', desc='availability and other losses (soiling, array, etc.)')

    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')


    def execute(self):

        CDF = self.probDist.CDF(self.V)
        self.AEP = self.lossFactor*np.trapz(self.P/1e3, CDF*365.0*24.0)  # in kWh




class FindControlSettings(TempDriver):
    """find control settings at a particular velocity"""

    VLoads = Array(iotype='in', units='m/s', desc='velocities to find control settings')
    ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')
    R = Float(iotype='in', units='m', desc='rotor radius')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    pitch_opt = Float(iotype='in', units='deg', desc='set pitch angle')

    tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
    cp_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
    dt = Slot(DrivetrainEfficiency, iotype='in')

    pitch_sweep_vec = Array(np.linspace(-5.0, 30.0, 40), iotype='in', desc='for var pitch need to sweep analysis to find appropriate pitch setting')
    
    # out
    Omega = Array(iotype='out', units='rpm', desc='corresponding rotor rotation speed')
    pitch = Array(iotype='out', units='deg', desc='corresponding pitch setting')


    def __getPitchToRegulatePower(self, Uinf, Omega, P):
        """private method
        finds pitch speed to achieve rated power
        at given wind and rotation speed.

        """

        pitchV = self.pitch_sweep_vec

        # manually run the workflow at different pitch settings
        P_sweep = run_workflow(self, np.concatenate((Uinf*np.ones(n), Omega*np.ones(n), pitchV)))
        P_sweep = np.array(P_sweep)

        # choose pitch to feather branch (and reverse order)
        idx = P_sweep.argmax()
        P_branch = P_sweep[:idx:-1]
        pitch_branch = pitchV[:idx:-1]

        # add drivetrain losses
        P_branch *= self.dt.efficiency(P_branch, self.machineType.ratedPower)

        return np.interp(P, P_branch, pitch_branch)



    def __omegaAndPitchForV(self, V):

        mt = self.machineType
        R = self.R

        # region 2 (or 2.5)
        if V <= self.ratedSpeed:

            if mt.varSpeed:
                Omega = V*self.tsr_opt/R*RS2RPM
                Omega = min(Omega, mt.maxOmega)
                Omega = max(Omega, mt.minOmega)
            else:
                Omega = mt.Omega

            if self.varPitch:
                pitch = self.pitch_opt
            else:
                pitch = mt.pitch

        else:  # region 3

            # fixed speed, fixed pitch
            if not mt.varSpeed and not mt.varPitch:
                Omega = mt.Omega
                pitch = mt.pitch

            # variable speed, fixed pitch
            elif mt.varSpeed and not mt.varPitch:

                pitch = mt.pitch

                # choose slowing down branch
                idx = self.tsr_vec < self.tsr_opt
                tsr_branch = self.tsr_vec[idx]
                cp_branch = self.cp_vec[idx]

                # add drivetrain losses
                P_branch = cp_branch * (0.5*self.rho*V**3*pi*R**2)
                P_branch *= self.dt.efficiency(P_branch, mt.ratedPower)

                tsr_reg = np.interp(mt.ratedPower, P_branch, tsr_branch)
                Omega = V*tsr_reg/R*RS2RPM


            # fixed speed, variable pitch
            elif not mt.varSpeed and mt.varPitch:
                Omega = mt.Omega
                pitch = self.__getPitchToRegulatePower(V, Omega, mt.ratedPower)

            # variable speed, variable pitch
            else:
                Omega = min(mt.maxOmega, self.ratedSpeed*self.tsr_opt/R*RS2RPM)
                pitch = self.__getPitchToRegulatePower(V, Omega, mt.ratedPower)

        return Omega, pitch



    def execute(self):

        n = len(self.VLoads)

        self.Omega = np.zeros(n)
        self.pitch = np.zeros(n)
        for i in range(n):
            self.Omega[i], self.pitch[i] = self.__omegaAndPitchForV(self.VLoads[i])





if __name__ == '__main__':

    import os



    basepath = '/Users/sning/Dropbox/NREL/5MW_files/5MW_AFFiles' + os.path.sep

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = basepath + 'Cylinder1.dat'
    airfoil_types[1] = basepath + 'Cylinder2.dat'
    airfoil_types[2] = basepath + 'DU40_A17.dat'
    airfoil_types[3] = basepath + 'DU35_A17.dat'
    airfoil_types[4] = basepath + 'DU30_A17.dat'
    airfoil_types[5] = basepath + 'DU25_A17.dat'
    airfoil_types[6] = basepath + 'DU21_A17.dat'
    airfoil_types[7] = basepath + 'NACA64_A17.dat'

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
    n = len(af_idx)

    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]


    rotor = HarpOpt(n)
    rotor.r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500, 28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500, 56.1667, 58.9000, 61.6333]
    rotor.ccblade.chord = [3.5420, 3.8540, 4.1670, 4.5570, 4.6520, 4.4580, 4.2490, 4.0070, 3.7480, 3.5020, 3.2560, 3.0100, 2.7640, 2.5180, 2.3130, 2.0860, 1.4190]
    rotor.ccblade.theta = [13.3080, 13.3080, 13.3080, 13.3080, 11.4800, 10.1620, 9.0110, 7.7950, 6.5440, 5.3610, 4.1880, 3.1250, 2.3190, 1.5260, 0.8630, 0.3700, 0.1060]
    rotor.Rhub = 1.5
    rotor.Rtip = 63.0
    rotor.precone = 2.5
    rotor.tilt = -5.0
    rotor.hubheight = 80.0

    rotor.airfoil_files = af
    rotor.yaw = 0.0
    rotor.B = 3
    rotor.rho = 1.225
    rotor.mu = 1.81206e-5
    rotor.shearExp = 0.2
    rotor.nSector = 4

    rotor.Uhub = 10.0
    tsr = 7.55
    rotor.pitch = 0.0
    rotor.Omega = rotor.Uhub*tsr/rotor.Rtip * 30.0/pi  # convert to RPM
    rotor.azimuth = 90


    # vals = np.array([0.3449E+01, 0.3213E+01, 0.2746E+01, 0.7742E+01, 0.5870E+01, 0.6142E+01, 0.5571E+01, 0.4311E+01, 0.3884E+01, 0.4012E+01, 0.3416E+01, 0.3479E+01, 0.3244E+01, 0.2865E+01, 0.2392E+01, 0.2036E+01, 0.1277E+01, 0.1331E+02, 0.1331E+02, 0.1331E+02, 0.1413E+02, 0.1092E+02, 0.9056E+01, 0.7722E+01, 0.7315E+01, 0.6026E+01, 0.5532E+01, 0.4739E+01, 0.4305E+01, 0.3731E+01, 0.2947E+01, 0.2085E+01, 0.1268E+01, 0.4426E+00])
    # chord = vals[:n]
    # theta = vals[n:]

    # import matplotlib.pyplot as plt
    # plt.plot(rotor.r, chord)
    # plt.figure()
    # plt.plot(rotor.r, theta)
    # plt.show()


    import time
    tt = time.time()
    rotor.run()

    print rotor.pyOpt_solution

    print "\n"
    print rotor.ccblade.chord
    print rotor.ccblade.theta
    print "Elapsed time: ", time.time()-tt, "seconds"

