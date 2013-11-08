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
from openmdao.main.api import VariableTree, Component, Assembly  # , Driver
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Str, List, Slot, Enum, Bool
# from openmdao.util.decorators import add_delegate
# from openmdao.main.hasparameters import HasParameters
# from openmdao.main.hasobjective import HasObjectives
# from openmdao.lib.drivers.api import SLSQPdriver
# from pyopt_driver.pyopt_driver import pyOptDriver
# from zope.interface import implements, Interface

# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi

from ccblade import CCAirfoil, CCBlade as CCBlade_PY


# -----------------
#  Helper Classes
# -----------------

# class ProbDist(Interface):

#     def CDF(x):
#         """cumulative distribution function"""
#         pass

#     def PDF(x):
#         """probability distribution function"""
#         pass


# class Weibull:
#     implements(ProbDist)

#     def __init__(self, A, k):
#         """Weibull probability distribution

#         Arguments
#         ---------
#         A : float (m/s)
#             scale factor
#         k : float
#             shape or form factor

#         """
#         self.A = A
#         self.k = k


#     def CDF(self, U):
#         return 1.0 - np.exp(-(U/self.A)**self.k)

#     def PDF(self, U):
#         k = self.k
#         A = self.A
#         return k/A*(U/A)**(k-1)*np.exp(-(U/A)**k)



# class Rayleigh:
#     implements(ProbDist)

#     def __init__(self, Ubar):
#         """Rayleigh probability distribution

#         Arguments
#         ---------
#         Ubar : float (m/s)
#             mean wind speed of distribution

#         """
#         A = 2.0*Ubar/pi**0.5
#         k = 2
#         self.prob = Weibull(A, k)


#     def CDF(self, U):
#         return self.prob.CDF(U)


#     def PDF(self, U):
#         return self.prob.PDF(U)



# # TODO: move to CommonSE
# def run_workflow(driver, x):
#     """manually run a workflow"""

#     driver.set_parameters(x)
#     driver.run_iteration()
#     return driver.eval_objectives()


# # TODO: delete once OpenMDAO adds this capability in stable branch
# @add_delegate(HasParameters, HasObjectives)
# class TempDriver(Driver):

#     def add_parameters(self, name, len, assembly):

#         assembly.set(name, np.zeros(len))

#         for i in range(len):
#             self.add_parameter(name + '[%s]' % i, low=-1e9, high=1e9)


#     def add_objectives(self, name, len):

#         for i in range(len):
#             self.add_objective(name + '[%s]' % i)



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


class PowerBase(Component):

    # in
    Uhub = Array(iotype='in', units='m/s', desc='hub height wind speed')
    Omega = Array(iotype='in', units='rpm', desc='rotor rotation speed')
    pitch = Array(iotype='in', units='deg', desc='blade pitch setting')
    dimensional = Bool(True)

    # out
    T = Array(iotype='out')
    Q = Array(iotype='out')
    P = Array(iotype='out', units='W', desc='corresponding power for wind speed (power curve)')
    CT = Array(iotype='out')
    CQ = Array(iotype='out')
    CP = Array(iotype='out')



class DistributedLoadsBase(Component):

    # in
    Uhub = Array(iotype='in', units='m/s', desc='hub height wind speed')
    Omega = Array(iotype='in', units='rpm', desc='rotor rotation speed')
    pitch = Array(iotype='in', units='deg', desc='blade pitch setting')
    azimuth = Array(iotype='in', units='deg', desc='azimuth location (0 vertical, right hand rule when looking downwind)')

    # out
    Np = Array(iotype='out')
    Tp = Array(iotype='out')



class DrivetrainEfficiencyBase(Component):

    power = Array(iotype='in')
    ratedPower = Float(iotype='in')

    eff = Array(iotype='out')


class CDFBase(Component):

    x = Array(iotype='in')

    F = Array(iotype='out')


class PDFBase(Component):

    x = Array(iotype='in')

    f = Array(iotype='out')




# -----------------------
#  Subclassed Components
# -----------------------


class CCBlade(PowerBase):
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

        CP, CT, CQ, dCP_ds, dCT_ds, dCQ_ds, dCP_dv, dCT_dv, dCQ_dv = ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=True)

        self.CP = CP
        self.CT = CT
        self.CQ = CQ





class CSMDrivetrain(DrivetrainEfficiencyBase):

    drivetrainType = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')



    def execute(self):

        drivetrainType = self.drivetrainType


        if drivetrainType == 'geared':
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif drivetrainType == 'single_stage':
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif drivetrainType == 'multi_drive':
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif drivetrainType == 'pm_direct_drive':
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899

        Pbar = self.power / self.ratedPower

        # TODO: think about these gradients.  may not be able to use abs and minimum

        # handle negative power case
        Pbar = np.abs(Pbar)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar = np.minimum(Pbar, 1.0)

        # compute efficiency
        self.eff = np.zeros_like(Pbar)
        idx = Pbar != 0

        self.eff[idx] = 1.0 - (constant/Pbar[idx] + linear + quadratic*Pbar[idx])



class WeibullCDF(CDFBase):

    A = Float(iotype='in', desc='scale factor')
    k = Float(iotype='in', desc='shape or form factor')

    def execute(self):

        self.F = 1.0 - np.exp(-(self.x/self.A)**self.k)


class RayleighCDF(CDFBase):

    Ubar = Float(iotype='in', desc='mean wind speed of distribution')

    def execute(self):

        self.F = 1.0 - np.exp(-pi/4.0*(self.x/self.Ubar)**2)






# -----------------
#  Components
# -----------------


class SetupPowerCurve(Component):
    """setup conditions for aerodynamics code (PowerBase)"""

    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')
    R = Float(iotype='in', units='m', desc='rotor radius')
    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    pitch_opt = Float(iotype='in', units='deg', desc='set pitch angle')
    step_size = Float(0.25, iotype='in', desc='step size in tip-speed ratio for sweep (if necessary)')
    include_region_25 = Bool(True, iotype='in', desc='applies only to var speed machines')

    tsr = Array(iotype='out', desc='array of tip-speed ratios to run RotorAeroAnalysis at')
    Uinf = Array(iotype='out', units='m/s', desc='array of freestream velocities to run RotorAeroAnalysis at')
    Omega = Array(iotype='out', units='rpm', desc='array of rotation speeds to run RotorAeroAnalysis at')
    pitch = Array(iotype='out', units='deg', desc='array of pitch angles to run RotorAeroAnalysis at')


    def execute(self):

        R = self.R
        mt = self.machineType

        if mt.varSpeed:

            if self.include_region_25:

                # at Vin
                tsr_low_Vin = mt.minOmega*RPM2RS*R/mt.Vin
                tsr_high_Vin = mt.maxOmega*RPM2RS*R/mt.Vin

                tsr_max = min(max(self.tsr_opt, tsr_low_Vin), tsr_high_Vin)

                # at Vout
                tsr_low_Vout = mt.minOmega*RPM2RS*R/mt.Vout
                tsr_high_Vout = mt.maxOmega*RPM2RS*R/mt.Vout

                tsr_min = max(min(self.tsr_opt, tsr_high_Vout), tsr_low_Vout)

            else:

                tsr_min = self.tsr_opt
                tsr_max = self.tsr_opt

            # a nominal rotation speed to use for this tip speed ratio (affects Reynolds number)
            omegaNom = 0.5*(mt.maxOmega + mt.minOmega)

        else:
            tsr_max = mt.Omega*RPM2RS*R/mt.Vin
            tsr_min = mt.Omega*RPM2RS*R/mt.Vout

            omegaNom = mt.Omega


        n = int(round((tsr_max - tsr_min)/self.step_size))
        n = max(n, 1)


        # compute nominal power curve
        self.tsr = np.linspace(tsr_min, tsr_max, n)
        self.Omega = omegaNom*np.ones(n)
        self.pitch = self.pitch_opt*np.ones(n)
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
        if (self.P[idx] <= mt.ratedPower):  # check if rated power not reached
            self.ratedSpeed = self.V[idx]  # speed at maximum power generation for this case
        else:
            self.ratedSpeed = np.interp(mt.ratedPower, self.P[:idx], self.V[:idx])

        # apply control regulation
        if mt.varSpeed or mt.varPitch:
            self.P[self.V > self.ratedSpeed] = mt.ratedPower




class AEP(Component):
    """annual energy production"""

    # V = Array(iotype='in', units='m/s', desc='power curve (wind speed)')
    CDF_V = Array(iotype='in')
    P = Array(iotype='in', units='m', desc='power curve (power)')
    # probDist = Slot(ProbDist, iotype='in', desc='wind speed probability distribution')
    lossFactor = Float(iotype='in', desc='availability and other losses (soiling, array, etc.)')

    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')


    def execute(self):

        # CDF = self.probDist.CDF(self.V)
        self.AEP = self.lossFactor*np.trapz(self.P/1e3, self.CDF_V*365.0*24.0)  # in kWh




class SetupPitchSweep(Component):
    """setup conditions for pitch sweep"""

    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')
    Uinf = Float(iotype='in', units='m/s')
    pitch_start = Float(-5.0, iotype='in')
    pitch_end = Float(30.0, iotype='in')
    pitch_npts = Int(30, iotype='in')
    R = Float(iotype='in', units='m', desc='rotor radius')
    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')

    V = Array(iotype='out', units='m/s', desc='array of freestream velocities to run RotorAeroAnalysis at')
    Omega = Array(iotype='out', units='rpm', desc='array of rotation speeds to run RotorAeroAnalysis at')
    pitch = Array(iotype='out', units='deg', desc='array of pitch angles to run RotorAeroAnalysis at')


    def execute(self):

        mt = self.machineType
        n = len(self.VLoads)

        if not mt.varPitch:  # no need to re-run analysis for a fixed pitch we already know all we need
            self.Uinf = np.array([])
            self.Omega = np.array([])
            self.pitch = np.array([])
            return

        self.V = self.Uinf*np.ones(n)
        self.pitch = np.linspace(self.pitch_start, self.pitch_end, self.pitch_npts)

        if not mt.varSpeed:  # fixed speed, variable pitch
            self.Omega = mt.Omega*np.ones(n)

        else:  # variable speed, variable pitch
            self.Omega = min(mt.maxOmega, self.ratedSpeed*self.tsr_opt/self.R*RS2RPM)*np.ones(n)






class VSFPSlowDown(Component):

    V = Float(iotype='in')
    R = Float(iotype='in', units='m', desc='rotor radius')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')

    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
    cp_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')

    tsr_branch = Array(iotype='out')
    P_branch = Array(iotype='out')


    def execute(self):

        # choose slowing down branch
        idx = self.tsr_array < self.tsr_opt
        self.tsr_branch = self.tsr_array[idx]
        cp_branch = self.cp_array[idx]

        self.P_branch = cp_branch * (0.5*self.rho*self.V**3*pi*self.R**2)



class FindControlSettings(Component):

    Uinf = Float(iotype='in')

    pitch_sweep = Array(iotype='in')
    P_sweep = Array(iotype='in')
    dt_eff_pitch_sweep = Array(iotype='in')

    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')
    R = Float(iotype='in', units='m', desc='rotor radius')
    ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    pitch_opt = Float(iotype='in', units='deg', desc='set pitch angle')


    tsr_sweep_slow = Array(iotype='in')
    P_sweep_slow = Array(iotype='in')
    dt_eff_slow = Array(iotype='in')



    Omega = Array(iotype='out')
    pitch = Array(iotype='out')


    def execute(self):

        mt = self.machineType
        R = self.R
        V = self.Uinf

        # region 2 (or 2.5)
        if V <= self.ratedSpeed:

            if mt.varSpeed:
                self.Omega = V*self.tsr_opt/R*RS2RPM
                self.Omega = min(self.Omega, mt.maxOmega)
                self.Omega = max(self.Omega, mt.minOmega)
            else:
                self.Omega = mt.Omega

            if self.varPitch:
                self.pitch = self.pitch_opt
            else:
                self.pitch = mt.pitch

        else:  # region 3

            # fixed speed, fixed pitch
            if not mt.varSpeed and not mt.varPitch:
                self.Omega = mt.Omega
                self.pitch = mt.pitch

            # variable speed, fixed pitch
            elif mt.varSpeed and not mt.varPitch:

                self.pitch = mt.pitch

                P_branch = self.dt_eff_slow * self.P_sweep_slow
                tsr_reg = np.interp(mt.ratedPower, P_branch, self.tsr_sweep_slow)
                self.Omega = V*tsr_reg/R*RS2RPM


                ## choose slowing down branch
                #idx = self.tsr_array < self.tsr_opt
                #tsr_branch = self.tsr_array[idx]
                #cp_branch = self.cp_array[idx]

                ## add drivetrain losses
                #P_branch = cp_branch * (0.5*self.rho*V**3*pi*R**2)
                #P_branch *= self.dt.efficiency(P_branch, mt.ratedPower)

                #tsr_reg = np.interp(mt.ratedPower, P_branch, tsr_branch)
                #self.Omega = V*tsr_reg/R*RS2RPM


            # fixed speed, variable pitch
            elif not mt.varSpeed and mt.varPitch:
                self.Omega = mt.Omega

                # choose pitch to feather branch (and reverse order)
                idx = self.P_sweep.argmax()
                P_branch = self.P_sweep[:idx:-1]  # TODO: I should only need a part of this
                pitch_branch = self.pitch_sweep[:idx:-1]

                # add drivetrain losses
                P_branch *= self.dt_eff_pitch_sweep

                self.pitch = np.interp(self.ratedPower, P_branch, pitch_branch)


            # variable speed, variable pitch
            else:
                self.Omega = min(mt.maxOmega, self.ratedSpeed*self.tsr_opt/R*RS2RPM)

                # choose pitch to feather branch (and reverse order)
                idx = self.P_sweep.argmax()
                P_branch = self.P_sweep[:idx:-1]
                pitch_branch = self.pitch_sweep[:idx:-1]

                # add drivetrain losses
                P_branch *= self.dt_eff_pitch_sweep

                self.pitch = np.interp(self.ratedPower, P_branch, pitch_branch)











# class FindControlSettings(TempDriver):
#     """find control settings at a particular velocity"""

#     VLoads = Array(iotype='in', units='m/s', desc='velocities to find control settings')
#     ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
#     machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
#     tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
#     pitch_opt = Float(iotype='in', units='deg', desc='set pitch angle')

#     tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
#     cp_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
#     dt = Slot(DrivetrainEfficiency, iotype='in')

#     pitch_sweep_vec = Array(np.linspace(-5.0, 30.0, 40), iotype='in', desc='for var pitch need to sweep analysis to find appropriate pitch setting')

#     # out
#     Omega = Array(iotype='out', units='rpm', desc='corresponding rotor rotation speed')
#     pitch = Array(iotype='out', units='deg', desc='corresponding pitch setting')


#     def __getPitchToRegulatePower(self, Uinf, Omega, P):
#         """private method
#         finds pitch speed to achieve rated power
#         at given wind and rotation speed.

#         """

#         pitchV = self.pitch_sweep_vec

#         # manually run the workflow at different pitch settings
#         P_sweep = run_workflow(self, np.concatenate((Uinf*np.ones(n), Omega*np.ones(n), pitchV)))
#         P_sweep = np.array(P_sweep)

#         # choose pitch to feather branch (and reverse order)
#         idx = P_sweep.argmax()
#         P_branch = P_sweep[:idx:-1]
#         pitch_branch = pitchV[:idx:-1]

#         # add drivetrain losses
#         P_branch *= self.dt.efficiency(P_branch, self.machineType.ratedPower)

#         return np.interp(P, P_branch, pitch_branch)



#     def __omegaAndPitchForV(self, V):

#         mt = self.machineType
#         R = self.R

#         # region 2 (or 2.5)
#         if V <= self.ratedSpeed:

#             if mt.varSpeed:
#                 Omega = V*self.tsr_opt/R*RS2RPM
#                 Omega = min(Omega, mt.maxOmega)
#                 Omega = max(Omega, mt.minOmega)
#             else:
#                 Omega = mt.Omega

#             if self.varPitch:
#                 pitch = self.pitch_opt
#             else:
#                 pitch = mt.pitch

#         else:  # region 3

#             # fixed speed, fixed pitch
#             if not mt.varSpeed and not mt.varPitch:
#                 Omega = mt.Omega
#                 pitch = mt.pitch

#             # variable speed, fixed pitch
#             elif mt.varSpeed and not mt.varPitch:

#                 pitch = mt.pitch

#                 # choose slowing down branch
#                 idx = self.tsr_vec < self.tsr_opt
#                 tsr_branch = self.tsr_vec[idx]
#                 cp_branch = self.cp_vec[idx]

#                 # add drivetrain losses
#                 P_branch = cp_branch * (0.5*self.rho*V**3*pi*R**2)
#                 P_branch *= self.dt.efficiency(P_branch, mt.ratedPower)

#                 tsr_reg = np.interp(mt.ratedPower, P_branch, tsr_branch)
#                 Omega = V*tsr_reg/R*RS2RPM


#             # fixed speed, variable pitch
#             elif not mt.varSpeed and mt.varPitch:
#                 Omega = mt.Omega
#                 pitch = self.__getPitchToRegulatePower(V, Omega, mt.ratedPower)

#             # variable speed, variable pitch
#             else:
#                 Omega = min(mt.maxOmega, self.ratedSpeed*self.tsr_opt/R*RS2RPM)
#                 pitch = self.__getPitchToRegulatePower(V, Omega, mt.ratedPower)

#         return Omega, pitch



#     def execute(self):

#         n = len(self.VLoads)

#         self.Omega = np.zeros(n)
#         self.pitch = np.zeros(n)
#         for i in range(n):
#             self.Omega[i], self.pitch[i] = self.__omegaAndPitchForV(self.VLoads[i])




class RotorAero(Assembly):

    # coefficient normalization
    R = Float(iotype='in', units='m', desc='rotor radius')  # TODO: R should be computed from the geometry
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')

    # operational conditions
    tsr_opt = Float(iotype='in', desc='optimal tip-speed ratio in Region 2 (only for variable speed machines)')
    pitch_opt = Float(iotype='in', units='deg', desc='set pitch angle')
    machineType = VarTree(MachineTypeBase(), iotype='in', desc='machine type')

    # options
    include_region_25 = Bool(True, iotype='in', desc='applies only to var speed machines')
    tsr_sweep_step_size = Float(0.25, iotype='in', desc='step size in tip-speed ratio for sweep (if necessary)')
    npts_power_curve = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')
    AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')

    def configure(self):

        self.add('setup', SetupPowerCurve())
        self.add('analysis1', PowerBase())
        self.add('aeroPC', AeroPowerCurve())
        self.add('dt1', DrivetrainEfficiencyBase())
        self.add('powerCurve', RegulatedPowerCurve())
        self.add('cdf', CDFBase())
        self.add('aep', AEP())


        self.driver.workflow.add(['setup', 'analysis1', 'aeroPC', 'dt1', 'powerCurve', 'cdf', 'aep'])

        # connections to setup
        self.connect('machineType', 'setup.machineType')
        self.connect('R', 'setup.R')
        self.connect('tsr_opt', 'setup.tsr_opt')
        self.connect('pitch_opt', 'setup.pitch_opt')
        self.connect('tsr_sweep_step_size', 'setup.step_size')
        self.connect('include_region_25', 'setup.include_region_25')


        # connections to analysis1
        self.connect('setup.Uinf', 'analysis1.Uhub')
        self.connect('setup.Omega', 'analysis1.Omega')
        self.connect('setup.pitch', 'analysis1.pitch')
        self.analysis1.dimensional = False


        # connections to aeroPC
        self.connect('setup.tsr', 'aeroPC.tsr_array')
        self.connect('analysis1.CP', 'aeroPC.cp_array')
        self.connect('tsr_opt', 'aeroPC.tsr_opt')
        self.connect('machineType', 'aeroPC.machineType')
        self.connect('R', 'aeroPC.R')
        self.connect('rho', 'aeroPC.rho')
        self.connect('npts_power_curve', 'aeroPC.npts')

        # connections to dt1
        self.connect('aeroPC.P', 'dt1.power')
        self.connect('machineType.ratedPower', 'dt1.ratedPower')


        # connections to powerCurve
        self.connect('aeroPC.V', 'powerCurve.Vaero')
        self.connect('aeroPC.P', 'powerCurve.Paero')
        self.connect('dt1.eff', 'powerCurve.eff')
        self.connect('machineType', 'powerCurve.machineType')


        # connections to CDF
        self.connect('powerCurve.V', 'cdf.x')


        # connections to AEP
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powerCurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')


        # -------- passthroughs --------

        self.create_passthrough('powerCurve.ratedSpeed')
        self.create_passthrough('powerCurve.V')
        self.create_passthrough('powerCurve.P')
        self.create_passthrough('aep.AEP')




if __name__ == '__main__':

    rho = 1.225

    rotor = RotorAero()

    # TODO: setup scripts so these are computed

    rotor.R = 63.0*np.cos(2.5*pi/180.0)
    rotor.rho = rho

    # ------ aero analysis --------
    analysis = CCBlade()

    analysis.r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
    analysis.chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    analysis.theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    analysis.Rhub = 1.5
    analysis.Rtip = 63.0
    analysis.hubheight = 80.0
    analysis.precone = 2.5
    analysis.tilt = -5.0
    analysis.yaw = 0.0
    analysis.B = 3
    analysis.rho = rho
    analysis.mu = 1.81206e-5
    analysis.shearExp = 0.2
    analysis.nSector = 4

    basepath = '/Users/Andrew/Dropbox/NREL/5MW_files/5MW_AFFiles/'

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

    n = len(analysis.r)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]

    analysis.airfoil_files = af

    rotor.replace('analysis1', analysis)
    # ----------------------------------------


    # ----- drivetrain efficiency -------
    dt = CSMDrivetrain()
    dt.drivetrainType = 'geared'

    rotor.replace('dt1', dt)
    # ----------------

    # ------- CDF -----------
    cdf = RayleighCDF()
    cdf.Ubar = 6.0

    rotor.replace('cdf', cdf)
    # ---------------------


    # ----- operational conditions -------
    machineType = VarSpeedVarPitch()
    machineType.Vin = 3.0
    machineType.Vout = 25.0
    machineType.ratedPower = 5e6
    machineType.minOmega = 0.0
    machineType.maxOmega = 12.0

    rotor.machineType = machineType

    rotor.tsr_opt = 7.55
    rotor.pitch_opt = 0.0
    # ----------------------------

    # ------- options -----------
    rotor.include_region_25 = True
    rotor.tsr_sweep_step_size = 0.25
    rotor.npts_power_curve = 200
    rotor.drivetrainType = 'geared'
    rotor.AEP_loss_factor = 1.0
    # ----------------------------

    rotor.run()

    print rotor.AEP

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P/1e6)
    plt.show()



    # import os



    # basepath = '/Users/sning/Dropbox/NREL/5MW_files/5MW_AFFiles' + os.path.sep

    # # load all airfoils
    # airfoil_types = [0]*8
    # airfoil_types[0] = basepath + 'Cylinder1.dat'
    # airfoil_types[1] = basepath + 'Cylinder2.dat'
    # airfoil_types[2] = basepath + 'DU40_A17.dat'
    # airfoil_types[3] = basepath + 'DU35_A17.dat'
    # airfoil_types[4] = basepath + 'DU30_A17.dat'
    # airfoil_types[5] = basepath + 'DU25_A17.dat'
    # airfoil_types[6] = basepath + 'DU21_A17.dat'
    # airfoil_types[7] = basepath + 'NACA64_A17.dat'

    # # place at appropriate radial stations
    # af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
    # n = len(af_idx)

    # af = [0]*n
    # for i in range(n):
    #     af[i] = airfoil_types[af_idx[i]]


    # rotor = HarpOpt(n)
    # rotor.r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500, 28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500, 56.1667, 58.9000, 61.6333]
    # rotor.ccblade.chord = [3.5420, 3.8540, 4.1670, 4.5570, 4.6520, 4.4580, 4.2490, 4.0070, 3.7480, 3.5020, 3.2560, 3.0100, 2.7640, 2.5180, 2.3130, 2.0860, 1.4190]
    # rotor.ccblade.theta = [13.3080, 13.3080, 13.3080, 13.3080, 11.4800, 10.1620, 9.0110, 7.7950, 6.5440, 5.3610, 4.1880, 3.1250, 2.3190, 1.5260, 0.8630, 0.3700, 0.1060]
    # rotor.Rhub = 1.5
    # rotor.Rtip = 63.0
    # rotor.precone = 2.5
    # rotor.tilt = -5.0
    # rotor.hubheight = 80.0

    # rotor.airfoil_files = af
    # rotor.yaw = 0.0
    # rotor.B = 3
    # rotor.rho = 1.225
    # rotor.mu = 1.81206e-5
    # rotor.shearExp = 0.2
    # rotor.nSector = 4

    # rotor.Uhub = 10.0
    # tsr = 7.55
    # rotor.pitch = 0.0
    # rotor.Omega = rotor.Uhub*tsr/rotor.Rtip * 30.0/pi  # convert to RPM
    # rotor.azimuth = 90


    # # vals = np.array([0.3449E+01, 0.3213E+01, 0.2746E+01, 0.7742E+01, 0.5870E+01, 0.6142E+01, 0.5571E+01, 0.4311E+01, 0.3884E+01, 0.4012E+01, 0.3416E+01, 0.3479E+01, 0.3244E+01, 0.2865E+01, 0.2392E+01, 0.2036E+01, 0.1277E+01, 0.1331E+02, 0.1331E+02, 0.1331E+02, 0.1413E+02, 0.1092E+02, 0.9056E+01, 0.7722E+01, 0.7315E+01, 0.6026E+01, 0.5532E+01, 0.4739E+01, 0.4305E+01, 0.3731E+01, 0.2947E+01, 0.2085E+01, 0.1268E+01, 0.4426E+00])
    # # chord = vals[:n]
    # # theta = vals[n:]

    # # import matplotlib.pyplot as plt
    # # plt.plot(rotor.r, chord)
    # # plt.figure()
    # # plt.plot(rotor.r, theta)
    # # plt.show()


    # import time
    # tt = time.time()
    # rotor.run()

    # print rotor.pyOpt_solution

    # print "\n"
    # print rotor.ccblade.chord
    # print rotor.ccblade.theta
    # print "Elapsed time: ", time.time()-tt, "seconds"

