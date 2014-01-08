#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from scipy import interpolate
# from scipy.optimize import brentq
# from scipy.integrate import quad
from math import pi
from openmdao.main.api import VariableTree, Component, Assembly, Driver, ImplicitComponent
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot, Enum
# from openmdao.util.decorators import add_delegate
# from openmdao.main.hasparameters import HasParameters
# from openmdao.main.hasobjective import HasObjective
from openmdao.lib.drivers.api import Brent
# from openmdao.main.hasobjectives import HasObjectives
# from openmdao.main.driver import Run_Once

from commonse.utilities import hstack, vstack, interp_with_deriv, linspace_with_deriv, CubicSplineSegment, smooth_max, smooth_min
from akima import Akima


# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi


# ---------------------
# Variable Trees
# ---------------------


class VarSpeedMachine(VariableTree):
    """variable speed machines"""

    Vin = Float(units='m/s', desc='cut-in wind speed')
    Vout = Float(units='m/s', desc='cut-out wind speed')
    ratedPower = Float(units='W', desc='rated power')
    minOmega = Float(units='deg', desc='minimum allowed rotor rotation speed')
    maxOmega = Float(units='deg', desc='maximum allowed rotor rotation speed')
    tsr = Float(desc='tip-speed ratio in Region 2 (should be optimized externally)')
    pitch = Float(units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')



class FixedSpeedMachine(VariableTree):
    """fixed speed machines"""

    Vin = Float(units='m/s', desc='cut-in wind speed')
    Vout = Float(units='m/s', desc='cut-out wind speed')
    ratedPower = Float(units='W', desc='rated power')
    Omega = Float(units='rpm', desc='fixed rotor rotation speed')
    pitch = Float(units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')



class RatedConditions(VariableTree):
    """aerodynamic conditions at the rated wind speed"""

    V = Float(units='m/s', desc='rated wind speed')
    Omega = Float(units='rpm', desc='rotor rotation speed at rated')
    pitch = Float(units='deg', desc='pitch setting at rated')
    T = Float(units='N', desc='rotor aerodynamic thrust at rated')
    Q = Float(units='N*m', desc='rotor aerodynamic torque at rated')


class AeroLoads(VariableTree):

    r = Array(units='m', desc='radial positions along blade going toward tip')
    Px = Array(units='N/m', desc='distributed loads in blade-aligned x-direction')
    Py = Array(units='N/m', desc='distributed loads in blade-aligned y-direction')
    Pz = Array(units='N/m', desc='distributed loads in blade-aligned z-direction')

    # corresponding setting for loads
    V = Float(units='m/s', desc='hub height wind speed')
    Omega = Float(units='rpm', desc='rotor rotation speed')
    pitch = Float(units='deg', desc='pitch angle')
    azimuth = Float(units='deg', desc='azimuthal angle')
    tilt = Float(units='deg', desc='tilt angle')



# ---------------------
# Base Components
# ---------------------

class GeomtrySetupBase(Component):
    """a base component that computes the rotor radius from the geometry"""

    R = Float(iotype='out', units='m', desc='rotor radius')


class AeroBase(Component):
    """A base component for a rotor aerodynamics code."""

    run_case = Enum('power', ('power', 'loads'), iotype='in')


    # --- use these if (run_case == 'power') ---

    # inputs
    Uhub = Array(iotype='in', units='m/s', desc='hub height wind speed')
    Omega = Array(iotype='in', units='rpm', desc='rotor rotation speed')
    pitch = Array(iotype='in', units='deg', desc='blade pitch setting')

    # outputs
    T = Array(iotype='out', units='N', desc='rotor aerodynamic thrust')
    Q = Array(iotype='out', units='N*m', desc='rotor aerodynamic torque')
    P = Array(iotype='out', units='W', desc='rotor aerodynamic power')


    # --- use these if (run_case == 'loads') ---
    # if you only use rotoraero.py and not rotor.py
    # (i.e., only care about power curves, and not structural loads)
    # then these second set of inputs/outputs are not needed

    # inputs
    V_load = Float(iotype='in', units='m/s', desc='hub height wind speed')
    Omega_load = Float(iotype='in', units='rpm', desc='rotor rotation speed')
    pitch_load = Float(iotype='in', units='deg', desc='blade pitch setting')
    azimuth_load = Float(iotype='in', units='deg', desc='blade azimuthal location')

    # outputs
    loads = VarTree(AeroLoads(), iotype='out', desc='loads in blade-aligned coordinate system')




class DrivetrainLossesBase(Component):
    """base component for drivetrain efficiency losses"""

    aeroPower = Array(iotype='in', units='W', desc='aerodynamic power')
    aeroTorque = Array(iotype='in', units='N*m', desc='aerodynamic torque')
    aeroThrust = Array(iotype='in', units='N', desc='aerodynamic thrust')
    ratedPower = Float(iotype='in', units='W', desc='rated power')

    power = Array(iotype='out', units='W', desc='total power after drivetrain losses')


class PDFBase(Component):
    """probability distribution function"""

    x = Array(iotype='in')

    f = Array(iotype='out')


class CDFBase(Component):
    """cumulative distribution function"""

    x = Array(iotype='in')

    F = Array(iotype='out')



# ---------------------
# Components
# ---------------------

class Coefficients(Component):
    """convert power, thrust, torque into nondimensional coefficient form"""

    # inputs
    V = Array(iotype='in', units='m/s', desc='wind speed')
    T = Array(iotype='in', units='N', desc='rotor aerodynamic thrust')
    Q = Array(iotype='in', units='N*m', desc='rotor aerodynamic torque')
    P = Array(iotype='in', units='W', desc='rotor aerodynamic power')

    # inputs used in normalization
    R = Float(iotype='in', units='m', desc='rotor radius')
    rho = Float(iotype='in', units='kg/m**3', desc='density of fluid')


    # outputs
    CT = Array(iotype='out', desc='rotor aerodynamic thrust')
    CQ = Array(iotype='out', desc='rotor aerodynamic torque')
    CP = Array(iotype='out', desc='rotor aerodynamic power')


    def execute(self):

        q = 0.5 * self.rho * self.V**2
        A = pi * self.R**2
        self.CP = self.P / (q * A * self.V)
        self.CT = self.T / (q * A)
        self.CQ = self.Q / (q * self.R * A)


    def linearize(self):
        # TODO: remove this after openmdao updates to cache J
        pass

    def provideJ(self):

        inputs = ('V', 'T', 'Q', 'P', 'R')
        outputs = ('CT', 'CQ', 'CP')

        V = self.V
        R = self.R
        CT = self.CT
        CQ = self.CQ
        CP = self.CP
        n = len(self.V)
        q = 0.5 * self.rho * V**2
        A = pi * R**2
        zeronn = np.zeros((n, n))

        dCT_dV = np.diag(-2.0*CT/V)
        dCT_dT = np.diag(1.0/(q*A))
        dCT_dR = -2.0*CT/R
        dCT = hstack((dCT_dV, dCT_dT, zeronn, zeronn, dCT_dR))

        dCQ_dV = np.diag(-2.0*CQ/V)
        dCQ_dQ = np.diag(1.0/(q*R*A))
        dCQ_dR = -3.0*CQ/R
        dCQ = hstack((dCQ_dV, zeronn, dCQ_dQ, zeronn, dCQ_dR))

        dCP_dV = np.diag(-3.0*CP/V)
        dCP_dP = np.diag(1.0/(q*A*V))
        dCP_dR = -2.0*CP/R
        dCP = hstack((dCP_dV, zeronn, zeronn, dCP_dP, dCP_dR))

        J = vstack((dCT, dCQ, dCP))

        return inputs, outputs, J




# class SetupTSRFixedSpeed(Component):
#     """find min and max tsr for a fixed speed machine"""

#     # inputs
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     control = VarTree(FixedSpeedMachine(), iotype='in')

#     # outputs
#     tsr_min = Float(iotype='out', desc='minimum tip speed ratio')
#     tsr_max = Float(iotype='out', desc='maximum tip speed ratio')


#     def execute(self):

#         ctrl = self.control

#         self.tsr_min = ctrl.Omega*RPM2RS*self.R/ctrl.Vout
#         self.tsr_max = ctrl.Omega*RPM2RS*self.R/ctrl.Vin


#     def linearize(self):
#         # TODO: remove this after openmdao updates to cache J
#         pass

#     def provideJ(self):

#         inputs = ('R',)
#         outputs = ('tsr_min', 'tsr_max')

#         ctrl = self.control
#         J = np.zeros((2, 1))
#         J[0] = ctrl.Omega*RPM2RS/ctrl.Vout
#         J[1] = ctrl.Omega*RPM2RS/ctrl.Vin

#         return inputs, outputs, J


# class SetupTSRVarSpeed(Component):
#     """find min and max tsr for a variable speed machine"""

#     # inputs
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     control = VarTree(VarSpeedMachine(), iotype='in')

#     # outputs
#     tsr_min = Float(iotype='out', desc='minimum tip speed ratio')
#     tsr_max = Float(iotype='out', desc='maximum tip speed ratio')
#     Omega_nominal = Float(iotype='out', units='rpm', desc='a nominal rotation speed to use in tsr sweep')


#     def execute(self):

#         ctrl = self.control
#         R = self.R

#         # at Vin
#         tsr_low_Vin = ctrl.minOmega*RPM2RS*R/ctrl.Vin
#         tsr_high_Vin = ctrl.maxOmega*RPM2RS*R/ctrl.Vin

#         self.tsr_max = min(max(ctrl.tsr, tsr_low_Vin), tsr_high_Vin)

#         # at Vout
#         tsr_low_Vout = ctrl.minOmega*RPM2RS*R/ctrl.Vout
#         tsr_high_Vout = ctrl.maxOmega*RPM2RS*R/ctrl.Vout

#         self.tsr_min = max(min(ctrl.tsr, tsr_high_Vout), tsr_low_Vout)

#         print self.tsr_min, self.tsr_max

#         # a nominal rotation speed to use for this tip speed ratio (small Reynolds number effect)
#         self.Omega_nominal = 0.5*(ctrl.maxOmega + ctrl.minOmega)



# class SetupRegion2(Component):

#     control = VarTree(VarSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')

#     # outputs
#     V_OmegaMax = Float(iotype='in', units='m/s', desc='speed at which maxOmega is reached')
#     Uinf = Array(iotype='out', shape=((1,)), units='m/s', desc='freestream velocities to run')
#     Omega = Array(iotype='out', shape=((1,)), units='rpm', desc='rotation speeds to run')
#     pitch = Array(iotype='out', shape=((1,)), units='deg', desc='pitch angles to run')

#     def execute(self):

#         ctrl = self.control

#         self.V_OmegaMax = ctrl.maxOmega*RPM2RS*self.R/ctrl.tsr

#         self.Uinf = np.array([self.Vomega_max])
#         self.Omega = np.array([ctrl.maxOmega])
#         self.pitch = np.array([ctrl.pitch])



class SetupRun(Component):

    control = VarTree(VarSpeedMachine(), iotype='in')
    R = Float(iotype='in', units='m', desc='rotor radius')
    npts = Int(20, iotype='in', desc='number of points to evalute aero code to generate power curve')

    # outputs
    Uhub = Array(iotype='out', units='m/s', desc='freestream velocities to run')
    Omega = Array(iotype='out', units='rpm', desc='rotation speeds to run')
    pitch = Array(iotype='out', units='deg', desc='pitch angles to run')


    def execute(self):

        ctrl = self.control
        n = self.npts
        R = self.R

        # # attempt to distribute points mostly before rated
        # cpguess = 0.5
        # Vr0 = (ctrl.ratedPower/(cpguess*0.5*rho*pi*R**2))**(1.0/3)
        # Vr0 *= 1.20

        # V1 = np.linspace(Vin, Vr0, 15)
        # V2 = np.linspace(Vr0, Vout, 6)
        # V = np.concatenate([V1, V2[1:]])

        V = np.linspace(ctrl.Vin, ctrl.Vout, n)
        Omega_d = ctrl.tsr*V/R*RS2RPM
        Omega, dOmega_dOmegad = smooth_min(Omega_d, ctrl.maxOmega, pct_offset=0.01)

        self.Uhub = V
        self.Omega = Omega
        self.pitch = ctrl.pitch*np.ones_like(V)

        dV = np.zeros((n, 2))
        dOmega_dtsr = dOmega_dOmegad * V/R*RS2RPM
        dOmega_dR = dOmega_dOmegad * -ctrl.tsr*V/R**2*RS2RPM
        dOmega = hstack([dOmega_dtsr, dOmega_dR])
        dpitch = np.zeros((n, 2))
        self.J = vstack([dV, dOmega, dpitch])


    def linearize(self):
        pass

    def provideJ(self):

        inputs = ('control.tsr', 'R')
        outputs = ('Uhub', 'Omega', 'pitch')

        return inputs, outputs, self.J




class RegulatedPowerCurve(ImplicitComponent):

    control = VarTree(VarSpeedMachine(), iotype='in')
    Vcoarse = Array(iotype='in', units='m/s', desc='wind speeds')
    Pcoarse = Array(iotype='in', units='W', desc='unregulated power curve (but after drivetrain losses)')
    Vrated = Float(iotype='state', units='m/s', desc='rated wind speed')
    npts = Int(200, iotype='in', desc='number of points for splined power curve')

    residual = Float(iotype="residual")
    V = Array(iotype='out', units='m/s', desc='wind speeds')
    P = Array(iotype='out', units='W', desc='power')


    def __init__(self):
        super(RegulatedPowerCurve, self).__init__()
        self.eval_only = True  # allows an external solver to converge this, otherwise it will converge itself to mimic an explicit comp

    def evaluate(self):

        ctrl = self.control
        n = self.npts
        Vrated = self.Vrated

        # residual
        spline = Akima(self.Vcoarse, self.Pcoarse)
        P, dres_dVrated, dres_dVcoarse, dres_dPcoarse = spline.interp(Vrated)
        self.residual = P - ctrl.ratedPower

        # functional

        # place half of points in region 2, half in region 3
        # even though region 3 is constant we still need lots of points there
        # because we will be integrating against a discretized wind
        # speed distribution

        # region 2
        V2, _, dV2_dVrated = linspace_with_deriv(ctrl.Vin, Vrated, n/2)
        P2, dP2_dV2, dP2_dVcoarse, dP2_dPcoarse = spline.interp(V2)

        # region 3
        V3, dV3_dVrated, _ = linspace_with_deriv(Vrated, ctrl.Vout, n/2+1)
        V3 = V3[1:]  # remove duplicate point
        dV3_dVrated = dV3_dVrated[1:]
        P3 = ctrl.ratedPower*np.ones_like(V3)

        # concatenate
        self.V = np.concatenate([V2, V3])
        self.P = np.concatenate([P2, P3])



        # gradients
        ncoarse = len(self.Vcoarse)

        dres = np.concatenate([dres_dVcoarse, dres_dPcoarse, np.array([dres_dVrated])])

        dV_dVrated = np.concatenate([dV2_dVrated, dV3_dVrated])
        dV = hstack([np.zeros((n, 2*ncoarse)), dV_dVrated])

        dP_dVcoarse = vstack([dP2_dVcoarse, np.zeros((n/2, ncoarse))])
        dP_dPcoarse = vstack([dP2_dPcoarse, np.zeros((n/2, ncoarse))])
        dP_dVrated = np.concatenate([dP2_dV2*dV2_dVrated, np.zeros(n/2)])
        dP = hstack([dP_dVcoarse, dP_dPcoarse, dP_dVrated])

        self.J = vstack([dres, dV, dP])


    def linearize(self):
        pass

    def provideJ(self):

        inputs = ('Vcoarse', 'Pcoarse', 'Vrated')
        outputs = ('residual', 'V', 'P')


        return inputs, outputs, self.J




# class SetupRun(Component):

#     control = VarTree(VarSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     V = Float(iotype='in')

#     # outputs
#     Uhub = Array(iotype='out', units='m/s', desc='freestream velocities to run')
#     Omega = Array(iotype='out', units='rpm', desc='rotation speeds to run')
#     pitch = Array(iotype='out', units='deg', desc='pitch angles to run')

#     def execute(self):

#         ctrl = self.control

#         V_OmegaMax = ctrl.maxOmega*RPM2RS*self.R/ctrl.tsr

#         if self.V <= V_OmegaMax:  # operate at optimal tsr
#             self.Uhub = np.array([self.V])
#             self.Omega = np.array([ctrl.tsr*self.V/self.R*RS2RPM])  # TODO: replace with smooth_max
#             self.pitch = np.array([ctrl.pitch])

#         else:
#             self.Uhub = np.array([self.V])
#             self.Omega = np.array([ctrl.maxOmega])
#             self.pitch = np.array([ctrl.pitch])


# class Residual(Component):

#     power = Array(iotype='in')
#     ratedPower = Float(iotype='in')

#     residual = Float(iotype='out')

#     def execute(self):
#         self.residual = self.power[0] - self.ratedPower



# class NominalRatedSpeed(Component):

#     ratedPowerAfterLosses = Float(iotype='in', units='W', desc='rated power after drivetrain losses')
#     control = VarTree(VarSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')

#     def execute(self):

#         self.Vr0 = (ratedPower/(cp*0.5*rho*pi*R**2*eta_Prated))**(1.0/3)





# class SetupSweepFixedSpeed(Component):
#     """setup input conditions to AeroBase for a tsr sweep"""

#     control = VarTree(FixedSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')

#     # outputs
#     tsr = Array(iotype='out', desc='array of tip-speed ratios to run AeroBase at')
#     Uinf = Array(iotype='out', units='m/s', desc='array of freestream velocities to run AeroBase at')
#     Omega = Array(iotype='out', units='rpm', desc='array of rotation speeds to run AeroBase at')
#     pitch = Array(iotype='out', units='deg', desc='array of pitch angles to run AeroBase at')

#     Uinf_full = Array(iotype='out')
#     Omega_full = Array(iotype='out')


#     def execute(self):

#         # constants (hard-coded for now)
#         npts = 200
#         npts_subset = 20

#         ctrl = self.control
#         R = self.R

#         tsr_min = ctrl.Omega*RPM2RS*R/ctrl.Vout
#         tsr_max = ctrl.Omega*RPM2RS*R/ctrl.Vin

#         # compute nominal power curve
#         self.tsr = np.linspace(tsr_min, tsr_max, npts_subset)
#         self.Omega = ctrl.Omega*np.ones(npts_subset)
#         self.pitch = ctrl.pitch*np.ones(npts_subset)
#         self.Uinf = self.Omega*RPM2RS*R/self.tsr

#         # points to evaluate at
#         self.Uinf_full = np.linspace(ctrl.Vin, ctrl.Vout, npts)
#         self.Omega_full = ctrl.Omega*np.ones(npts)



# class SetupSweepVarSpeed(Component):

#     control = VarTree(VarSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')

#     # outputs
#     tsr = Array(iotype='out', desc='array of tip-speed ratios to run AeroBase at')
#     Uinf = Array(iotype='out', units='m/s', desc='array of freestream velocities to run AeroBase at')
#     Omega = Array(iotype='out', units='rpm', desc='array of rotation speeds to run AeroBase at')
#     pitch = Array(iotype='out', units='deg', desc='array of pitch angles to run AeroBase at')

#     Uinf_full = Array(iotype='out')
#     Omega_full = Array(iotype='out')

#     npts = Int(200, iotype='in', desc='number of points in full power curve')
#     npts_subset = Int(20, iotype='in', desc='number of points rotoraero is evaluted at to generated lambda-cp curve')


#     def execute(self):

#         ctrl = self.control
#         R = self.R
#         npts = self.npts
#         npts_subset = self.npts_subset

#         # evaluate from Vin to Vout
#         V = np.linspace(ctrl.Vin, ctrl.Vout, npts)

#         # Omega desired (always at optimal tip-speed ratio)
#         Omega_d = V*ctrl.tsr/R*RS2RPM

#         # apply min and max (in a smooth manner)
#         if ctrl.minOmega != 0.0:
#             Omega, dOmega_dOmegad = smooth_min(Omega_d, ctrl.minOmega, pct_offset=0.05, min_on_right=False)
#         else:
#             Omega = np.copy(Omega_d)
#             dOmega_dOmegad = np.ones_like(Omega_d)

#         Omega, dOmega_dOmegad = smooth_max(Omega, ctrl.maxOmega, pct_offset=0.05, dyd=dOmega_dOmegad)

#         # chain rule
#         dOmegad_dR = -V*ctrl.tsr/R**2*RS2RPM
#         dOmega_dR = dOmega_dOmegad*dOmegad_dR

#         # corresponding tip speed ratios for actual Omega
#         tsr = Omega*RPM2RS*R/V
#         dtsrfull_dR = dOmega_dR*RPM2RS*R/V + Omega*RPM2RS/V

#         # do actual evaluation at a smaller number of points that spans the range of interest
#         self.tsr, dtsr_dtsrs, dtsr_dtsrf = linspace_with_deriv(tsr[-1], tsr[0], npts_subset)

#         # to get corresponding Omega need to interp, but interp but be done with sorted, unique entries
#         tsr_array, indices = np.unique(tsr, return_index=True)
#         Omega_array = Omega[indices]

#         self.Omega, dOmega_dtsr, dOmega_dtsrfull_sub, dOmega_dOmegafull_sub \
#             = interp_with_deriv(self.tsr, tsr_array, Omega_array)
#         dOmega_dtsrfull = np.zeros((npts_subset, npts))
#         dOmega_dtsrfull[:, indices] = dOmega_dtsrfull_sub
#         dOmega_dOmegafull = np.zeros((npts_subset, npts))
#         dOmega_dOmegafull[:, indices] = dOmega_dOmegafull_sub

#         # set pitch and freestream velocity
#         self.pitch = ctrl.pitch*np.ones_like(self.tsr)
#         self.Uinf = self.Omega*RPM2RS*R/self.tsr

#         # save the desired arrays for the full power curve
#         self.Uinf_full = V
#         self.Omega_full = Omega

#         # save derivatives
#         self.dOmegafull_dR = dOmega_dR
#         self.dtsr_dR = dtsr_dtsrs*dtsrfull_dR[-1] + dtsr_dtsrf*dtsrfull_dR[0]
#         self.dOmega_dR = np.dot(dOmega_dtsr, self.dtsr_dR) + np.dot(dOmega_dtsrfull, dtsrfull_dR) + np.dot(dOmega_dOmegafull, self.dOmegafull_dR)
#         self.dUinf_dR = self.dOmega_dR*RPM2RS*R/self.tsr - self.Omega*RPM2RS*R/self.tsr**2*self.dtsr_dR + self.Omega*RPM2RS/self.tsr

#     def linearize(self):
#         pass  # TODO: remove this method

#     def provideJ(self):

#         inputs = ('R')
#         outputs = ('tsr', 'Uinf', 'Omega', 'Omega_full')

#         J = np.concatenate([self.dtsr_dR, self.dUinf_dR, self.dOmega_dR, self.dOmegafull_dR])
#         J = J[:, np.newaxis]  # make column vector

#         return inputs, outputs, J


# class AeroPowerCurve(Component):
#     """create aerodynamic power curve for variable speed machines (no drivetrain losses, no regulation)"""

#     # inputs
#     tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
#     CP_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
#     CT_array = Array(iotype='in', desc='corresponding thrust coefficients')
#     Uinf_full = Array(iotype='in')
#     Omega_full = Array(iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
#     npts = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')

#     # outputs
#     # V = Array(iotype='out', units='m/s', desc='aerodynamic power curve (wind speeds)')
#     # Omega = Array(iotype='out', units='rpm', desc='rotation speed curve')
#     P = Array(iotype='out', units='W', desc='aerodynamic power curve (corresponding power)')
#     T = Array(iotype='out', units='N', desc='thrust curve')
#     Q = Array(iotype='out', units='N*m', desc='torque curve')


#     def execute(self):

#         R = self.R
#         tsr_full = self.Omega_full*RPM2RS*R/self.Uinf_full

#         tsr_full[-1] += 1e-6  # TODO: remove these rounding errors
#         tsr_full[:-1] -= 1e-6
#         spline = interpolate.interp1d(self.tsr_array, self.CP_array, kind='cubic')
#         cp = spline(tsr_full)


#         spline = interpolate.interp1d(self.tsr_array, self.CT_array, kind='cubic')
#         ct = spline(tsr_full)

#         # convert to dimensional form
#         V = self.Uinf_full
#         q = 0.5*self.rho*V**2
#         A = pi*R**2
#         self.P = cp * q * V * A
#         self.T = ct * q * A
#         self.Q = self.P / (self.Omega_full*RPM2RS)



# class AeroPowerCurveVarSpeed(Component):
#     """create aerodynamic power curve for variable speed machines (no drivetrain losses, no regulation)"""

#     # inputs
#     tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
#     CP_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
#     CT_array = Array(iotype='in', desc='corresponding thrust coefficients')
#     control = VarTree(VarSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
#     npts = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')

#     # outputs
#     V = Array(iotype='out', units='m/s', desc='aerodynamic power curve (wind speeds)')
#     Omega = Array(iotype='out', units='rpm', desc='rotation speed curve')
#     P = Array(iotype='out', units='W', desc='aerodynamic power curve (corresponding power)')
#     T = Array(iotype='out', units='N', desc='thrust curve')
#     Q = Array(iotype='out', units='N*m', desc='torque curve')


#     def execute(self):

#         R = self.R
#         ctrl = self.control


#         # evaluate from cut-in to cut-out
#         V = np.linspace(ctrl.Vin, ctrl.Vout, self.npts)

#         # evalaute nondimensional power curve using a spline
#         if len(self.tsr_array) == 1:
#             cp = self.CP_array[0]
#             ct = self.CT_array[0]
#             self.Omega = V*self.tsr_array[0]/R

#         else:
#             tsr = ctrl.tsr*np.ones(len(V))
#             min_tsr = ctrl.minOmega*RPM2RS*R/V
#             max_tsr = ctrl.maxOmega*RPM2RS*R/V
#             tsr = np.minimum(np.maximum(tsr, min_tsr), max_tsr)

#             self.Omega = V*tsr/R

#             spline = interpolate.interp1d(self.tsr_array, self.CP_array, kind='cubic')
#             cp = spline(tsr)

#             spline = interpolate.interp1d(self.tsr_array, self.CT_array, kind='cubic')
#             ct = spline(tsr)

#         # convert to dimensional form
#         self.V = V
#         self.P = cp * 0.5*self.rho*V**3 * pi*R**2
#         self.T = ct * 0.5*self.rho*V**2 * pi*R**2
#         self.Q = self.P / self.Omega
#         self.Omega *= RS2RPM




# class AeroPowerCurveFixedSpeed(Component):
#     """create aerodynamic power curve for fixed speed machines (no drivetrain losses, no regulation)"""

#     # inputs
#     tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
#     cp_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
#     control = VarTree(FixedSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')
#     rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
#     npts = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')

#     # outputs
#     V = Array(iotype='out', units='m/s', desc='aerodynamic power curve (wind speeds)')
#     P = Array(iotype='out', units='W', desc='aerodynamic power curve (corresponding power)')


#     def execute(self):

#         R = self.R
#         ctrl = self.control

#         # evaluate from cut-in to cut-out`
#         V = np.linspace(ctrl.Vin, ctrl.Vout, self.npts)
#         tsr = ctrl.Omega*RPM2RS*R/V

#         spline = interpolate.interp1d(self.tsr_array, self.cp_array, kind='cubic')
#         cp = spline(tsr)

#         # convert to dimensional form
#         self.V = V
#         self.P = cp * 0.5*self.rho*V**3 * pi*R**2


#     def provideJ(self):

#         inputs = ('tsr_array', 'cp_array', 'R')
#         outputs = ('V', 'P')

#         # TODO: need gradients of spline (and control points).  either Akima or cubic.
#         dP_dtsrarray = dcp_dtsrarray * 0.5*self.rho*V**3 * pi*R**2
#         dP_dcparray = dcp_dcparray * 0.5*self.rho*V**3 * pi*R**2
#         dP_dR = (2*cp + dcp_dtsr*tsr) * 0.5*self.rho*V**3 * pi*R
#         n = len(self.tsr_array)
#         dV = np.zeros(2*n+1)


# class RegulatedPowerCurve(Component):
#     """power curve after drivetrain efficiency losses and control regulation"""

#     # inputs
#     V = Array(iotype='in', units='m/s', desc='wind speeds')
#     P0 = Array(iotype='in', units='W', desc='unregulated power curve (but after drivetrain losses)')
#     ratedPower = Float(iotype='in', units='W', desc='rated power')

#     # outputs
#     ratedSpeed = Float(iotype='out', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
#     P = Array(iotype='out', units='W', desc='power curve (power)')

#     def execute(self):


#         self.P, self.dP_dP0 = smooth_max(self.P0, self.ratedPower, pct_offset=0.001)

#         # compute rated speed
#         self.drs_dp = np.zeros_like(self.P)
#         idx = np.argmax(self.P0)
#         self.ratedSpeed, drs_drp, self.drs_dp[:idx], drs_dv = interp_with_deriv(self.ratedPower, self.P0[:idx], self.V[:idx])



#     def linearize(self):
#         # TODO: remove this after openmdao updates to cache J
#         pass

#     def provideJ(self):
#         inputs = ('P0', )
#         outputs = ('ratedSpeed', 'P')

#         J = vstack([self.drs_dp, np.diag(self.dP_dP0)])

#         return inputs, outputs, J


class AEP(Component):
    """annual energy production"""

    # inputs
    CDF_V = Array(iotype='in')
    P = Array(iotype='in', units='W', desc='power curve (power)')
    lossFactor = Float(iotype='in', desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

    # outputs
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')


    def execute(self):

        self.AEP = self.lossFactor*np.trapz(self.P, self.CDF_V)/1e3*365.0*24.0  # in kWh


    def linearize(self):
        # TODO: remove this after openmdao updates to cache J
        pass

    def provideJ(self):
        inputs = ('CDF_V', 'P', 'lossFactor')
        outputs = ('AEP',)

        P = self.P
        CDF = self.CDF_V
        factor = self.lossFactor/1e3*365.0*24.0

        n = len(P)
        dAEP_dP = np.gradient(CDF)
        dAEP_dP[0] /= 2
        dAEP_dP[-1] /= 2
        dAEP_dP *= factor

        dAEP_dCDF = -np.gradient(P)
        dAEP_dCDF[0] = -0.5*(P[0] + P[1])
        dAEP_dCDF[-1] = 0.5*(P[-1] + P[-2])
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([self.AEP/self.lossFactor])

        J = np.zeros((1, 2*n+1))
        J[0, 0:n] = dAEP_dCDF
        J[0, n:2*n] = dAEP_dP
        J[0, 2*n] = dAEP_dlossFactor

        return inputs, outputs, J


# class SetupRatedConditionsVarSpeed(Component):
#     """setup to run AeroBase at rated conditions for a variable speed machine"""

#     # inputs
#     ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
#     control = VarTree(VarSpeedMachine(), iotype='in')
#     R = Float(iotype='in', units='m', desc='rotor radius')

#     # outputs (1D arrays because AeroBase expects Array input)
#     V_rated = Array(np.zeros(1), iotype='out', shape=((1,)), units='m/s', desc='rated speed')
#     Omega_rated = Array(np.zeros(1), iotype='out', shape=((1,)), units='rpm', desc='rotation speed at rated')
#     pitch_rated = Array(np.zeros(1), iotype='out', shape=((1,)), units='deg', desc='pitch setting at rated')


#     def execute(self):

#         ctrl = self.control

#         self.V_rated = np.array([self.ratedSpeed])
#         self.Omega_rated = np.array([min(ctrl.maxOmega, self.ratedSpeed*ctrl.tsr/self.R*RS2RPM)])
#         self.pitch_rated = np.array([ctrl.pitch])




# ---------------------
# Assemblies
# ---------------------


class NewAssembly(Assembly):

    # --- inputs ---
    # rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    control = VarTree(VarSpeedMachine(), iotype='in')

    # options
    npts_coarse_power_curve = Int(20, iotype='in', desc='number of points to evaluate aero analysis at')
    npts_spline_power_curve = Int(200, iotype='in', desc='number of points to use in fitting spline to power curve')
    AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')

    # --- slots (must replace) ---
    geom = Slot(GeomtrySetupBase)
    analysis = Slot(AeroBase)
    dt = Slot(DrivetrainLossesBase)

    def configure(self):

        self.add('geom', GeomtrySetupBase())
        self.add('setup', SetupRun())
        self.add('analysis', AeroBase())
        self.add('dt', DrivetrainLossesBase())
        self.add('powercurve', RegulatedPowerCurve())
        self.add('brent', Brent())
        self.add('cdf', CDFBase())
        self.add('aep', AEP())

        self.brent.workflow.add(['powercurve'])

        self.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'brent', 'cdf', 'aep'])

        # connections to setup
        self.connect('control', 'setup.control')
        self.connect('geom.R', 'setup.R')
        self.connect('npts_coarse_power_curve', 'setup.npts')

        # connections to analysis
        self.connect('setup.Uhub', 'analysis.Uhub')
        self.connect('setup.Omega', 'analysis.Omega')
        self.connect('setup.pitch', 'analysis.pitch')
        self.analysis.run_case = 'power'

        # connections to drivetrain
        self.connect('analysis.P', 'dt.aeroPower')
        self.connect('analysis.Q', 'dt.aeroTorque')
        self.connect('analysis.T', 'dt.aeroThrust')
        self.connect('control.ratedPower', 'dt.ratedPower')

        # connections to powercurve
        self.connect('control', 'powercurve.control')
        self.connect('setup.Uhub', 'powercurve.Vcoarse')
        self.connect('dt.power', 'powercurve.Pcoarse')
        self.connect('npts_spline_power_curve', 'powercurve.npts')

        # setup Brent method to find rated speed
        self.connect('control.Vin', 'brent.lower_bound')
        self.connect('control.Vout', 'brent.upper_bound')
        self.brent.add_parameter('powercurve.Vrated', low=-1e-15, high=1e15)
        self.brent.add_constraint('powercurve.residual = 0')

        # connections to cdf
        self.connect('powercurve.V', 'cdf.x')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')


        # self.add('setup', SetupRun())
        # self.add('analysis', AeroBase())
        # self.add('dt', DrivetrainLossesBase())
        # self.add('residual', Residual())
        # self.add('brent', Brent())


        # # connectiont to setup
        # self.connect('control', 'setup.control')
        # self.connect('geom.R', 'setup.R')

        # # connections to analysis
        # self.connect('setup.Uhub', 'analysis.Uhub')
        # self.connect('setup.Omega', 'analysis.Omega')
        # self.connect('setup.pitch', 'analysis.pitch')
        # self.analysis.run_case = 'power'

        # # connections to drivetrain
        # self.connect('analysis.P', 'dt.aeroPower')
        # self.connect('analysis.Q', 'dt.aeroTorque')
        # self.connect('analysis.T', 'dt.aeroThrust')
        # self.connect('control.ratedPower', 'dt.ratedPower')

        # # connections to residual
        # # self.dt.power = np.zeros(1)
        # self.connect('dt.power', 'residual.power')
        # self.connect('control.ratedPower', 'residual.ratedPower')

        # # setup Brent solver
        # self.connect('control.Vin', 'brent.lower_bound')
        # self.connect('control.Vout', 'brent.upper_bound')
        # # self.brent.lower_bound = self.control.Vin
        # # self.brent.upper_bound = self.control.Vout  # TODO: generalize Brent so this can be iterated on
        # self.brent.add_parameter(['setup.V'], low=-1e15, high=1e15)
        # self.brent.add_constraint('residual.residual = 0')
        # self.brent.workflow.add(['setup', 'analysis', 'dt', 'residual'])

        # self.driver.workflow.add(['geom', 'brent'])








class RotorAeroVS(Assembly):

    # --- inputs ---
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    control = VarTree(VarSpeedMachine(), iotype='in')

    # options
    tsr_sweep_step_size = Float(0.25, iotype='in', desc='step size in tip-speed ratio for sweep (if necessary)')
    npts_power_curve = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')
    AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')


    # --- slots (must replace) ---
    geom = Slot(GeomtrySetupBase)
    analysis = Slot(AeroBase)
    analysis2 = Slot(AeroBase)  # TODO: figure out how to replace both at same time
    dt = Slot(DrivetrainLossesBase)
    cdf = Slot(CDFBase)

    # --- outputs ---
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')
    V = Array(iotype='out', units='m/s', desc='wind speeds (power curve)')
    P = Array(iotype='out', units='W', desc='power (power curve)')
    rated = VarTree(RatedConditions(), iotype='out')



    def replace(self, name, obj):

        if name == 'analysis2':
            obj.T = np.zeros(1)
            obj.Q = np.zeros(1)

        super(RotorAeroVS, self).replace(name, obj)


    def configure(self):

        self.add('geom', GeomtrySetupBase())
        self.add('setup', SetupSweepVarSpeed())
        self.add('analysis', AeroBase())
        self.add('coeff', Coefficients())
        self.add('aeroPC', AeroPowerCurve())
        self.add('dt', DrivetrainLossesBase())
        self.add('powercurve', RegulatedPowerCurve())
        self.add('cdf', CDFBase())
        self.add('aep', AEP())
        self.add('rated_pre', SetupRatedConditionsVarSpeed())
        self.add('analysis2', AeroBase())

        self.driver.workflow.add(['geom', 'setup', 'analysis', 'coeff',
            'aeroPC', 'dt', 'powercurve', 'cdf', 'aep', 'rated_pre', 'analysis2'])

        # connections to setup
        self.connect('control', 'setup.control')
        self.connect('geom.R', 'setup.R')

        # connections to analysis
        self.connect('setup.Uinf', 'analysis.Uhub')
        self.connect('setup.Omega', 'analysis.Omega')
        self.connect('setup.pitch', 'analysis.pitch')
        self.analysis.run_case = 'power'

        # connections to coefficients
        self.connect('setup.Uinf', 'coeff.V')
        self.connect('analysis.T', 'coeff.T')
        self.connect('analysis.Q', 'coeff.Q')
        self.connect('analysis.P', 'coeff.P')
        self.connect('geom.R', 'coeff.R')
        self.connect('rho', 'coeff.rho')

        # connections to aeroPC
        self.connect('setup.tsr', 'aeroPC.tsr_array')
        self.connect('coeff.CP', 'aeroPC.CP_array')
        self.connect('coeff.CT', 'aeroPC.CT_array')
        self.connect('setup.Uinf_full', 'aeroPC.Uinf_full')
        self.connect('setup.Omega_full', 'aeroPC.Omega_full')
        self.connect('geom.R', 'aeroPC.R')
        self.connect('rho', 'aeroPC.rho')
        self.connect('npts_power_curve', 'aeroPC.npts')  # TODO: I think this isn't do anything anymore


        # connections to drivetrain
        self.connect('aeroPC.P', 'dt.aeroPower')
        self.connect('aeroPC.Q', 'dt.aeroTorque')
        self.connect('aeroPC.T', 'dt.aeroThrust')
        self.connect('control.ratedPower', 'dt.ratedPower')

        # connections to powercurve
        self.connect('setup.Uinf_full', 'powercurve.V')
        self.connect('dt.power', 'powercurve.P0')
        self.connect('control.ratedPower', 'powercurve.ratedPower')

        # connections to cdf
        self.connect('setup.Uinf_full', 'cdf.x')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to rated_pre
        self.connect('powercurve.ratedSpeed', 'rated_pre.ratedSpeed')
        self.connect('control', 'rated_pre.control')
        self.connect('geom.R', 'rated_pre.R')

        # connections to analysis2
        self.connect('rated_pre.V_rated', 'analysis2.Uhub')
        self.connect('rated_pre.Omega_rated', 'analysis2.Omega')
        self.connect('rated_pre.pitch_rated', 'analysis2.pitch')
        self.analysis2.run_case = 'power'
        self.analysis2.T = np.zeros(1)
        self.analysis2.Q = np.zeros(1)

        # connections to outputs
        self.connect('setup.Uinf_full', 'V')
        self.connect('powercurve.P', 'P')
        self.connect('aep.AEP', 'AEP')
        self.connect('rated_pre.V_rated[0]', 'rated.V')
        self.connect('rated_pre.Omega_rated[0]', 'rated.Omega')
        self.connect('rated_pre.pitch_rated[0]', 'rated.pitch')
        self.connect('analysis2.T[0]', 'rated.T')
        self.connect('analysis2.Q[0]', 'rated.Q')






# --------------------
# Not using right now
# ---------------------




class SetupPitchSearchVS(Component):

    control = VarTree(VarSpeedMachine(), iotype='in')

    R = Float(iotype='in', units='m', desc='rotor radius')
    ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')

    Omega = Float(iotype='out')
    pitch_min = Float(iotype='out')
    pitch_max = Float(iotype='out')

    def execute(self):

        ctrl = self.control
        self.Omega = min(ctrl.maxOmega, self.ratedSpeed*ctrl.tsr/self.R*RS2RPM)

        # pitch to feather
        self.pitch_min = ctrl.pitch
        self.pitch_max = 40.0


# @add_delegate(HasParameters, HasObjective)
# class Brent(Driver):
#     """root finding using Brent's method."""

#     fstar = Float(0.0, iotype='in', desc='function value sought (i.e., find root of f-fstar)')
#     xlow = Float(iotype='in')
#     xhigh = Float(iotype='in')

#     xstar = Float(iotype='out', desc='value sought.  find xstar s.t. f(xstar) = 0')


#     def eval(self, x):
#         """evaluate f(x)"""

#         self.set_parameter_by_name('x', x)
#         self.run_iteration()
#         obj = self.get_objectives()
#         f = obj['f'].evaluate(self.parent)

#         return f


#     def _error(self, x):
#         """solve: f - fstar = 0"""

#         f = self.eval(x)

#         return f - self.fstar


#     def execute(self):

#         # bounds
#         # params = self.get_parameters()
#         # xlow = params['x'].low
#         # xhigh = params['x'].high
#         xlow = self.xlow
#         xhigh = self.xhigh

#         # TODO: better error handling.  ideally, if this failed we would attempt to find
#         # bounds ourselves rather than throwing an error or returning a value at the bounds
#         if self.eval(xlow)*self.eval(xhigh) > 0:
#             raise Exception('bounds do not bracket a root')

#         # Brent's method
#         xstar = brentq(self._error, xlow, xhigh)

#         # set result
#         self.set_parameter_by_name('x', xstar)
#         self.xstar = xstar




# class RotorAeroVSVP(RotorAeroVS):

#     V_load = Float(iotype='in')
#     azimuth_load = Float(iotype='in')

#     def configure(self):
#         super(RotorAeroVSVP, self).configure()

#         # --- residual workflow ---

#         self.add('setuppitch', SetupPitchSearchVS())
#         # self.add('analysis3', AeroBase())  # TODO: replace back later
#         # self.add('dt2', DrivetrainLossesBase())
#         self.add('analysis3', CCBlade())
#         self.add('dt2', CSMDrivetrain())
#         self.add('brent', Brent())

#         # connections to setuppitch
#         self.connect('control', 'setuppitch.control')
#         self.connect('R', 'setuppitch.R')
#         self.connect('powercurve.ratedSpeed', 'setuppitch.ratedSpeed')

#         # connections to analysis3
#         self.analysis3.Uhub = np.zeros(1)
#         self.analysis3.Omega = np.zeros(1)
#         self.analysis3.pitch = np.zeros(1)
#         self.analysis3.run_case = 'power'
#         self.connect('V_load', 'analysis3.Uhub[0]')
#         self.connect('setuppitch.Omega', 'analysis3.Omega[0]')
#         self.brent.add_parameter('analysis3.pitch[0]', name='x', low=0.0, high=40.0)

#         # connections to drivetrain
#         self.connect('analysis3.P', 'dt2.aeroPower')
#         self.connect('control.ratedPower', 'dt2.ratedPower')

#         # brent setup
#         self.brent.workflow.add(['setuppitch', 'analysis3', 'dt2'])
#         self.brent.add_objective('dt2.power[0]', name='f')
#         self.connect('control.ratedPower', 'brent.fstar')
#         self.connect('setuppitch.pitch_min', 'brent.xlow')
#         self.connect('setuppitch.pitch_max', 'brent.xhigh')


#         # --- functional workflow ---

#         # self.add('analysis4', AeroBase())  TODO
#         self.add('analysis4', CCBlade())

#         self.driver.workflow.add(['brent', 'analysis4'])

#         # connections to analysis4
#         self.analysis4.run_case = 'loads'
#         self.connect('V_load', 'analysis4.V_load')
#         self.connect('setuppitch.Omega', 'analysis4.Omega_load')
#         self.connect('brent.xstar', 'analysis4.pitch_load')
#         self.connect('azimuth_load', 'analysis4.azimuth_load')

#         self.create_passthrough('analysis4.Np')
#         self.create_passthrough('analysis4.Tp')




#         # TODO: remove later
#         self.connect('r', 'analysis3.r')
#         self.connect('chord', 'analysis3.chord')
#         self.connect('theta', 'analysis3.theta')
#         self.connect('Rhub', 'analysis3.Rhub')
#         self.connect('Rtip', 'analysis3.Rtip')
#         self.connect('hubheight', 'analysis3.hubheight')
#         self.connect('airfoil_files', 'analysis3.airfoil_files')
#         self.connect('precone', 'analysis3.precone')
#         self.connect('tilt', 'analysis3.tilt')
#         self.connect('yaw', 'analysis3.yaw')
#         self.connect('B', 'analysis3.B')
#         self.connect('rho', 'analysis3.rho')
#         self.connect('mu', 'analysis3.mu')
#         self.connect('shearExp', 'analysis3.shearExp')
#         self.connect('nSector', 'analysis3.nSector')
#         self.connect('drivetrainType', 'dt2.drivetrainType')
#         self.connect('r', 'analysis4.r')
#         self.connect('chord', 'analysis4.chord')
#         self.connect('theta', 'analysis4.theta')
#         self.connect('Rhub', 'analysis4.Rhub')
#         self.connect('Rtip', 'analysis4.Rtip')
#         self.connect('hubheight', 'analysis4.hubheight')
#         self.connect('airfoil_files', 'analysis4.airfoil_files')
#         self.connect('precone', 'analysis4.precone')
#         self.connect('tilt', 'analysis4.tilt')
#         self.connect('yaw', 'analysis4.yaw')
#         self.connect('B', 'analysis4.B')
#         self.connect('rho', 'analysis4.rho')
#         self.connect('mu', 'analysis4.mu')
#         self.connect('shearExp', 'analysis4.shearExp')
#         self.connect('nSector', 'analysis4.nSector')


# def _setvar(comp, name, value):
#     vars = name.split('.')
#     base = comp
#     for i in range(len(vars)-1):
#         base = getattr(base, vars[i])

#     setattr(base, vars[-1], value)

# if __name__ == '__main__':

#     sr = SetupRun()
#     sr.control.Vin = 3.0
#     sr.control.Vout = 25.0
#     sr.control.tsr = 7.55
#     sr.control.maxOmega = 12.0
#     # sr.control.ratedPower = 5e6
#     # sr.control.Omega = 12.0
#     # sr.control.pitch = 1.0
#     sr.R = 62.9400379597

#     # setattr(sr, 'control.tsr', 8.0)
#     _setvar(sr, 'control.tsr', 8.0)
#     _setvar(sr, 'R', 3.0)

#     print sr.control.tsr
#     print sr.R
#     # sr.check_gradient()



#     # rpg = RegulatedPowerCurve()

#     # rpg.V = np.array([3.0, 3.11055276382, 3.22110552764, 3.33165829146, 3.44221105528, 3.5527638191, 3.66331658291, 3.77386934673, 3.88442211055, 3.99497487437, 4.10552763819, 4.21608040201, 4.32663316583, 4.43718592965, 4.54773869347, 4.65829145729, 4.76884422111, 4.87939698492, 4.98994974874, 5.10050251256, 5.21105527638, 5.3216080402, 5.43216080402, 5.54271356784, 5.65326633166, 5.76381909548, 5.8743718593, 5.98492462312, 6.09547738693, 6.20603015075, 6.31658291457, 6.42713567839, 6.53768844221, 6.64824120603, 6.75879396985, 6.86934673367, 6.97989949749, 7.09045226131, 7.20100502513, 7.31155778894, 7.42211055276, 7.53266331658, 7.6432160804, 7.75376884422, 7.86432160804, 7.97487437186, 8.08542713568, 8.1959798995, 8.30653266332, 8.41708542714, 8.52763819095, 8.63819095477, 8.74874371859, 8.85929648241, 8.96984924623, 9.08040201005, 9.19095477387, 9.30150753769, 9.41206030151, 9.52261306533, 9.63316582915, 9.74371859296, 9.85427135678, 9.9648241206, 10.0753768844, 10.1859296482, 10.2964824121, 10.4070351759, 10.5175879397, 10.6281407035, 10.7386934673, 10.8492462312, 10.959798995, 11.0703517588, 11.1809045226, 11.2914572864, 11.4020100503, 11.5125628141, 11.6231155779, 11.7336683417, 11.8442211055, 11.9547738693, 12.0653266332, 12.175879397, 12.2864321608, 12.3969849246, 12.5075376884, 12.6180904523, 12.7286432161, 12.8391959799, 12.9497487437, 13.0603015075, 13.1708542714, 13.2814070352, 13.391959799, 13.5025125628, 13.6130653266, 13.7236180905, 13.8341708543, 13.9447236181, 14.0552763819, 14.1658291457, 14.2763819095, 14.3869346734, 14.4974874372, 14.608040201, 14.7185929648, 14.8291457286, 14.9396984925, 15.0502512563, 15.1608040201, 15.2713567839, 15.3819095477, 15.4924623116, 15.6030150754, 15.7135678392, 15.824120603, 15.9346733668, 16.0452261307, 16.1557788945, 16.2663316583, 16.3768844221, 16.4874371859, 16.5979899497, 16.7085427136, 16.8190954774, 16.9296482412, 17.040201005, 17.1507537688, 17.2613065327, 17.3718592965, 17.4824120603, 17.5929648241, 17.7035175879, 17.8140703518, 17.9246231156, 18.0351758794, 18.1457286432, 18.256281407, 18.3668341709, 18.4773869347, 18.5879396985, 18.6984924623, 18.8090452261, 18.9195979899, 19.0301507538, 19.1407035176, 19.2512562814, 19.3618090452, 19.472361809, 19.5829145729, 19.6934673367, 19.8040201005, 19.9145728643, 20.0251256281, 20.135678392, 20.2462311558, 20.3567839196, 20.4673366834, 20.5778894472, 20.6884422111, 20.7989949749, 20.9095477387, 21.0201005025, 21.1306532663, 21.2412060302, 21.351758794, 21.4623115578, 21.5728643216, 21.6834170854, 21.7939698492, 21.9045226131, 22.0150753769, 22.1256281407, 22.2361809045, 22.3467336683, 22.4572864322, 22.567839196, 22.6783919598, 22.7889447236, 22.8994974874, 23.0100502513, 23.1206030151, 23.2311557789, 23.3417085427, 23.4522613065, 23.5628140704, 23.6733668342, 23.783919598, 23.8944723618, 24.0050251256, 24.1155778894, 24.2261306533, 24.3366834171, 24.4472361809, 24.5577889447, 24.6683417085, 24.7788944724, 24.8894472362, 25.0])
#     # rpg.P0 = np.array([22025.3980736, 31942.1181193, 42589.3997593, 53993.208026, 66179.5079517, 79174.2645691, 93003.4429104, 107693.008008, 123268.924895, 139757.158602, 157183.674164, 175574.436612, 194955.410978, 215352.562295, 236791.855596, 259299.255912, 282900.728277, 307622.237722, 333489.749281, 360529.227985, 388766.638867, 418227.946959, 448939.117295, 480926.114905, 514214.904824, 548831.452082, 584801.721713, 622151.678749, 660907.288223, 701094.515166, 742739.324611, 785867.681592, 830505.551139, 876678.898286, 924413.688065, 973735.885508, 1024671.45565, 1077246.36352, 1131486.57415, 1187418.05257, 1245066.76383, 1304458.67294, 1365619.74494, 1428575.94487, 1493353.23775, 1559977.58862, 1628474.96251, 1698871.32446, 1771192.63949, 1845464.87264, 1921713.98894, 1999965.95343, 2080246.73113, 2162582.28708, 2246998.58631, 2333521.59385, 2422177.27474, 2512991.594, 2605990.51668, 2701200.00779, 2798646.03239, 2898354.55549, 3000351.54213, 3104662.13686, 3211234.53779, 3320020.79612, 3430990.56802, 3544080.65515, 3659191.80334, 3776184.86917, 3894876.35452, 4015033.31145, 4136367.65351, 4258680.0611, 4382273.3287, 4507146.67229, 4631519.40922, 4757024.27387, 4883654.30446, 5011339.43105, 5139999.82993, 5269555.60053, 5399916.0227, 5530964.29104, 5662579.748, 5794641.68428, 5927024.45923, 6059593.49951, 6192213.29006, 6324748.31256, 6457056.63766, 6588976.52285, 6720342.40998, 6850988.7409, 6980745.41449, 7109366.13506, 7236541.44259, 7361959.96889, 7485310.3458, 7606287.48886, 7724653.16081, 7840209.47866, 7952759.11774, 8062104.75335, 8168052.88536, 8270559.25623, 8369773.80644, 8465859.48794, 8558979.25269, 8649296.05265, 8736964.98421, 8822040.79359, 8904508.92236, 8984353.43362, 9061558.39044, 9136107.8559, 9207987.76139, 9277282.30024, 9344220.50785, 9409043.04585, 9471990.57587, 9533303.75954, 9593223.25849, 9651987.57125, 9709773.37109, 9766687.94617, 9822834.90328, 9878317.84921, 9933240.39074, 9987706.13466, 10041818.6878, 10095670.7381, 10149288.0659, 10202671.1196, 10255820.2999, 10308736.0076, 10361418.6433, 10413868.6078, 10466086.3017, 10518071.9311, 10569819.99, 10621318.4776, 10672555.0406, 10723517.3258, 10774192.9802, 10824569.6505, 10874634.9836, 10924376.6264, 10973782.2235, 11022835.9676, 11071511.5564, 11119780.7013, 11167615.1132, 11214986.5034, 11261866.583, 11308227.063, 11354039.6548, 11399276.0692, 11443908.0176, 11487907.5393, 11531251.3389, 11573919.5895, 11615892.5469, 11657150.4667, 11697673.6046, 11737442.2163, 11776436.5575, 11814636.8837, 11852023.4508, 11888576.5144, 11924276.3301, 11959104.5046, 11993052.6433, 12026116.8385, 12058293.2041, 12089577.8538, 12119966.9011, 12149456.4599, 12178042.6439, 12205721.5667, 12232489.3421, 12258342.0838, 12283275.9055, 12307286.921, 12330371.2701, 12352526.2533, 12373750.7632, 12394043.8081, 12413404.3963, 12431831.5361, 12449324.2358, 12465881.5035, 12481502.3477, 12496185.7765, 12509930.7982, 12522736.4212, 12534601.6536, 12545525.5037, 12555506.9798, 12564567.6944])
#     # rpg.ratedPower = 50e6

#     # rpg.run()
#     # print rpg.P0[78:82]

#     # # print rpg.P0

#     # # print rpg.P

#     # Pvec = 5e6*np.linspace(0.99, 1.01, 100)
#     # out = []

#     # for p in Pvec:
#     #     rpg.P0[79] = p

#     #     rpg.run()

#     #     out.append(rpg.P[79])

#     # import matplotlib.pyplot as plt
#     # plt.plot(Pvec, out)
#     # plt.show()



