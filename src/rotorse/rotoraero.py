#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi
from openmdao.main.api import VariableTree, Component, Assembly, ImplicitComponent
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot, Enum
from openmdao.lib.drivers.api import Brent

from commonse.utilities import hstack, vstack, linspace_with_deriv, smooth_min
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
    # tilt = Float(units='deg', desc='tilt angle')



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

    # TODO: remove when OpenMDAO fixes this
    missing_deriv_policy = 'assume_zero'


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

# This Component is now no longer used, but I'll keep it around for now in case that changes.
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


    def list_deriv_vars(self):

        inputs = ('V', 'T', 'Q', 'P', 'R')
        outputs = ('CT', 'CQ', 'CP')

        return inputs, outputs


    def provideJ(self):

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

        return J




class SetupRun(Component):
    """determines approriate conditions to run AeroBase code across the power curve"""

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

        # velocity sweep
        V = np.linspace(ctrl.Vin, ctrl.Vout, n)

        # corresponding rotation speed
        Omega_d = ctrl.tsr*V/R*RS2RPM
        Omega, dOmega_dOmegad = smooth_min(Omega_d, ctrl.maxOmega, pct_offset=0.01)

        # store values
        self.Uhub = V
        self.Omega = Omega
        self.pitch = ctrl.pitch*np.ones_like(V)

        # gradients
        dV = np.zeros((n, 2))
        dOmega_dtsr = dOmega_dOmegad * V/R*RS2RPM
        dOmega_dR = dOmega_dOmegad * -ctrl.tsr*V/R**2*RS2RPM
        dOmega = hstack([dOmega_dtsr, dOmega_dR])
        dpitch = np.zeros((n, 2))
        self.J = vstack([dV, dOmega, dpitch])


    def list_deriv_vars(self):

        inputs = ('control.tsr', 'R')
        outputs = ('Uhub', 'Omega', 'pitch')

        return inputs, outputs

    def provideJ(self):

        return self.J





class RegulatedPowerCurve(ImplicitComponent):
    """Fit a spline to the coarse sampled power curve (and thrust curve),
    find rated speed through a residual convergence strategy,
    then compute the regulated power curve and rated conditions"""

    # inputs
    control = VarTree(VarSpeedMachine(), iotype='in')
    Vcoarse = Array(iotype='in', units='m/s', desc='wind speeds')
    Pcoarse = Array(iotype='in', units='W', desc='unregulated power curve (but after drivetrain losses)')
    Tcoarse = Array(iotype='in', units='N', desc='unregulated thrust curve')
    R = Float(iotype='in', units='m', desc='rotor radius')
    npts = Int(200, iotype='in', desc='number of points for splined power curve')

    # state
    Vrated = Float(iotype='state', units='m/s', desc='rated wind speed')

    # residual
    residual = Float(iotype="residual")

    # outputs
    V = Array(iotype='out', units='m/s', desc='wind speeds')
    P = Array(iotype='out', units='W', desc='power')
    ratedConditions = VarTree(RatedConditions(), iotype='out', desc='conditions at rated speed')



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

        # rated speed conditions
        Omega_d = ctrl.tsr*Vrated/self.R*RS2RPM
        OmegaRated, dOmegaRated_dOmegad = smooth_min(Omega_d, ctrl.maxOmega, pct_offset=0.01)

        splineT = Akima(self.Vcoarse, self.Tcoarse)
        Trated, dT_dVrated, dT_dVcoarse, dT_dTcoarse = splineT.interp(Vrated)

        self.ratedConditions.V = Vrated
        self.ratedConditions.Omega = OmegaRated
        self.ratedConditions.pitch = ctrl.pitch
        self.ratedConditions.T = Trated
        self.ratedConditions.Q = ctrl.ratedPower / (self.ratedConditions.Omega * RPM2RS)


        # gradients
        ncoarse = len(self.Vcoarse)

        dres = np.concatenate([[0.0], dres_dVcoarse, dres_dPcoarse, np.zeros(ncoarse), np.array([dres_dVrated]), [0.0]])

        dV_dVrated = np.concatenate([dV2_dVrated, dV3_dVrated])
        dV = hstack([np.zeros((n, 1)), np.zeros((n, 3*ncoarse)), dV_dVrated, np.zeros((n, 1))])

        dP_dVcoarse = vstack([dP2_dVcoarse, np.zeros((n/2, ncoarse))])
        dP_dPcoarse = vstack([dP2_dPcoarse, np.zeros((n/2, ncoarse))])
        dP_dVrated = np.concatenate([dP2_dV2*dV2_dVrated, np.zeros(n/2)])
        dP = hstack([np.zeros((n, 1)), dP_dVcoarse, dP_dPcoarse, np.zeros((n, ncoarse)), dP_dVrated, np.zeros((n, 1))])

        drV = np.concatenate([[0.0], np.zeros(3*ncoarse), [1.0, 0.0]])
        drOmega = np.concatenate([[dOmegaRated_dOmegad*Vrated/self.R*RS2RPM], np.zeros(3*ncoarse),
            [dOmegaRated_dOmegad*ctrl.tsr/self.R*RS2RPM, -dOmegaRated_dOmegad*ctrl.tsr*Vrated/self.R**2*RS2RPM]])
        drpitch = np.zeros(3*ncoarse+3)
        drT = np.concatenate([[0.0], dT_dVcoarse, np.zeros(ncoarse), dT_dTcoarse, [dT_dVrated, 0.0]])
        drQ = -ctrl.ratedPower / (self.ratedConditions.Omega**2 * RPM2RS) * drOmega

        self.J = vstack([dres, dV, dP, drV, drOmega, drpitch, drT, drQ])


    def list_deriv_vars(self):

        inputs = ('control.tsr', 'Vcoarse', 'Pcoarse', 'Tcoarse', 'Vrated', 'R')
        outputs = ('residual', 'V', 'P', 'ratedConditions.V', 'ratedConditions.Omega',
            'ratedConditions.pitch', 'ratedConditions.T', 'ratedConditions.Q')

        return inputs, outputs

    def provideJ(self):

        return self.J






class AEP(Component):
    """integrate to find annual energy production"""

    # inputs
    CDF_V = Array(iotype='in')
    P = Array(iotype='in', units='W', desc='power curve (power)')
    lossFactor = Float(iotype='in', desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

    # outputs
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')


    def execute(self):

        self.AEP = self.lossFactor*np.trapz(self.P, self.CDF_V)/1e3*365.0*24.0  # in kWh


    def list_deriv_vars(self):

        inputs = ('CDF_V', 'P', 'lossFactor')
        outputs = ('AEP',)

        return inputs, outputs


    def provideJ(self):

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

        return J



# ---------------------
# Assemblies
# ---------------------


class RotorAeroVS(Assembly):

    # --- inputs ---
    # rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    control = VarTree(VarSpeedMachine(), iotype='in')

    # options
    npts_coarse_power_curve = Int(20, iotype='in', desc='number of points to evaluate aero analysis at')
    npts_spline_power_curve = Int(200, iotype='in', desc='number of points to use in fitting spline to power curve')
    AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')

    # slots (must replace)
    geom = Slot(GeomtrySetupBase)
    analysis = Slot(AeroBase)
    dt = Slot(DrivetrainLossesBase)
    cdf = Slot(CDFBase)

    # --- outputs ---
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')
    V = Array(iotype='out', units='m/s', desc='wind speeds (power curve)')
    P = Array(iotype='out', units='W', desc='power (power curve)')
    ratedConditions = VarTree(RatedConditions(), iotype='out')
    diameter = Float(iotype='out', units='m')


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
        self.connect('analysis.T', 'powercurve.Tcoarse')
        self.connect('geom.R', 'powercurve.R')
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


        # connections to outputs
        self.connect('powercurve.V', 'V')
        self.connect('powercurve.P', 'P')
        self.connect('aep.AEP', 'AEP')
        self.connect('powercurve.ratedConditions', 'ratedConditions')
        self.connect('2*geom.R', 'diameter')




