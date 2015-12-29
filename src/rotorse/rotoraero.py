#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi
from openmdao.core.component import Component
from openmdao.components.execcomp import ExecComp
from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
# from openmdao.main.api import VariableTree, Component, Assembly, ImplicitComponent
# from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot, Enum
# from openmdao.lib.drivers.api import Brent

from commonse.utilities import hstack, vstack, linspace_with_deriv, smooth_min, trapz_deriv
from akima import Akima


# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi

# ---------------------
# Base Components
# ---------------------

class GeomtrySetupBase(Component):
    def __init__(self):
        super(GeomtrySetupBase).__init__()
        """a base component that computes the rotor radius from the geometry"""

        self.add_output('R', units='m', desc='rotor radius')


class AeroBase(Component):
    def __init__(self):
        super(AeroBase).__init__()
        """A base component for a rotor aerodynamics code."""

        self.add_param('run_case', val=Enum('power', ('power', 'loads')))


        # --- use these if (run_case == 'power') ---

        # inputs
        self.add_param('Uhub', units='m/s', desc='hub height wind speed')
        self.add_param('Omega', units='rpm', desc='rotor rotation speed')
        self.add_param('pitch', units='deg', desc='blade pitch setting')

        # outputs
        self.add_param('T', units='N', desc='rotor aerodynamic thrust')
        self.add_param('Q', units='N*m', desc='rotor aerodynamic torque')
        self.add_param('P', units='W', desc='rotor aerodynamic power')


        # --- use these if (run_case == 'loads') ---
        # if you only use rotoraero.py and not rotor.py
        # (i.e., only care about power curves, and not structural loads)
        # then these second set of inputs/outputs are not needed

        # inputs
        self.add_param('V_load', units='m/s', desc='hub height wind speed')
        self.add_param('Omega_load', units='rpm', desc='rotor rotation speed')
        self.add_param('pitch_load', units='deg', desc='blade pitch setting')
        self.add_param('azimuth_load', units='deg', desc='blade azimuthal location')

        # outputs
        # loads = VarTree(AeroLoads(), iotype='out', desc='loads in blade-aligned coordinate system')
        self.add_output('loads.r', units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads.Px', units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads.Py', units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads.Pz', units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_output('loads.V', units='m/s', desc='hub height wind speed')
        self.add_output('loads.Omega', units='rpm', desc='rotor rotation speed')
        self.add_output('loads.pitch', units='deg', desc='pitch angle')
        self.add_output('loads.azimuth', units='deg', desc='azimuthal angle')




class DrivetrainLossesBase(Component):
    def __init__(self):
        super(DrivetrainLossesBase).__init__()
        """base component for drivetrain efficiency losses"""

        self.add_param('aeroPower', units='W', desc='aerodynamic power')
        self.add_param('aeroTorque', units='N*m', desc='aerodynamic torque')
        self.add_param('aeroThrust', units='N', desc='aerodynamic thrust')
        self.add_param('ratedPower', units='W', desc='rated power')

        self.add_output('power', units='W', desc='total power after drivetrain losses')
        self.add_output('rpm',units='rpm', desc='rpm curve after drivetrain losses')


class PDFBase(Component):
    def __init__(self):
        super(PDFBase).__init__()
        """probability distribution function"""

        self.add_param('x')

        self.add_output('f')


class CDFBase(Component):
    def __init__(self):
        super(CDFBase).__init__()
        """cumulative distribution function"""
        self.add_param('x')

        self.add_output('F')


# ---------------------
# Components
# ---------------------

# class MaxTipSpeed(Component):

#     R = Float(iotype='in', units='m', desc='rotor radius')
#     Vtip_max = Float(iotype='in', units='m/s', desc='maximum tip speed')

#     Omega_max = Float(iotype='out', units='rpm', desc='maximum rotation speed')

#     def execute(self):

#         self.Omega_max = self.Vtip_max/self.R * RS2RPM


#     def list_deriv_vars(self):

#         inputs = ('R', 'Vtip_max')
#         outputs = ('Omega_max',)

#         return inputs, outputs


#     def provideJ(self):

#         J = np.array([[-self.Vtip_max/self.R**2*RS2RPM, RS2RPM/self.R]])

#         return J


# This Component is now no longer used, but I'll keep it around for now in case that changes.
class Coefficients(Component):
    def __init__(self):
        super(Coefficients).__init__()
        """convert power, thrust, torque into nondimensional coefficient form"""

        # inputs
        self.add_param('V', units='m/s', desc='wind speed')
        self.add_param('T', units='N', desc='rotor aerodynamic thrust')
        self.add_param('Q', units='N*m', desc='rotor aerodynamic torque')
        self.add_param('P', units='W', desc='rotor aerodynamic power')

        # inputs used in normalization
        self.add_param('R', units='m', desc='rotor radius')
        self.add_param('rho', units='kg/m**3', desc='density of fluid')


        # outputs
        self.add_output('CT', desc='rotor aerodynamic thrust')
        self.add_output('CQ', desc='rotor aerodynamic torque')
        self.add_output('CP', desc='rotor aerodynamic power')


    def solve_nonlinear(self, params, unknowns, resids):

        q = 0.5 * self.rho * self.V**2
        A = pi * self.R**2
        self.CP = self.P / (q * A * self.V)
        self.CT = self.T / (q * A)
        self.CQ = self.Q / (q * self.R * A)


    def list_deriv_vars(self):

        inputs = ('V', 'T', 'Q', 'P', 'R')
        outputs = ('CT', 'CQ', 'CP')

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

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



class SetupRunFixedSpeed(Component):
    def __init__(self):
        super(SetupRunFixedSpeed).__init__()
        """determines approriate conditions to run AeroBase code across the power curve"""

        self.add_param('control.Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control.Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control.ratedPower', units='W', desc='rated power')
        self.add_param('control.Omega', units='rpm', desc='fixed rotor rotation speed')
        self.add_param('control.pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_param('control.npts', val=20, desc='number of points to evalute aero code to generate power curve')

        # outputs
        self.add_output('Uhub', units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', units='deg', desc='pitch angles to run')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

        ctrl = self.control
        n = self.npts

        # velocity sweep
        V = np.linspace(ctrl.Vin, ctrl.Vout, n)

        # store values
        self.Uhub = V
        self.Omega = ctrl.Omega*np.ones_like(V)
        self.pitch = ctrl.pitch*np.ones_like(V)



    def list_deriv_vars(self):

        inputs = ('',)  # everything is constant
        outputs = ('',)

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        return []



class SetupRunVarSpeed(Component):
    def __init__(self):
        super(SetupRunVarSpeed).__init__()
        """determines approriate conditions to run AeroBase code across the power curve"""

        self.add_param('control.Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control.Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control.ratedPower', units='W', desc='rated power')
        self.add_param('control.minOmega', units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control.maxOmega', units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control.tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control.pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('R', units='m', desc='rotor radius')
        self.add_param('npts', val=20, desc='number of points to evalute aero code to generate power curve')

        # outputs
        self.add_output('Uhub', units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', units='deg', desc='pitch angles to run')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

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
        Omega, dOmega_dOmegad, dOmega_dmaxOmega = smooth_min(Omega_d, ctrl.maxOmega, pct_offset=0.01)

        # store values
        self.Uhub = V
        self.Omega = Omega
        self.pitch = ctrl.pitch*np.ones_like(V)

        # gradients
        dV = np.zeros((n, 3))
        dOmega_dtsr = dOmega_dOmegad * V/R*RS2RPM
        dOmega_dR = dOmega_dOmegad * -ctrl.tsr*V/R**2*RS2RPM
        dOmega = hstack([dOmega_dtsr, dOmega_dR, dOmega_dmaxOmega])
        dpitch = np.zeros((n, 3))
        self.J = vstack([dV, dOmega, dpitch])


    def list_deriv_vars(self):

        inputs = ('control.tsr', 'R', 'control.maxOmega')
        outputs = ('Uhub', 'Omega', 'pitch')

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        return self.J





class UnregulatedPowerCurve(Component):
    def __init__(self):
        super(UnregulatedPowerCurve).__init__()
        # inputs
        self.add_param('control.Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control.Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control.ratedPower', units='W', desc='rated power')
        self.add_param('control.Omega', units='rpm', desc='fixed rotor rotation speed')
        self.add_param('control.pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_param('control.npts', val=20, desc='number of points to evalute aero code to generate power curve')

        self.add_param('Vcoarse', units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', units='N', desc='unregulated thrust curve')
        self.add_param('npts', val=200, desc='number of points for splined power curve')


        # outputs
        self.add_output('V', units='m/s', desc='wind speeds')
        self.add_output('P', units='W', desc='power')

        missing_deriv_policy = 'assume_zero'


    def solve_nonlinear(self, params, unknowns, resids):

        ctrl = self.control
        n = self.npts

        # finer power curve
        self.V, _, _ = linspace_with_deriv(ctrl.Vin, ctrl.Vout, n)
        spline = Akima(self.Vcoarse, self.Pcoarse)
        self.P, dP_dV, dP_dVcoarse, dP_dPcoarse = spline.interp(self.V)

        self.J = hstack([dP_dVcoarse, dP_dPcoarse])

    def list_deriv_vars(self):

        inputs = ('Vcoarse', 'Pcoarse')
        outputs = ('P')

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        return self.J


class RegulatedPowerCurve(Component): # Implicit COMPONENT


    def __init__(self):
        super(RegulatedPowerCurve, self).__init__()
        self.eval_only = True  # allows an external solver to converge this, otherwise it will converge itself to mimic an explicit comp
        """Fit a spline to the coarse sampled power curve (and thrust curve),
        find rated speed through a residual convergence strategy,
        then compute the regulated power curve and rated conditions"""

        # inputs
        self.add_param('control.Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control.Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control.ratedPower', units='W', desc='rated power')
        self.add_param('control.minOmega', units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control.maxOmega', units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control.tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control.pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('Vcoarse', units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', units='N', desc='unregulated thrust curve')
        self.add_param('R', units='m', desc='rotor radius')
        self.add_param('npts', val=200, desc='number of points for splined power curve')

        # state
        self.add_state('Vrated', units='m/s', desc='rated wind speed')

        # residual
        self.resids('residual')

        # outputs
        self.add_output('V', units='m/s', desc='wind speeds')
        self.add_output('P', units='W', desc='power')

        self.add_output('ratedConditions.V', units='m/s', desc='rated wind speed')
        self.add_output('ratedConditions.Omega', units='rpm', desc='rotor rotation speed at rated')
        self.add_output('ratedConditions.pitch', units='deg', desc='pitch setting at rated')
        self.add_output('ratedConditions.T', units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('ratedConditions.Q', units='N*m', desc='rotor aerodynamic torque at rated')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

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
        OmegaRated, dOmegaRated_dOmegad, dOmegaRated_dmaxOmega \
            = smooth_min(Omega_d, ctrl.maxOmega, pct_offset=0.01)

        splineT = Akima(self.Vcoarse, self.Tcoarse)
        Trated, dT_dVrated, dT_dVcoarse, dT_dTcoarse = splineT.interp(Vrated)

        self.ratedConditions.V = Vrated
        self.ratedConditions.Omega = OmegaRated
        self.ratedConditions.pitch = ctrl.pitch
        self.ratedConditions.T = Trated
        self.ratedConditions.Q = ctrl.ratedPower / (self.ratedConditions.Omega * RPM2RS)


        # gradients
        ncoarse = len(self.Vcoarse)

        dres = np.concatenate([[0.0], dres_dVcoarse, dres_dPcoarse, np.zeros(ncoarse), np.array([dres_dVrated]), [0.0, 0.0]])

        dV_dVrated = np.concatenate([dV2_dVrated, dV3_dVrated])
        dV = hstack([np.zeros((n, 1)), np.zeros((n, 3*ncoarse)), dV_dVrated, np.zeros((n, 2))])

        dP_dVcoarse = vstack([dP2_dVcoarse, np.zeros((n/2, ncoarse))])
        dP_dPcoarse = vstack([dP2_dPcoarse, np.zeros((n/2, ncoarse))])
        dP_dVrated = np.concatenate([dP2_dV2*dV2_dVrated, np.zeros(n/2)])
        dP = hstack([np.zeros((n, 1)), dP_dVcoarse, dP_dPcoarse, np.zeros((n, ncoarse)), dP_dVrated, np.zeros((n, 2))])

        drV = np.concatenate([[0.0], np.zeros(3*ncoarse), [1.0, 0.0, 0.0]])
        drOmega = np.concatenate([[dOmegaRated_dOmegad*Vrated/self.R*RS2RPM], np.zeros(3*ncoarse),
            [dOmegaRated_dOmegad*ctrl.tsr/self.R*RS2RPM, -dOmegaRated_dOmegad*ctrl.tsr*Vrated/self.R**2*RS2RPM,
            dOmegaRated_dmaxOmega]])
        drpitch = np.zeros(3*ncoarse+4)
        drT = np.concatenate([[0.0], dT_dVcoarse, np.zeros(ncoarse), dT_dTcoarse, [dT_dVrated, 0.0, 0.0]])
        drQ = -ctrl.ratedPower / (self.ratedConditions.Omega**2 * RPM2RS) * drOmega

        self.J = vstack([dres, dV, dP, drV, drOmega, drpitch, drT, drQ])


    def list_deriv_vars(self):

        inputs = ('control.tsr', 'Vcoarse', 'Pcoarse', 'Tcoarse', 'Vrated', 'R', 'control.maxOmega')
        outputs = ('residual', 'V', 'P', 'ratedConditions.V', 'ratedConditions.Omega',
            'ratedConditions.pitch', 'ratedConditions.T', 'ratedConditions.Q')

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        return self.J






class AEP(Component):
    def __init__(self):
        super(AEP).__init__()
        """integrate to find annual energy production"""

        # inputs
        self.add_param('CDF_V', desc='cumulative distribution function evaluated at each wind speed')
        self.add_param('P', units='W', desc='power curve (power)')
        self.add_param('lossFactor', desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', units='kW*h', desc='annual energy production')


    def solve_nonlinear(self, params, unknowns, resids):

        self.AEP = self.lossFactor*np.trapz(self.P, self.CDF_V)/1e3*365.0*24.0  # in kWh


    def list_deriv_vars(self):

        inputs = ('CDF_V', 'P', 'lossFactor')
        outputs = ('AEP',)

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        factor = self.lossFactor/1e3*365.0*24.0

        dAEP_dP, dAEP_dCDF = trapz_deriv(self.P, self.CDF_V)
        dAEP_dP *= factor
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([self.AEP/self.lossFactor])

        n = len(self.P)
        J = np.zeros((1, 2*n+1))
        J[0, 0:n] = dAEP_dCDF
        J[0, n:2*n] = dAEP_dP
        J[0, 2*n] = dAEP_dlossFactor

        return J



# ---------------------
# Assemblies
# ---------------------


def common_io(assembly, varspeed, varpitch):

    regulated = varspeed or varpitch

    # add inputs
    assembly.add_param('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at')
    assembly.add_param('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve')
    assembly.add_param('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)')
    if varspeed:
        assembly.add_param('control.Vin', units='m/s', desc='cut-in wind speed')
        assembly.add_param('control.Vout', units='m/s', desc='cut-out wind speed')
        assembly.add_param('control.ratedPower', units='W', desc='rated power')
        assembly.add_param('control.minOmega', units='rpm', desc='minimum allowed rotor rotation speed')
        assembly.add_param('control.maxOmega', units='rpm', desc='maximum allowed rotor rotation speed')
        assembly.add_param('control.tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        assembly.add_param('control.pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
    else:
        assembly.add_param('control.Vin', units='m/s', desc='cut-in wind speed')
        assembly.add_param('control.Vout', units='m/s', desc='cut-out wind speed')
        assembly.add_param('control.ratedPower', units='W', desc='rated power')
        assembly.add_param('control.Omega', units='rpm', desc='fixed rotor rotation speed')
        assembly.add_param('control.pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        assembly.add_param('control.npts', val=20, desc='number of points to evalute aero code to generate power curve')


    # add slots (must replace)
    assembly.add('geom', Slot(GeomtrySetupBase))
    assembly.add('analysis', Slot(AeroBase))
    assembly.add('dt', Slot(DrivetrainLossesBase))
    assembly.add('cdf', Slot(CDFBase))


    # add outputs
    assembly.add_output('AEP', units='kW*h', desc='annual energy production')
    assembly.add_output('V', units='m/s', desc='wind speeds (power curve)')
    assembly.add_output('P', units='W', desc='power (power curve)')
    assembly.add_output('diameter', units='m', desc='rotor diameter')
    if regulated:
        assembly.add_output('ratedConditions.V', units='m/s', desc='rated wind speed')
        assembly.add_output('ratedConditions.Omega', units='rpm', desc='rotor rotation speed at rated')
        assembly.add_output('ratedConditions.pitch', units='deg', desc='pitch setting at rated')
        assembly.add_output('ratedConditions.T', units='N', desc='rotor aerodynamic thrust at rated')
        assembly.add_output('ratedConditions.Q', units='N*m', desc='rotor aerodynamic torque at rated')


def common_configure(assembly, varspeed, varpitch):

    regulated = varspeed or varpitch

    # add components
    assembly.add('geom', GeomtrySetupBase())

    if varspeed:
        assembly.add('setup', SetupRunVarSpeed())
    else:
        assembly.add('setup', SetupRunFixedSpeed())

    assembly.add('analysis', AeroBase())
    assembly.add('dt', DrivetrainLossesBase())

    if varspeed or varpitch:
        assembly.add('powercurve', RegulatedPowerCurve())
        assembly.add('brent', Brent())
        assembly.brent.workflow.add(['powercurve'])
    else:
        assembly.add('powercurve', UnregulatedPowerCurve())

    assembly.add('cdf', CDFBase())
    assembly.add('aep', AEP())

    if regulated:
        assembly.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'brent', 'cdf', 'aep'])
    else:
        assembly.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'powercurve', 'cdf', 'aep'])


    # connections to setup
    assembly.connect('control', 'setup.control')
    assembly.connect('npts_coarse_power_curve', 'setup.npts')
    if varspeed:
        assembly.connect('geom.R', 'setup.R')


    # connections to analysis
    assembly.connect('setup.Uhub', 'analysis.Uhub')
    assembly.connect('setup.Omega', 'analysis.Omega')
    assembly.connect('setup.pitch', 'analysis.pitch')
    assembly.analysis.run_case = 'power'


    # connections to drivetrain
    assembly.connect('analysis.P', 'dt.aeroPower')
    assembly.connect('analysis.Q', 'dt.aeroTorque')
    assembly.connect('analysis.T', 'dt.aeroThrust')
    assembly.connect('control.ratedPower', 'dt.ratedPower')


    # connections to powercurve
    assembly.connect('control', 'powercurve.control')
    assembly.connect('setup.Uhub', 'powercurve.Vcoarse')
    assembly.connect('dt.power', 'powercurve.Pcoarse')
    assembly.connect('analysis.T', 'powercurve.Tcoarse')
    assembly.connect('npts_spline_power_curve', 'powercurve.npts')

    if regulated:
        assembly.connect('geom.R', 'powercurve.R')

        # setup Brent method to find rated speed
        assembly.connect('control.Vin', 'brent.lower_bound')
        assembly.connect('control.Vout', 'brent.upper_bound')
        assembly.brent.add_parameter('powercurve.Vrated', low=-1e-15, high=1e15)
        assembly.brent.add_constraint('powercurve.residual = 0')
        assembly.brent.invalid_bracket_return = 1.0


    # connections to cdf
    assembly.connect('powercurve.V', 'cdf.x')


    # connections to aep
    assembly.connect('cdf.F', 'aep.CDF_V')
    assembly.connect('powercurve.P', 'aep.P')
    assembly.connect('AEP_loss_factor', 'aep.lossFactor')


    # connections to outputs
    assembly.connect('powercurve.V', 'V')
    assembly.connect('powercurve.P', 'P')
    assembly.connect('aep.AEP', 'AEP')
    assembly.connect('2*geom.R', 'diameter')
    if regulated:
        assembly.connect('powercurve.ratedConditions', 'ratedConditions')



class RotorAeroVSVP(Group):

    def configure(self):
        varspeed = True
        varpitch = True
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroVSFP(Group):

    def configure(self):
        varspeed = True
        varpitch = False
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroFSVP(Group):

    def configure(self):
        varspeed = False
        varpitch = True
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroFSFP(Group):

    def configure(self):
        varspeed = False
        varpitch = False
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)





# class RotorAeroVS(Assembly):

#     # --- inputs ---
#     # rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
#     control = VarTree(VarSpeedMachine(), iotype='in')

#     # options
#     npts_coarse_power_curve = Int(20, iotype='in', desc='number of points to evaluate aero analysis at')
#     npts_spline_power_curve = Int(200, iotype='in', desc='number of points to use in fitting spline to power curve')
#     AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')

#     # slots (must replace)
#     geom = Slot(GeomtrySetupBase)
#     analysis = Slot(AeroBase)
#     dt = Slot(DrivetrainLossesBase)
#     cdf = Slot(CDFBase)

#     # --- outputs ---
#     AEP = Float(iotype='out', units='kW*h', desc='annual energy production')
#     V = Array(iotype='out', units='m/s', desc='wind speeds (power curve)')
#     P = Array(iotype='out', units='W', desc='power (power curve)')
#     ratedConditions = VarTree(RatedConditions(), iotype='out')
#     diameter = Float(iotype='out', units='m')


#     def configure(self):

#         self.add('geom', GeomtrySetupBase())
#         self.add('setup', SetupRun())
#         self.add('analysis', AeroBase())
#         self.add('dt', DrivetrainLossesBase())
#         self.add('powercurve', RegulatedPowerCurve())
#         self.add('brent', Brent())
#         self.add('cdf', CDFBase())
#         self.add('aep', AEP())

#         self.brent.workflow.add(['powercurve'])

#         self.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'brent', 'cdf', 'aep'])

#         # connections to setup
#         self.connect('control', 'setup.control')
#         self.connect('geom.R', 'setup.R')
#         self.connect('npts_coarse_power_curve', 'setup.npts')

#         # connections to analysis
#         self.connect('setup.Uhub', 'analysis.Uhub')
#         self.connect('setup.Omega', 'analysis.Omega')
#         self.connect('setup.pitch', 'analysis.pitch')
#         self.analysis.run_case = 'power'

#         # connections to drivetrain
#         self.connect('analysis.P', 'dt.aeroPower')
#         self.connect('analysis.Q', 'dt.aeroTorque')
#         self.connect('analysis.T', 'dt.aeroThrust')
#         self.connect('control.ratedPower', 'dt.ratedPower')

#         # connections to powercurve
#         self.connect('control', 'powercurve.control')
#         self.connect('setup.Uhub', 'powercurve.Vcoarse')
#         self.connect('dt.power', 'powercurve.Pcoarse')
#         self.connect('analysis.T', 'powercurve.Tcoarse')
#         self.connect('geom.R', 'powercurve.R')
#         self.connect('npts_spline_power_curve', 'powercurve.npts')

#         # setup Brent method to find rated speed
#         self.connect('control.Vin', 'brent.lower_bound')
#         self.connect('control.Vout', 'brent.upper_bound')
#         self.brent.add_parameter('powercurve.Vrated', low=-1e-15, high=1e15)
#         self.brent.add_constraint('powercurve.residual = 0')

#         # connections to cdf
#         self.connect('powercurve.V', 'cdf.x')

#         # connections to aep
#         self.connect('cdf.F', 'aep.CDF_V')
#         self.connect('powercurve.P', 'aep.P')
#         self.connect('AEP_loss_factor', 'aep.lossFactor')


#         # connections to outputs
#         self.connect('powercurve.V', 'V')
#         self.connect('powercurve.P', 'P')
#         self.connect('aep.AEP', 'AEP')
#         self.connect('powercurve.ratedConditions', 'ratedConditions')
#         self.connect('2*geom.R', 'diameter')

