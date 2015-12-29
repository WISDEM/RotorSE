#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi
from openmdao.api import Component
from openmdao.api import ExecComp, IndepVarComp, Group, NLGaussSeidel
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
from openmdao.api import IndepVarComp, Component, Problem, Group, SqliteRecorder, BaseRecorder
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
# from openmdao.main.api import VariableTree, Component, Assembly, ImplicitComponent
# from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot, Enum
# from openmdao.lib.drivers.api import Brent
from openmdao.api import ScipyGMRES
from brent import Brent

from commonse.utilities import hstack, vstack, linspace_with_deriv, smooth_min, trapz_deriv
from akima import Akima
from enum import Enum

# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi

# ---------------------
# Base Components
# ---------------------




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

        rho = params['rho']
        V = params['V']
        R = params['R']
        T = params['T']
        Q = params['Q']
        P = params['P']

        q = 0.5 * rho * V**2
        A = pi * R**2
        unknowns['CP'] = P / (q * A * V)
        unknowns['CT'] = T / (q * A)
        unknowns['CQ'] = Q / (q * R * A)


    def list_deriv_vars(self):

        inputs = ('V', 'T', 'Q', 'P', 'R')
        outputs = ('CT', 'CQ', 'CP')

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):
        J = {}

        rho = params['rho']
        V = params['V']
        R = params['R']

        n = len(V)
        CP = unknowns['CP']
        CT = unknowns['CT']
        CQ = unknowns['CQ']

        q = 0.5 * rho * V**2
        A = pi * R**2
        zeronn = np.zeros((n, n))

        J['CT', 'V'] = np.diag(-2.0*CT/V)
        J['CT', 'T'] = np.diag(1.0/(q*A))
        J['CT', 'R'] = -2.0*CT/R

        J['CQ', 'V'] = np.diag(-2.0*CQ/V)
        J['CQ', 'V'] = np.diag(1.0/(q*R*A))
        J['CQ', 'V'] = -3.0*CQ/R

        J['CP', 'V'] = np.diag(-3.0*CP/V)
        J['CP', 'P'] = np.diag(1.0/(q*A*V))
        J['CP', 'R'] = -2.0*CP/R

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

    def solve_nonlinear(self, params, unknowns, resids):
        ctrl = params['control']
        n = ctrl.n

        # velocity sweep
        V = np.linspace(ctrl.Vin, ctrl.Vout, n)

        # store values
        unknowns['Uhub'] = V
        unknowns['Omega'] = ctrl.Omega*np.ones_like(V)
        unknowns['pitch'] = ctrl.pitch*np.ones_like(V)

class SetupRunVarSpeed(Component):
    def __init__(self):
        super(SetupRunVarSpeed, self).__init__()
        """determines approriate conditions to run AeroBase code across the power curve"""

        self.add_param('control:Vin', shape=1, units='m/s', desc='cut-in wind speed')
        self.add_param('control:Vout', shape=1, units='m/s', desc='cut-out wind speed')
        self.add_param('control:ratedPower', shape=1, units='W', desc='rated power')
        self.add_param('control:minOmega', shape=1, units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control:maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control:tsr', shape=1, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control:pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('R', shape=1, units='m', desc='rotor radius')
        self.add_param('npts', shape=1, val=20, desc='number of points to evalute aero code to generate power curve')

        # outputs
        self.add_output('Uhub', shape=20, units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', shape=20, units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', shape=20, units='deg', desc='pitch angles to run')

    def solve_nonlinear(self, params, unknowns, resids):

        n = params['npts']
        R = params['R']

        # # attempt to distribute points mostly before rated
        # cpguess = 0.5
        # Vr0 = (ctrl.ratedPower/(cpguess*0.5*rho*pi*R**2))**(1.0/3)
        # Vr0 *= 1.20

        # V1 = np.linspace(Vin, Vr0, 15)
        # V2 = np.linspace(Vr0, Vout, 6)
        # V = np.concatenate([V1, V2[1:]])

        # velocity sweep
        V = np.linspace(params['control:Vin'], params['control:Vout'], n)

        # corresponding rotation speed
        Omega_d = params['control:tsr']*V/R*RS2RPM
        Omega, dOmega_dOmegad, dOmega_dmaxOmega = smooth_min(Omega_d, params['control:maxOmega'], pct_offset=0.01)

        # store values
        unknowns['Uhub'] = V
        unknowns['Omega'] = Omega
        unknowns['pitch'] = params['control:pitch']*np.ones_like(V)

        # gradients
        J = {}
        J['Omega', 'tsr'] = dOmega_dOmegad * V/R*RS2RPM
        J['Omega', 'R'] = dOmega_dOmegad * -params['control:tsr']*V/R**2*RS2RPM
        J['Omega', 'control.maxOmega'] = dOmega_dmaxOmega

        self.J = J

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

        self.add_param('Vcoarse', shape=20, units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', shape=20, units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', shape=20, units='N', desc='unregulated thrust curve')
        self.add_param('npts', val=200, desc='number of points for splined power curve')


        # outputs
        self.add_output('V', units='m/s', desc='wind speeds')
        self.add_output('P', units='W', desc='power')


    def solve_nonlinear(self, params, unknowns, resids):

        ctrl = params['control']
        n = ctrl.n

        # finer power curve
        V, _, _ = linspace_with_deriv(ctrl.Vin, ctrl.Vout, n)
        unknowns['V'] = V
        spline = Akima(params['Vcoarse'], params['Pcoarse'])
        P, dP_dV, dP_dVcoarse, dP_dPcoarse = spline.interp(self.V)
        unknowns['P'] = P
        J = {}
        J['P', 'Vcoarse'] = dP_dVcoarse
        J['P', 'Pcoarse'] = dP_dPcoarse
        self.J = J

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
        self.add_param('control:Vin', shape=1, units='m/s', desc='cut-in wind speed')
        self.add_param('control:Vout', shape=1, units='m/s', desc='cut-out wind speed')
        self.add_param('control:ratedPower', shape=1, units='W', desc='rated power')
        self.add_param('control:minOmega', shape=1, units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control:maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control:tsr', shape=1, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control:pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('Vcoarse', shape=20, units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', shape=20, units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', shape=20, units='N', desc='unregulated thrust curve')
        self.add_param('R', shape=1, units='m', desc='rotor radius')
        self.add_param('npts', val=200, desc='number of points for splined power curve')

        # state
        self.add_state('Vrated', val=11.0, units='m/s', desc='rated wind speed', lower=-1e-15, upper=1e15)


        # residual
        # self.add_state('residual', shape=1)

        # outputs
        self.add_output('V', shape=200, units='m/s', desc='wind speeds')
        self.add_output('P', shape=200, units='W', desc='power')

        self.add_output('ratedConditions:V', shape=1, units='m/s', desc='rated wind speed')
        self.add_output('ratedConditions:Omega', shape=1, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('ratedConditions:pitch', shape=1, units='deg', desc='pitch setting at rated')
        self.add_output('ratedConditions:T', shape=1, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('ratedConditions:Q', shape=1, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_output('azimuth', shape=1, units='deg', desc='azimuth load')

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def apply_nonlinear(self, params, unknowns, resids):

        n = params['npts']
        Vrated = unknowns['Vrated']

        # residual
        spline = Akima(params['Vcoarse'], params['Pcoarse'])
        P, dres_dVrated, dres_dVcoarse, dres_dPcoarse = spline.interp(Vrated)
        # resids['residual'] = P - ctrl.ratedPower
        resids = P - params['control:ratedPower']
        # functional

        # place half of points in region 2, half in region 3
        # even though region 3 is constant we still need lots of points there
        # because we will be integrating against a discretized wind
        # speed distribution

        # region 2
        V2, _, dV2_dVrated = linspace_with_deriv(params['control:Vin'], Vrated, n/2)
        P2, dP2_dV2, dP2_dVcoarse, dP2_dPcoarse = spline.interp(V2)

        # region 3
        V3, dV3_dVrated, _ = linspace_with_deriv(Vrated, params['control:Vout'], n/2+1)
        V3 = V3[1:]  # remove duplicate point
        dV3_dVrated = dV3_dVrated[1:]
        P3 = params['control:ratedPower']*np.ones_like(V3)

        # concatenate
        unknowns['V'] = np.concatenate([V2, V3])
        unknowns['P'] = np.concatenate([P2, P3])

        R = params['R']
        # rated speed conditions
        Omega_d = params['control:tsr']*Vrated/R*RS2RPM
        OmegaRated, dOmegaRated_dOmegad, dOmegaRated_dmaxOmega \
            = smooth_min(Omega_d, params['control:maxOmega'], pct_offset=0.01)

        splineT = Akima(params['Vcoarse'], params['Tcoarse'])
        Trated, dT_dVrated, dT_dVcoarse, dT_dTcoarse = splineT.interp(Vrated)

        unknowns['ratedConditions:V'] = Vrated
        unknowns['ratedConditions:Omega'] = OmegaRated
        unknowns['ratedConditions:pitch'] = params['control:pitch']
        unknowns['ratedConditions:T'] = Trated
        unknowns['ratedConditions:Q'] = params['control:ratedPower'] / (OmegaRated * RPM2RS)
        unknowns['azimuth'] = 180.0

        # gradients
        ncoarse = len(params['Vcoarse'])

        dres = np.concatenate([[0.0], dres_dVcoarse, dres_dPcoarse, np.zeros(ncoarse), np.array([dres_dVrated]), [0.0, 0.0]])

        dV_dVrated = np.concatenate([dV2_dVrated, dV3_dVrated])
        dV = hstack([np.zeros((n, 1)), np.zeros((n, 3*ncoarse)), dV_dVrated, np.zeros((n, 2))])

        dP_dVcoarse = vstack([dP2_dVcoarse, np.zeros((n/2, ncoarse))])
        dP_dPcoarse = vstack([dP2_dPcoarse, np.zeros((n/2, ncoarse))])
        dP_dVrated = np.concatenate([dP2_dV2*dV2_dVrated, np.zeros(n/2)])
        dP = hstack([np.zeros((n, 1)), dP_dVcoarse, dP_dPcoarse, np.zeros((n, ncoarse)), dP_dVrated, np.zeros((n, 2))])

        drV = np.concatenate([[0.0], np.zeros(3*ncoarse), [1.0, 0.0, 0.0]])
        drOmega = np.concatenate([[dOmegaRated_dOmegad*Vrated/R*RS2RPM], np.zeros(3*ncoarse),
            [dOmegaRated_dOmegad*params['control:tsr']/R*RS2RPM, -dOmegaRated_dOmegad*params['control:tsr']*Vrated/R**2*RS2RPM,
            dOmegaRated_dmaxOmega]])
        drpitch = np.zeros(3*ncoarse+4)
        drT = np.concatenate([[0.0], dT_dVcoarse, np.zeros(ncoarse), dT_dTcoarse, [dT_dVrated, 0.0, 0.0]])
        drQ = -params['control:ratedPower'] / (OmegaRated**2 * RPM2RS) * drOmega

        J = {}

        J['V', 'Vrated'] = dV_dVrated
        J['P', 'Vrated'] = dP_dVrated
        J['P', 'Vcoarse'] = dP_dVcoarse
        J['P', 'Pcoarse'] = dP_dPcoarse
        J['ratedConditions.V', 'Vrated'] = 1
        J['ratedConditions.Omega', 'control.tsr'] = dOmegaRated_dOmegad*Vrated/R*RS2RPM
        J['ratedConditions.Omega', 'Vrated'] = dOmegaRated_dOmegad*params['control:tsr']/R*RS2RPM
        J['ratedConditions.Omega', 'R'] = -dOmegaRated_dOmegad*params['control:tsr']*Vrated/R**2*RS2RPM
        J['ratedConditions.Omega', 'control.maxOmega'] = dOmegaRated_dmaxOmega
        J['ratedConditions.T', 'Vcoarse'] = dT_dVcoarse
        J['ratedConditions.T', 'Tcoarse'] = dT_dTcoarse
        J['ratedConditions.T', 'Vrated'] = dT_dVrated
        J['ratedConditions.Q', 'control.maxOmega'] = drQ ## TODO: Fix

        self.J = J

    def jacobian(self, params, unknowns, resids):

        return self.J

class RegulatedPowerCurveGroup(Group):
    def __init__(self):
        super(RegulatedPowerCurveGroup, self).__init__()

        sub = self.add('sub', Group(), promotes=['*'])

        sub.add('powercurve', RegulatedPowerCurve(), promotes=['*'])

        sub.nl_solver = Brent()
        sub.ln_solver = ScipyGMRES()
        # self.connect('control:Vin', 'brent.lower_bound')
        # self.connect('control:Vout', 'brent.upper_bound')
        # self.brent.add_parameter('powercurve.Vrated', low=-1e-15, high=1e15)
        # self.brent.add_constraint('powercurve.residual = 0')
        # self.brent.invalid_bracket_return = 1.0
        sub.nl_solver.options['lower_bound'] = 3.0
        sub.nl_solver.options['upper_bound'] = 11.73845
        sub.nl_solver.options['state_var'] = 'Vrated'
        sub.nl_solver.options['xtol'] = 1

class AEP(Component):
    def __init__(self):
        super(AEP, self).__init__()
        """integrate to find annual energy production"""

        # inputs
        self.add_param('CDF_V', shape=200, desc='cumulative distribution function evaluated at each wind speed')
        self.add_param('P', shape=200, units='W', desc='power curve (power)')
        self.add_param('lossFactor', shape=1, desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', shape=1, units='kW*h', desc='annual energy production')


    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['AEP'] = params['lossFactor']*np.trapz(params['P'], params['CDF_V'])/1e3*365.0*24.0  # in kWh

    def list_deriv_vars(self):

        inputs = ('CDF_V', 'P', 'lossFactor')
        outputs = ('AEP',)

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        lossFactor = params['lossFactor']
        P = params['P']
        factor = lossFactor/1e3*365.0*24.0

        dAEP_dP, dAEP_dCDF = trapz_deriv(P, params['CDF_V'])
        dAEP_dP *= factor
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([unknowns['AEP']/lossFactor])

        J = {}
        J['AEP', 'CDF_V'] = dAEP_dCDF
        J['AEP', 'P'] = dAEP_dP
        J['AEP', 'lossFactor'] = dAEP_dlossFactor

        return J



# ---------------------
# Assemblies
# ---------------------

#
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

