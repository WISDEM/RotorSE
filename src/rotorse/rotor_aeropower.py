#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

from __future__ import print_function
import numpy as np
import os
from openmdao.api import IndepVarComp, Component, Group, Problem, Brent, ScipyGMRES
from scipy.optimize import brentq

from ccblade.ccblade_component import CCBladeGeometry, CCBladePower

from commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from commonse.utilities import vstack, trapz_deriv, linspace_with_deriv, smooth_min, smooth_abs
from commonse.environment import PowerWind
#from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from akima import Akima
from rotor_geometry import RotorGeometry, NREL5MW, DTU10MW, DRIVETRAIN_TYPE

from rotorse import RPM2RS, RS2RPM


# ---------------------
# Base Components
# ---------------------


class DrivetrainLossesBase(Component):
    """base component for drivetrain efficiency losses"""
    def __init__(self, npower):
        super(DrivetrainLossesBase, self).__init__()
        self.add_param('aeroPower', val=np.zeros(npower), units='W', desc='aerodynamic power')
        self.add_param('aeroTorque', val=np.zeros(npower), units='N*m', desc='aerodynamic torque')
        self.add_param('aeroThrust', val=np.zeros(npower), units='N', desc='aerodynamic thrust')
        self.add_param('ratedPower', val=0.0, units='W', desc='rated power')

        self.add_output('power', val=np.zeros(npower), units='W', desc='total power after drivetrain losses')
        self.add_output('rpm', val=np.zeros(npower), units='rpm', desc='rpm curve after drivetrain losses')

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

'''
# This Component is now no longer used, but I'll keep it around for now in case that changes.
class Coefficients(Component):
    def __init__(self, npts_coarse_power_curve):
        super(Coefficients, self).__init__()
        """convert power, thrust, torque into nondimensional coefficient form"""

        # inputs
        self.add_param('V', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='wind speed')
        self.add_param('T', val=np.zeros(npts_coarse_power_curve), units='N', desc='rotor aerodynamic thrust')
        self.add_param('Q', val=np.zeros(npts_coarse_power_curve), units='N*m', desc='rotor aerodynamic torque')
        self.add_param('P', val=np.zeros(npts_coarse_power_curve), units='W', desc='rotor aerodynamic power')

        # inputs used in normalization
        self.add_param('R', val=0.0, units='m', desc='rotor radius')
        self.add_param('rho', val=0.0, units='kg/m**3', desc='density of fluid')


        # outputs
        self.add_output('CT', val=np.zeros(npts_coarse_power_curve), desc='rotor aerodynamic thrust')
        self.add_output('CQ', val=np.zeros(npts_coarse_power_curve), desc='rotor aerodynamic torque')
        self.add_output('CP', val=np.zeros(npts_coarse_power_curve), desc='rotor aerodynamic power')

	self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'


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


    def linearize(self, params, unknowns, resids):
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


        J['CT', 'V'] = np.diag(-2.0*CT/V)
        J['CT', 'T'] = np.diag(1.0/(q*A))
        J['CT', 'Q'] = np.zeros((n, n))
        J['CT', 'P'] = np.zeros((n, n))
        J['CT', 'R'] = -2.0*CT/R

        J['CQ', 'V'] = np.diag(-2.0*CQ/V)
        J['CQ', 'T'] = np.zeros((n, n))
        J['CQ', 'Q'] = np.diag(1.0/(q*R*A))
        J['CQ', 'P'] = np.zeros((n, n))
        J['CQ', 'R'] = -3.0*CQ/R

        J['CP', 'V'] = np.diag(-3.0*CP/V)
        J['CP', 'T'] = np.zeros((n, n))
        J['CP', 'Q'] = np.zeros((n, n))
        J['CP', 'P'] = np.diag(1.0/(q*A*V))
        J['CP', 'R'] = -2.0*CP/R

        return J



class SetupRunFixedSpeed(Component):
    def __init__(self, npts):
        super(SetupRunFixedSpeed, self).__init__()
        self.npts = npts
        """determines approriate conditions to run AeroBase code across the power curve"""

        self.add_param('control_Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control_Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control_ratedPower', units='W', desc='rated power')
        self.add_param('control_Omega', units='rpm', desc='fixed rotor rotation speed')
        self.add_param('control_pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        # outputs
        self.add_output('Uhub', units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', units='deg', desc='pitch angles to run')

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        # velocity sweep
        V = np.linspace(ctrl.Vin, ctrl.Vout, self.npts)

        # store values
        unknowns['Uhub'] = V
        unknowns['Omega'] = ctrl.Omega*np.ones_like(V)
        unknowns['pitch'] = ctrl.pitch*np.ones_like(V)
'''

class SetupRunVarSpeed(Component):
    def __init__(self, npts_coarse_power_curve):
        super(SetupRunVarSpeed, self).__init__()
        self.npts = npts_coarse_power_curve
        """determines approriate conditions to run AeroBase code across the power curve"""

        self.add_param('control_Vin', val=0.0, units='m/s', desc='cut-in wind speed')
        self.add_param('control_Vout', val=0.0, units='m/s', desc='cut-out wind speed')
        self.add_param('control_maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control_tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control_pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('R', val=0.0, units='m', desc='rotor radius')

        # outputs
        self.add_output('Uhub', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', val=np.zeros(npts_coarse_power_curve), units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', val=np.zeros(npts_coarse_power_curve), units='deg', desc='pitch angles to run')


    def solve_nonlinear(self, params, unknowns, resids):

        R = params['R']

        # # attempt to distribute points mostly before rated
        # cpguess = 0.5
        # Vr0 = (ctrl.ratedPower/(cpguess*0.5*rho*pi*R**2))**(1.0/3)
        # Vr0 *= 1.20

        # V1 = np.linspace(Vin, Vr0, 15)
        # V2 = np.linspace(Vr0, Vout, 6)
        # V = np.concatenate([V1, V2[1:]])

        # velocity sweep
        V, dV_dVin, dV_dVout = linspace_with_deriv(params['control_Vin'], params['control_Vout'], self.npts)
        
        # corresponding rotation speed
        Omega_d = params['control_tsr']*V/R*RS2RPM
        Omega, dOmega_dOmegad, dOmega_dmaxOmega = smooth_min(Omega_d, params['control_maxOmega'], pct_offset=0.01)

        # store values
        unknowns['Uhub'] = V
        unknowns['Omega'] = Omega
        unknowns['pitch'] = params['control_pitch']*np.ones_like(V)

        # gradients
        J = {}
        J['Omega', 'control_tsr'] = dOmega_dOmegad * V/R*RS2RPM
        J['Omega', 'R'] = dOmega_dOmegad * -params['control_tsr']*V/R**2*RS2RPM
        J['Omega', 'control_maxOmega'] = dOmega_dmaxOmega
        J['Omega', 'control_Vin'] = dOmega_dOmegad * params['control_tsr']/R*RS2RPM * dV_dVin
        J['Omega', 'control_Vout'] = dOmega_dOmegad * params['control_tsr']/R*RS2RPM * dV_dVout
        J['Uhub', 'control_tsr'] = np.zeros(len(V))
        J['Uhub', 'R'] = np.zeros(len(V))
        J['Uhub', 'control_pitch'] = np.zeros(len(V))
        J['Uhub', 'control_Vin']   = dV_dVin
        J['Uhub', 'control_Vout']   = dV_dVout
        J['pitch', 'control_tsr'] = np.zeros(len(V))
        J['pitch', 'R'] = np.zeros(len(V))
        J['pitch', 'control_pitch'] = np.ones(len(V))
        J['pitch', 'control_Vin'] = np.zeros(len(V))
        J['pitch', 'control_Vout'] = np.zeros(len(V))

        self.J = J

    def list_deriv_vars(self):

        inputs = ('control_tsr', 'R', 'control_maxOmega', 'control_Vin', 'control_Vout', 'control_pitch')
        outputs = ('Uhub', 'Omega', 'pitch', )

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        return self.J




'''
class UnregulatedPowerCurve(Component):
    def __init__(self, npts_coarse_power_curve, npts_spline_power_curve):
        super(UnregulatedPowerCurve, self).__init__()
        self.npts = npts_spline_power_curve
        
        # inputs
        self.add_param('control_Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control_Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control_Omega', units='rpm', desc='fixed rotor rotation speed')
        self.add_param('control_pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('Vcoarse', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', val=np.zeros(npts_coarse_power_curve), units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', val=np.zeros(npts_coarse_power_curve), units='N', desc='unregulated thrust curve')

        # outputs
        self.add_output('V', units='m/s', desc='wind speeds')
        self.add_output('P', units='W', desc='power')


    def solve_nonlinear(self, params, unknowns, resids):

        ctrl = params['control']

        # finer power curve
        V, _, _ = linspace_with_deriv(ctrl.Vin, ctrl.Vout, self.npts)
        unknowns['V'] = V
        spline = Akima(params['Vcoarse'], params['Pcoarse'])
        P, dP_dV, dP_dVcoarse, dP_dPcoarse = spline.interp(self.V)
        unknowns['P'] = P

        J = {}
        J['P', 'Vcoarse'] = dP_dVcoarse
        J['P', 'Pcoarse'] = dP_dPcoarse
        self.J = J

    def linearize(self, params, unknowns, resids):

        return self.J
'''

class RegulatedPowerCurve(Component): # Implicit COMPONENT


    def __init__(self, npts_coarse_power_curve, npts_spline_power_curve):
        super(RegulatedPowerCurve, self).__init__()

        self.npts = int(npts_spline_power_curve)
        self.eval_only = True  # allows an external solver to converge this, otherwise it will converge itself to mimic an explicit comp
        """Fit a spline to the coarse sampled power curve (and thrust curve),
        find rated speed through a residual convergence strategy,
        then compute the regulated power curve and rated conditions"""

        # inputs
        self.add_param('control_Vin', val=0.0, units='m/s', desc='cut-in wind speed')
        self.add_param('control_Vout', val=0.0, units='m/s', desc='cut-out wind speed')
        self.add_param('control_ratedPower', val=0.0, units='W', desc='rated power')
        self.add_param('control_minOmega', val=0.0, units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control_maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control_tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control_pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('Vcoarse', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', val=np.zeros(npts_coarse_power_curve), units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', val=np.zeros(npts_coarse_power_curve), units='N', desc='unregulated thrust curve')
        self.add_param('R', val=0.0, units='m', desc='rotor radius')

        # state
        #self.add_state('Vrated', val=11.0, units='m/s', desc='rated wind speed', lower=-1e-15, upper=1e15)
        self.add_output('Vrated', val=11.0, units='m/s', desc='rated wind speed')

        # residual
        # self.add_state('residual', val=0.0)

        # outputs
        self.add_output('V', val=np.zeros(npts_spline_power_curve), units='m/s', desc='wind speeds')
        self.add_output('P', val=np.zeros(npts_spline_power_curve), units='W', desc='power')

        self.add_output('rated_V', val=0.0, units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_output('rated_T', val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q', val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_output('azimuth', val=0.0, units='deg', desc='azimuth load')


        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['form'] = 'central'
        self.deriv_options['type'] = 'fd'
        self.deriv_options['check_step_calc'] = 'relative'
        self.deriv_options['check_form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        #Vrated = unknowns['Vrated']

        # residual
        spline = Akima(params['Vcoarse'], params['Pcoarse'])
        def myfun(myv):
            P, _, _, _ = spline.interp(myv)
            return (P - params['control_ratedPower'])
        if (myfun(params['control_Vout']) > params['control_ratedPower']):
            Vrated = brentq(lambda x: myfun(x), params['control_Vin'], params['control_Vout'])
        else:
            Vrated = params['control_Vout']
        P, dres_dVrated, dres_dVcoarse, dres_dPcoarse = spline.interp(Vrated)
        unknowns['Vrated'] = Vrated
        # functional

        # place half of points in region 2, half in region 3
        # even though region 3 is constant we still need lots of points there
        # because we will be integrating against a discretized wind
        # speed distribution

        # region 2
        V2, _, dV2_dVrated = linspace_with_deriv(params['control_Vin'], Vrated, self.npts/2)
        P2, dP2_dV2, dP2_dVcoarse, dP2_dPcoarse = spline.interp(V2)

        # region 3
        V3, dV3_dVrated, _ = linspace_with_deriv(Vrated, params['control_Vout'], self.npts/2+1)
        V3 = V3[1:]  # remove duplicate point
        dV3_dVrated = dV3_dVrated[1:]
        P3 = params['control_ratedPower']*np.ones_like(V3)

        # concatenate
        unknowns['V'] = np.concatenate([V2, V3])
        unknowns['P'] = np.concatenate([P2, P3])

        R = params['R']
        # rated speed conditions
        Omega_d = params['control_tsr']*Vrated/R*RS2RPM
        OmegaRated, dOmegaRated_dOmegad, dOmegaRated_dmaxOmega \
            = smooth_min(Omega_d, params['control_maxOmega'], pct_offset=0.01)

        splineT = Akima(params['Vcoarse'], params['Tcoarse'])
        Trated, dT_dVrated, dT_dVcoarse, dT_dTcoarse = splineT.interp(Vrated)

        unknowns['rated_V'] = Vrated
        unknowns['rated_Omega'] = OmegaRated
        unknowns['rated_pitch'] = params['control_pitch']
        unknowns['rated_T'] = Trated
        unknowns['rated_Q'] = params['control_ratedPower'] / (OmegaRated * RPM2RS)
        unknowns['azimuth'] = 180.0


        # gradients
        ncoarse = len(params['Vcoarse'])

        dV_dVrated = np.concatenate([dV2_dVrated, dV3_dVrated])

        dP_dVcoarse = vstack([dP2_dVcoarse, np.zeros((int(self.npts/2), ncoarse))])
        dP_dPcoarse = vstack([dP2_dPcoarse, np.zeros((int(self.npts/2), ncoarse))])
        dP_dVrated = np.concatenate([dP2_dV2*dV2_dVrated, np.zeros(int(self.npts/2))])

        drOmega = np.concatenate([[dOmegaRated_dOmegad*Vrated/R*RS2RPM], np.zeros(3*ncoarse),
            [dOmegaRated_dOmegad*params['control_tsr']/R*RS2RPM, -dOmegaRated_dOmegad*params['control_tsr']*Vrated/R**2*RS2RPM,
            dOmegaRated_dmaxOmega]])
        drQ = -params['control_ratedPower'] / (OmegaRated**2 * RPM2RS) * drOmega

        J = {}

	#J['Vrated', 'Vcoarse'] = np.reshape(dres_dVcoarse, (1, len(dres_dVcoarse)))
        #J['Vrated', 'Pcoarse'] = np.reshape(dres_dPcoarse, (1, len(dres_dPcoarse)))
        #J['Vrated', 'Vrated'] = dres_dVrated
	#J['Vrated', 'control_ratedPower'] = -1
	#J['Vrated', 'control_tsr'] = 0
	


        J['V', 'Vrated'] = dV_dVrated
        J['P', 'Vrated'] = dP_dVrated
        J['P', 'Vcoarse'] = dP_dVcoarse
        J['P', 'Pcoarse'] = dP_dPcoarse
        J['rated_V', 'Vrated'] = 1.0
        J['rated_Omega', 'control_tsr'] = dOmegaRated_dOmegad*Vrated/R*RS2RPM
        J['rated_Omega', 'Vrated'] = dOmegaRated_dOmegad*params['control_tsr']/R*RS2RPM
        J['rated_Omega', 'R'] = -dOmegaRated_dOmegad*params['control_tsr']*Vrated/R**2*RS2RPM
        J['rated_Omega', 'control_maxOmega'] = dOmegaRated_dmaxOmega
        J['rated_T', 'Vcoarse'] = np.reshape(dT_dVcoarse, (1, len(dT_dVcoarse)))
        J['rated_T', 'Tcoarse'] = np.reshape(dT_dTcoarse, (1, len(dT_dTcoarse)))
        J['rated_T', 'Vrated'] = dT_dVrated
        J['rated_Q', 'control_tsr'] = drQ[0]
        J['rated_Q', 'Vrated'] = drQ[-3]
        J['rated_Q', 'R'] = drQ[-2]
        J['rated_Q', 'control_maxOmega'] = drQ[-1]

        self.J = J

    '''
    def linearize(self, params, unknowns, resids):

        return self.J

    def apply_nonlinear(self, params, unknowns, resids):
        Vrated = unknowns['Vrated']

        # residual
        spline = Akima(params['Vcoarse'], params['Pcoarse'])
        P, dres_dVrated, dres_dVcoarse, dres_dPcoarse = spline.interp(Vrated)
        # resids['residual'] = P - ctrl.ratedPower
        resids['Vrated'] = P - params['control_ratedPower']
        # functional

        # place half of points in region 2, half in region 3
        # even though region 3 is constant we still need lots of points there
        # because we will be integrating against a discretized wind
        # speed distribution

        # region 2
        V2, _, dV2_dVrated = linspace_with_deriv(params['control_Vin'], Vrated, self.npts/2)
        P2, dP2_dV2, dP2_dVcoarse, dP2_dPcoarse = spline.interp(V2)

        # region 3
        V3, dV3_dVrated, _ = linspace_with_deriv(Vrated, params['control_Vout'], self.npts/2+1)
        V3 = V3[1:]  # remove duplicate point
        dV3_dVrated = dV3_dVrated[1:]
        P3 = params['control_ratedPower']*np.ones_like(V3)

        # concatenate
        unknowns['V'] = np.concatenate([V2, V3])
        unknowns['P'] = np.concatenate([P2, P3])

        R = params['R']
        # rated speed conditions
        Omega_d = params['control_tsr']*Vrated/R*RS2RPM
        OmegaRated, dOmegaRated_dOmegad, dOmegaRated_dmaxOmega \
            = smooth_min(Omega_d, params['control_maxOmega'], pct_offset=0.01)

        splineT = Akima(params['Vcoarse'], params['Tcoarse'])
        Trated, dT_dVrated, dT_dVcoarse, dT_dTcoarse = splineT.interp(Vrated)

        unknowns['rated_V'] = Vrated
        unknowns['rated_Omega'] = OmegaRated
        unknowns['rated_pitch'] = params['control_pitch']
        unknowns['rated_T'] = Trated
        unknowns['rated_Q'] = params['control_ratedPower'] / (OmegaRated * RPM2RS)
        unknowns['azimuth'] = 180.0
    '''

    
# class RegulatedPowerCurveGroup(Group): # EMG: Remove? doesn't appear to be in use
#     def __init__(self, npts_coarse_power_curve, npts_spline_power_curve):
#         super(RegulatedPowerCurveGroup, self).__init__()
#         self.add('powercurve_comp', RegulatedPowerCurve(npts_coarse_power_curve, npts_spline_power_curve), promotes=['*'])
#         self.nl_solver = Brent()
#         self.ln_solver = ScipyGMRES()
#         self.nl_solver.options['var_lower_bound'] = 'Vin'
#         self.nl_solver.options['var_upper_bound'] = 'Vout'
#         self.nl_solver.options['state_var'] = 'Vrated'

#         self.deriv_options['form'] = 'central'
#         self.deriv_options['check_form'] = 'central'
#         self.deriv_options['type'] = 'fd'
#         self.deriv_options['step_calc'] = 'relative'

#     def list_deriv_vars(self):

#         inputs = ('control_tsr', 'Vcoarse', 'Pcoarse', 'Tcoarse', 'Vrated', 'R', 'control_maxOmega')
#         outputs = ('Vrated', 'V', 'P', 'rated_V', 'rated_Omega',
#             'rated_pitch', 'rated_T', 'rated_Q')

#         return inputs, outputs

    
class AEP(Component):
    def __init__(self, npts_spline_power_curve):
        super(AEP, self).__init__()
        """integrate to find annual energy production"""

        # inputs
        self.add_param('CDF_V', val=np.zeros(npts_spline_power_curve), units='m/s', desc='cumulative distribution function evaluated at each wind speed')
        self.add_param('P', val=np.zeros(npts_spline_power_curve), units='W', desc='power curve (power)')
        self.add_param('lossFactor', val=0.0, desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production')

        #self.deriv_options['step_size'] = 1.0
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'	

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['AEP'] = params['lossFactor']*np.trapz(params['P'], params['CDF_V'])/1e3*365.0*24.0  # in kWh

    def list_deriv_vars(self):

        inputs = ('CDF_V', 'P', 'lossFactor')
        outputs = ('AEP',)

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        lossFactor = params['lossFactor']
        P = params['P']
        factor = lossFactor/1e3*365.0*24.0

        dAEP_dP, dAEP_dCDF = trapz_deriv(P, params['CDF_V'])
        dAEP_dP *= factor
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([unknowns['AEP']/lossFactor])

        J = {}
        J['AEP', 'CDF_V'] = np.reshape(dAEP_dCDF, (1, len(dAEP_dCDF)))
        J['AEP', 'P'] = np.reshape(dAEP_dP, (1, len(dAEP_dP)))
        J['AEP', 'lossFactor'] = dAEP_dlossFactor

        return J


class CSMDrivetrain(DrivetrainLossesBase):
    def __init__(self, n):
        super(CSMDrivetrain, self).__init__(n)
        """drivetrain losses from NREL cost and scaling model"""

        self.add_param('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True)
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        drivetrainType = params['drivetrainType']
        aeroPower = params['aeroPower']
        aeroTorque = params['aeroTorque']
        ratedPower = params['ratedPower']

        if drivetrainType == DRIVETRAIN_TYPE['GEARED']:
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif drivetrainType == DRIVETRAIN_TYPE['SINGLE_STAGE']:
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif drivetrainType == DRIVETRAIN_TYPE['MULTI_DRIVE']:
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif drivetrainType == DRIVETRAIN_TYPE['PM_DIRECT_DRIVE']:
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899


        Pbar0 = aeroPower / ratedPower

        # handle negative power case (with absolute value)
        Pbar1, dPbar1_dPbar0 = smooth_abs(Pbar0, dx=0.01)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar, dPbar_dPbar1, _ = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant/Pbar + linear + quadratic*Pbar)

        unknowns['power'] = aeroPower * eff

        # gradients
        dPbar_dPa = dPbar_dPbar1*dPbar1_dPbar0/ratedPower
        dPbar_dPr = -dPbar_dPbar1*dPbar1_dPbar0*Pbar0/ratedPower

        deff_dPa = dPbar_dPa*(constant/Pbar**2 - quadratic)
        deff_dPr = dPbar_dPr*(constant/Pbar**2 - quadratic)

        dP_dPa = eff + aeroPower*deff_dPa
        dP_dPr = aeroPower*deff_dPr
        
        J = {}
        J['power', 'aeroPower'] = np.diag(dP_dPa)
        J['power', 'ratedPower'] = dP_dPr
        self.J = J


    def list_deriv_vars(self):

        inputs = ('aeroPower', 'ratedPower')
        outputs = ('power',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        return self.J
    

class OutputsAero(Component):
    def __init__(self, npts_spline_power_curve):
        super(OutputsAero, self).__init__()

        # --- outputs ---
        self.add_param('AEP_in', val=0.0, units='kW*h', desc='annual energy production')
        self.add_param('V_in', val=np.zeros(npts_spline_power_curve), units='m/s', desc='wind speeds (power curve)')
        self.add_param('P_in', val=np.zeros(npts_spline_power_curve), units='W', desc='power (power curve)')

        self.add_param('rated_V_in', val=0.0, units='m/s', desc='rated wind speed')
        self.add_param('rated_Omega_in', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_param('rated_pitch_in', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_param('rated_T_in', val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_param('rated_Q_in', val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_param('diameter_in', val=0.0, units='m', desc='rotor diameter')
        self.add_param('V_extreme_in', val=0.0, units='m/s', desc='survival wind speed')
        self.add_param('T_extreme_in', val=0.0, units='N', desc='thrust at survival wind condition')
        self.add_param('Q_extreme_in', val=0.0, units='N*m', desc='thrust at survival wind condition')

        # internal use outputs
        self.add_param('precurveTip_in', val=0.0, units='m', desc='tip location in x_b')
        self.add_param('presweepTip_in', val=0.0, units='m', desc='tip location in y_b')  # TODO: connect later

        # --- outputs ---
        self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production')
        self.add_output('V', val=np.zeros(npts_spline_power_curve), units='m/s', desc='wind speeds (power curve)')
        self.add_output('P', val=np.zeros(npts_spline_power_curve), units='W', desc='power (power curve)')

        self.add_output('rated_V', val=0.0, units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_output('rated_T', val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q', val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_output('diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_output('V_extreme', val=0.0, units='m/s', desc='survival wind speed')
        self.add_output('T_extreme', val=0.0, units='N', desc='thrust at survival wind condition')
        self.add_output('Q_extreme', val=0.0, units='N*m', desc='thrust at survival wind condition')

        # internal use outputs
        self.add_output('precurveTip', val=0.0, units='m', desc='tip location in x_b')
        self.add_output('presweepTip', val=0.0, units='m', desc='tip location in y_b')  # TODO: connect later

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['AEP'] = params['AEP_in']
        unknowns['V'] = params['V_in']
        unknowns['P'] = params['P_in']
        unknowns['rated_V'] = params['rated_V_in']
        unknowns['rated_Omega'] = params['rated_Omega_in']
        unknowns['rated_pitch'] = params['rated_pitch_in']
        unknowns['rated_T'] = params['rated_T_in']
        unknowns['rated_Q'] = params['rated_Q_in']
        unknowns['diameter'] = params['diameter_in']
        unknowns['V_extreme'] = params['V_extreme_in']
        unknowns['T_extreme'] = params['T_extreme_in']
        unknowns['Q_extreme'] = params['Q_extreme_in']
        unknowns['precurveTip'] = params['precurveTip_in']
        unknowns['presweepTip'] = params['presweepTip_in']

    def linearize(self, params, unknowns,resids):
        J = {}
        J['AEP', 'AEP_in'] = 1
        J['V', 'V_in'] = np.diag(np.ones(len(params['V_in'])))
        J['P', 'P_in'] = np.diag(np.ones(len(params['P_in'])))
        J['rated_V', 'rated_V_in'] = 1
        J['rated_Omega', 'rated_Omega_in'] = 1
        J['rated_pitch', 'rated_pitch_in'] = 1
        J['rated_T', 'rated_T_in'] = 1
        J['rated_Q', 'rated_Q_in'] = 1
        J['diameter', 'diameter_in'] = 1
        J['V_extreme', 'V_extreme_in'] = 1
        J['T_extreme', 'T_extreme_in'] = 1
        J['Q_extreme', 'Q_extreme_in'] = 1
        J['precurveTip', 'precurveTip_in'] = 1
        J['presweepTip', 'presweepTip_in'] = 1

        return J

class RotorAeroPower(Group):
    def __init__(self, RefBlade, npts_coarse_power_curve=20, npts_spline_power_curve=200):
        super(RotorAeroPower, self).__init__()

        #self.add('rho', IndepVarComp('rho', val=1.225), promotes=['*'])
        #self.add('mu', IndepVarComp('mu', val=1.81e-5), promotes=['*'])
        #self.add('shearExp', IndepVarComp('shearExp', val=0.2), promotes=['*'])

        self.add('cdf_reference_height_wind_speed', IndepVarComp('cdf_reference_height_wind_speed', val=0.0, units='m', desc='reference hub height for IEC wind speed (used in CDF calculation)'), promotes=['*'])

        self.add('tiploss', IndepVarComp('tiploss', True, pass_by_obj=True), promotes=['*'])
        self.add('hubloss', IndepVarComp('hubloss', True, pass_by_obj=True), promotes=['*'])
        self.add('wakerotation', IndepVarComp('wakerotation', True, pass_by_obj=True), promotes=['*'])
        self.add('usecd', IndepVarComp('usecd', True, pass_by_obj=True), promotes=['*'])
        #self.add('airfoil_files', IndepVarComp('airfoil_files', AirfoilProperties.airfoil_files, pass_by_obj=True), promotes=['*'])
        
        # --- control ---
        self.add('c_Vin', IndepVarComp('control_Vin', val=0.0, units='m/s', desc='cut-in wind speed'), promotes=['*'])
        self.add('c_Vout', IndepVarComp('control_Vout', val=0.0, units='m/s', desc='cut-out wind speed'), promotes=['*'])
        self.add('c_ratedPower', IndepVarComp('control_ratedPower', val=0.0,  units='W', desc='rated power'), promotes=['*'])
        self.add('c_minOmega', IndepVarComp('control_minOmega', val=0.0, units='rpm', desc='minimum allowed rotor rotation speed'), promotes=['*'])
        self.add('c_maxOmega', IndepVarComp('control_maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed'), promotes=['*'])
        self.add('c_tsr', IndepVarComp('control_tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)'), promotes=['*'])
        self.add('c_pitch', IndepVarComp('control_pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)'), promotes=['*'])

        # --- drivetrain efficiency ---
        self.add('drivetrainType', IndepVarComp('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True), promotes=['*'])


        # --- options ---
        self.add('nSector', IndepVarComp('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True), promotes=['*'])
        self.add('AEP_loss_factor', IndepVarComp('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)'), promotes=['*'])

        self.add('shape_parameter', IndepVarComp('shape_parameter', val=0.0), promotes=['*'])

        
        # --- Rotor Aero & Power ---
        self.add('rotorGeom', RotorGeometry(RefBlade), promotes=['*'])

        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRunVarSpeed(npts_coarse_power_curve))
        self.add('analysis', CCBladePower(RefBlade.npts, npts_coarse_power_curve))
        self.add('dt', CSMDrivetrain(npts_coarse_power_curve))
        self.add('powercurve', RegulatedPowerCurve(npts_coarse_power_curve, npts_spline_power_curve))
        self.add('wind', PowerWind(1))
        # self.add('cdf', WeibullWithMeanCDF(npts_spline_power_curve))
        self.add('cdf', RayleighCDF(npts_spline_power_curve))
        self.add('aep', AEP(npts_spline_power_curve))

        self.add('outputs_aero', OutputsAero(npts_spline_power_curve), promotes=['*'])

        # # connectiosn to tipspeed
        # self.connect('geom.R', 'tipspeed.R')
        # self.connect('max_tip_speed', 'tipspeed.Vtip_max')
        # self.connect('tipspeed.Omega_max', 'control_maxOmega')

        # connections to setup
        # self.connect('control', 'setup.control')
        self.connect('control_Vin', 'setup.control_Vin')
        self.connect('control_Vout', 'setup.control_Vout')
        self.connect('control_maxOmega', 'setup.control_maxOmega')
        self.connect('control_pitch', 'setup.control_pitch')
        self.connect('control_tsr', 'setup.control_tsr')
        self.connect('geom.R', 'setup.R')

        # connections to analysis
        self.connect('r_pts', 'analysis.r')
        self.connect('chord', 'analysis.chord')
        self.connect('theta', 'analysis.theta')
        self.connect('precurve', 'analysis.precurve')
        self.connect('precurve_tip', 'analysis.precurveTip')
        self.connect('Rhub', 'analysis.Rhub')
        self.connect('Rtip', 'analysis.Rtip')
        self.connect('hub_height', 'analysis.hubHt')
        self.connect('precone', 'analysis.precone')
        self.connect('tilt', 'analysis.tilt')
        self.connect('yaw', 'analysis.yaw')
        self.connect('airfoils', 'analysis.airfoils')
        self.connect('nBlades', 'analysis.B')
        #self.connect('rho', 'analysis.rho')
        #self.connect('mu', 'analysis.mu')
        #self.connect('shearExp', 'analysis.shearExp')
        self.connect('nSector', 'analysis.nSector')
        self.connect('setup.Uhub', 'analysis.Uhub')
        self.connect('setup.Omega', 'analysis.Omega')
        self.connect('setup.pitch', 'analysis.pitch')
        self.connect('tiploss', 'analysis.tiploss')
        self.connect('hubloss', 'analysis.hubloss')
        self.connect('wakerotation', 'analysis.wakerotation')
        self.connect('usecd', 'analysis.usecd')

        # connections to drivetrain
        self.connect('analysis.P', 'dt.aeroPower')
        self.connect('analysis.Q', 'dt.aeroTorque')
        self.connect('analysis.T', 'dt.aeroThrust')
        self.connect('control_ratedPower', 'dt.ratedPower')
        self.connect('drivetrainType', 'dt.drivetrainType')

        # connections to powercurve
        self.connect('control_Vin', 'powercurve.control_Vin')
        self.connect('control_Vout', 'powercurve.control_Vout')
        self.connect('control_maxOmega', 'powercurve.control_maxOmega')
        self.connect('control_minOmega', 'powercurve.control_minOmega')
        self.connect('control_pitch', 'powercurve.control_pitch')
        self.connect('control_ratedPower', 'powercurve.control_ratedPower')
        self.connect('control_tsr', 'powercurve.control_tsr')
        self.connect('setup.Uhub', 'powercurve.Vcoarse')
        self.connect('dt.power', 'powercurve.Pcoarse')
        self.connect('analysis.T', 'powercurve.Tcoarse')
        self.connect('geom.R', 'powercurve.R')

        # # setup Brent method to find rated speed
        # self.connect('control_Vin', 'brent.lower_bound')
        # self.connect('control_Vout', 'brent.upper_bound')
        # self.brent.add_param('powercurve.Vrated', low=-1e-15, high=1e15)
        # self.brent.add_constraint('powercurve.residual = 0')
        # self.brent.invalid_bracket_return = 1.0

        # connections to wind
        #self.wind.z = np.zeros(1)
        #self.wind.U = np.zeros(1)
        # self.connect('cdf_reference_mean_wind_speed', 'wind.Uref')
        self.connect('turbineclass.V_mean', 'wind.Uref')
        self.connect('cdf_reference_height_wind_speed', 'wind.zref')
        self.connect('wind_zvec', 'wind.z')
        self.connect('analysis.shearExp', 'wind.shearExp')

        # connections to cdf
        self.connect('powercurve.V', 'cdf.x')
        self.connect('wind.U', 'cdf.xbar', src_indices=[0])
        self.connect('shape_parameter', 'cdf.k')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to outputs
        self.connect('powercurve.V', 'V_in')
        self.connect('powercurve.P', 'P_in')
        self.connect('aep.AEP', 'AEP_in')
        self.connect('powercurve.rated_V', 'rated_V_in')
        self.connect('powercurve.rated_Omega', 'rated_Omega_in')
        self.connect('powercurve.rated_pitch', 'rated_pitch_in')
        self.connect('powercurve.rated_T', 'rated_T_in')
        self.connect('powercurve.rated_Q', 'rated_Q_in')


        # connect to outputs
        self.connect('geom.diameter', 'diameter_in')
        self.connect('turbineclass.V_extreme', 'V_extreme_in')
        self.connect('precurve_tip', 'precurveTip_in')
        self.connect('presweep_tip', 'presweepTip_in')


if __name__ == '__main__':
    # myref = NREL5MW()
    myref = DTU10MW()
    
    rotor = Problem()
    npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
    
    rotor.root = RotorAeroPower(myref, npts_coarse_power_curve, npts_spline_power_curve)
    
    #rotor.setup(check=False)
    rotor.setup()
    
    # === blade grid ===
    rotor['hubFraction'] = myref.hubFraction #0.025  # (Float): hub location as fraction of radius
    rotor['bladeLength'] = myref.bladeLength #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    # rotor['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    rotor['precone'] = myref.precone #2.5  # (Float, deg): precone angle
    rotor['tilt'] = myref.tilt #5.0  # (Float, deg): shaft tilt
    rotor['yaw'] = 0.0  # (Float, deg): yaw error
    rotor['nBlades'] = myref.nBlades #3  # (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    rotor['r_max_chord'] = myref.r_max_chord #0.23577  # (Float): location of max chord on unit radius
    rotor['chord_in'] = myref.chord #np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in'] = myref.theta #np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in'] = myref.precurve #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in'] = myref.presweep #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    # rotor['delta_precurve_in'] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
    rotor['sparT_in'] = myref.spar_thickness #np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in'] = myref.te_thickness #np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # ------------------
    
    # === atmosphere ===
    rotor['analysis.rho'] = 1.225  # (Float, kg/m**3): density of air
    rotor['analysis.mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['hub_height'] = myref.hub_height #90.0
    rotor['analysis.shearExp'] = 0.25  # (Float): shear exponent
    rotor['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['cdf_reference_height_wind_speed'] = myref.hub_height #90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    # ----------------------
    
    # === control ===
    rotor['control_Vin'] = myref.control_Vin #3.0  # (Float, m/s): cut-in wind speed
    rotor['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
    rotor['control_ratedPower'] = myref.rating #5e6  # (Float, W): rated power
    rotor['control_minOmega'] = myref.control_minOmega #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control_maxOmega'] = myref.control_maxOmega #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control_tsr'] = myref.control_tsr #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control_pitch'] = myref.control_pitch #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType'] = myref.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    # ----------------------

    # === run and outputs ===
    rotor.run()
    print(rotor['r_pts'])

    print('AEP =', rotor['AEP'])
    print('diameter =', rotor['diameter'])
    print('ratedConditions.V =', rotor['rated_V'])
    print('ratedConditions.Omega =', rotor['rated_Omega'])
    print('ratedConditions.pitch =', rotor['rated_pitch'])
    print('ratedConditions.T =', rotor['rated_T'])
    print('ratedConditions.Q =', rotor['rated_Q'])
    #for io in rotor.root.unknowns:
    #    print(io + ' ' + str(rotor.root.unknowns[io]))



    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rotor['V'], rotor['P']/1e6)
    plt.xlabel('wind speed (m/s)')
    plt.xlabel('power (W)')

    plt.show()
    # ----------------
    '''
    f = open('deriv_aeropower.dat','w')
    #out = rotor.check_total_derivatives(f)
    out = rotor.check_partial_derivatives(f, compact_print=True)
    f.close()
    tol = 1e-4
    for comp in out.keys():
        for k in out[comp].keys():
            if ( (out[comp][k]['rel error'][0] > tol) and (out[comp][k]['abs error'][0] > tol) ):
                print k
    '''

