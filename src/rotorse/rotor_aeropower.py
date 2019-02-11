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
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import PchipInterpolator

from ccblade.ccblade_component import CCBladeGeometry, CCBladePower
from ccblade import CCAirfoil, CCBlade

from commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from commonse.utilities import vstack, trapz_deriv, linspace_with_deriv, smooth_min, smooth_abs
from commonse.environment import PowerWind
#from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from akima import Akima
from rotorse.rotor_geometry import RotorGeometry, TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE
from rotorse import RPM2RS, RS2RPM

from rotorse.rotor_geometry_yaml import ReferenceBlade

import time
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


class RegulatedPowerCurve(Component): # Implicit COMPONENT

    def __init__(self, naero, n_pc, n_pc_spline, regulation_reg_II5 = True, regulation_reg_III = True):
        super(RegulatedPowerCurve, self).__init__()

        # parameters
        self.add_param('control_Vin',        val=0.0, units='m/s',  desc='cut-in wind speed')
        self.add_param('control_Vout',       val=0.0, units='m/s',  desc='cut-out wind speed')
        self.add_param('control_ratedPower', val=0.0, units='W',    desc='electrical rated power')
        self.add_param('control_minOmega',   val=0.0, units='rpm',  desc='minimum allowed rotor rotation speed')
        self.add_param('control_maxOmega',   val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
        self.add_param('control_maxTS',      val=0.0, units='m/s',  desc='maximum allowed blade tip speed')
        self.add_param('control_tsr',        val=0.0,               desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control_pitch',      val=0.0, units='deg',  desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_param('drivetrainType',     val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True)
        
        self.add_param('r',         val=np.zeros(naero), units='m',   desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_param('chord',     val=np.zeros(naero), units='m',   desc='chord length at each section')
        self.add_param('theta',     val=np.zeros(naero), units='deg', desc='twist angle at each section (positive decreases angle of attack)')
        self.add_param('Rhub',      val=0.0,             units='m',   desc='hub radius')
        self.add_param('Rtip',      val=0.0,             units='m',   desc='tip radius')
        self.add_param('hubHt',     val=0.0,             units='m',   desc='hub height')
        self.add_param('precone',   val=0.0,             units='deg', desc='precone angle', )
        self.add_param('tilt',      val=0.0,             units='deg', desc='shaft tilt', )
        self.add_param('yaw',       val=0.0,             units='deg', desc='yaw error', )
        self.add_param('precurve',      val=np.zeros(naero),    units='m', desc='precurve at each section')
        self.add_param('precurveTip',   val=0.0,                units='m', desc='precurve at tip')

        self.add_param('airfoils',  val=[0]*naero,                      desc='CCAirfoil instances', pass_by_obj=True)
        self.add_param('B',         val=0,                              desc='number of blades', pass_by_obj=True)
        self.add_param('rho',       val=0.0,        units='kg/m**3',    desc='density of air')
        self.add_param('mu',        val=0.0,        units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_param('shearExp',  val=0.0,                            desc='shear exponent')
        self.add_param('nSector',   val=4,                              desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True)
        self.add_param('tiploss',   val=True,                           desc='include Prandtl tip loss model', pass_by_obj=True)
        self.add_param('hubloss',   val=True,                           desc='include Prandtl hub loss model', pass_by_obj=True)
        self.add_param('wakerotation', val=True,                        desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)', pass_by_obj=True)
        self.add_param('usecd',     val=True,                           desc='use drag coefficient in computing induction factors', pass_by_obj=True)

        # outputs
        self.add_output('V',        val=np.zeros(n_pc), units='m/s',  desc='wind vector')
        self.add_output('Omega',    val=np.zeros(n_pc), units='rpm',  desc='rotor rotational speed')
        self.add_output('pitch',    val=np.zeros(n_pc), units='deg',  desc='rotor pitch schedule')
        self.add_output('P',        val=np.zeros(n_pc), units='W',    desc='rotor electrical power')
        self.add_output('T',        val=np.zeros(n_pc), units='N',    desc='rotor aerodynamic thrust')
        self.add_output('Q',        val=np.zeros(n_pc), units='N*m',  desc='rotor aerodynamic torque')
        self.add_output('M',        val=np.zeros(n_pc), units='N*m',  desc='blade root moment')
        self.add_output('Cp',       val=np.zeros(n_pc),               desc='rotor electrical power coefficient')

        self.add_output('V_spline', val=np.zeros(n_pc_spline), units='m/s',  desc='wind vector')
        self.add_output('P_spline', val=np.zeros(n_pc_spline), units='W',    desc='rotor electrical power')
        
        self.add_output('rated_V',     val=0.0, units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        self.naero                      = naero
        self.n_pc                       = n_pc
        self.n_pc_spline                = n_pc_spline
        self.regulation_reg_II5         = regulation_reg_II5
        self.regulation_reg_III         = regulation_reg_III
        self.deriv_options['form']      = 'central'
        self.deriv_options['step_calc'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):

        # Init BEM solver
        # chord = [2.6001099, 2.62242763, 2.67996425, 2.90085584, 3.17531587, 3.42387123, 3.64330915, 3.83052412, 3.98212681, 4.10594413, 4.19584165, 4.26048096, 4.28580291, 4.29671995, 4.27768742, 4.23738792, 4.17189564, 4.08924059, 3.98766963, 3.87233115, 3.74355951, 3.60392971, 3.45902389, 3.3113602, 3.1659319, 3.02649254, 2.89457583, 2.77070512, 2.6548174, 2.54683906, 2.44672653, 2.3545666, 2.27039966, 2.19417705, 2.12588872, 2.06554844, 2.01316757, 1.96875011, 1.93235573, 1.90390608, 1.88232097, 1.8583458, 1.82120335, 1.75844402, 1.65851675, 1.50997905, 1.30133456, 1.02070032, 0.65731181, 0.20224083]
        # self.ccblade = CCBlade(params['r'], chord, params['theta'], params['airfoils'], params['Rhub'], params['Rtip'], params['B'], params['rho'], params['mu'], params['precone'], params['tilt'], params['yaw'], params['shearExp'], params['hubHt'], params['nSector'])        
        self.ccblade = CCBlade(params['r'], params['chord'], params['theta'], params['airfoils'], params['Rhub'], params['Rtip'], params['B'], params['rho'], params['mu'], params['precone'], params['tilt'], params['yaw'], params['shearExp'], params['hubHt'], params['nSector'])        

        # BEM solver optimization/root finding wrappers
        def P_residual_U(Uhub):
            Omega = min(Omega_max, Uhub*params['control_tsr']/params['Rtip']*30./np.pi)
            pitch = params['control_pitch']
            P_aero, _, _, _ = self.ccblade.evaluate([Uhub], [Omega], [pitch], coefficients=False)
            P, _  = CSMDrivetrain(P_aero, params['control_ratedPower'], params['drivetrainType'])
            return params['control_ratedPower'] - P[0]

        def P_residual_pitch(pitch, Uhub):
            Omega = min(Omega_max, Uhub*params['control_tsr']/params['Rtip']*30./np.pi)
            P_aero, _, _, _ = self.ccblade.evaluate([Uhub], [Omega], [pitch], coefficients=False)
            P, _  = CSMDrivetrain(P_aero, params['control_ratedPower'], params['drivetrainType'])
            return params['control_ratedPower'] - P[0]

        def P_max(pitch, Uhub):
            P_aero, _, _, _ = self.ccblade.evaluate([Uhub], [Omega_max], [pitch], coefficients=False)
            P, _  = CSMDrivetrain(P_aero, params['control_ratedPower'], params['drivetrainType'])
            return abs(P - params['control_ratedPower'])


        # Region II.5 wind speed
        Omega_max   = min(params['control_maxOmega'], params['control_maxTS']/params['Rtip']*30./np.pi)
        U_reg       = Omega_max*np.pi/30. * params['Rtip'] / params['control_tsr']

        # Region III wind speed
        try:
            U_rated = brentq(lambda x: P_residual_U(x), params['control_Vin'],params['control_Vout'], xtol=1e-8)
        except ValueError:
            U_rated = params['control_Vout']

        # Region wind grids
        odd = 0
        if self.n_pc % 2 != 0:
            odd = 1
        if U_reg < U_rated:
            regionII5 = True
            U_II  = np.linspace(params['control_Vin'],U_reg, int(self.n_pc/2)+odd)
            U_II5 = np.linspace(U_reg, U_rated, 4)[1:]
            U_III = np.linspace(U_rated, params['control_Vout'], int(self.n_pc/2)-2)[1:]

        else:
            regionII5 = False
            U_II  = np.linspace(params['control_Vin'],U_rated, int(self.n_pc/2)+odd+1)
            U_II5 = []
            U_III = np.linspace(U_rated, params['control_Vout'], int(self.n_pc/2))[1:]

        # Region II Pitch, Rotor Speed
        Omega_II = U_II*params['control_tsr']/params['Rtip']*30./np.pi
        pitch_II = np.array([params['control_pitch']]*len(U_II))

        # Region II5 Pitch, Rotor Speed
        options = {}
        options['disp'] = False
        if regionII5:
            Omega_II5 = [Omega_max]*len(U_II5)
            pitch_II5 = [params['control_pitch']]*len(U_II5)
            if self.regulation_reg_II5:
                for i, Ui in enumerate(U_II5):
                    pitch_II5i = minimize_scalar(lambda x: P_max(x, Ui), bounds=[-10., 10.], method='bounded', options=options)['x']
                    try:
                        pitch_II5[i] = pitch_II5i[0]
                    except:
                        pitch_II5[i] = pitch_II5i
        else:
            pitch_II5 = []
            Omega_II5 = []


        # Region III Pitch, Rotor Speed
        Omega_III = [Omega_max]*len(U_III)
        pitch_III = [params['control_pitch']]*len(U_III)
        if self.regulation_reg_III:
            for i, Ui in enumerate(U_III):
                try:
                    pitch_III[i] = brentq(lambda x: P_residual_pitch(x, Ui), params['control_pitch'], 90., xtol=1e-4)
                except ValueError:
                    if P_residual_pitch(params['control_pitch'], Ui) > 0.:
                        try:
                            pitch_III[i] = brentq(lambda x: P_residual_pitch(x, Ui), -45., params['control_pitch'], xtol=1e-4)
                        except:
                            pitch_III[i] = params['control_pitch']
                    else:
                        pitch_III[i] = 90.

        # BEM solution for full operating range
        U     = np.concatenate((U_II, U_II5, U_III))
        Omega = np.concatenate((Omega_II, Omega_II5, Omega_III))
        pitch = np.concatenate((pitch_II, pitch_II5, pitch_III))

        P_aero, T, Q, M, Cp_aero, _, _, _ = self.ccblade.evaluate(U, Omega, pitch, coefficients=True)
        P, eff  = CSMDrivetrain(P_aero, params['control_ratedPower'], params['drivetrainType'])
        Cp = Cp_aero*eff

        # If above rated regulation not determined, P(U>U_rated) = P_rated
        if not self.regulation_reg_III:
            P = np.array([min(Pi, params['control_ratedPower']) for Pi in P])

        # Fit spline to powercurve for higher grid density
        spline   = PchipInterpolator(U, P)
        V_spline = np.linspace(params['control_Vin'],params['control_Vout'], num=self.n_pc_spline)
        P_spline = spline(V_spline)

        # outputs
        unknowns['V']       = U
        unknowns['Omega']   = Omega
        unknowns['pitch']   = pitch

        unknowns['P']       = P
        unknowns['T']       = T
        unknowns['Q']       = Q
        unknowns['M']       = M
        unknowns['Cp']      = Cp

        idx_rated = list(U).index(U_rated)
        unknowns['rated_V']     = U_rated
        unknowns['rated_Omega'] = Omega[idx_rated]
        unknowns['rated_pitch'] = pitch[idx_rated]
        unknowns['rated_T'   ]  = T[idx_rated]
        unknowns['rated_Q']     = Q[idx_rated]
        
        unknowns['V_spline']    = V_spline
        unknowns['P_spline']    = P_spline


class AEP(Component):
    def __init__(self, n_pc_spline):
        super(AEP, self).__init__()
        """integrate to find annual energy production"""

        # inputs
        self.add_param('CDF_V', val=np.zeros(n_pc_spline), units='m/s', desc='cumulative distribution function evaluated at each wind speed')
        self.add_param('P', val=np.zeros(n_pc_spline), units='W', desc='power curve (power)')
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


def CSMDrivetrain(aeroPower, ratedPower, drivetrainType):

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

    return aeroPower * eff, eff


class OutputsAero(Component):
    def __init__(self, npts_coarse_power_curve):
        super(OutputsAero, self).__init__()

        # --- outputs ---
        self.add_param('AEP_in', val=0.0, units='kW*h', desc='annual energy production')
        self.add_param('V_in', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='wind speeds (power curve)')
        self.add_param('P_in', val=np.zeros(npts_coarse_power_curve), units='W', desc='power (power curve)')

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
        self.add_output('V', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='wind speeds (power curve)')
        self.add_output('P', val=np.zeros(npts_coarse_power_curve), units='W', desc='power (power curve)')

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
        # unknowns['diameter'] = params['diameter_in']
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
    def __init__(self, RefBlade, npts_coarse_power_curve=20, npts_spline_power_curve=200, regulation_reg_III=True):
        super(RotorAeroPower, self).__init__()

        self.add('rho', IndepVarComp('rho', val=1.225), promotes=['*'])
        self.add('mu', IndepVarComp('mu', val=1.81e-5), promotes=['*'])
        self.add('shearExp', IndepVarComp('shearExp', val=0.2), promotes=['*'])

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
        self.add('c_maxTS', IndepVarComp('control_maxTS', val=0.0, units='m/s', desc='maximum allowed blade tip speed'), promotes=['*'])
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
        self.add('powercurve', RegulatedPowerCurve(RefBlade.npts, npts_coarse_power_curve, npts_spline_power_curve, regulation_reg_III))
        self.add('wind', PowerWind(1))
        # self.add('cdf', WeibullWithMeanCDF(npts_coarse_power_curve))
        self.add('cdf', RayleighCDF(npts_spline_power_curve))
        self.add('aep', AEP(npts_spline_power_curve))

        self.add('outputs_aero', OutputsAero(npts_coarse_power_curve), promotes=['*'])


        # connections to analysis
        self.connect('r_pts', 'powercurve.r')
        self.connect('chord', 'powercurve.chord')
        self.connect('theta', 'powercurve.theta')
        self.connect('precurve', 'powercurve.precurve')
        self.connect('precurve_tip', 'powercurve.precurveTip')
        self.connect('Rhub', 'powercurve.Rhub')
        self.connect('Rtip', 'powercurve.Rtip')
        self.connect('hub_height', 'powercurve.hubHt')
        self.connect('precone', 'powercurve.precone')
        self.connect('tilt', 'powercurve.tilt')
        self.connect('yaw', 'powercurve.yaw')
        self.connect('airfoils', 'powercurve.airfoils')
        self.connect('nBlades', 'powercurve.B')
        self.connect('rho', 'powercurve.rho')
        self.connect('mu', 'powercurve.mu')
        self.connect('shearExp', 'powercurve.shearExp')
        self.connect('nSector', 'powercurve.nSector')
        self.connect('tiploss', 'powercurve.tiploss')
        self.connect('hubloss', 'powercurve.hubloss')
        self.connect('wakerotation', 'powercurve.wakerotation')
        self.connect('usecd', 'powercurve.usecd')

        # connections to powercurve
        self.connect('drivetrainType', 'powercurve.drivetrainType')
        self.connect('control_Vin', 'powercurve.control_Vin')
        self.connect('control_Vout', 'powercurve.control_Vout')
        self.connect('control_maxTS', 'powercurve.control_maxTS')
        self.connect('control_maxOmega', 'powercurve.control_maxOmega')
        self.connect('control_minOmega', 'powercurve.control_minOmega')
        self.connect('control_pitch', 'powercurve.control_pitch')
        self.connect('control_ratedPower', 'powercurve.control_ratedPower')
        self.connect('control_tsr', 'powercurve.control_tsr')

        # connections to wind
        # self.connect('cdf_reference_mean_wind_speed', 'wind.Uref')
        self.connect('turbineclass.V_mean', 'wind.Uref')
        self.connect('cdf_reference_height_wind_speed', 'wind.zref')
        self.connect('wind_zvec', 'wind.z')
        self.connect('shearExp', 'wind.shearExp')

        # connections to cdf
        self.connect('powercurve.V_spline', 'cdf.x')
        self.connect('wind.U', 'cdf.xbar', src_indices=[0])
        self.connect('shape_parameter', 'cdf.k')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P_spline', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connect to outputs
        self.connect('geom.diameter', 'diameter_in')
        self.connect('turbineclass.V_extreme', 'V_extreme_in')
        self.connect('precurve_tip', 'precurveTip_in')
        self.connect('presweep_tip', 'presweepTip_in')

        self.connect('powercurve.V', 'V_in')
        self.connect('powercurve.P', 'P_in')
        self.connect('aep.AEP', 'AEP_in')
        self.connect('powercurve.rated_V', 'rated_V_in')
        self.connect('powercurve.rated_Omega', 'rated_Omega_in')
        self.connect('powercurve.rated_pitch', 'rated_pitch_in')
        self.connect('powercurve.rated_T', 'rated_T_in')
        self.connect('powercurve.rated_Q', 'rated_Q_in')


if __name__ == '__main__':
    tt = time.time()
    # myref = NREL5MW()
    myref = TUM3_35MW()
    # myref = DTU10MW()
    
    rotor = Problem()
    npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    npts_spline_power_curve = 2000  # (Int): number of points to use in fitting spline to power curve
    regulation_reg_II5 = False # calculate Region 2.5 pitch schedule, False will not maximize power in region 2.5
    regulation_reg_III = False # calculate Region 3 pitch schedule, False will return erroneous Thrust, Torque, and Moment for above rated
    
    rotor.root = RotorAeroPower(myref, npts_coarse_power_curve, npts_spline_power_curve, regulation_reg_II5, regulation_reg_III)
    
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
    rotor['rho'] = 1.225  # (Float, kg/m**3): density of air
    rotor['mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['hub_height'] = myref.hub_height #90.0
    rotor['shearExp'] = 0.0  # (Float): shear exponent
    rotor['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['cdf_reference_height_wind_speed'] = myref.hub_height #90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    # ----------------------
    
    # === control ===
    rotor['control_Vin'] = myref.control_Vin #3.0  # (Float, m/s): cut-in wind speed
    rotor['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
    rotor['control_ratedPower'] = myref.rating #5e6  # (Float, W): rated power
    rotor['control_minOmega'] = myref.control_minOmega #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control_maxOmega'] = myref.control_maxOmega #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control_maxTS'] = myref.control_maxTS
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

    print('Run time = ', time.time()-tt)
    print('AEP =', rotor['AEP'])
    print('diameter =', rotor['diameter'])
    print('ratedConditions.V =', rotor['rated_V'])
    print('ratedConditions.Omega =', rotor['rated_Omega'])
    print('ratedConditions.pitch =', rotor['rated_pitch'])
    print('ratedConditions.T =', rotor['rated_T'])
    print('ratedConditions.Q =', rotor['rated_Q'])


    # rotor2 = Problem()
    # npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    
    # rotor2.root = RotorAeroPower(myref, npts_coarse_power_curve, True)
    
    # #rotor.setup(check=False)
    # rotor2.setup()
    
    # # === blade grid ===
    # rotor2['hubFraction'] = myref.hubFraction #0.025  # (Float): hub location as fraction of radius
    # rotor2['bladeLength'] = myref.bladeLength #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    # # rotor2['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    # rotor2['precone'] = myref.precone #2.5  # (Float, deg): precone angle
    # rotor2['tilt'] = myref.tilt #5.0  # (Float, deg): shaft tilt
    # rotor2['yaw'] = 0.0  # (Float, deg): yaw error
    # rotor2['nBlades'] = myref.nBlades #3  # (Int): number of blades
    # # ------------------
    
    # # === blade geometry ===
    # rotor2['r_max_chord'] = myref.r_max_chord #0.23577  # (Float): location of max chord on unit radius
    # rotor2['chord_in'] = myref.chord #np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    # rotor2['theta_in'] = myref.theta #np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    # rotor2['precurve_in'] = myref.precurve #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    # rotor2['presweep_in'] = myref.presweep #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    # # rotor2['delta_precurve_in'] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
    # rotor2['sparT_in'] = myref.spar_thickness #np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    # rotor2['teT_in'] = myref.te_thickness #np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # # ------------------
    
    # # === atmosphere ===
    # rotor2['rho'] = 1.225  # (Float, kg/m**3): density of air
    # rotor2['mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    # rotor2['hub_height'] = myref.hub_height #90.0
    # rotor2['shearExp'] = 0.0  # (Float): shear exponent
    # rotor2['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    # rotor2['cdf_reference_height_wind_speed'] = myref.hub_height #90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    # # ----------------------
    
    # # === control ===
    # rotor2['control_Vin'] = myref.control_Vin #3.0  # (Float, m/s): cut-in wind speed
    # rotor2['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
    # rotor2['control_ratedPower'] = myref.rating #5e6  # (Float, W): rated power
    # rotor2['control_minOmega'] = myref.control_minOmega #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    # rotor2['control_maxOmega'] = myref.control_maxOmega #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    # rotor2['control_maxTS'] = 80.
    # rotor2['control_tsr'] = myref.control_tsr #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    # rotor2['control_pitch'] = myref.control_pitch #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    # # ----------------------

    # # === aero and structural analysis options ===
    # rotor2['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    # rotor2['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    # rotor2['drivetrainType'] = myref.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    # # ----------------------

    # # === run and outputs ===
    # rotor2.run()

    # print('Run time = ', time.time()-tt)
    # print('AEP =', rotor2['AEP'])
    # print('diameter =', rotor2['diameter'])
    # print('ratedConditions.V =', rotor2['rated_V'])
    # print('ratedConditions.Omega =', rotor2['rated_Omega'])
    # print('ratedConditions.pitch =', rotor2['rated_pitch'])
    # print('ratedConditions.T =', rotor2['rated_T'])
    # print('ratedConditions.Q =', rotor2['rated_Q'])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(rotor['V'], rotor['P']/1e6, label='200')
    # plt.plot(rotor2['V'], rotor2['P']/1e6, label='20')
    # plt.xlabel('wind speed (m/s)')
    # plt.ylabel('power (W)')
    # plt.show()
