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
from scipy.optimize import brentq, minimize_scalar, minimize
from scipy.interpolate import PchipInterpolator

from ccblade.ccblade_component import CCBladeGeometry, CCBladePower
from ccblade import CCAirfoil, CCBlade

from commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from commonse.utilities import vstack, trapz_deriv, linspace_with_deriv, smooth_min, smooth_abs
from commonse.environment import PowerWind
#from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from akima import Akima

from rotorse import RPM2RS, RS2RPM
from rotorse.rotor_geometry import RotorGeometry, TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE
from rotorse.rotor_geometry_yaml import ReferenceBlade

import time
# ---------------------
# Components
# ---------------------


class RegulatedPowerCurve(Component): # Implicit COMPONENT

    def __init__(self, naero, n_pc, n_pc_spline, regulation_reg_II5 = True, regulation_reg_III = False):
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
        self.add_param('drivetrainEff',     val=0.0,               desc='overwrite drivetrain model with a given efficiency, used for FAST analysis')
        
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
        self.add_output('V',        val=np.zeros(n_pc), units='m/s',    desc='wind vector')
        self.add_output('Omega',    val=np.zeros(n_pc), units='rpm',    desc='rotor rotational speed')
        self.add_output('pitch',    val=np.zeros(n_pc), units='deg',    desc='rotor pitch schedule')
        self.add_output('P',        val=np.zeros(n_pc), units='W',      desc='rotor electrical power')
        self.add_output('T',        val=np.zeros(n_pc), units='N',      desc='rotor aerodynamic thrust')
        self.add_output('Q',        val=np.zeros(n_pc), units='N*m',    desc='rotor aerodynamic torque')
        self.add_output('M',        val=np.zeros(n_pc), units='N*m',    desc='blade root moment')
        self.add_output('Cp',       val=np.zeros(n_pc),                 desc='rotor electrical power coefficient')
        self.add_output('V_spline', val=np.zeros(n_pc_spline), units='m/s',  desc='wind vector')
        self.add_output('P_spline', val=np.zeros(n_pc_spline), units='W',    desc='rotor electrical power')
        self.add_output('V_R25',       val=0.0, units='m/s', desc='region 2.5 transition wind speed')
        self.add_output('rated_V',     val=0.0, units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0, units='N',   desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')
        self.add_output('ax_induct_cutin',   val=np.zeros(naero), desc='rotor axial induction at cut-in wind speed along blade span')
        self.add_output('tang_induct_cutin', val=np.zeros(naero), desc='rotor tangential induction at cut-in wind speed along blade span')
        self.add_output('aoa_cutin',         val=np.zeros(naero), desc='angle of attack distribution along blade span at cut-in wind speed')
        self.add_output('cl_cutin',          val=np.zeros(naero), desc='lift coefficient distribution along blade span at cut-in wind speed')
        self.add_output('cd_cutin',          val=np.zeros(naero), desc='drag coefficient distribution along blade span at cut-in wind speed')

        self.naero                      = naero
        self.n_pc                       = n_pc
        self.n_pc_spline                = n_pc_spline
        self.lock_pitchII               = False
        self.regulation_reg_II5         = regulation_reg_II5
        self.regulation_reg_III         = regulation_reg_III
        self.deriv_options['form']      = 'central'
        self.deriv_options['step_calc'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):
                
        self.ccblade = CCBlade(params['r'], params['chord'], params['theta'], params['airfoils'], params['Rhub'], params['Rtip'], params['B'], params['rho'], params['mu'], params['precone'], params['tilt'], params['yaw'], params['shearExp'], params['hubHt'], params['nSector'])
        
        Uhub    = np.linspace(params['control_Vin'],params['control_Vout'], self.n_pc)
        
        P_aero   = np.zeros_like(Uhub)
        Cp_aero  = np.zeros_like(Uhub)
        P       = np.zeros_like(Uhub)
        Cp      = np.zeros_like(Uhub)
        T       = np.zeros_like(Uhub)
        Q       = np.zeros_like(Uhub)
        M       = np.zeros_like(Uhub)
        Omega   = np.zeros_like(Uhub)
        pitch   = np.zeros_like(Uhub) + params['control_pitch']

        Omega_max = min([params['control_maxTS'] / params['Rtip'], params['control_maxOmega']*np.pi/30.])
        
        # Region II
        for i in range(len(Uhub)):
            Omega[i] = Uhub[i] * params['control_tsr'] / params['Rtip']
        
        P_aero, T, Q, M, Cp_aero, _, _, _ = self.ccblade.evaluate(Uhub, Omega * 30. / np.pi, pitch, coefficients=True)
        P, eff  = CSMDrivetrain(P_aero, params['control_ratedPower'], params['drivetrainType'])
        Cp      = Cp_aero*eff
        
        
        # search for Region 2.5 bounds
        for i in range(len(Uhub)):
        
            if Omega[i] > Omega_max and P[i] < params['control_ratedPower']:
                Omega[i]        = Omega_max
                Uhub[i]         = Omega[i] * params['Rtip'] / params['control_tsr']
                P_aero[i], T[i], Q[i], M[i], Cp_aero[i], _, _, _ = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch[i]], coefficients=True)
                P[i], eff       = CSMDrivetrain(P_aero[i], params['control_ratedPower'], params['drivetrainType'])
                Cp[i]           = Cp_aero[i]*eff
                regionIIhalf    = True
                i_IIhalf_start  = i

                unknowns['V_R25'] = Uhub[i]
                break


            if P[i] > params['control_ratedPower']:
                
                regionIIhalf = False
                break

        
        def maxPregionIIhalf(pitch, Uhub, Omega):
            Uhub_i  = Uhub
            Omega_i = Omega
            pitch   = pitch
                        
            P, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
            return -P
        
        # Solve for regoin 2.5 pitch
        options             = {}
        options['disp']     = False
        options['xatol']    = 1.e-2
        if regionIIhalf == True:
            for i in range(i_IIhalf_start + 1, len(Uhub)):   
                Omega[i]    = Omega_max
                pitch0      = pitch[i-1]
                
                bnds        = [pitch0 - 10., pitch0 + 10.]
                pitch_regionIIhalf = minimize_scalar(lambda x: maxPregionIIhalf(x, Uhub[i], Omega[i]), bounds=bnds, method='bounded', options=options)['x']
                pitch[i]    = pitch_regionIIhalf
                
                P_aero[i], T[i], Q[i], M[i], Cp_aero[i], _, _, _ = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch[i]], coefficients=True)
                
                P[i], eff  = CSMDrivetrain(P_aero[i], params['control_ratedPower'], params['drivetrainType'])
                Cp[i]      = Cp_aero[i]*eff

                if P[i] > params['control_ratedPower']:    
                    break    
                        
        options             = {}
        options['disp']     = False
        def constantPregionIII(pitch, Uhub, Omega):
            Uhub_i  = Uhub
            Omega_i = Omega
            pitch   = pitch           
            P_aero, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
            P, eff          = CSMDrivetrain(P_aero, params['control_ratedPower'], params['drivetrainType'])
            return abs(P - params['control_ratedPower'])
            

        
        if regionIIhalf == True:
            # Rated conditions
            
            def min_Uhub_rated_II12(min_params):
                return min_params[1]
                
            def get_Uhub_rated_II12(min_params):

                Uhub_i  = min_params[1]
                Omega_i = Omega_max
                pitch   = min_params[0]           
                P_aero_i, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
                P_i,eff          = CSMDrivetrain(P_aero_i, params['control_ratedPower'], params['drivetrainType'])
                return abs(P_i - params['control_ratedPower'])

            x0              = [pitch[i] + 2. , Uhub[i]]
            bnds            = [(pitch0, pitch0 + 10.),(Uhub[i-1],Uhub[i+1])]
            const           = {}
            const['type']   = 'eq'
            const['fun']    = get_Uhub_rated_II12
            params_rated    = minimize(min_Uhub_rated_II12, x0, method='SLSQP', tol = 1.e-2, bounds=bnds, constraints=const)
            U_rated         = params_rated.x[1]
            
            if not np.isnan(U_rated):
                Uhub[i]         = U_rated
                pitch[i]        = params_rated.x[0]
            else:
                print('Regulation trajectory is struggling to find a solution for rated wind speed. Check rotor_aeropower.py')
                U_rated         = Uhub[i]
            
            Omega[i]        = Omega_max
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], _, _, _ = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch0], coefficients=True)
            P_i, eff        = CSMDrivetrain(P_aero[i], params['control_ratedPower'], params['drivetrainType'])
            Cp[i]           = Cp_aero[i]*eff
            P[i]            = params['control_ratedPower']
            
            
        else:
            # Rated conditions
            def get_Uhub_rated_noII12(pitch, Uhub):
                Uhub_i  = Uhub
                Omega_i = min([Uhub_i * params['control_tsr'] / params['Rtip'], Omega_max])
                pitch_i = pitch           
                P_aero_i, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch_i], coefficients=False)
                P_i, eff          = CSMDrivetrain(P_aero_i, params['control_ratedPower'], params['drivetrainType'])
                return abs(P_i - params['control_ratedPower'])
            
            bnds     = [Uhub[i-1], Uhub[i+1]]
            U_rated  = minimize_scalar(lambda x: get_Uhub_rated_noII12(pitch[i], x), bounds=bnds, tol = 1.e-2, method='bounded', options=options)['x']
            
            if not np.isnan(U_rated):
                Uhub[i]         = U_rated
            else:
                print('Regulation trajectory is struggling to find a solution for rated wind speed. Check rotor_aeropower.py. For now, U rated is assumed equal to ' + str(Uhub[i]) + ' m/s')
                U_rated         = Uhub[i]
            
            
            
            
            
            Omega[i] = min([Uhub[i] * params['control_tsr'] / params['Rtip'], Omega_max])
            pitch0   = pitch[i]
            
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], _, _, _ = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch0], coefficients=True)
            P[i], eff    = CSMDrivetrain(P_aero[i], params['control_ratedPower'], params['drivetrainType'])
            Cp[i]        = Cp_aero[i]*eff
        
        
        for j in range(i + 1,len(Uhub)):
            Omega[j] = Omega[i]
            if self.regulation_reg_III == True:
                
                pitch0   = pitch[j-1]
                bnds     = [pitch0, pitch0 + 15.]
                pitch_regionIII = minimize_scalar(lambda x: constantPregionIII(x, Uhub[j], Omega[j]), bounds=bnds, method='bounded', options=options)['x']
                pitch[j]        = pitch_regionIII
                
                P_aero[j], T[j], Q[j], M[j], Cp_aero[j], _, _, _ = self.ccblade.evaluate([Uhub[j]], [Omega[j] * 30. / np.pi], [pitch[j]], coefficients=True)
                P[j], eff       = CSMDrivetrain(P_aero[j], params['control_ratedPower'], params['drivetrainType'])
                Cp[j]           = Cp_aero[j]*eff


                if abs(P[j] - params['control_ratedPower']) > 1e+4:
                    print('The pitch in region III is not being determined correctly at wind speed ' + str(Uhub[j]) + ' m/s')
                    P[j]        = params['control_ratedPower']
                    T[j]        = T[j-1]
                    Q[j]        = P[j] / Omega[j]
                    M[j]        = M[j-1]
                    pitch[j]    = pitch[j-1]
                    Cp[j]       = P[j] / (0.5 * params['rho'] * np.pi * params['Rtip']**2 * Uhub[i]**3)

                P[j] = params['control_ratedPower']
                
            else:
                P[j]        = params['control_ratedPower']
                T[j]        = 0
                Q[j]        = Q[i]
                M[j]        = 0
                pitch[j]    = 0
                Cp[j]       = P[j] / (0.5 * params['rho'] * np.pi * params['Rtip']**2 * Uhub[i]**3)

        
        unknowns['T']       = T
        unknowns['Q']       = Q
        unknowns['Omega']   = Omega * 30. / np.pi


        unknowns['P']       = P  
        unknowns['Cp']      = Cp  
        unknowns['V']       = Uhub
        unknowns['M']       = M
        unknowns['pitch']   = pitch
        
        
        self.ccblade.induction_inflow = True
        a_regII, ap_regII, alpha_regII, cl_regII, cd_regII = self.ccblade.distributedAeroLoads(Uhub[0], Omega[0] * 30. / np.pi, pitch[0], 0.0)
        
        # Fit spline to powercurve for higher grid density
        spline   = PchipInterpolator(Uhub, P)
        V_spline = np.linspace(params['control_Vin'],params['control_Vout'], num=self.n_pc_spline)
        P_spline = spline(V_spline)
        
        # outputs
        idx_rated = list(Uhub).index(U_rated)
        unknowns['rated_V']     = U_rated
        unknowns['rated_Omega'] = Omega[idx_rated] * 30. / np.pi
        unknowns['rated_pitch'] = pitch[idx_rated]
        unknowns['rated_T']     = T[idx_rated]
        unknowns['rated_Q']     = Q[idx_rated]
        unknowns['V_spline']    = V_spline
        unknowns['P_spline']    = P_spline
        unknowns['ax_induct_cutin']   = a_regII
        unknowns['tang_induct_cutin'] = ap_regII
        unknowns['aoa_cutin']         = alpha_regII
        unknowns['cl_cutin']         = cl_regII
        unknowns['cd_cutin']         = cd_regII



class Cp_Ct_Cq_Tables(Component):

    def __init__(self, naero):
        super(Cp_Ct_Cq_Tables, self).__init__()
        
        n_pitch = 5#51
        n_tsr   = 9
        n_U     = 1
        
        # parameters        
        self.add_param('control_Vin',   val=0.0,             units='m/s',       desc='cut-in wind speed')
        self.add_param('control_Vout',  val=0.0,             units='m/s',       desc='cut-out wind speed')
        self.add_param('r',             val=np.zeros(naero), units='m',         desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_param('chord',         val=np.zeros(naero), units='m',         desc='chord length at each section')
        self.add_param('theta',         val=np.zeros(naero), units='deg',       desc='twist angle at each section (positive decreases angle of attack)')
        self.add_param('Rhub',          val=0.0,             units='m',         desc='hub radius')
        self.add_param('Rtip',          val=0.0,             units='m',         desc='tip radius')
        self.add_param('hubHt',         val=0.0,             units='m',         desc='hub height')
        self.add_param('precone',       val=0.0,             units='deg',       desc='precone angle')
        self.add_param('tilt',          val=0.0,             units='deg',       desc='shaft tilt')
        self.add_param('yaw',           val=0.0,             units='deg',       desc='yaw error')
        self.add_param('precurve',      val=np.zeros(naero), units='m',         desc='precurve at each section')
        self.add_param('precurveTip',   val=0.0,             units='m',         desc='precurve at tip')
        self.add_param('airfoils',      val=[0]*naero,                          desc='CCAirfoil instances', pass_by_obj=True)
        self.add_param('B',             val=0,                                  desc='number of blades', pass_by_obj=True)
        self.add_param('rho',           val=0.0,             units='kg/m**3',   desc='density of air')
        self.add_param('mu',            val=0.0,             units='kg/(m*s)',  desc='dynamic viscosity of air')
        self.add_param('shearExp',      val=0.0,                                desc='shear exponent')
        self.add_param('nSector',       val=4,                                  desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True)
        self.add_param('tiploss',       val=True,                               desc='include Prandtl tip loss model', pass_by_obj=True)
        self.add_param('hubloss',       val=True,                               desc='include Prandtl hub loss model', pass_by_obj=True)
        self.add_param('wakerotation',  val=True,                               desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)', pass_by_obj=True)
        self.add_param('usecd',         val=True,                               desc='use drag coefficient in computing induction factors', pass_by_obj=True)

        
        self.add_param('pitch_vector',  val=np.zeros(n_pitch), units='deg',     desc='pitch vector')
        self.add_param('tsr_vector',    val=np.zeros(n_tsr),                    desc='tsr vector')
        self.add_param('U_vector',      val=np.zeros(n_U),     units='m/s',     desc='wind vector')

        # outputs
        self.add_output('Cp_aero_table',        val=np.zeros((n_tsr, n_pitch, n_U)),           desc='table of aero power coefficient')
        self.add_output('Ct_aero_table',        val=np.zeros((n_tsr, n_pitch, n_U)),           desc='table of aero thrust coefficient')
        self.add_output('Cq_aero_table',        val=np.zeros((n_tsr, n_pitch, n_U)),           desc='table of aero torque coefficient')


        self.naero   = naero
        self.n_pitch = n_pitch
        self.n_tsr   = n_tsr
        self.n_U     = n_U
        
    def solve_nonlinear(self, params, unknowns, resids):
        
        self.ccblade = CCBlade(params['r'], params['chord'], params['theta'], params['airfoils'], params['Rhub'], params['Rtip'], params['B'], params['rho'], params['mu'], params['precone'], params['tilt'], params['yaw'], params['shearExp'], params['hubHt'], params['nSector'])
        
        if max(params['U_vector']) == 0.:
            params['U_vector']    = np.linspace(params['control_Vin'],params['control_Vout'], self.n_U)
        
        if max(params['tsr_vector']) == 0.:
            params['tsr_vector'] = np.linspace(3.,11., self.n_tsr)
        
        if max(params['pitch_vector']) == 0.:
            params['pitch_vector'] = np.linspace(-10., 40., self.n_pitch)
        
        R = params['Rtip']
        
        Cp_aero_table = np.zeros((self.n_tsr, self.n_pitch, self.n_U))
        Ct_aero_table = np.zeros((self.n_tsr, self.n_pitch, self.n_U))
        Cq_aero_table = np.zeros((self.n_tsr, self.n_pitch, self.n_U))
        
        for i in range(self.n_U):
            for j in range(self.n_tsr):
                U     =  params['U_vector'][i] * np.ones(self.n_pitch)
                Omega = params['tsr_vector'][j] *  params['U_vector'][i] / R * 30. / np.pi * np.ones(self.n_pitch)
                _, _, _, _, unknowns['Cp_aero_table'][j,:,i], unknowns['Ct_aero_table'][j,:,i], unknowns['Cq_aero_table'][j,:,i], _ = self.ccblade.evaluate(U, Omega, params['pitch_vector'], coefficients=True)
                
        


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
    
    elif drivetrainType == DRIVETRAIN_TYPE['CONSTANT_EFF']:
        constant = 0.00000  
        linear = 0.10000
        quadratic = 0.0000

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
        self.add_param('Cp_in', val=np.zeros(npts_coarse_power_curve), units='W', desc='power (power curve)')

        self.add_param('rated_V_in', val=0.0, units='m/s', desc='rated wind speed')
        self.add_param('rated_Omega_in', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_param('rated_pitch_in', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_param('rated_T_in', val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_param('rated_Q_in', val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        # self.add_param('diameter_in', val=0.0, units='m', desc='rotor diameter')
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
        self.add_output('Cp', val=np.zeros(npts_coarse_power_curve), units='W', desc='power (power curve)')

        self.add_output('rated_V', val=0.0, units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_output('rated_T', val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q', val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        # self.add_output('diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_output('V_extreme', val=0.0, units='m/s', desc='survival wind speed')
        self.add_output('T_extreme', val=0.0, units='N', desc='thrust at survival wind condition')
        self.add_output('Q_extreme', val=0.0, units='N*m', desc='thrust at survival wind condition')

        # internal use outputs
        self.add_output('precurveTip', val=0.0, units='m', desc='tip location in x_b')
        self.add_output('presweepTip', val=0.0, units='m', desc='tip location in y_b')  # TODO: connect later

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['AEP'] = params['AEP_in']
        unknowns['V'] = params['V_in']
        unknowns['Cp'] = params['Cp_in']
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
        J['Cp', 'Cp_in'] = np.diag(np.ones(len(params['Cp_in'])))
        J['rated_V', 'rated_V_in'] = 1
        J['rated_Omega', 'rated_Omega_in'] = 1
        J['rated_pitch', 'rated_pitch_in'] = 1
        J['rated_T', 'rated_T_in'] = 1
        J['rated_Q', 'rated_Q_in'] = 1
        # J['diameter', 'diameter_in'] = 1
        J['V_extreme', 'V_extreme_in'] = 1
        J['T_extreme', 'T_extreme_in'] = 1
        J['Q_extreme', 'Q_extreme_in'] = 1
        J['precurveTip', 'precurveTip_in'] = 1
        J['presweepTip', 'presweepTip_in'] = 1

        return J

class RotorAeroPower(Group):
    def __init__(self, RefBlade, npts_coarse_power_curve=20, npts_spline_power_curve=200, regulation_reg_II5=True, regulation_reg_III=True):
        super(RotorAeroPower, self).__init__()

        NPTS = len(RefBlade['pf']['s'])

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
        self.add('powercurve', RegulatedPowerCurve(NPTS, npts_coarse_power_curve, npts_spline_power_curve, regulation_reg_II5, regulation_reg_III))
        self.add('wind', PowerWind(1))
        self.add('cdf', WeibullWithMeanCDF(npts_spline_power_curve))
        # self.add('cdf', RayleighCDF(npts_spline_power_curve))
        self.add('aep', AEP(npts_spline_power_curve))

        self.add('outputs_aero', OutputsAero(npts_coarse_power_curve), promotes=['*'])


        # connections to analysis
        self.connect('r_pts',       'powercurve.r')
        self.connect('chord',       'powercurve.chord')
        self.connect('theta',       'powercurve.theta')
        self.connect('precurve',    'powercurve.precurve')
        self.connect('precurve_tip','powercurve.precurveTip')
        self.connect('Rhub',        'powercurve.Rhub')
        self.connect('Rtip',        'powercurve.Rtip')
        self.connect('precone',     'powercurve.precone')
        self.connect('tilt',        'powercurve.tilt')
        self.connect('yaw',         'powercurve.yaw')
        self.connect('airfoils',    'powercurve.airfoils')
        self.connect('nBlades',     'powercurve.B')
        self.connect('nSector',     'powercurve.nSector')

        self.connect('tiploss',     'powercurve.tiploss')
        self.connect('hubloss',     'powercurve.hubloss')
        self.connect('wakerotation','powercurve.wakerotation')
        self.connect('usecd',       'powercurve.usecd')

        # connections to powercurve
        self.connect('drivetrainType',      'powercurve.drivetrainType')
        self.connect('control_Vin',         'powercurve.control_Vin')
        self.connect('control_Vout',        'powercurve.control_Vout')
        self.connect('control_maxTS',       'powercurve.control_maxTS')
        self.connect('control_maxOmega',    'powercurve.control_maxOmega')
        self.connect('control_minOmega',    'powercurve.control_minOmega')
        self.connect('control_pitch',       'powercurve.control_pitch')
        self.connect('control_ratedPower',  'powercurve.control_ratedPower')
        self.connect('control_tsr',         'powercurve.control_tsr')

        self.connect('hub_height',  'powercurve.hubHt')
        self.connect('rho',         'powercurve.rho')
        self.connect('mu',          'powercurve.mu')
        self.connect('shearExp',    'powercurve.shearExp')

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
        # self.connect('geom.diameter', 'diameter_in')
        self.connect('turbineclass.V_extreme50', 'V_extreme_in')
        self.connect('precurve_tip', 'precurveTip_in')
        self.connect('presweep_tip', 'presweepTip_in')

        self.connect('powercurve.V', 'V_in')
        self.connect('powercurve.P', 'P_in')
        self.connect('powercurve.Cp', 'Cp_in')
        self.connect('aep.AEP', 'AEP_in')
        self.connect('powercurve.rated_V', 'rated_V_in')
        self.connect('powercurve.rated_Omega', 'rated_Omega_in')
        self.connect('powercurve.rated_pitch', 'rated_pitch_in')
        self.connect('powercurve.rated_T', 'rated_T_in')
        self.connect('powercurve.rated_Q', 'rated_Q_in')


def Init_RotorAeropower_wRefBlade(rotor, blade):
    # === blade grid ===
    rotor['hubFraction']      = blade['config']['hubD']/2./blade['pf']['r'][-1] #0.025  # (Float): hub location as fraction of radius
    rotor['bladeLength']      = blade['ctrl_pts']['bladeLength'] #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    rotor['precone']          = blade['config']['cone_angle'] #2.5  # (Float, deg): precone angle
    rotor['tilt']             = blade['config']['tilt_angle'] #5.0  # (Float, deg): shaft tilt
    rotor['yaw']              = 0.0  # (Float, deg): yaw error
    rotor['nBlades']          = blade['config']['number_of_blades'] #3  # (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    rotor['r_max_chord']      = blade['ctrl_pts']['r_max_chord']  # 0.23577 #(Float): location of max chord on unit radius
    rotor['chord_in']         = np.array(blade['ctrl_pts']['chord_in']) # np.array([3.2612, 4.3254, 4.5709, 3.7355, 2.69923333, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in']         = np.array(blade['ctrl_pts']['theta_in']) # np.array([0.0, 13.2783, 12.30514836,  6.95106536,  2.72696309, -0.0878099]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in']      = np.array(blade['ctrl_pts']['precurve_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in']      = np.array(blade['ctrl_pts']['presweep_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['sparT_in']         = np.array(blade['ctrl_pts']['sparT_in']) # np.array([0.0, 0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in']           = np.array(blade['ctrl_pts']['teT_in']) # np.array([0.0, 0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # ------------------
    
    # === atmosphere ===
    rotor['rho']              = 1.225  # (Float, kg/m**3): density of air
    rotor['mu']               = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['hub_height']       = blade['config']['hub_height']  # (Float, m): hub height
    rotor['shearExp']         = 0.25  # (Float): shear exponent
    rotor['shape_parameter']  = 2.0
    rotor['turbine_class']    = TURBINE_CLASS[blade['config']['turbine_class'].upper()] #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['cdf_reference_height_wind_speed'] = blade['config']['hub_height']  # (Float, m): hub height
    # ----------------------
    
    # === control ===
    rotor['control_Vin']      = blade['config']['Vin'] #3.0  # (Float, m/s): cut-in wind speed
    rotor['control_Vout']     = blade['config']['Vout'] #25.0  # (Float, m/s): cut-out wind speed
    rotor['control_ratedPower']   = blade['config']['rating'] #5e6  # (Float, W): rated power
    rotor['control_minOmega'] = blade['config']['minOmega'] #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control_maxOmega'] = blade['config']['maxOmega'] #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control_maxTS']    = blade['config']['maxTS']
    rotor['control_tsr']      = blade['config']['tsr'] #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control_pitch']    = blade['config']['pitch'] #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType']   = DRIVETRAIN_TYPE[blade['config']['drivetrain'].upper()] #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    # ----------------------
    return rotor

if __name__ == '__main__':


    tt = time.time()

    # Turbine Ontology input
    fname_input  = "turbine_inputs/nrel5mw_mod_update.yaml"
    # fname_output = "turbine_inputs/nrel5mw_mod_out.yaml"
    fname_schema = "turbine_inputs/IEAontology_schema.yaml"
    
    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose = True
    refBlade.NINPUT  = 5
    refBlade.NPTS    = 50
    refBlade.spar_var = ['Spar_Cap_SS', 'Spar_Cap_PS']
    refBlade.te_var   = 'TE_reinforcement'
    refBlade.validate     = False
    refBlade.fname_schema = fname_schema
    
    blade = refBlade.initialize(fname_input)
    rotor = Problem()
    npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    npts_spline_power_curve = 2000  # (Int): number of points to use in fitting spline to power curve
    regulation_reg_II5 = True # calculate Region 2.5 pitch schedule, False will not maximize power in region 2.5
    regulation_reg_III = True # calculate Region 3 pitch schedule, False will return erroneous Thrust, Torque, and Moment for above rated
    
    rotor.root = RotorAeroPower(blade, npts_coarse_power_curve, npts_spline_power_curve, regulation_reg_II5, regulation_reg_III)
    
    #rotor.setup(check=False)
    rotor.setup()
    rotor = Init_RotorAeropower_wRefBlade(rotor, blade)

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
