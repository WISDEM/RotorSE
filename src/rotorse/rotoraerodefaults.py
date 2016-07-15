#!/usr/bin/env python
# encoding: utf-8
"""
aerodefaults.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""
from __future__ import print_function
import numpy as np
from math import pi, gamma
from openmdao.api import Component, Group
from ccblade import CCAirfoil, CCBlade as CCBlade_PY

from commonse.utilities import sind, cosd, smooth_abs, smooth_min, hstack, vstack, linspace_with_deriv
from rotoraero import GeometrySetupBase, AeroBase, DrivetrainLossesBase, CDFBase, \
    VarSpeedMachine, FixedSpeedMachine, RatedConditions, common_configure
from akima import Akima
from enum import Enum
import copy, time, os

# ---------------------
# Map Design Variables to Discretization
# ---------------------

class GeometrySpline(Component):
    def __init__(self, naero):
        super(GeometrySpline, self).__init__()
        self.add_param('r_af', shape=naero, units='m', desc='locations where airfoils are defined on unit radius')

        self.add_param('idx_cylinder', val=0, desc='location where cylinder section ends on unit radius')
        self.add_param('r_max_chord', shape=1, desc='position of max chord on unit radius')

        self.add_param('Rhub', shape=1, units='m', desc='blade hub radius')
        self.add_param('Rtip', shape=1, units='m', desc='blade tip radius')

        self.add_param('chord_sub', shape=4, units='m', desc='chord at control points')
        self.add_param('theta_sub', shape=4, units='deg', desc='twist at control points')

        self.add_output('r', shape=naero, units='m', desc='chord at airfoil locations')
        self.add_output('chord', shape=naero, units='m', desc='chord at airfoil locations')
        self.add_output('theta', shape=naero, units='deg', desc='twist at airfoil locations')
        self.add_output('precurve', shape=naero, units='m', desc='precurve at airfoil locations')
        # self.add_output('r_af_spacing', shape=naero-1)  # deprecated: not used anymore

        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['type'] = 'fd'
        
    def solve_nonlinear(self, params, unknowns, resids):
        chord_sub = params['chord_sub']
        theta_sub = params['theta_sub']
        r_max_chord = params['r_max_chord']

        nc = len(chord_sub)
        nt = len(theta_sub)
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        idxc = params['idx_cylinder']
        r_max_chord = Rhub + (Rtip-Rhub)*r_max_chord
        r_af = params['r_af']
        r_cylinder = Rhub + (Rtip-Rhub)*r_af[idxc]

        # chord parameterization
        rc_outer, drc_drcmax, drc_drtip = linspace_with_deriv(r_max_chord, Rtip, nc-1)
        r_chord = np.concatenate([[Rhub], rc_outer])
        drc_drcmax = np.concatenate([[0.0], drc_drcmax])
        drc_drtip = np.concatenate([[0.0], drc_drtip])
        drc_drhub = np.concatenate([[1.0], np.zeros(nc-1)])

        # theta parameterization
        r_theta, drt_drcyl, drt_drtip = linspace_with_deriv(r_cylinder, Rtip, nt)

        # spline
        chord_spline = Akima(r_chord, chord_sub)
        theta_spline = Akima(r_theta, theta_sub)

        r = Rhub + (Rtip-Rhub)*r_af
        unknowns['r'] = r
        chord, dchord_dr, dchord_drchord, dchord_dchordsub = chord_spline.interp(r)
        theta_outer, dthetaouter_dr, dthetaouter_drtheta, dthetaouter_dthetasub = theta_spline.interp(r[idxc:])
        unknowns['chord'] = chord

        theta_inner = theta_outer[0] * np.ones(idxc)
        unknowns['theta'] = np.concatenate([theta_inner, theta_outer])

        # unknowns['r_af_spacing'] = np.diff(r_af)

        unknowns['precurve'] = np.zeros_like(unknowns['chord'])  # TODO: for now I'm forcing this to zero, just for backwards compatibility

        # gradients (TODO: rethink these a bit or use Tapenade.)
        n = len(r_af)
        dr_draf = (Rtip-Rhub)*np.ones(n)
        dr_dRhub = 1.0 - r_af
        dr_dRtip = r_af
        # dr = hstack([np.diag(dr_draf), np.zeros((n, 1)), dr_dRhub, dr_dRtip, np.zeros((n, nc+nt))])

        dchord_draf = dchord_dr * dr_draf
        dchord_drmaxchord0 = np.dot(dchord_drchord, drc_drcmax)
        dchord_drmaxchord = dchord_drmaxchord0 * (Rtip-Rhub)
        dchord_drhub = np.dot(dchord_drchord, drc_drhub) + dchord_drmaxchord0*(1.0 - params['r_max_chord']) + dchord_dr*dr_dRhub
        dchord_drtip = np.dot(dchord_drchord, drc_drtip) + dchord_drmaxchord0*(params['r_max_chord']) + dchord_dr*dr_dRtip

        dthetaouter_dcyl = np.dot(dthetaouter_drtheta, drt_drcyl)
        dthetaouter_draf = dthetaouter_dr*dr_draf[idxc:]
        dthetaouter_drhub = dthetaouter_dr*dr_dRhub[idxc:]
        dthetaouter_drtip = dthetaouter_dr*dr_dRtip[idxc:] + np.dot(dthetaouter_drtheta, drt_drtip)

        dtheta_draf = np.concatenate([np.zeros(idxc), dthetaouter_draf])
        dtheta_drhub = np.concatenate([dthetaouter_drhub[0]*np.ones(idxc), dthetaouter_drhub])
        dtheta_drtip = np.concatenate([dthetaouter_drtip[0]*np.ones(idxc), dthetaouter_drtip])
        sub = dthetaouter_dthetasub[0, :]
        dtheta_dthetasub = vstack([np.dot(np.ones((idxc, 1)), sub[np.newaxis, :]), dthetaouter_dthetasub])

        dtheta_draf = np.diag(dtheta_draf)
        dtheta_dcyl = np.concatenate([dthetaouter_dcyl[0]*np.ones(idxc), dthetaouter_dcyl])
        dtheta_draf[idxc:, idxc] += dthetaouter_dcyl*(Rtip-Rhub)
        dtheta_drhub += dtheta_dcyl*(1.0 - r_af[idxc])
        dtheta_drtip += dtheta_dcyl*r_af[idxc]

        drafs_dr = np.zeros((n-1, n))
        for i in range(n-1):
            drafs_dr[i, i] = -1.0
            drafs_dr[i, i+1] = 1.0

        J = {}
        J['r', 'r_af'] = np.diag(dr_draf)
        J['r', 'Rhub'] = dr_dRhub
        J['r', 'Rtip'] = dr_dRtip
        J['chord', 'r_af'] = np.diag(dchord_draf)
        J['chord', 'r_max_chord'] = dchord_drmaxchord
        J['chord', 'Rhub'] = dchord_drhub
        J['chord', 'Rtip'] = dchord_drtip
        J['chord', 'chord_sub'] =dchord_dchordsub
        J['theta', 'r_af'] = dtheta_draf
        J['theta', 'Rhub'] =dtheta_drhub
        J['theta', 'Rtip'] =dtheta_drtip
        J['theta', 'theta_sub'] =dtheta_dthetasub
        # J['r_af_spacing', 'r_af'] = drafs_dr

        self.J = J

    def linearize(self, params, unknowns, resids):

        return self.J

# ---------------------
# Default Implementations of Base Classes
# ---------------------

class CCBladeGeometry(GeometrySetupBase):
    def __init__(self):
        super(CCBladeGeometry, self).__init__()
        self.add_param('Rtip', shape=1, units='m', desc='tip radius')
        self.add_param('precurveTip', val=0.0, units='m', desc='tip radius')
        self.add_param('precone', val=0.0, desc='precone angle', units='deg')
        self.add_output('diameter', shape=1, units='m')

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['R'] = params['Rtip']*cosd(params['precone']) + params['precurveTip']*sind(params['precone'])
        unknowns['diameter'] = unknowns['R'] * 2.0

    def linearize(self, params, unknowns, resids):

        J_sub = np.array([[cosd(params['precone']), sind(params['precone']),
            (-params['Rtip']*sind(params['precone']) + params['precurveTip']*sind(params['precone']))*pi/180.0]])
        J = {}
        J['R', 'Rtip'] = J_sub[0][0]
        J['R', 'precurveTip'] = J_sub[0][1]
        J['R', 'precone'] = J_sub[0][2]
        J['diameter', 'Rtip'] = 2.0*J_sub[0][0]
        J['diameter', 'precurveTip'] = 2.0*J_sub[0][1]
        J['diameter', 'precone'] = 2.0*J_sub[0][2]

        return J

class CCBladeAirfoils(Component):
    def __init__(self, naero, nstr, num_airfoils, airfoils_dof):
        super(CCBladeAirfoils, self).__init__()
        self.add_param('airfoil_parameterization', val=np.zeros((num_airfoils, airfoils_dof)))
        self.add_param('afOptions', val={}, pass_by_obj=True)
        self.add_param('airfoil_types', shape=naero, desc='names of airfoil file', pass_by_obj=True)
        self.add_param('af_idx', val=np.zeros(naero), pass_by_obj=True)
        self.add_param('af_str_idx', val=np.zeros(nstr), pass_by_obj=True)
        self.add_output('af', shape=naero, desc='airfoils along aero sections', pass_by_obj=True)
        self.add_output('af_str', shape=nstr, desc='airfoils along structural section', pass_by_obj=True)
        self.add_output('dummy', shape=1)

        self.naero = naero
        self.nstr = nstr
        self.num_airfoils = num_airfoils
        self.airfoil_dof = airfoils_dof

    def solve_nonlinear(self, params, unknowns, resids):
        airfoil_types = params['airfoil_types']
        af = [0]*self.naero
        af_str = [0]*self.nstr
        afinit = CCAirfoil.initFromAerodynFile
        if not np.any(params['airfoil_parameterization']):
            for i in range(self.naero):
                af[i] = afinit(airfoil_types[params['af_idx'][i]])
            for i in range(self.nstr):
                af_str[i] = afinit(airfoil_types[params['af_str_idx'][i]])
        else:
            # change = np.zeros(self.num_airfoils)
            # for j in range(self.num_airfoils):
            #     index = np.where(af_idx >= j+2)[0][0]
            #     change[j] = max(abs(params['airfoil_types'][index].Saf - self.airfoil_parameterization[j]))

            from airfoilprep import AirfoilAnalysis
            afanalysis = AirfoilAnalysis(None, params['afOptions'])

            for i in range(len(airfoil_types)):
                if i < 2:
                    airfoil_types[i] = afinit(airfoil_types[i])
                else:
                    airfoil_types[i] = CCAirfoil.initFromAirfoilAnalysis(params['airfoil_parameterization'][i-2], afanalysis)
            for i in range(self.naero):
                af[i] = airfoil_types[params['af_idx'][i]]
            for i in range(self.nstr):
                af_str[i] = airfoil_types[params['af_str_idx'][i]]

        unknowns['af'] = copy.deepcopy(af)
        unknowns['af_str'] = copy.deepcopy(af_str)

    def linearize(self, params, unknowns, resids):
        J = {}
        J['dummy', 'airfoil_parameterization'] = np.zeros((1, self.num_airfoils*self.airfoil_dof))
        return J

class AirfoilSpline(Component):
    def __init__(self, n, nstr, num_airfoils, airfoils_dof):
        super(AirfoilSpline, self).__init__()
        self.add_param('airfoil_parameterization', val=np.zeros((num_airfoils, airfoils_dof)))
        self.add_param('af_idx', val=np.zeros(n), pass_by_obj=True)
        self.add_param('af_str_idx', val=np.zeros(nstr), pass_by_obj=True)
        self.add_param('idx_cylinder_str', val=1, pass_by_obj=True)
        self.add_param('idx_cylinder_aero', val=1, pass_by_obj=True)
        self.add_output('airfoil_parameterization_full', val=np.zeros((n, airfoils_dof)))
        self.add_output('airfoil_str_parameterization_full', val=np.zeros((nstr, airfoils_dof)))
        self.n = n
        self.nstr = nstr
        self.num_airfoils = num_airfoils
        self.airfoil_dof = airfoils_dof

    def solve_nonlinear(self, params, unknowns, resids):
        self.airfoil_parameterization = params['airfoil_parameterization']
        n = self.n
        nstr = self.nstr
        CST = np.zeros((n,self.airfoil_dof))
        af_idx = params['af_idx']
        self.daf_daf = np.zeros((n*self.airfoil_dof,self.num_airfoils*self.airfoil_dof))
        self.daf_daf_str = np.zeros((nstr*self.airfoil_dof,self.num_airfoils*self.airfoil_dof))
        aero_idx = params['idx_cylinder_aero']
        for i in range(n-aero_idx):
            for j in range(self.airfoil_dof):
                CST[i+aero_idx][j] = self.airfoil_parameterization[af_idx[i+aero_idx]-2][j]
            self.daf_daf[np.ix_(range((i+aero_idx)*self.airfoil_dof, (i+aero_idx)*self.airfoil_dof+self.airfoil_dof), range((af_idx[i+aero_idx]-2)*self.airfoil_dof,((af_idx[i+aero_idx]-2)*self.airfoil_dof)+self.airfoil_dof))] += np.diag(np.ones(self.airfoil_dof))
        unknowns['airfoil_parameterization_full'] = CST

        airfoil_types_str = np.zeros((8,self.airfoil_dof))
        for z in range(self.num_airfoils):
            airfoil_types_str[z+2, :] = self.airfoil_parameterization[z]
        pro_str = [0]*nstr
        af_str_idx = params['af_str_idx']
        for i in range(nstr):
            pro_str[i] = airfoil_types_str[af_str_idx[i]]
        str_idx = params['idx_cylinder_str']
        for i in range(nstr-str_idx):
            self.daf_daf_str[np.ix_(range((i+str_idx)*self.airfoil_dof, (i+str_idx)*self.airfoil_dof+self.airfoil_dof), range((af_str_idx[i+str_idx]-2)*self.airfoil_dof,((af_str_idx[i+str_idx]-2)*self.airfoil_dof)+self.airfoil_dof))] += np.diag(np.ones(self.airfoil_dof))
        unknowns['airfoil_str_parameterization_full'] = np.asarray(pro_str)

    def linearize(self, params, unknowns, resids):
        J = {}
        J['airfoil_parameterization_full', 'airfoil_parameterization'] = self.daf_daf
        J['airfoil_str_parameterization_full', 'airfoil_parameterization'] = self.daf_daf_str
        return J

class CCBlade(AeroBase):
    def __init__(self, run_case, n, n2, num_airfoils, airfoils_dof):
        super(CCBlade, self).__init__(n, n2)
        """blade element momentum code"""

        # (potential) variables
        self.add_param('r', shape=n, units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_param('chord', shape=n, units='m', desc='chord length at each section')
        self.add_param('theta', shape=n,  units='deg', desc='twist angle at each section (positive decreases angle of attack)')
        self.add_param('Rhub', shape=1, units='m', desc='hub radius')
        self.add_param('Rtip', shape=1, units='m', desc='tip radius')
        self.add_param('hubHt', val=np.zeros(1), units='m', desc='hub height')
        self.add_param('precone', shape=1, desc='precone angle', units='deg')
        self.add_param('tilt', shape=1, desc='shaft tilt', units='deg')
        self.add_param('yaw', shape=1, desc='yaw error', units='deg')

        # TODO: I've not hooked up the gradients for these ones yet.
        self.add_param('precurve', shape=n, units='m', desc='precurve at each section')
        self.add_param('precurveTip', val=0.0, units='m', desc='precurve at tip')

        # parameters
        self.add_param('airfoil_parameterization', val=np.zeros((n, airfoils_dof)))
        self.add_param('afOptions', val={}, pass_by_obj=True)
        self.add_param('af', shape=n, desc='names of airfoil file', pass_by_obj=True)
        self.add_param('B', val=3, desc='number of blades', pass_by_obj=True)
        self.add_param('rho', val=1.225, units='kg/m**3', desc='density of air')
        self.add_param('mu', val=1.81206e-5, units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_param('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True)
        self.add_param('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True)
        self.add_param('tiploss', val=True, desc='include Prandtl tip loss model', pass_by_obj=True)
        self.add_param('hubloss', val=True, desc='include Prandtl hub loss model', pass_by_obj=True)
        self.add_param('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)', pass_by_obj=True)
        self.add_param('usecd', val=True, desc='use drag coefficient in computing induction factors', pass_by_obj=True)

        self.add_param('run_case', val=Enum('power', 'loads'), pass_by_obj=True)


        # inputs
        # self.add_param('V_load', shape=1, units='m/s', desc='hub height wind speed')
        # self.add_param('Omega_load', shape=1, units='rpm', desc='rotor rotation speed')
        # self.add_param('pitch_load', shape=1, units='deg', desc='blade pitch setting')
        # self.add_param('azimuth_load', shape=1, units='deg', desc='blade azimuthal location')
        #
        # # outputs
        # self.add_output('loads:r', shape=n+2, units='m', desc='radial positions along blade going toward tip')
        # self.add_output('loads:Px', shape=n+2, units='N/m', desc='distributed loads in blade-aligned x-direction')
        # self.add_output('loads:Py', shape=n+2, units='N/m', desc='distributed loads in blade-aligned y-direction')
        # self.add_output('loads:Pz', shape=n+2, units='N/m', desc='distributed loads in blade-aligned z-direction')
        #
        # # corresponding setting for loads
        # self.add_output('loads:V', shape=1, units='m/s', desc='hub height wind speed')
        # self.add_output('loads:Omega', shape=1, units='rpm', desc='rotor rotation speed')
        # self.add_output('loads:pitch', shape=1, units='deg', desc='pitch angle')
        # self.add_output('loads:azimuth', shape=1, units='deg', desc='azimuthal angle')

        self.run_case = run_case
        self.num_airfoils = num_airfoils
        self.airfoils_dof = airfoils_dof
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        self.r = params['r']
        self.chord = params['chord']
        self.theta = params['theta']
        self.Rhub = params['Rhub']
        self.Rtip = params['Rtip']
        self.hubHt = params['hubHt']
        self.precone = params['precone']
        self.tilt = params['tilt']
        self.yaw = params['yaw']
        self.precurve = params['precurve']
        self.precurveTip = params['precurveTip']
        self.af = params['af'] #airfoil_files']
        self.airfoil_parameterization = params['airfoil_parameterization']
        self.afOptions = params['afOptions']
        self.B = params['B']
        self.rho = params['rho']
        self.mu = params['mu']
        self.shearExp = params['shearExp']
        self.nSector = params['nSector']
        self.tiploss = params['tiploss']
        self.hubloss = params['hubloss']
        self.wakerotation = params['wakerotation']
        self.usecd = params['usecd']
        self.Uhub = params['Uhub']
        self.Omega = params['Omega']
        self.pitch = params['pitch']
        self.V_load = params['V_load']
        self.Omega_load = params['Omega_load']
        self.pitch_load = params['pitch_load']
        self.azimuth_load = params['azimuth_load']
        self.parallel = False
        try:
            computeGradient = self.afOptions['GradientOptions']['ComputeGradient']
        except:
            computeGradient = False
        self.ccblade = CCBlade_PY(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip, self.B,
                    self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubHt,
                    self.nSector, self.precurve, self.precurveTip, tiploss=self.tiploss, hubloss=self.hubloss,
                    wakerotation=self.wakerotation, usecd=self.usecd, derivatives=computeGradient)
        # try:
        if self.run_case == 'power':
            # power, thrust, torque
            if computeGradient:
                self.P, self.T, self.Q, self.dP, self.dT, self.dQ = self.ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=False)
            else:
                self.P, self.T, self.Q = self.ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=False)
            unknowns['T'] = self.T
            unknowns['Q'] = self.Q
            unknowns['P'] = self.P
        elif self.run_case == 'loads':
            # distributed loads
            if self.parallel:
                ccblade_loads = self.ccblade.distributedAeroLoadsParallel
            else:
                ccblade_loads = self.ccblade.distributedAeroLoads
            if computeGradient:
                Np, Tp, self.dNp, self.dTp = ccblade_loads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)
            else:
                Np, Tp = ccblade_loads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)
            # concatenate loads at root/tip
            unknowns['loads:r'] = np.concatenate([[self.Rhub], self.r, [self.Rtip]])
            Np = np.concatenate([[0.0], Np, [0.0]])
            Tp = np.concatenate([[0.0], Tp, [0.0]])

            # conform to blade-aligned coordinate system
            unknowns['loads:Px'] = Np
            unknowns['loads:Py'] = -Tp
            unknowns['loads:Pz'] = 0*Np

            # return other outputs needed
            unknowns['loads:V'] = self.V_load
            unknowns['loads:Omega'] = self.Omega_load
            unknowns['loads:pitch'] = self.pitch_load
            unknowns['loads:azimuth'] = self.azimuth_load

    def linearize(self, params, unknowns, resids):
        if not self.afOptions['GradientOptions']['ComputeGradient']:
            self.ccblade = CCBlade_PY(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip, self.B,
                self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubHt,
                self.nSector, self.precurve, self.precurveTip, tiploss=self.tiploss, hubloss=self.hubloss,
                wakerotation=self.wakerotation, usecd=self.usecd, derivatives=True)
            if self.run_case == 'power':
                # power, thrust, torque
                self.P, self.T, self.Q, self.dP, self.dT, self.dQ = self.ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=False)

            elif self.run_case == 'loads':
                # distributed loads
                Np, Tp, self.dNp, self.dTp = self.ccblade.distributedAeroLoads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)

        J = {}
        if self.run_case == 'power':

            dP = self.dP
            dT = self.dT
            dQ = self.dQ

            J['P', 'precone'] = dP['dprecone']
            J['P', 'tilt'] = dP['dtilt']
            J['P', 'hubHt'] = dP['dhubHt']
            J['P', 'Rhub'] = dP['dRhub']
            J['P', 'Rtip'] = dP['dRtip']
            J['P', 'yaw'] = dP['dyaw']
            J['P', 'Uhub'] = dP['dUinf']
            J['P', 'Omega'] = dP['dOmega']
            J['P', 'pitch'] =  dP['dpitch']
            J['P', 'r'] = dP['dr']
            J['P', 'chord'] = dP['dchord']
            J['P', 'theta'] = dP['dtheta']
            J['P', 'precurve'] = dP['dprecurve']
            J['P', 'precurveTip'] = dP['dprecurveTip']

            J['T', 'precone'] = dT['dprecone']
            J['T', 'tilt'] = dT['dtilt']
            J['T', 'hubHt'] = dT['dhubHt']
            J['T', 'Rhub'] = dT['dRhub']
            J['T', 'Rtip'] = dT['dRtip']
            J['T', 'yaw'] = dT['dyaw']
            J['T', 'Uhub'] = dT['dUinf']
            J['T', 'Omega'] = dT['dOmega']
            J['T', 'pitch'] =  dT['dpitch']
            J['T', 'r'] = dT['dr']
            J['T', 'chord'] = dT['dchord']
            J['T', 'theta'] = dT['dtheta']
            J['T', 'precurve'] = dT['dprecurve']
            J['T', 'precurveTip'] = dT['dprecurveTip']

            J['Q', 'precone'] = dQ['dprecone']
            J['Q', 'tilt'] = dQ['dtilt']
            J['Q', 'hubHt'] = dQ['dhubHt']
            J['Q', 'Rhub'] = dQ['dRhub']
            J['Q', 'Rtip'] = dQ['dRtip']
            J['Q', 'yaw'] = dQ['dyaw']
            J['Q', 'Uhub'] = dQ['dUinf']
            J['Q', 'Omega'] = dQ['dOmega']
            J['Q', 'pitch'] =  dQ['dpitch']
            J['Q', 'r'] = dQ['dr']
            J['Q', 'chord'] = dQ['dchord']
            J['Q', 'theta'] = dQ['dtheta']
            J['Q', 'precurve'] = dQ['dprecurve']
            J['Q', 'precurveTip'] = dQ['dprecurveTip']


            J['P', 'airfoil_parameterization'] = dP['dSaf']
            J['T', 'airfoil_parameterization'] = dT['dSaf']
            J['Q', 'airfoil_parameterization'] = dQ['dSaf']

        elif self.run_case == 'loads':

            dNp = self.dNp
            dTp = self.dTp
            n = len(self.r)

            dr_dr = vstack([np.zeros(n), np.eye(n), np.zeros(n)])
            dr_dRhub = np.zeros(n+2)
            dr_dRtip = np.zeros(n+2)
            dr_dRhub[0] = 1.0
            dr_dRtip[-1] = 1.0

            dV = np.zeros(4*n+10)
            dV[3*n+6] = 1.0
            dOmega = np.zeros(4*n+10)
            dOmega[3*n+7] = 1.0
            dpitch = np.zeros(4*n+10)
            dpitch[3*n+8] = 1.0
            dazimuth = np.zeros(4*n+10)
            dazimuth[3*n+9] = 1.0

            zero = np.zeros(17)
            J['loads:r', 'r'] = dr_dr
            J['loads:r', 'Rhub'] = dr_dRhub
            J['loads:r', 'Rtip'] = dr_dRtip
            J['loads:Px', 'r'] = np.vstack([zero, dNp['dr'], zero])
            J['loads:Px', 'chord'] = np.vstack([zero, dNp['dchord'], zero])
            J['loads:Px', 'theta'] = np.vstack([zero, dNp['dtheta'], zero])
            J['loads:Px', 'Rhub'] = np.concatenate([[0.0], np.squeeze(dNp['dRhub']), [0.0]])
            J['loads:Px', 'Rtip'] = np.concatenate([[0.0], np.squeeze(dNp['dRtip']), [0.0]])
            J['loads:Px', 'hubHt'] = np.concatenate([[0.0], np.squeeze(dNp['dhubHt']), [0.0]])
            J['loads:Px', 'precone'] = np.concatenate([[0.0], np.squeeze(dNp['dprecone']), [0.0]])
            J['loads:Px', 'tilt'] = np.concatenate([[0.0], np.squeeze(dNp['dtilt']), [0.0]])
            J['loads:Px', 'yaw'] = np.concatenate([[0.0], np.squeeze(dNp['dyaw']), [0.0]])
            J['loads:Px', 'V_load'] = np.concatenate([[0.0], np.squeeze(dNp['dUinf']), [0.0]])
            J['loads:Px', 'Omega_load'] = np.concatenate([[0.0], np.squeeze(dNp['dOmega']), [0.0]])
            J['loads:Px', 'pitch_load'] = np.concatenate([[0.0], np.squeeze(dNp['dpitch']), [0.0]])
            J['loads:Px', 'azimuth_load'] = np.concatenate([[0.0], np.squeeze(dNp['dazimuth']), [0.0]])
            J['loads:Px', 'precurve'] = np.vstack([zero, dNp['dprecurve'], zero])
            J['loads:Py', 'r'] = np.vstack([zero, -dTp['dr'], zero])
            J['loads:Py', 'chord'] = np.vstack([zero, -dTp['dchord'], zero])
            J['loads:Py', 'theta'] = np.vstack([zero, -dTp['dtheta'], zero])
            J['loads:Py', 'Rhub'] = np.concatenate([[0.0], -np.squeeze(dTp['dRhub']), [0.0]])
            J['loads:Py', 'Rtip'] = np.concatenate([[0.0], -np.squeeze(dTp['dRtip']), [0.0]])
            J['loads:Py', 'hubHt'] = np.concatenate([[0.0], -np.squeeze(dTp['dhubHt']), [0.0]])
            J['loads:Py', 'precone'] = np.concatenate([[0.0], -np.squeeze(dTp['dprecone']), [0.0]])
            J['loads:Py', 'tilt'] = np.concatenate([[0.0], -np.squeeze(dTp['dtilt']), [0.0]])
            J['loads:Py', 'yaw'] = np.concatenate([[0.0], -np.squeeze(dTp['dyaw']), [0.0]])
            J['loads:Py', 'V_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dUinf']), [0.0]])
            J['loads:Py', 'Omega_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dOmega']), [0.0]])
            J['loads:Py', 'pitch_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dpitch']), [0.0]])
            J['loads:Py', 'azimuth_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dazimuth']), [0.0]])
            J['loads:Py', 'precurve'] = np.vstack([zero, -dTp['dprecurve'], zero])
            J['loads:V', 'V_load'] = 1.0
            J['loads:Omega', 'Omega_load'] = 1.0
            J['loads:pitch', 'pitch_load'] = 1.0
            J['loads:azimuth', 'azimuth_load'] = 1.0

            zero_Saf = np.zeros((n*self.airfoils_dof))
            J['loads:Px', 'airfoil_parameterization'] = np.vstack([zero_Saf, dNp['dSaf'], zero_Saf])
            J['loads:Py', 'airfoil_parameterization'] = np.vstack([zero_Saf, -dTp['dSaf'], zero_Saf])

        return J



class CSMDrivetrain(DrivetrainLossesBase):
    def __init__(self, n):
        super(CSMDrivetrain, self).__init__(n)
        """drivetrain losses from NREL cost and scaling model"""

        self.add_param('drivetrainType', val=Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):
        drivetrainType = params['drivetrainType']
        aeroPower = params['aeroPower']
        aeroTorque = params['aeroTorque']
        ratedPower = params['ratedPower']

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
        dPbar_dPr = -dPbar_dPbar1*dPbar1_dPbar0*aeroPower/ratedPower**2

        deff_dPa = dPbar_dPa*(constant/Pbar**2 - quadratic)
        deff_dPr = dPbar_dPr*(constant/Pbar**2 - quadratic)

        dP_dPa = eff + aeroPower*deff_dPa
        dP_dPr = aeroPower*deff_dPr
        
        J = {}
        J['power', 'aeroPower'] = np.diag(dP_dPa)
        J['power', 'ratedPower'] = dP_dPr
        self.J = J


    def linearize(self, params, unknowns, resids):

        return self.J




class WeibullCDF(CDFBase):
    def __init__(self, nspline):
        super(WeibullCDF, self).__init__(nspline)
        """Weibull cumulative distribution function"""

        self.add_param('A', shape=1, desc='scale factor')
        self.add_param('k', shape=1, desc='shape or form factor')

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['F'] = 1.0 - np.exp(-(params['x']/params['A'])**params['k'])

    def linearize(self, params, unknowns, resids):

        x = params['x']
        A = params['A']
        k = params['k']
        J = {}
        J['F', 'x'] = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        return J


class WeibullWithMeanCDF(CDFBase):
    def __init__(self, nspline):
        super(WeibullWithMeanCDF, self).__init__(nspline)
        """Weibull cumulative distribution function"""

        self.add_param('xbar', shape=1, desc='mean value of distribution')
        self.add_param('k', shape=1, desc='shape or form factor')

    def solve_nonlinear(self, params, unknowns, resids):
        A = params['xbar'] / gamma(1.0 + 1.0/params['k'])

        unknowns['F'] = 1.0 - np.exp(-(params['x']/A)**params['k'])

    def linearize(self, params, unknowns, resids):

        x = params['x']
        k = params['k']
        A = params['xbar'] / gamma(1.0 + 1.0/k)
        dx = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        dxbar = -np.exp(-(x/A)**k)*(x/A)**(k-1)*k*x/A**2/gamma(1.0 + 1.0/k)

        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar
        J['F', 'k'] = np.exp(-(x/A)**k)*(x/A)**k*np.log(x/A) # TODO Check derivative

        return J


class RayleighCDF(CDFBase):
    def __init__(self, nspline):
        super(RayleighCDF,  self).__init__(nspline)

        # variables
        self.add_param('xbar', shape=1, units='m/s', desc='reference wind speed (usually at hub height)')

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['F'] = 1.0 - np.exp(-pi/4.0*(params['x']/params['xbar'])**2)

    def linearize(self, params, unknowns, resids):

        x = params['x']
        xbar = params['xbar']
        dx = np.diag(np.exp(-pi/4.0*(x/xbar)**2)*pi*x/(2.0*xbar**2))
        dxbar = -np.exp(-pi/4.0*(x/xbar)**2)*pi*x**2/(2.0*xbar**3)
        
        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar
        
        return J

def common_io_with_ccblade(group, varspeed, varpitch, cdf_type):

    regulated = varspeed or varpitch

    # add inputs
    group.add_param('r_af', units='m', desc='locations where airfoils are defined on unit radius')
    group.add_param('r_max_chord')
    group.add_param('chord_sub', units='m', desc='chord at control points')
    group.add_param('theta_sub', units='deg', desc='twist at control points')
    group.add_param('Rhub', units='m', desc='hub radius')
    group.add_param('Rtip', units='m', desc='tip radius')
    group.add_param('hubHt', units='m')
    group.add_param('precone', desc='precone angle', units='deg')
    group.add_param('tilt', val=0.0, desc='shaft tilt', units='deg')
    group.add_param('yaw', val=0.0, desc='yaw error', units='deg')
    group.add_param('airfoil_files', desc='names of airfoil file')
    group.add_param('idx_cylinder', desc='location where cylinder section ends on unit radius')
    group.add_param('B', val=3, desc='number of blades')
    group.add_param('rho', val=1.225, units='kg/m**3', desc='density of air')
    group.add_param('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air')
    group.add_param('shearExp', val=0.2, desc='shear exponent')
    group.add_param('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
    group.add_param('tiploss', val=True, desc='include Prandtl tip loss model')
    group.add_param('hubloss', val=True, desc='include Prandtl hub loss model')
    group.add_param('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
    group.add_param('usecd', val=True, desc='use drag coefficient in computing induction factors')
    group.add_param('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at')
    group.add_param('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve')
    group.add_param('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)')

    if varspeed:
        group.add_param('control:Vin', units='m/s', desc='cut-in wind speed')
        group.add_param('control:Vout', units='m/s', desc='cut-out wind speed')
        group.add_param('control:ratedPower', units='W', desc='rated power')
        group.add_param('control:minOmega', units='rpm', desc='minimum allowed rotor rotation speed')
        group.add_param('control:maxOmega', units='rpm', desc='maximum allowed rotor rotation speed')
        group.add_param('control:tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        group.add_param('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
    else:
        group.add_param('control:Vin', units='m/s', desc='cut-in wind speed')
        group.add_param('control:Vout', units='m/s', desc='cut-out wind speed')
        group.add_param('control:ratedPower', units='W', desc='rated power')
        group.add_param('control:Omega', units='rpm', desc='fixed rotor rotation speed')
        group.add_param('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        group.add_param('control:npts', val=20, desc='number of points to evalute aero code to generate power curve')

    group.add_param('drivetrainType', val=Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive')))
    group.add_param('cdf_mean_wind_speed', units='m/s', desc='mean wind speed of site cumulative distribution function')

    if cdf_type == 'weibull':
        group.add_param('weibull_shape_factor', desc='(shape factor of weibull distribution)')

    # outputs
    group.add_output('AEP', units='kW*h', desc='annual energy production')
    group.add_output('V', units='m/s', desc='wind speeds (power curve)')
    group.add_output('P', units='W', desc='power (power curve)')
    group.add_output('diameter', units='m')
    if regulated:
        group.add_output('ratedConditions:V', units='m/s', desc='rated wind speed')
        group.add_output('ratedConditions:Omega', units='rpm', desc='rotor rotation speed at rated')
        group.add_output('ratedConditions:pitch', units='deg', desc='pitch setting at rated')
        group.add_output('ratedConditions:T', units='N', desc='rotor aerodynamic thrust at rated')
        group.add_output('ratedConditions:Q', units='N*m', desc='rotor aerodynamic torque at rated')



def common_configure_with_ccblade(group, varspeed, varpitch, cdf_type):
    common_configure(group, varspeed, varpitch)

    # put in parameterization for CCBlade
    group.add('spline', GeometrySpline())
    group.replace('geom', CCBladeGeometry())
    group.replace('analysis', CCBlade())
    group.replace('dt', CSMDrivetrain())
    if cdf_type == 'rayleigh':
        group.replace('cdf', RayleighCDF())
    elif cdf_type == 'weibull':
        group.replace('cdf', WeibullWithMeanCDF())


    # add spline to workflow
    group.driver.workflow.add('spline')

    # connections to spline
    group.connect('r_af', 'spline.r_af')
    group.connect('r_max_chord', 'spline.r_max_chord')
    group.connect('chord_sub', 'spline.chord_sub')
    group.connect('theta_sub', 'spline.theta_sub')
    group.connect('idx_cylinder', 'spline.idx_cylinder')
    group.connect('Rhub', 'spline.Rhub')
    group.connect('Rtip', 'spline.Rtip')

    # connections to geom
    group.connect('Rtip', 'geom.Rtip')
    group.connect('precone', 'geom.precone')

    # connections to analysis
    group.connect('spline.r', 'analysis.r')
    group.connect('spline.chord', 'analysis.chord')
    group.connect('spline.theta', 'analysis.theta')
    group.connect('spline.precurve', 'analysis.precurve')
    group.connect('Rhub', 'analysis.Rhub')
    group.connect('Rtip', 'analysis.Rtip')
    group.connect('hubHt', 'analysis.hubHt')
    group.connect('precone', 'analysis.precone')
    group.connect('tilt', 'analysis.tilt')
    group.connect('yaw', 'analysis.yaw')
    group.connect('airfoil_files', 'analysis.airfoil_files')
    group.connect('B', 'analysis.B')
    group.connect('rho', 'analysis.rho')
    group.connect('mu', 'analysis.mu')
    group.connect('shearExp', 'analysis.shearExp')
    group.connect('nSector', 'analysis.nSector')
    group.connect('tiploss', 'analysis.tiploss')
    group.connect('hubloss', 'analysis.hubloss')
    group.connect('wakerotation', 'analysis.wakerotation')
    group.connect('usecd', 'analysis.usecd')

    # connections to dt
    group.connect('drivetrainType', 'dt.drivetrainType')
    group.dt.missing_deriv_policy = 'assume_zero'  # TODO: openmdao bug remove later

    # connnections to cdf
    group.connect('cdf_mean_wind_speed', 'cdf.xbar')
    if cdf_type == 'weibull':
        group.connect('weibull_shape_factor', 'cdf.k')

class RotorAeroVSVPWithCCBlade(Group):
    def __init__(self, cdf_type='weibull'):
        super(RotorAeroVSVPWithCCBlade, self).__init__()
        self.cdf_type = cdf_type

    def configure(self):
        varspeed = True
        varpitch = True
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)


class RotorAeroVSFPWithCCBlade(Group):

    def __init__(self, cdf_type='weibull'):
        self.cdf_type = cdf_type
        super(RotorAeroVSFPWithCCBlade, self).__init__()

    def configure(self):
        varspeed = True
        varpitch = False
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)



class RotorAeroFSVPWithCCBlade(Group):

    def __init__(self, cdf_type='weibull'):
        self.cdf_type = cdf_type
        super(RotorAeroFSVPWithCCBlade, self).__init__()

    def configure(self):
        varspeed = False
        varpitch = True
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)



class RotorAeroFSFPWithCCBlade(Group):

    def __init__(self, cdf_type='weibull'):
        self.cdf_type = cdf_type
        super(RotorAeroFSFPWithCCBlade, self).__init__()

    def configure(self):
        varspeed = False
        varpitch = False
        common_io_with_ccblade(self, varspeed, varpitch, self.cdf_type)
        common_configure_with_ccblade(self, varspeed, varpitch, self.cdf_type)



if __name__ == '__main__':

    optimize = True

    import os

    # --- instantiate rotor ----
    cdf_type = 'weibull'
    rotor = RotorAeroVSVPWithCCBlade(cdf_type)
    # --------------------------------

    # --- rotor geometry ------------
    rotor.r_max_chord = 0.23577  # (Float): location of second control point (generally also max chord)
    rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]  # (Array, m): chord at control points
    rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]  # (Array, deg): twist at control points
    rotor.Rhub = 1.5  # (Float, m): hub radius
    rotor.Rtip = 63.0  # (Float, m): tip radius
    rotor.precone = 2.5  # (Float, deg): precone angle
    rotor.tilt = -5.0  # (Float, deg): shaft tilt
    rotor.yaw = 0.0  # (Float, deg): yaw error
    rotor.B = 3  # (Int): number of blades
    # -------------------------------------

    # --- airfoils ------------
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = basepath + os.path.sep + 'Cylinder1.dat'
    airfoil_types[1] = basepath + os.path.sep + 'Cylinder2.dat'
    airfoil_types[2] = basepath + os.path.sep + 'DU40_A17.dat'
    airfoil_types[3] = basepath + os.path.sep + 'DU35_A17.dat'
    airfoil_types[4] = basepath + os.path.sep + 'DU30_A17.dat'
    airfoil_types[5] = basepath + os.path.sep + 'DU25_A17.dat'
    airfoil_types[6] = basepath + os.path.sep + 'DU21_A17.dat'
    airfoil_types[7] = basepath + os.path.sep + 'NACA64_A17.dat'

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]

    rotor.airfoil_files = af  # (List): paths to AeroDyn-style airfoil files
    rotor.r_af = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
        0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943,
        0.93333333, 0.97777724])    # (Array, m): locations where airfoils are defined on unit radius
    rotor.idx_cylinder = 3  # (Int): index in r_af where cylinder section ends
    # -------------------------------------

    # --- site characteristics --------
    rotor.rho = 1.225  # (Float, kg/m**3): density of air
    rotor.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor.shearExp = 0.2  # (Float): shear exponent
    rotor.hubHt = 80.0  # (Float, m)
    rotor.cdf_mean_wind_speed = 6.0  # (Float, m/s): mean wind speed of site cumulative distribution function
    rotor.weibull_shape_factor = 2.0  # (Float): shape factor of weibull distribution
    # -------------------------------------


    # --- control settings ------------
    rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed
    rotor.control.Vout = 25.0  # (Float, m/s): cut-out wind speed
    rotor.control.ratedPower = 5e6  # (Float, W): rated power
    rotor.control.pitch = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor.control.minOmega = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor.control.maxOmega = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor.control.tsr = 7.55  # **dv** (Float): tip-speed ratio in Region 2 (should be optimized externally)
    # -------------------------------------

    # --- drivetrain model for efficiency --------
    rotor.drivetrainType = 'geared'
    # -------------------------------------


    # --- analysis options ------------
    rotor.nSector = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor.npts_coarse_power_curve = 20  # (Int): number of points to evaluate aero analysis at
    rotor.npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
    rotor.AEP_loss_factor = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor.tiploss = True  # (Bool): include Prandtl tip loss model
    rotor.hubloss = True  # (Bool): include Prandtl hub loss model
    rotor.wakerotation = True  # (Bool): include effect of wake rotation (i.e., tangential induction factor is nonzero)
    rotor.usecd = True  # (Bool): use drag coefficient in computing induction factors
    # -------------------------------------

    # --- run ------------
    rotor.run()

    AEP0 = rotor.AEP
    print('AEP0 =', AEP0)

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P/1e6)
    plt.xlabel('wind speed (m/s)')
    plt.ylabel('power (MW)')
    plt.show()

    # --------------------------


    if optimize:

        # --- optimizer imports ---
        from pyopt_driver.pyopt_driver import pyOptDriver
        from openmdao.lib.casehandlers.api import DumpCaseRecorder
        # ----------------------

        # --- Setup Pptimizer ---
        rotor.replace('driver', pyOptDriver())
        rotor.driver.optimizer = 'SNOPT'
        rotor.driver.options = {'Major feasibility tolerance': 1e-6,
                               'Minor feasibility tolerance': 1e-6,
                               'Major optimality tolerance': 1e-5,
                               'Function precision': 1e-8}
        # ----------------------

        # --- Objective ---
        rotor.driver.add_objective('-aep.AEP/%f' % AEP0)
        # ----------------------

        # --- Design Variables ---
        rotor.driver.add_parameter('r_max_chord', low=0.1, high=0.5)
        rotor.driver.add_parameter('chord_sub', low=0.4, high=5.3)
        rotor.driver.add_parameter('theta_sub', low=-10.0, high=30.0)
        rotor.driver.add_parameter('control.tsr', low=3.0, high=14.0)
        # ----------------------

        # --- recorder ---
        rotor.recorders = [DumpCaseRecorder()]
        # ----------------------

        # --- Constraints ---
        rotor.driver.add_constraint('1.0 >= 0.0')  # dummy constraint, OpenMDAO bug when using pyOpt
        # ----------------------

        # --- run opt ---
        rotor.run()
        # ---------------


