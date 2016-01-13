#!/usr/bin/env python
# encoding: utf-8
"""
aerodefaults.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi, gamma
from openmdao.api import Component, Group
from utilities import sind, cosd, smooth_abs, smooth_min, hstack, vstack, linspace_with_deriv
from rotoraero import common_configure
from akima import Akima
from enum import Enum

# ---------------------
# Map Design Variables to Discretization
# ---------------------

class PDFBase(Component):
    def __init__(self):
        super(PDFBase, self).__init__()
        """probability distribution function"""

        self.add_param('x')

        self.add_output('f')

class GeometrySpline(Component):
    def __init__(self):
        super(GeometrySpline, self).__init__()
        self.add_param('r_af', shape=17, units='m', desc='locations where airfoils are defined on unit radius')

        self.add_param('idx_cylinder', val=0, desc='location where cylinder section ends on unit radius')
        self.add_param('r_max_chord', shape=1, desc='position of max chord on unit radius')

        self.add_param('Rhub', shape=1, units='m', desc='blade hub radius')
        self.add_param('Rtip', shape=1, units='m', desc='blade tip radius')

        self.add_param('chord_sub', shape=4, units='m', desc='chord at control points')
        self.add_param('theta_sub', shape=4, units='deg', desc='twist at control points')

        self.add_output('r', shape=17, units='m', desc='chord at airfoil locations')
        self.add_output('chord', shape=17, units='m', desc='chord at airfoil locations')
        self.add_output('theta', shape=17, units='deg', desc='twist at airfoil locations')
        self.add_output('precurve', shape=17, units='m', desc='precurve at airfoil locations')
        # self.add_output('r_af_spacing', shape=16)  # deprecated: not used anymore

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        
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


    def list_deriv_vars(self):

        inputs = ('r_af', 'r_max_chord', 'Rhub', 'Rtip', 'chord_sub', 'theta_sub')
        outputs = ('r', 'chord', 'theta', 'precurve')

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        return self.J

# ---------------------
# Default Implementations of Base Classes
# ---------------------
class CCBladeGeometry(Component):
    def __init__(self):
        super(CCBladeGeometry, self).__init__()
        self.add_param('Rtip', shape=1, units='m', desc='tip radius')
        self.add_param('precurveTip', val=0.0, units='m', desc='tip radius')
        self.add_param('precone', val=0.0, desc='precone angle', units='deg')
        self.add_output('R', shape=1, units='m', desc='rotor radius')
        self.add_output('diameter', shape=1, units='m')

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):

        self.Rtip = params['Rtip']
        self.precurveTip = params['precurveTip']
        self.precone = params['precone']

        self.R = self.Rtip*cosd(self.precone) + self.precurveTip*sind(self.precone)

        unknowns['R'] = self.R
        unknowns['diameter'] = self.R*2

    def list_deriv_vars(self):

        inputs = ('Rtip', 'precurveTip', 'precone')
        outputs = ('R',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        J_sub = np.array([[cosd(self.precone), sind(self.precone),
            (-self.Rtip*sind(self.precone) + self.precurveTip*sind(self.precone))*pi/180.0]])
        J = {}
        J['R', 'Rtip'] = J_sub[0][0]
        J['R', 'precurveTip'] = J_sub[0][1]
        J['R', 'precone'] = J_sub[0][2]
        J['diameter', 'Rtip'] = 2.0*J_sub[0][0]
        J['diameter', 'precurveTip'] = 2.0*J_sub[0][1]
        J['diameter', 'precone'] = 2.0*J_sub[0][2]
        J['diameter', 'R'] = 2.0

        return J


# ## TODO
# class CCBlade(Component):
#     def __init__(self, run_case, n, n2):
#         super(CCBlade, self).__init__()
#         """blade element momentum code"""
#
#         # (potential) variables
#         self.add_param('r', shape=n, units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
#         self.add_param('chord', shape=n, units='m', desc='chord length at each section')
#         self.add_param('theta', shape=n,  units='deg', desc='twist angle at each section (positive decreases angle of attack)')
#         self.add_param('Rhub', shape=1, units='m', desc='hub radius')
#         self.add_param('Rtip', shape=1, units='m', desc='tip radius')
#         self.add_param('hubHt', shape=1, units='m', desc='hub height')
#         self.add_param('precone', shape=1, desc='precone angle', units='deg')
#         self.add_param('tilt', shape=1, desc='shaft tilt', units='deg')
#         self.add_param('yaw', shape=1, desc='yaw error', units='deg')
#
#         # TODO: I've not hooked up the gradients for these ones yet.
#         self.add_param('precurve', shape=n, units='m', desc='precurve at each section')
#         self.add_param('precurveTip', val=0.0, units='m', desc='precurve at tip')
#
#         # parameters
#         # self.add_param('airfoil_files', shape=n, desc='names of airfoil file', pass_by_obj=True)
#         self.add_param('airfoil_parameterization', val=np.zeros((n, 8)))
#         self.add_param('airfoil_analysis_options', val={})
#         self.add_param('B', val=3, desc='number of blades', pass_by_obj=True)
#         self.add_param('rho', val=1.225, units='kg/m**3', desc='density of air')
#         self.add_param('mu', val=1.81206e-5, units='kg/(m*s)', desc='dynamic viscosity of air')
#         self.add_param('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True)
#         self.add_param('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True)
#         self.add_param('tiploss', val=True, desc='include Prandtl tip loss model', pass_by_obj=True)
#         self.add_param('hubloss', val=True, desc='include Prandtl hub loss model', pass_by_obj=True)
#         self.add_param('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)', pass_by_obj=True)
#         self.add_param('usecd', val=True, desc='use drag coefficient in computing induction factors', pass_by_obj=True)
#
#         self.add_param('run_case', val=Enum('power', 'loads'), pass_by_obj=True)
#
#
#         # --- use these if (run_case == 'power') ---
#
#         # inputs
#         self.add_param('Uhub', shape=n2, units='m/s', desc='hub height wind speed')
#         self.add_param('Omega', shape=n2, units='rpm', desc='rotor rotation speed')
#         self.add_param('pitch', shape=n2, units='deg', desc='blade pitch setting')
#
#         # outputs
#         self.add_output('T', shape=n2, units='N', desc='rotor aerodynamic thrust')
#         self.add_output('Q', shape=n2, units='N*m', desc='rotor aerodynamic torque')
#         self.add_output('P', shape=n2, units='W', desc='rotor aerodynamic power')
#
#
#         # --- use these if (run_case == 'loads') ---
#         # if you only use rotoraero.py and not rotor.py
#         # (i.e., only care about power curves, and not structural loads)
#         # then these second set of inputs/outputs are not needed
#
#         # inputs
#         self.add_param('V_load', shape=1, units='m/s', desc='hub height wind speed')
#         self.add_param('Omega_load', shape=1, units='rpm', desc='rotor rotation speed')
#         self.add_param('pitch_load', shape=1, units='deg', desc='blade pitch setting')
#         self.add_param('azimuth_load', shape=1, units='deg', desc='blade azimuthal location')
#
#         # outputs
#         self.add_output('loads:r', shape=19, units='m', desc='radial positions along blade going toward tip')
#         self.add_output('loads:Px', shape=19, units='N/m', desc='distributed loads in blade-aligned x-direction')
#         self.add_output('loads:Py', shape=19, units='N/m', desc='distributed loads in blade-aligned y-direction')
#         self.add_output('loads:Pz', shape=19, units='N/m', desc='distributed loads in blade-aligned z-direction')
#
#         # corresponding setting for loads
#         self.add_output('loads:V', shape=1, units='m/s', desc='hub height wind speed')
#         self.add_output('loads:Omega', shape=1, units='rpm', desc='rotor rotation speed')
#         self.add_output('loads:pitch', shape=1, units='deg', desc='pitch angle')
#         self.add_output('loads:azimuth', shape=1, units='deg', desc='azimuthal angle')
#
#         self.run_case = run_case
#         self.fd_options['form'] = 'central'
#         self.fd_options['step_type'] = 'relative'
#
#     def solve_nonlinear(self, params, unknowns, resids):
#
#         self.r = params['r']
#         self.chord = params['chord']
#         self.theta = params['theta']
#         self.Rhub = params['Rhub']
#         self.Rtip = params['Rtip']
#         self.hubHt = params['hubHt']
#         self.precone = params['precone']
#         self.tilt = params['tilt']
#         self.yaw = params['yaw']
#         self.precurve = params['precurve']
#         self.precurveTip = params['precurveTip']
#         # self.airfoil_files = params['airfoil_files']
#         self.airfoil_parameterization = params['airfoil_parameterization']
#         self.airfoil_analysis_options = params['airfoil_analysis_options']
#         self.B = params['B']
#         self.rho = params['rho']
#         self.mu = params['mu']
#         self.shearExp = params['shearExp']
#         self.nSector = params['nSector']
#         self.tiploss = params['tiploss']
#         self.hubloss = params['hubloss']
#         self.wakerotation = params['wakerotation']
#         self.usecd = params['usecd']
#         self.Uhub = params['Uhub']
#         self.Omega = params['Omega']
#         self.pitch = params['pitch']
#         self.V_load = params['V_load']
#         self.Omega_load = params['Omega_load']
#         self.pitch_load = params['pitch_load']
#         self.azimuth_load = params['azimuth_load']
#
#
#         if len(self.precurve) == 0:
#             self.precurve = np.zeros_like(self.r)
#
#         # airfoil files
#         n = len(self.airfoil_files)
#         af = [0]*n
#         afinit = CCAirfoil.initFromAerodynFile
#         for i in range(n):
#             af[i] = afinit(self.airfoil_files[i])
#
#         self.ccblade = CCBlade_PY(self.r, self.chord, self.theta, af, self.Rhub, self.Rtip, self.B,
#             self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubHt,
#             self.nSector, self.precurve, self.precurveTip, tiploss=self.tiploss, hubloss=self.hubloss,
#             wakerotation=self.wakerotation, usecd=self.usecd, derivatives=True)
#
#
#         if self.run_case == 'power':
#
#             # power, thrust, torque
#
#             self.P, self.T, self.Q, self.dP, self.dT, self.dQ \
#                 = self.ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=False)
#             unknowns['T'] = self.T
#             unknowns['Q'] = self.Q
#             unknowns['P'] = self.P
#
#         elif self.run_case == 'loads':
#             # distributed loads
#             Np, Tp, self.dNp, self.dTp \
#                 = self.ccblade.distributedAeroLoads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)
#
#             # concatenate loads at root/tip
#             unknowns['loads:r'] = np.concatenate([[self.Rhub], self.r, [self.Rtip]])
#             Np = np.concatenate([[0.0], Np, [0.0]])
#             Tp = np.concatenate([[0.0], Tp, [0.0]])
#
#             # conform to blade-aligned coordinate system
#             unknowns['loads:Px'] = Np
#             unknowns['loads:Py'] = -Tp
#             unknowns['loads:Pz'] = 0*Np
#
#             # return other outputs needed
#             unknowns['loads:V'] = self.V_load
#             unknowns['loads:Omega'] = self.Omega_load
#             unknowns['loads:pitch'] = self.pitch_load
#             unknowns['loads:azimuth'] = self.azimuth_load
#
#
#
#     def list_deriv_vars(self):
#
#         if self.run_case == 'power':
#             inputs = ('precone', 'tilt', 'hubHt', 'Rhub', 'Rtip', 'yaw',
#                 'Uhub', 'Omega', 'pitch', 'r', 'chord', 'theta', 'precurve', 'precurveTip')
#             outputs = ('P', 'T', 'Q')
#
#         elif self.run_case == 'loads':
#
#             inputs = ('r', 'chord', 'theta', 'Rhub', 'Rtip', 'hubHt', 'precone',
#                 'tilt', 'yaw', 'V_load', 'Omega_load', 'pitch_load', 'azimuth_load', 'precurve')
#             outputs = ('loads:r', 'loads:Px', 'loads:Py', 'loads:Pz', 'loads:V',
#                 'loads:Omega', 'loads:pitch', 'loads:azimuth')
#
#         return inputs, outputs
#
#
#     def linearize(self, params, unknowns, resids):
#
#         if self.run_case == 'power':
#
#             dP = self.dP
#             dT = self.dT
#             dQ = self.dQ
#
#             J = {}
#             J['P', 'precone'] = dP['dprecone']
#             J['P', 'tilt'] = dP['dtilt']
#             J['P', 'hubHt'] = dP['dhubHt']
#             J['P', 'Rhub'] = dP['dRhub']
#             J['P', 'Rtip'] = dP['dRtip']
#             J['P', 'yaw'] = dP['dyaw']
#             J['P', 'Uhub'] = dP['dUinf']
#             J['P', 'Omega'] = dP['dOmega']
#             J['P', 'pitch'] =  dP['dpitch']
#             J['P', 'r'] = dP['dr']
#             J['P', 'chord'] = dP['dchord']
#             J['P', 'theta'] = dP['dtheta']
#             J['P', 'precurve'] = dP['dprecurve']
#             J['P', 'precurveTip'] = dP['dprecurveTip']
#
#             J['T', 'precone'] = dT['dprecone']
#             J['T', 'tilt'] = dT['dtilt']
#             J['T', 'hubHt'] = dT['dhubHt']
#             J['T', 'Rhub'] = dT['dRhub']
#             J['T', 'Rtip'] = dT['dRtip']
#             J['T', 'yaw'] = dT['dyaw']
#             J['T', 'Uhub'] = dT['dUinf']
#             J['T', 'Omega'] = dT['dOmega']
#             J['T', 'pitch'] =  dT['dpitch']
#             J['T', 'r'] = dT['dr']
#             J['T', 'chord'] = dT['dchord']
#             J['T', 'theta'] = dT['dtheta']
#             J['T', 'precurve'] = dT['dprecurve']
#             J['T', 'precurveTip'] = dT['dprecurveTip']
#
#             J['Q', 'precone'] = dQ['dprecone']
#             J['Q', 'tilt'] = dQ['dtilt']
#             J['Q', 'hubHt'] = dQ['dhubHt']
#             J['Q', 'Rhub'] = dQ['dRhub']
#             J['Q', 'Rtip'] = dQ['dRtip']
#             J['Q', 'yaw'] = dQ['dyaw']
#             J['Q', 'Uhub'] = dQ['dUinf']
#             J['Q', 'Omega'] = dQ['dOmega']
#             J['Q', 'pitch'] =  dQ['dpitch']
#             J['Q', 'r'] = dQ['dr']
#             J['Q', 'chord'] = dQ['dchord']
#             J['Q', 'theta'] = dQ['dtheta']
#             J['Q', 'precurve'] = dQ['dprecurve']
#             J['Q', 'precurveTip'] = dQ['dprecurveTip']
#
#         elif self.run_case == 'loads':
#
#             dNp = self.dNp
#             dTp = self.dTp
#             n = len(self.r)
#
#             dr_dr = vstack([np.zeros(n), np.eye(n), np.zeros(n)])
#             dr_dRhub = np.zeros(n+2)
#             dr_dRtip = np.zeros(n+2)
#             dr_dRhub[0] = 1.0
#             dr_dRtip[-1] = 1.0
#
#             dV = np.zeros(4*n+10)
#             dV[3*n+6] = 1.0
#             dOmega = np.zeros(4*n+10)
#             dOmega[3*n+7] = 1.0
#             dpitch = np.zeros(4*n+10)
#             dpitch[3*n+8] = 1.0
#             dazimuth = np.zeros(4*n+10)
#             dazimuth[3*n+9] = 1.0
#
#             J = {}
#             zero = np.zeros(17)
#             J['loads:r', 'r'] = dr_dr
#             J['loads:r', 'Rhub'] = dr_dRhub
#             J['loads:r', 'Rtip'] = dr_dRtip
#             J['loads:Px', 'r'] = np.vstack([zero, dNp['dr'], zero])
#             J['loads:Px', 'chord'] = np.vstack([zero, dNp['dchord'], zero])
#             J['loads:Px', 'theta'] = np.vstack([zero, dNp['dtheta'], zero])
#             J['loads:Px', 'Rhub'] = np.concatenate([[0.0], np.squeeze(dNp['dRhub']), [0.0]])
#             J['loads:Px', 'Rtip'] = np.concatenate([[0.0], np.squeeze(dNp['dRtip']), [0.0]])
#             J['loads:Px', 'hubHt'] = np.concatenate([[0.0], np.squeeze(dNp['dhubHt']), [0.0]])
#             J['loads:Px', 'precone'] = np.concatenate([[0.0], np.squeeze(dNp['dprecone']), [0.0]])
#             J['loads:Px', 'tilt'] = np.concatenate([[0.0], np.squeeze(dNp['dtilt']), [0.0]])
#             J['loads:Px', 'yaw'] = np.concatenate([[0.0], np.squeeze(dNp['dyaw']), [0.0]])
#             J['loads:Px', 'V_load'] = np.concatenate([[0.0], np.squeeze(dNp['dUinf']), [0.0]])
#             J['loads:Px', 'Omega_load'] = np.concatenate([[0.0], np.squeeze(dNp['dOmega']), [0.0]])
#             J['loads:Px', 'pitch_load'] = np.concatenate([[0.0], np.squeeze(dNp['dpitch']), [0.0]])
#             J['loads:Px', 'azimuth_load'] = np.concatenate([[0.0], np.squeeze(dNp['dazimuth']), [0.0]])
#             J['loads:Px', 'precurve'] = np.vstack([zero, dNp['dprecurve'], zero])
#             J['loads:Py', 'r'] = np.vstack([zero, -dTp['dr'], zero])
#             J['loads:Py', 'chord'] = np.vstack([zero, -dTp['dchord'], zero])
#             J['loads:Py', 'theta'] = np.vstack([zero, -dTp['dtheta'], zero])
#             J['loads:Py', 'Rhub'] = np.concatenate([[0.0], -np.squeeze(dTp['dRhub']), [0.0]])
#             J['loads:Py', 'Rtip'] = np.concatenate([[0.0], -np.squeeze(dTp['dRtip']), [0.0]])
#             J['loads:Py', 'hubHt'] = np.concatenate([[0.0], -np.squeeze(dTp['dhubHt']), [0.0]])
#             J['loads:Py', 'precone'] = np.concatenate([[0.0], -np.squeeze(dTp['dprecone']), [0.0]])
#             J['loads:Py', 'tilt'] = np.concatenate([[0.0], -np.squeeze(dTp['dtilt']), [0.0]])
#             J['loads:Py', 'yaw'] = np.concatenate([[0.0], -np.squeeze(dTp['dyaw']), [0.0]])
#             J['loads:Py', 'V_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dUinf']), [0.0]])
#             J['loads:Py', 'Omega_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dOmega']), [0.0]])
#             J['loads:Py', 'pitch_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dpitch']), [0.0]])
#             J['loads:Py', 'azimuth_load'] = np.concatenate([[0.0], -np.squeeze(dTp['dazimuth']), [0.0]])
#             J['loads:Py', 'precurve'] = np.vstack([zero, -dTp['dprecurve'], zero])
#             J['loads:V', 'V_load'] = 1.0
#             J['loads:Omega', 'Omega_load'] = 1.0
#             J['loads:pitch', 'pitch_load'] = 1.0
#             J['loads:azimuth', 'azimuth_load'] = 1.0
#
#         return J



class CSMDrivetrain(Component):
    def __init__(self, n):
        super(CSMDrivetrain, self).__init__()
        """drivetrain losses from NREL cost and scaling model"""

        self.add_param('drivetrainType', val=Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True)


        self.add_param('aeroPower', shape=n, units='W', desc='aerodynamic power')
        self.add_param('aeroTorque', shape=n, units='N*m', desc='aerodynamic torque')
        self.add_param('aeroThrust', shape=n, units='N', desc='aerodynamic thrust')
        self.add_param('ratedPower', shape=1, units='W', desc='rated power')

        self.add_output('power', shape=n, units='W', desc='total power after drivetrain losses')
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

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


    def list_deriv_vars(self):

        inputs = ('aeroPower', 'ratedPower')
        outputs = ('power',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        return self.J




class WeibullCDF(Component):
    def __init__(self, n):
        super(WeibullCDF, self).__init__()
        """Weibull cumulative distribution function"""

        self.add_param('A', shape=1, desc='scale factor')
        self.add_param('k', shape=1, desc='shape or form factor')
        self.add_param('x', shape=n)

        self.add_output('F', shape=n)

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['F'] = 1.0 - np.exp(-(params['x']/params['A'])**params['k'])

    def list_deriv_vars(self):
        inputs = ('x',)
        outputs = ('F',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        x = params['x']
        A = params['A']
        k = params['k']
        J = {}
        J['F', 'x'] = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        return J


class WeibullWithMeanCDF(Component):
    def __init__(self, n):
        super(WeibullWithMeanCDF, self).__init__()
        """Weibull cumulative distribution function"""

        self.add_param('xbar', shape=1, desc='mean value of distribution')
        self.add_param('k', shape=1, desc='shape or form factor')
        self.add_param('x', shape=n)

        self.add_output('F', shape=n)
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):

        A = params['xbar'] / gamma(1.0 + 1.0/params['k'])

        unknowns['F'] = 1.0 - np.exp(-(params['x']/A)**params['k'])


    def list_deriv_vars(self):

        inputs = ('x', 'xbar')
        outputs = ('F',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        x = params['x']
        k = params['k']
        A = params['xbar'] / gamma(1.0 + 1.0/k)
        dx = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        dxbar = -np.exp(-(x/A)**k)*(x/A)**(k-1)*k*x/A**2/gamma(1.0 + 1.0/k)
        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar

        return J


# class RayleighCDF(Component):
#     def __init(self):
#         super(RayleighCDF, self).__init__()
#
#         """Rayleigh cumulative distribution function"""
#
#         self.add_param('xbar', shape=1, desc='mean value of distribution')
#         self.add_param('x', shape=200)
#
#         self.add_output('F', shape=20)
#         self.fd_options['form'] = 'central'
#         self.fd_options['step_type'] = 'relative'
#
#     def solve_nonlinear(self, params, unknowns, resids):
#
#         unknowns['F'] = 1.0 - np.exp(-pi/4.0*(params['x']/params['xbar'])**2)
#
#     def list_deriv_vars(self):
#
#         inputs = ('x', 'xbar')
#         outputs = ('F',)
#
#         return inputs, outputs
#
#     def linearize(self, params, unknowns, resids):
#
#         x = params['x']
#         xbar = params['xbar']
#         dx = np.diag(np.exp(-pi/4.0*(x/xbar)**2)*pi*x/(2.0*xbar**2))
#         dxbar = -np.exp(-pi/4.0*(x/xbar)**2)*pi*x**2/(2.0*xbar**3)
#         J = {}
#         J['F', 'x'] = dx
#         J['F', 'xbar'] = dxbar
#
#         return J

class RayleighCDF(Component):
    def __init__(self):
        super(RayleighCDF,  self).__init__()

        # variables
        self.add_param('xbar', shape=1, units='m/s', desc='reference wind speed (usually at hub height)')
        self.add_param('x', shape=200,  units='m/s', desc='corresponding reference height')

        # out
        self.add_output('F', shape=200, units='m/s', desc='magnitude of wind speed at each z location')
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

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

    rotor = Problem()
    rotor.root = RotorSE()

    rotor.setup()

    # === blade grid ===
    rotor['initial_aero_grid'] = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
        0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): initial aerodynamic grid on unit radius
    rotor['initial_str_grid'] = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
        0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
        0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
        0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
        0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
        1.0])  # (Array): initial structural grid on unit radius
    rotor['idx_cylinder_aero'] = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
    rotor['idx_cylinder_str'] = 14  # (Int): first idx in r_str_unit of non-cylindrical section
    rotor['hubFraction'] = 0.025  # (Float): hub location as fraction of radius
    # ------------------

    # === blade geometry ===
    rotor['r_aero'] = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): new aerodynamic grid on unit radius
    rotor['r_max_chord'] = 0.23577  # (Float): location of max chord on unit radius
    rotor['chord_sub'] = np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_sub'] = np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_sub'] = np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    # rotor['delta_precurve_sub'] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
    rotor['sparT'] = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT'] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    rotor['bladeLength'] = 61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    # rotor['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    rotor['precone'] = 2.5  # (Float, deg): precone angle
    rotor['tilt'] = 5.0  # (Float, deg): shaft tilt
    rotor['yaw'] = 0.0  # (Float, deg): yaw error
    rotor['nBlades'] = 3  # (Int): number of blades
    # ------------------

    # === airfoil files ===
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]
    rotor['airfoil_files'] = af  # (List): names of airfoil file
    # ----------------------

    # === atmosphere ===
    rotor['rho'] = 1.225  # (Float, kg/m**3): density of air
    rotor['mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['shearExp'] = 0.25  # (Float): shear exponent
    rotor['hubHt'] = 90.0  # (Float, m): hub height
    rotor['turbine_class'] = 'I'  # (Enum): IEC turbine class
    rotor['turbulence_class'] = 'B'  # (Enum): IEC turbulence class class
    rotor['cdf_reference_height_wind_speed'] = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    rotor['g'] = 9.81  # (Float, m/s**2): acceleration of gravity
    # ----------------------

    # === control ===
    rotor['control:Vin'] = 3.0  # (Float, m/s): cut-in wind speed
    rotor['control:Vout'] = 25.0  # (Float, m/s): cut-out wind speed
    rotor['control:ratedPower'] = 5e6  # (Float, W): rated power
    rotor['control:minOmega'] = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control:maxOmega'] = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control:tsr'] = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control:pitch'] = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor['pitch_extreme'] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    rotor['azimuth_extreme'] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    rotor['VfactorPC'] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['npts_coarse_power_curve'] = 20  # (Int): number of points to evaluate aero analysis at
    rotor['npts_spline_power_curve'] = 200  # (Int): number of points to use in fitting spline to power curve
    rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType'] = 'geared'  # (Enum)
    rotor['nF'] = 5  # (Int): number of natural frequencies to compute
    rotor['dynamic_amplication_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------

    # === materials and composite layup  ===
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(rotor['initial_str_grid'])
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    rotor['leLoc'] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
        0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
    rotor['sector_idx_strain_spar'] = [2]*ncomp  # (Array): index of sector for spar (PreComp definition of sector)
    rotor['sector_idx_strain_te'] = [3]*ncomp  # (Array): index of sector for trailing-edge (PreComp definition of sector)
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    rotor['chord_str_ref'] = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
        3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
         4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
         4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
         3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
         2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)

    for i in range(ncomp):

        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))

    rotor['materials'] = np.array(materials)  # (List): list of all Orthotropic2DMaterial objects used in defining the geometry
    rotor['upperCS'] = np.array(upper)  # (List): list of CompositeSection objections defining the properties for upper surface
    rotor['lowerCS'] = np.array(lower)  # (List): list of CompositeSection objections defining the properties for lower surface
    rotor['websCS'] = np.array(webs)  # (List): list of CompositeSection objections defining the properties for shear webs
    rotor['profile'] = np.array(profile)  # (List): airfoil shape at each radial position
    # --------------------------------------


    # === fatigue ===
    rotor['rstar_damage'] = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    rotor['Mxb_damage'] = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    rotor['Myb_damage'] = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    rotor['strain_ult_spar'] = 1.0e-2  # (Float): ultimate strain in spar cap
    rotor['strain_ult_te'] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    rotor['eta_damage'] = 1.35*1.3*1.0  # (Float): safety factor for fatigue
    rotor['m_damage'] = 10.0  # (Float): slope of S-N curve for fatigue analysis
    rotor['N_damage'] = 365*24*3600*20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------

    # from myutilities import plt

    # === run and outputs ===
    rotor.run()

    print 'AEP =', rotor['AEP']
    print 'diameter =', rotor['diameter']
    print 'ratedConditions:V =', rotor['ratedConditions:V']
    print 'ratedConditions:Omega =', rotor['ratedConditions:Omega']
    print 'ratedConditions:pitch =', rotor['ratedConditions:pitch']
    print 'ratedConditions:T =', rotor['ratedConditions:T']
    print 'ratedConditions:Q =', rotor['ratedConditions:Q']
    print 'mass_one_blade =', rotor['mass_one_blade']
    print 'mass_all_blades =', rotor['mass_all_blades']
    print 'I_all_blades =', rotor['I_all_blades']
    print 'freq =', rotor['freq']
    print 'tip_deflection =', rotor['tip_deflection']
    print 'root_bending_moment =', rotor['root_bending_moment']

