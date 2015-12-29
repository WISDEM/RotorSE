#!/usr/bin/env python
# encoding: utf-8
"""
aerodefaults.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi, gamma
# from openmdao.main.datatypes.api import Int, Float, Array, Str, List, Enum, VarTree, Bool
# from openmdao.main.api import Component, Assembly
from openmdao.api import IndepVarComp, Component, Problem, Group, SqliteRecorder, BaseRecorder
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


from ccblade import CCAirfoil, CCBlade as CCBlade_PY
from commonse.utilities import sind, cosd, smooth_abs, smooth_min, hstack, vstack, linspace_with_deriv
from rotoraero import common_configure
from akima import Akima
from enum import Enum
from scipy.optimize import brentq


from scipy.optimize import brentq

# class Brent(Component):
#     """Root finding using Brent's method."""
#     def __init__(self):
#         super(Brent, self).__init__()
#             self.workflow = CyclicWorkflow()
#             self.xstar = self._param = None
#
#             self.add_param('lower_bound', val=0., desc="lower bound for the root search")
#             self.add_param('upper_bound', val=100., desc="upper bound for the root search")
#             self.add_param('xtol', val=0.0, desc='The routine converges when a root is known to lie within xtol of the value return. Should be >= 0. '
#                          'The routine modifies this to take into account the relative precision of doubles.')
#             self.add_param('rtol', val=0.0, desc='The routine converges when a root is known to lie within rtol times the value returned of '
#                          'the value returned. Should be >= 0. Defaults to np.finfo(float).eps * 2.')
#             self.add_param('maxiter', val=100, desc='if convergence is not achieved in maxiter iterations, and error is raised. Must be >= 0.')
#             self.add_param('iprint', val=0, desc='Set to 1 to print out itercount and residual.')
#             self.add_param('f_resize_bracket', Slot(object,
#                                    desc='user supplied function to handle resizing bracket.  Form of function is: \
#                                    lower_new, upper_new, continue = f_resize_bracket(lower, upper, iteration) \
#                                    inputs include the current lower and upper bracket and the current iteration \
#                                    count starting from 1.  Outputs include a new lower and upper bracket, and a \
#                                    boolean flag on whether or not to terminate calling resize bracket')
#
#             self.add_param('invalid_bracket_return', val=-1,
#                                            desc='user supplied value to handle what value should be returned \
#                                                  when a suitable bracket cannot be found. sets the "zero" as a \
#                                                  linear combination of the lower and upper brackets. \
#                                                  Must be between 0 and 1 or an error will be thrown. \
#                                                  root = lower + invalid_bracket_return*(upper-lower)')
#     def _eval(self, x):
#         """Callback function for evaluating f(x)"""
#         self._param.set(x)
#         self.run_iteration()
#         return self.eval_eq_constraints(self.parent)[0]
#
#     def solv(self):
#
#         bracket_invalid = self._eval(self.lower_bound)*self._eval(self.upper_bound) > 0
#
#         # check if user supplied function to handle resizing bracket
#         if bracket_invalid and self.f_resize_bracket:
#
#             # try to resize bracket to find a valid bracket.
#             iteration = 1
#             should_continue = True
#             bracket_invalid = True
#
#             while bracket_invalid and should_continue:
#                 self.lower_bound, self.upper_bound, should_continue = \
#                     self.f_resize_bracket(self.lower_bound, self.upper_bound, iteration)
#
#                 bracket_invalid = self._eval(self.lower_bound)*self._eval(self.upper_bound) > 0
#                 iteration += 1
#
#         if bracket_invalid:  # if bracket is still invalid, see if user has specified what to return
#
#             if self.invalid_bracket_return >= 0.0 and self.invalid_bracket_return <= 1.0:
#                 xstar = self.lower_bound + self.invalid_bracket_return*(self.upper_bound-self.lower_bound)
#                 brent_iterations = 'valid bracket not found.  returning user specified value'
#
#             else:
#                 self.raise_exception('bounds (low=%s, high=%s) do not bracket a root' %
#                                  (self.lower_bound, self.upper_bound))
#
#         else:
#
#             kwargs = {'maxiter': self.maxiter, 'a': self.lower_bound,
#                       'b': self.upper_bound, 'full_output': True}
#
#             if self.xtol > 0:
#                 kwargs['xtol'] = self.xtol
#             if self.rtol > 0:
#                 kwargs['rtol'] = self.rtol
#
#             # Brent's method
#             xstar, r = brentq(self._eval, **kwargs)
#             brent_iterations = r.iterations
#
#
#         # Propagate solution back into the model
#         self._param.set(xstar)
#         self.run_iteration()
#
#         if self.iprint == 1:
#             print 'iterations:', brent_iterations
#             print 'residual:', self.eval_eq_constraints()
#
#     def check_config(self, strict=False):
#         '''Make sure we have 1 parameter and 1 constraint'''
#
#         super(Brent, self).check_config(strict=strict)
#
#         params = self.get_parameters().values()
#         if len(params) != 1:
#             self.raise_exception("Brent driver must have 1 parameter, "
#                                  "but instead it has %d" % len(params))
#
#         constraints = self.get_eq_constraints()
#         if len(constraints) != 1:
#             self.raise_exception("Brent driver must have 1 equality constraint, "
#                                  "but instead it has %d" % len(constraints))
#         self._param = params[0]

class Brent(Component):
    """root finding using Brent's method."""
    def __init__(self):
        super(Brent, self).__init__()

        self.add_param('lower_bound', val=0.)
        self.add_param('upper_bound', val=100., desc="upper bound for the root search")

        self.add_param('xtol', val=0.0, desc='The routine converges when a root is known to lie within xtol of the value return. Should be >= 0. '
            'The routine modifies this to take into account the relative precision of doubles.')

        self.add_param('rtol', val=0.0, desc='The routine converges when a root is known to lie within rtol times the value returned of '
            'the value returned. Should be >= 0. Defaults to np.finfo(float).eps * 2.')

        self.add_param('maxiter', val=100, desc='if convergence is not achieved in maxiter iterations, and error is raised. Must be >= 0.')
        self.add_output('xstar', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):

        xlow = params['lower_bound']
        xhigh = params['upper_bound']
        maxiter = params['maxiter']

        if self._eval(xlow)*self._eval(xhigh) > 0:
            raise Exception('bounds do not bracket a root')

        kwargs = {'maxiter': maxiter, 'a': xlow, 'b': xhigh}
        if self.xtol > 0:
            kwargs['xtol'] = self.xtol
        if self.rtol > 0:
            kwargs['rtol'] = self.rtol


        # Brent's method
        xstar = brentq(self._eval, **kwargs)

        # set result
        #param.set(xstar)
        unknowns['xstar'] = xstar


# ---------------------
# Map Design Variables to Discretization
# ---------------------

class PDFBase(Component):
    def __init__(self):
        super(PDFBase).__init__()
        """probability distribution function"""

        self.add_param('x')

        self.add_output('f')

class GeometrySpline(Component):
    def __init__(self):
        super(GeometrySpline).__init__()
        self.add_param('r_af', units='m', desc='locations where airfoils are defined on unit radius')

        self.add_param('idx_cylinder', desc='location where cylinder section ends on unit radius')
        self.add_param('r_max_chord', desc='position of max chord on unit radius')

        self.add_param('Rhub', units='m', desc='blade hub radius')
        self.add_param('Rtip', units='m', desc='blade tip radius')

        self.add_param('chord_sub', units='m', desc='chord at control points')
        self.add_param('theta_sub', units='deg', desc='twist at control points')

        self.add_output('r', units='m', desc='chord at airfoil locations')
        self.add_output('chord', units='m', desc='chord at airfoil locations')
        self.add_output('theta', units='deg', desc='twist at airfoil locations')
        self.add_output('precurve', units='m', desc='precurve at airfoil locations')
        self.add_output('r_af_spacing')  # deprecated: not used anymore


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
        chord, dchord_dr, dchord_drchord, dchord_dchordsub = chord_spline.interp(r)
        theta_outer, dthetaouter_dr, dthetaouter_drtheta, dthetaouter_dthetasub = theta_spline.interp(r[idxc:])
        unknowns['chord'] = chord

        theta_inner = theta_outer[0] * np.ones(idxc)
        unknowns['theta'] = np.concatenate([theta_inner, theta_outer])

        unknowns['r_af_spacing'] = np.diff(r_af)

        unknowns['precurve'] = np.zeros_like(unknowns['chord'])  # TODO: for now I'm forcing this to zero, just for backwards compatibility

        # gradients (TODO: rethink these a bit or use Tapenade.)
        n = len(r_af)
        dr_draf = (Rtip-Rhub)*np.ones(n)
        dr_dRhub = 1.0 - r_af
        dr_dRtip = r_af
        dr = hstack([np.diag(dr_draf), np.zeros((n, 1)), dr_dRhub, dr_dRtip, np.zeros((n, nc+nt))])

        dchord_draf = dchord_dr * dr_draf
        dchord_drmaxchord0 = np.dot(dchord_drchord, drc_drcmax)
        dchord_drmaxchord = dchord_drmaxchord0 * (Rtip-Rhub)
        dchord_drhub = np.dot(dchord_drchord, drc_drhub) + dchord_drmaxchord0*(1.0 - r_max_chord) + dchord_dr*dr_dRhub
        dchord_drtip = np.dot(dchord_drchord, drc_drtip) + dchord_drmaxchord0*(r_max_chord) + dchord_dr*dr_dRtip
        dchord = hstack([np.diag(dchord_draf), dchord_drmaxchord, dchord_drhub, dchord_drtip, dchord_dchordsub, np.zeros((n, nt))])

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

        dtheta = hstack([dtheta_draf, np.zeros((n, 1)), dtheta_drhub, dtheta_drtip, np.zeros((n, nc)), dtheta_dthetasub])

        drafs_dr = np.zeros((n-1, n))
        for i in range(n-1):
            drafs_dr[i, i] = -1.0
            drafs_dr[i, i+1] = 1.0
        drafs = hstack([drafs_dr, np.zeros((n-1, 3+nc+nt))])

        dprecurve = np.zeros((len(unknowns['precurve']), n+3+nc+nt))
        J = {}
        J['r', 'r_af'] = np.diag(dr_draf)
        J['r', 'Rhub'] = dr_dRhub
        J['r', 'Rtip'] = dr_dRtip
        J['chord', 'r_af'] = np.diag(dchord_draf)
        J['chord', 'r_max_chord'] = dchord_drmaxchord
        J['chord', 'Rhub'] =dchord_drhub
        J['chord', 'Rtip'] =dchord_drtip
        J['chord', 'chord_sub'] =dchord_dchordsub
        J['theta', 'r_af'] = dtheta_draf
        J['theta', 'Rhub'] =dtheta_drhub
        J['theta', 'Rtip'] =dtheta_drtip
        J['theta', 'theta_sub'] =dtheta_dthetasub
        J['r_af_spacing', 'r_af'] = drafs_dr

        self.J = J


    def list_deriv_vars(self):

        inputs = ('r_af', 'r_max_chord', 'Rhub', 'Rtip', 'chord_sub', 'theta_sub')
        outputs = ('r', 'chord', 'theta', 'r_af_spacing', 'precurve')

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

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
        self.add_output('diameter', shape=1)

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

    def jacobian(self, params, unknowns, resids):

        J = np.array([[cosd(self.precone), sind(self.precone),
            (-self.Rtip*sind(self.precone) + self.precurveTip*sind(self.precone))*pi/180.0]])

        J['diameter', 'R'] = 2.0

        return J


## TODO
class CCBlade(Component):
    def __init__(self, run_case, n):
        super(CCBlade, self).__init__()
        """blade element momentum code"""

        # (potential) variables
        self.add_param('r', shape=17, units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_param('chord', shape=17, units='m', desc='chord length at each section')
        self.add_param('theta', shape=17,  units='deg', desc='twist angle at each section (positive decreases angle of attack)')
        self.add_param('Rhub', shape=1, units='m', desc='hub radius')
        self.add_param('Rtip', shape=1, units='m', desc='tip radius')
        self.add_param('hubHt', shape=1, units='m', desc='hub height')
        self.add_param('precone', shape=1, desc='precone angle', units='deg')
        self.add_param('tilt', shape=1, desc='shaft tilt', units='deg')
        self.add_param('yaw', shape=1, desc='yaw error', units='deg')

        # TODO: I've not hooked up the gradients for these ones yet.
        self.add_param('precurve', shape=17, units='m', desc='precurve at each section')
        self.add_param('precurveTip', val=0.0, units='m', desc='precurve at tip')

        # parameters
        self.add_param('airfoil_files', shape=17, desc='names of airfoil file', pass_by_obj=True)
        self.add_param('B', val=3, desc='number of blades')
        self.add_param('rho', val=1.225, units='kg/m**3', desc='density of air')
        self.add_param('mu', val=1.81206e-5, units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_param('shearExp', val=0.2, desc='shear exponent')
        self.add_param('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_param('tiploss', val=True, desc='include Prandtl tip loss model')
        self.add_param('hubloss', val=True, desc='include Prandtl hub loss model')
        self.add_param('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_param('usecd', val=True, desc='use drag coefficient in computing induction factors')

        missing_deriv_policy = 'assume_zero'


        # self.add_param('run_case', val=Enum('power', 'loads'))
        self.add_param('run_case', val=Enum('power', 'loads'))


        # --- use these if (run_case == 'power') ---

        # inputs
        self.add_param('Uhub', shape=n, units='m/s', desc='hub height wind speed')
        self.add_param('Omega', shape=n, units='rpm', desc='rotor rotation speed')
        self.add_param('pitch', shape=n, units='deg', desc='blade pitch setting')

        # outputs
        self.add_output('T', shape=n, units='N', desc='rotor aerodynamic thrust')
        self.add_output('Q', shape=n, units='N*m', desc='rotor aerodynamic torque')
        self.add_output('P', shape=n, units='W', desc='rotor aerodynamic power')


        # --- use these if (run_case == 'loads') ---
        # if you only use rotoraero.py and not rotor.py
        # (i.e., only care about power curves, and not structural loads)
        # then these second set of inputs/outputs are not needed

        # inputs
        self.add_param('V_load', shape=1, units='m/s', desc='hub height wind speed')
        self.add_param('Omega_load', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_param('pitch_load', shape=1, units='deg', desc='blade pitch setting')
        self.add_param('azimuth_load', shape=1, units='deg', desc='blade azimuthal location')

        # outputs
        # loads = VarTree(AeroLoads(), iotype='out', desc='loads in blade-aligned coordinate system')
        self.add_output('loads:r', shape=19, units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads:Px', shape=19, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads:Py', shape=19, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads:Pz', shape=19, units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_output('loads:V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_output('loads:Omega', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_output('loads:pitch', shape=1, units='deg', desc='pitch angle')
        self.add_output('loads:azimuth', shape=1, units='deg', desc='azimuthal angle')

        self.run_case = run_case

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
        self.airfoil_files = params['airfoil_files']
        self.B = params['B']
        self.rho = params['rho']
        self.mu = params['mu']
        self.shearExp = params['shearExp']
        self.nSector = params['nSector']
        self.tiploss = params['tiploss']
        self.hubloss = params['hubloss']
        self.wakerotation = params['wakerotation']
        self.usecd = params['usecd']
        # self.run_case = params['run_case']
        self.Uhub = params['Uhub']
        self.Omega = params['Omega']
        self.pitch = params['pitch']
        self.V_load = params['V_load']
        self.Omega_load = params['Omega_load']
        self.pitch_load = params['pitch_load']
        self.azimuth_load = params['azimuth_load']


        if len(self.precurve) == 0:
            self.precurve = np.zeros_like(self.r)

        # airfoil files
        n = len(self.airfoil_files)
        af = [0]*n
        afinit = CCAirfoil.initFromAerodynFile
        for i in range(n):
            af[i] = afinit(self.airfoil_files[i])

        self.ccblade = CCBlade_PY(self.r, self.chord, self.theta, af, self.Rhub, self.Rtip, self.B,
            self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubHt,
            self.nSector, self.precurve, self.precurveTip, tiploss=self.tiploss, hubloss=self.hubloss,
            wakerotation=self.wakerotation, usecd=self.usecd, derivatives=True)


        if self.run_case == 'power':

            # power, thrust, torque

            self.P, self.T, self.Q, self.dP, self.dT, self.dQ \
                = self.ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=False)
            unknowns['T'] = self.T
            unknowns['Q'] = self.Q
            unknowns['P'] = self.P

        elif self.run_case == 'loads':
            if self.azimuth_load == 180.0:
                pass
            # distributed loads
            Np, Tp, self.dNp, self.dTp \
                = self.ccblade.distributedAeroLoads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)

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



    def list_deriv_vars(self):

        if self.run_case == 'power':
            inputs = ('precone', 'tilt', 'hubHt', 'Rhub', 'Rtip', 'yaw',
                'Uhub', 'Omega', 'pitch', 'r', 'chord', 'theta', 'precurve', 'precurveTip')
            outputs = ('P', 'T', 'Q')

        elif self.run_case == 'loads':

            inputs = ('r', 'chord', 'theta', 'Rhub', 'Rtip', 'hubHt', 'precone',
                'tilt', 'yaw', 'V_load', 'Omega_load', 'pitch_load', 'azimuth_load', 'precurve')
            outputs = ('loads.r', 'loads.Px', 'loads.Py', 'loads.Pz', 'loads.V',
                'loads.Omega', 'loads.pitch', 'loads.azimuth')

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        if self.run_case == 'power':

            dP = self.dP
            dT = self.dT
            dQ = self.dQ

            jP = hstack([dP['dprecone'], dP['dtilt'], dP['dhubHt'], dP['dRhub'], dP['dRtip'],
                dP['dyaw'], dP['dUinf'], dP['dOmega'], dP['dpitch'], dP['dr'], dP['dchord'], dP['dtheta'],
                dP['dprecurve'], dP['dprecurveTip']])
            jT = hstack([dT['dprecone'], dT['dtilt'], dT['dhubHt'], dT['dRhub'], dT['dRtip'],
                dT['dyaw'], dT['dUinf'], dT['dOmega'], dT['dpitch'], dT['dr'], dT['dchord'], dT['dtheta'],
                dT['dprecurve'], dT['dprecurveTip']])
            jQ = hstack([dQ['dprecone'], dQ['dtilt'], dQ['dhubHt'], dQ['dRhub'], dQ['dRtip'],
                dQ['dyaw'], dQ['dUinf'], dQ['dOmega'], dQ['dpitch'], dQ['dr'], dQ['dchord'], dQ['dtheta'],
                dQ['dprecurve'], dQ['dprecurveTip']])

            J = vstack([jP, jT, jQ])


        elif self.run_case == 'loads':

            dNp = self.dNp
            dTp = self.dTp
            n = len(self.r)

            dr_dr = vstack([np.zeros(n), np.eye(n), np.zeros(n)])
            dr_dRhub = np.zeros(n+2)
            dr_dRtip = np.zeros(n+2)
            dr_dRhub[0] = 1.0
            dr_dRtip[-1] = 1.0
            dr = hstack([dr_dr, np.zeros((n+2, 2*n)), dr_dRhub, dr_dRtip, np.zeros((n+2, 8+n))])

            jNp = hstack([dNp['dr'], dNp['dchord'], dNp['dtheta'], dNp['dRhub'], dNp['dRtip'],
                dNp['dhubHt'], dNp['dprecone'], dNp['dtilt'], dNp['dyaw'], dNp['dUinf'],
                dNp['dOmega'], dNp['dpitch'], dNp['dazimuth'], dNp['dprecurve']])
            jTp = hstack([dTp['dr'], dTp['dchord'], dTp['dtheta'], dTp['dRhub'], dTp['dRtip'],
                dTp['dhubHt'], dTp['dprecone'], dTp['dtilt'], dTp['dyaw'], dTp['dUinf'],
                dTp['dOmega'], dTp['dpitch'], dTp['dazimuth'], dTp['dprecurve']])
            dPx = vstack([np.zeros(4*n+10), jNp, np.zeros(4*n+10)])
            dPy = vstack([np.zeros(4*n+10), -jTp, np.zeros(4*n+10)])
            dPz = np.zeros((n+2, 4*n+10))

            dV = np.zeros(4*n+10)
            dV[3*n+6] = 1.0
            dOmega = np.zeros(4*n+10)
            dOmega[3*n+7] = 1.0
            dpitch = np.zeros(4*n+10)
            dpitch[3*n+8] = 1.0
            dazimuth = np.zeros(4*n+10)
            dazimuth[3*n+9] = 1.0

            J = vstack([dr, dPx, dPy, dPz, dV, dOmega, dpitch, dazimuth])


        return J



class CSMDrivetrain(Component):
    def __init__(self):
        super(CSMDrivetrain, self).__init__()
        """drivetrain losses from NREL cost and scaling model"""

        self.add_param('drivetrainType', val=Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'))


        self.add_param('aeroPower', shape=20, units='W', desc='aerodynamic power')
        self.add_param('aeroTorque', shape=20, units='N*m', desc='aerodynamic torque')
        self.add_param('aeroThrust', shape=20, units='N', desc='aerodynamic thrust')
        self.add_param('ratedPower', shape=1, units='W', desc='rated power')

        self.add_output('power', shape=20, units='W', desc='total power after drivetrain losses')
        # self.add_output('rpm', shape=1, units='rpm', desc='rpm curve after drivetrain losses')

        missing_deriv_policy = 'assume_zero'

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
        # J['power', 'Pa'] = np.diag(dP_dPa)
        J['power', 'Pa'] = dP_dPa
        J['power', 'Pr'] = dP_dPr
        self.J = J


    def list_deriv_vars(self):

        inputs = ('aeroPower', 'ratedPower')
        outputs = ('power',)

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        return self.J




class WeibullCDF(Component):
    def __init__(self):
        super(WeibullCDF).__init__()
        """Weibull cumulative distribution function"""

        self.add_param('A', desc='scale factor')
        self.add_param('k', desc='shape or form factor')
        self.add_param('x')

        self.add_output('F')

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['F'] = 1.0 - np.exp(-(params['x']/params['A'])**params['k'])

    def list_deriv_vars(self):
        inputs = ('x',)
        outputs = ('F',)

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        x = params['x']
        A = params['A']
        k = params['k']
        J = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        J = {}
        # J['']
        return J


class WeibullWithMeanCDF(Component):
    def __init__(self):
        super(WeibullWithMeanCDF).__init__()
        """Weibull cumulative distribution function"""

        self.add_param('xbar', desc='mean value of distribution')
        self.add_param('k', desc='shape or form factor')
        self.add_param('x')

        self.add_output('F')

    def solve_nonlinear(self, params, unknowns, resids):

        A = params['xbar'] / gamma(1.0 + 1.0/params['k'])

        unknowns['F'] = 1.0 - np.exp(-(params['x']/A)**params['k'])


    def list_deriv_vars(self):

        inputs = ('x', 'xbar')
        outputs = ('F',)

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        x = params['x']
        k = params['k']
        A = params['xbar'] / gamma(1.0 + 1.0/k)
        dx = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        dxbar = -np.exp(-(x/A)**k)*(x/A)**(k-1)*k*x/A**2/gamma(1.0 + 1.0/k)
        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar

        return J


class RayleighCDF(Component):
    def __init(self):
        super(RayleighCDF, self).__init__()

        """Rayleigh cumulative distribution function"""

        self.add_param('xbar', shape=1, desc='mean value of distribution')
        self.add_param('x', shape=200)

        self.add_output('F', shape=20)

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['F'] = 1.0 - np.exp(-pi/4.0*(params['x']/params['xbar'])**2)

    def list_deriv_vars(self):

        inputs = ('x', 'xbar')
        outputs = ('F',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        x = params['x']
        xbar = params['xbar']
        dx = np.diag(np.exp(-pi/4.0*(x/xbar)**2)*pi*x/(2.0*xbar**2))
        dxbar = -np.exp(-pi/4.0*(x/xbar)**2)*pi*x**2/(2.0*xbar**3)
        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar

        return J

class RayleighCDF2(Component):
    def __init__(self):
        super(RayleighCDF2,  self).__init__()

        # variables
        self.add_param('xbar', shape=1, units='m/s', desc='reference wind speed (usually at hub height)')
        self.add_param('x', shape=200,  units='m/s', desc='corresponding reference height')

        # out
        self.add_output('F', shape=200, units='m/s', desc='magnitude of wind speed at each z location')


    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['F'] = 1.0 - np.exp(-pi/4.0*(params['x']/params['xbar'])**2)

    def jacobian(self, params, unknowns, resids):
        x = params['x']
        xbar = params['xbar']
        dx = np.diag(np.exp(-pi/4.0*(x/xbar)**2)*pi*x/(2.0*xbar**2))
        dxbar = -np.exp(-pi/4.0*(x/xbar)**2)*pi*x**2/(2.0*xbar**3)
        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar

def common_io_with_ccblade(assembly, varspeed, varpitch, cdf_type):

    regulated = varspeed or varpitch

    # add inputs
    assembly.add_param('r_af', units='m', desc='locations where airfoils are defined on unit radius')
    assembly.add_param('r_max_chord')
    assembly.add_param('chord_sub', units='m', desc='chord at control points')
    assembly.add_param('theta_sub', units='deg', desc='twist at control points')
    assembly.add_param('Rhub', units='m', desc='hub radius')
    assembly.add_param('Rtip', units='m', desc='tip radius')
    assembly.add_param('hubHt', units='m')
    assembly.add_param('precone', desc='precone angle', units='deg')
    assembly.add_param('tilt', val=0.0, desc='shaft tilt', units='deg')
    assembly.add_param('yaw', val=0.0, desc='yaw error', units='deg')
    assembly.add_param('airfoil_files', desc='names of airfoil file')
    assembly.add_param('idx_cylinder', desc='location where cylinder section ends on unit radius')
    assembly.add_param('B', val=3, desc='number of blades')
    assembly.add_param('rho', val=1.225, units='kg/m**3', desc='density of air')
    assembly.add_param('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air')
    assembly.add_param('shearExp', val=0.2, desc='shear exponent')
    assembly.add_param('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
    assembly.add_param('tiploss', val=True, desc='include Prandtl tip loss model')
    assembly.add_param('hubloss', val=True, desc='include Prandtl hub loss model')
    assembly.add_param('wakerotation', val=True, desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
    assembly.add_param('usecd', val=True, desc='use drag coefficient in computing induction factors')
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

    assembly.add_param('drivetrainType', val=Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive')))
    assembly.add_param('cdf_mean_wind_speed', units='m/s', desc='mean wind speed of site cumulative distribution function')

    if cdf_type == 'weibull':
        assembly.add_param('weibull_shape_factor', desc='(shape factor of weibull distribution)')

    # outputs
    assembly.add_output('AEP', units='kW*h', desc='annual energy production')
    assembly.add_output('V', units='m/s', desc='wind speeds (power curve)')
    assembly.add_output('P', units='W', desc='power (power curve)')
    assembly.add_output('diameter', units='m')
    if regulated:
        assembly.add_output('ratedConditions.V', units='m/s', desc='rated wind speed')
        assembly.add_output('ratedConditions.Omega', units='rpm', desc='rotor rotation speed at rated')
        assembly.add_output('ratedConditions.pitch', units='deg', desc='pitch setting at rated')
        assembly.add_output('ratedConditions.T', units='N', desc='rotor aerodynamic thrust at rated')
        assembly.add_output('ratedConditions.Q', units='N*m', desc='rotor aerodynamic torque at rated')



def common_configure_with_ccblade(assembly, varspeed, varpitch, cdf_type):
    common_configure(assembly, varspeed, varpitch)

    # put in parameterization for CCBlade
    assembly.add('spline', GeometrySpline())
    assembly.replace('geom', CCBladeGeometry())
    assembly.replace('analysis', CCBlade())
    assembly.replace('dt', CSMDrivetrain())
    if cdf_type == 'rayleigh':
        assembly.replace('cdf', RayleighCDF())
    elif cdf_type == 'weibull':
        assembly.replace('cdf', WeibullWithMeanCDF())


    # add spline to workflow
    assembly.driver.workflow.add('spline')

    # connections to spline
    assembly.connect('r_af', 'spline.r_af')
    assembly.connect('r_max_chord', 'spline.r_max_chord')
    assembly.connect('chord_sub', 'spline.chord_sub')
    assembly.connect('theta_sub', 'spline.theta_sub')
    assembly.connect('idx_cylinder', 'spline.idx_cylinder')
    assembly.connect('Rhub', 'spline.Rhub')
    assembly.connect('Rtip', 'spline.Rtip')

    # connections to geom
    assembly.connect('Rtip', 'geom.Rtip')
    assembly.connect('precone', 'geom.precone')

    # connections to analysis
    assembly.connect('spline.r', 'analysis.r')
    assembly.connect('spline.chord', 'analysis.chord')
    assembly.connect('spline.theta', 'analysis.theta')
    assembly.connect('spline.precurve', 'analysis.precurve')
    assembly.connect('Rhub', 'analysis.Rhub')
    assembly.connect('Rtip', 'analysis.Rtip')
    assembly.connect('hubHt', 'analysis.hubHt')
    assembly.connect('precone', 'analysis.precone')
    assembly.connect('tilt', 'analysis.tilt')
    assembly.connect('yaw', 'analysis.yaw')
    assembly.connect('airfoil_files', 'analysis.airfoil_files')
    assembly.connect('B', 'analysis.B')
    assembly.connect('rho', 'analysis.rho')
    assembly.connect('mu', 'analysis.mu')
    assembly.connect('shearExp', 'analysis.shearExp')
    assembly.connect('nSector', 'analysis.nSector')
    assembly.connect('tiploss', 'analysis.tiploss')
    assembly.connect('hubloss', 'analysis.hubloss')
    assembly.connect('wakerotation', 'analysis.wakerotation')
    assembly.connect('usecd', 'analysis.usecd')

    # connections to dt
    assembly.connect('drivetrainType', 'dt.drivetrainType')
    assembly.dt.missing_deriv_policy = 'assume_zero'  # TODO: openmdao bug remove later

    # connnections to cdf
    assembly.connect('cdf_mean_wind_speed', 'cdf.xbar')
    if cdf_type == 'weibull':
        assembly.connect('weibull_shape_factor', 'cdf.k')



class RotorAeroVSVPWithCCBlade(Group):

    def __init__(self, cdf_type='weibull'):
        self.cdf_type = cdf_type
        super(RotorAeroVSVPWithCCBlade, self).__init__()

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
    print 'AEP0 =', AEP0

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

