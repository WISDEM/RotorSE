__author__ = 'ryanbarr'

import warnings
from math import cos, sin, pi, sqrt, acos, exp
import numpy as np
import _bem
from openmdao.api import Component, ExecComp, IndepVarComp, Group, Problem, ScipyGMRES, ParallelGroup
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver #  Uncomment for optimization
from openmdao.core.mpi_wrap import MPI
from zope.interface import Interface, implements
from scipy.interpolate import RectBivariateSpline, bisplev
from airfoilprep import Airfoil
from vector_brent_omdao import Brent

class CCInit(Component):
    """
    CCInit
    Inputs: Rtip, precone, precurveTip
    Outputs: rotorR
    """
    def __init__(self):
        super(CCInit, self).__init__()
        self.add_param('Rtip', val=0.0)
        self.add_param('precone', val=0.0, units='deg')
        self.add_param('precurveTip', val=0.0)
        self.add_output('rotorR', shape=1)
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['rotorR'] = params['Rtip']*cos(np.radians(params['precone'])) + params['precurveTip']*sin(np.radians(params['precone']))
    def linearize(self, params, unknowns, resids):
        J = {}
        J['rotorR', 'precone'] = -params['Rtip']*sin(np.radians(params['precone'])) + params['precurveTip']*cos(np.radians(params['precone']))
        J['rotorR', 'precurveTip'] = sin(np.radians(params['precone']))
        J['rotorR', 'Rtip'] = cos(np.radians(params['precone']))
        return J

class WindComponents(Component):
    """
    WindComponents
    Inputs: r, precurve, Uinf, presweep, Uinf, precone, azimuth, tilt, yaw, Omega, shearExp, hubHt
    Outputs: Vx, Vy
    """
    def __init__(self, n):
        super(WindComponents, self).__init__()
        self.add_param('r', val=np.zeros(n))
        self.add_param('precurve', val=np.zeros(n))
        self.add_param('presweep', shape=n)
        self.add_param('Uinf', shape=1)
        self.add_param('precone', shape=1, units='deg')
        self.add_param('azimuth', shape=1, units='rad')
        self.add_param('tilt', shape=1, units='deg')
        self.add_param('yaw', shape=1, units='deg')
        self.add_param('Omega', shape=1)
        self.add_param('shearExp', shape=1)
        self.add_param('hubHt', shape=1)
        self.add_output('Vx', shape=n)
        self.add_output('Vy', shape=n)
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['Vx'], unknowns['Vy'] = _bem.windcomponents(params['r'], params['precurve'], params['presweep'], np.radians(params['precone']), np.radians(params['yaw']), np.radians(params['tilt']), params['azimuth'], params['Uinf'], params['Omega'], params['hubHt'], params['shearExp'])
    # def list_deriv_vars(self):
    #     inputs = ('r', 'precurve', 'presweep', 'Uinf', 'precone', 'azimuth', 'tilt', 'yaw', 'Omega', 'hubHt')
    #     outputs = ('Vx', 'Vy')
    #     return inputs, outputs
    def linearize(self, params, unknowns, resids):
        J = {}
        # y = [r, precurve, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]  (derivative order)
        n = self.n
        dy_dy = np.eye(3*n+7)
        _, Vxd, _, Vyd = _bem.windcomponents_dv(params['r'], dy_dy[:, :n], params['precurve'], dy_dy[:, n:2*n],
            params['presweep'], dy_dy[:, 2*n:3*n], params['precone'], dy_dy[:, 3*n], params['yaw'], dy_dy[:, 3*n+3],
            params['tilt'], dy_dy[:, 3*n+1], params['azimuth'], dy_dy[:, 3*n+4], params['Uinf'], dy_dy[:, 3*n+5],
            params['Omega'], dy_dy[:, 3*n+6], params['hubHt'], dy_dy[:, 3*n+2], params['shearExp'])
        dVx_dr = np.diag(Vxd[:n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dr = np.diag(Vyd[:n, :])
        dVx_dcurve = Vxd[n:2*n, :].T  # tri-diagonal  (note: dVx_j / dcurve_i  i==row)
        dVy_dcurve = Vyd[n:2*n, :].T  # off-diagonal are actually all zero, but leave for convenience
        dVx_dsweep = np.diag(Vxd[2*n:3*n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dsweep = np.diag(Vyd[2*n:3*n, :])
        # w = [r, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]
        dVx_dw = np.vstack((dVx_dr, dVx_dsweep, Vxd[3*n:, :]))
        dVy_dw = np.vstack((dVy_dr, dVy_dsweep, Vyd[3*n:, :]))
        J['Vx', 'r'] = Vxd[:n, :]
        J['Vy', 'r'] = Vyd[:n, :]
        J['Vx', 'presweep'] = Vxd[2*n:3*n, :]
        J['Vy', 'presweep'] = Vyd[2*n:3*n, :]
        J['Vx', 'precone'] = np.degrees(dVx_dw[2])
        J['Vy', 'precone'] = np.degrees(dVy_dw[2])
        J['Vx', 'tilt'] = np.degrees(dVx_dw[3])
        J['Vy', 'tilt'] = np.degrees(dVy_dw[3])
        J['Vx', 'hubHt'] = dVx_dw[4]
        J['Vy', 'hubHt'] = dVy_dw[4]
        J['Vx', 'yaw'] = np.degrees(dVx_dw[5])
        J['Vy', 'yaw'] = np.degrees(dVy_dw[5])
        J['Vx', 'azimuth'] = dVx_dw[6]
        J['Vy', 'azimuth'] = dVy_dw[6]
        J['Vx', 'Uinf'] = dVx_dw[7]
        J['Vy', 'Uinf'] = dVy_dw[7]
        J['Vx', 'Omega'] = dVx_dw[8]
        J['Vy', 'Omega'] = dVy_dw[8]
        J['Vx', 'precurve'] = dVx_dcurve
        J['Vy', 'precurve'] = dVy_dcurve
        return J

class FlowCondition(Component):
    """
    FlowCondition
    Inputs: pitch, Vx, Vy, chord, theta, rho, mu, a, ap, phi, da_dx, dap_dx
    Outputs: alpha, Re, W
    """
    def __init__(self, n):
        super(FlowCondition, self).__init__()
        self.add_param('pitch', shape=1, units='deg')
        self.add_param('Vx', shape=n)
        self.add_param('Vy', shape=n)
        self.add_param('chord', shape=n)
        self.add_param('theta', shape=n, units='deg')
        self.add_param('rho', shape=1)
        self.add_param('mu', shape=1)
        self.add_param('a', shape=n, val=np.ones(n)*0.3)
        self.add_param('ap', val=np.ones(n)*-0.3)
        self.add_param('phi', shape=n, units='rad')
        self.add_param('da_dx', val=np.zeros((n,9)))
        self.add_param('dap_dx', val=np.zeros((n,9)))
        self.add_output('alpha', shape=n, units='deg')
        self.add_output('Re', shape=n)
        self.add_output('W', shape=n)
        self.add_output('dalpha_dx', val=np.zeros((n,9)))
        self.add_output('dRe_dx', val=np.zeros((n,9)))
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True
        self.size = n
    def solve_nonlinear(self, params, unknowns, resids):
        n = self.size
        alpha, W, Re = np.zeros(n), np.zeros(n), np.zeros(n)
        dalpha_dx, dRe_dx = np.zeros((n,9)), np.zeros((n,9))
        for i in range(n):
            alpha[i], W[i], Re[i] = _bem.relativewind(params['phi'][i], params['a'][i], params['ap'][i], params['Vx'][i], params['Vy'][i], np.radians(params['pitch']), params['chord'][i], np.radians(params['theta'][i]), params['rho'], params['mu'])
            dalpha_dx[i] = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            dRe_dx[i] = np.array([0.0, Re[i]/params['chord'][i], 0.0, Re[i]*params['Vx'][i]/W[i]**2, Re[i]*params['Vy'][i]/W[i]**2, 0.0, 0.0, 0.0, 0.0])
        unknowns['dalpha_dx'] = np.degrees(dalpha_dx)
        unknowns['dRe_dx'] = dRe_dx
        unknowns['alpha'] = np.degrees(alpha)
        unknowns['W'] = W
        unknowns['Re'] = Re
    def list_deriv_vars(self):
        inputs = ('Vx', 'Vy', 'theta', 'pitch', 'rho', 'mu', 'phi', 'a', 'ap', 'chord')
        outputs = ('alpha', 'W', 'Re')
        return inputs, outputs
    def linearize(self, params, unknowns, resids):
        J = {}
        n = self.size
        for i in range(n):
            dalpha_dphi, dalpha_dtheta, dalpha_dpitch, dRe_dchord, dRe_dVx, dRe_dVy, dRe_da, dRe_dap, dRe_drho, dRe_dmu, \
            dW_dVx, dW_dVy, dW_da, dW_dap = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), \
                np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
            dx_dx = np.eye(9)
            alpha, dalpha_dx, W, dW_dx, Re, dRe_dx = _bem.relativewind_dv(params['phi'][i], dx_dx[0, :],
                params['a'][i], params['da_dx'][i], params['ap'][i], params['dap_dx'][i], params['Vx'][i], dx_dx[3, :], params['Vy'][i], dx_dx[4, :],
                params['pitch'], dx_dx[8, :], params['chord'][i], dx_dx[1, :], np.radians(params['theta'][i]), dx_dx[2, :],
                params['rho'], params['mu'])
            Vx = params['Vx'][i]
            Vy = params['Vy'][i]
            if abs(params['a'][i]) > 10:
                dW_da[i] = 0.0
                dW_dap[i] = Vy / cos(params['phi'][i])
                dW_dVx[i] = 0.0
                dW_dVy[i] = (1+params['ap'][i])/cos(params['phi'][i])
            elif abs(params['ap'][i]) > 10:
                dW_da[i] = -params['Vx'][i] / sin(params['phi'][i])
                dW_dap[i] = 0.0
                dW_dVx[i] = (1-params['a'][i])/sin(params['phi'][i])
                dW_dVy[i] = 0.0
            else:
                dW_da[i] = Vx**2*(params['a'][i] - 1) / (sqrt((Vx*(1 - params['a'][i]))**2 + (Vy*(1+params['ap'][i]))**2))
                dW_dap[i] = Vy**2*(params['ap'][i] + 1) / (sqrt((Vx*(1 - params['a'][i]))**2 + (Vy*(1+params['ap'][i]))**2))
                dW_dVx[i] = Vx*((1-params['a'][i])**2) / (sqrt((Vx*(1 - params['a'][i]))**2 + (Vy*(1+params['ap'][i]))**2))
                dW_dVy[i] = Vy*((1 + params['ap'][i])**2) / (sqrt((Vx*(1 - params['a'][i]))**2 + (Vy*(1+params['ap'][i]))**2))
            dalpha_dphi[i], dalpha_dtheta[i], dalpha_dpitch[i], dRe_dchord[i], dRe_dVx[i], dRe_dVy[i], dRe_da[i], dRe_dap[i], dRe_drho[i], dRe_dmu[i],  = 1.0*pi/180, -1.0, -1.0, Re/params['chord'][i], params['rho'] * dW_dVx[i] * params['chord'][i] / params['mu'], params['rho'] * dW_dVy[i] * params['chord'][i] / params['mu'], \
                params['rho'] * dW_da[i] * params['chord'][i] / params['mu'], params['rho'] * dW_dap[i] * params['chord'][i] / params['mu'], Re / params['rho'], -Re / params['mu']

        J['alpha', 'phi'] = np.diag(dalpha_dphi)
        J['alpha', 'theta'] = np.degrees(np.diag(dalpha_dtheta))
        J['alpha', 'pitch'] = np.degrees(dalpha_dpitch)
        J['Re', 'chord'] = np.diag(dRe_dchord)
        J['Re', 'Vx'] = np.diag(dRe_dVx)
        J['Re', 'Vy'] = np.diag(dRe_dVy)
        J['Re', 'a'] = np.diag(dRe_da)
        J['Re', 'ap'] = np.diag(dRe_dap)
        J['Re', 'rho'] = dRe_drho
        J['Re', 'mu'] = dRe_dmu
        J['W', 'Vx'] = np.diag(dW_dVx)
        J['W', 'Vy'] = np.diag(dW_dVy)
        J['W', 'a'] = np.diag(dW_da)
        J['W', 'ap'] = np.diag(dW_dap)

        return J

class AirfoilComp(Component):
    """
    AirfoilComp
    Inputs: alpha, Re, af, cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_Re
    Outputs: cl, cd
    """
    def __init__(self, n):
        super(AirfoilComp, self).__init__()
        self.add_param('alpha', shape=n, units='deg')
        self.add_param('Re', shape=n)
        self.add_param('af', val=np.ones(n), pass_by_obj=True)
        self.add_output('cl', shape=n)
        self.add_output('cd', shape=n)
        self.add_output('dcl_dalpha', shape=n)
        self.add_output('dcl_dRe', shape=n)
        self.add_output('dcd_dalpha', shape=n)
        self.add_output('dcd_dRe', shape=n)
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
        self.size = n
    def solve_nonlinear(self, params, unknowns, resids):
        for i in range(self.size):
            unknowns['cl'][i], unknowns['cd'][i] = params['af'][i].evaluate(np.radians(params['alpha'][i]), params['Re'][i])
            unknowns['dcl_dalpha'][i], unknowns['dcl_dRe'][i], unknowns['dcd_dalpha'][i], unknowns['dcd_dRe'][i] = params['af'][i].derivatives(np.radians(params['alpha'][i]), params['Re'][i])
    # def list_deriv_vars(self):
    #     inputs = ('alpha', 'Re')
    #     outputs = ('cl', 'cd')
    #     return inputs, outputs
    def linearize(self, params, unknowns, resids):
        J = {}
        J['cl', 'alpha'] = np.radians(np.diag(unknowns['dcl_dalpha']))
        J['cl', 'Re'] = np.diag(unknowns['dcl_dRe'])
        J['cd', 'alpha'] = np.radians(np.diag(unknowns['dcd_dalpha']))
        J['cd', 'Re'] = np.diag(unknowns['dcd_dRe'])
        return J

class BEM(Component):
    """
    BEM
    Inputs: pitch, Rtip, Vx, Vy, Omega, r, chord, theta, rho, mu, Rhub, alpha,
    cl, cd, B, bemoptions, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dRe_dx
    Outputs: phi, a, ap, da_dx, dap_dx
    """
    def __init__(self, n):
        super(BEM, self).__init__()

        self.add_param('pitch', shape=1, units='deg')
        self.add_param('Rtip', shape=1)
        self.add_param('Vx', shape=n)
        self.add_param('Vy', shape=n)
        self.add_param('Omega', shape=1)
        self.add_param('r', shape=n)
        self.add_param('chord', shape=n)
        self.add_param('theta', shape=n, units='deg')
        self.add_param('rho', shape=1)
        self.add_param('mu', shape=1)
        self.add_param('Rhub', shape=1)
        self.add_param('alpha', shape=n, units='deg')
        self.add_param('cl', shape=n)
        self.add_param('cd', shape=n)
        self.add_param('B', val=3, pass_by_obj=True)
        self.add_param('bemoptions', val={}, pass_by_obj=True)
        self.add_param('dcl_dalpha', shape=n)
        self.add_param('dcd_dalpha', shape=n)
        self.add_param('dalpha_dx', val=np.zeros((n,9)))
        self.add_param('dcl_dRe', shape=n)
        self.add_param('dcd_dRe', shape=n)
        self.add_param('dRe_dx', val=np.zeros((n,9)))
        self.add_output('a', shape=n, val=np.ones(n)*0.3)
        self.add_output('ap', shape=n, val=np.ones(n)*-0.3)
        self.add_output('da_dx', val=np.zeros((n,9)))
        self.add_output('dap_dx', val=np.zeros((n,9)))
        self.add_state('phi', shape=n, val=np.ones(n)*1.0e-7, units='rad')
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.size = n
        # self.fd_options['force_fd'] = True
    def solve_nonlinear(self, params, unknowns, resids):
        n = self.size
        self.dR_dx, self.da_dx, self.dap_dx = np.zeros((n,9)), np.zeros((n,9)), np.zeros((n,9))
        for i in range(n):
            dx_dx = np.eye(9)
            dcl_dx = params['dcl_dalpha'][i]*params['dalpha_dx'][i] + params['dcl_dRe'][i]*params['dRe_dx'][i]
            dcd_dx = params['dcd_dalpha'][i]*params['dalpha_dx'][i] + params['dcd_dRe'][i]*params['dRe_dx'][i]
            if not (params['Omega'] != 0):
                dR_dx = np.zeros(9)
                dR_dx[0] = 1.0  # just to prevent divide by zero
                da_dx = np.zeros(9)
                dap_dx = np.zeros(9)
            else:
                # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------
                fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(params['r'][i], params['chord'][i], params['Rhub'], params['Rtip'],
                 unknowns['phi'][i], params['cl'][i], params['cd'][i], params['B'], params['Vx'][i], params['Vy'][i], dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
                 dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **params['bemoptions'])
                unknowns['a'][i] = a
                unknowns['ap'][i] = ap
                unknowns['da_dx'][i] = da_dx
                unknowns['dap_dx'][i] = dap_dx
            self.dR_dx[i], self.da_dx[i],self.dap_dx[i] = dR_dx, da_dx, dap_dx
    def apply_nonlinear(self, params, unknowns, resids):
        n = self.size
        self.dR_dx = np.zeros((n,9))
        self.da_dx = np.zeros((n,9))
        self.dap_dx = np.zeros((n,9))
        for i in range(self.size):
            dx_dx = np.eye(9)
            dcl_dx = params['dcl_dalpha'][i]*params['dalpha_dx'][i] + params['dcl_dRe'][i]*params['dRe_dx'][i]
            dcd_dx = params['dcd_dalpha'][i]*params['dalpha_dx'][i] + params['dcd_dRe'][i]*params['dRe_dx'][i]
            if not (params['Omega'] != 0):
                dR_dx = np.zeros(9)
                dR_dx[0] = 1.0  # just to prevent divide by zero
                da_dx = np.zeros(9)
                dap_dx = np.zeros(9)
            else:
                # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------
                fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(params['r'][i], params['chord'][i], params['Rhub'], params['Rtip'],
                 unknowns['phi'][i], params['cl'][i], params['cd'][i], params['B'], params['Vx'][i], params['Vy'][i], dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
                 dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **params['bemoptions'])
                resids['phi'][i] = fzero
                resids['a'][i] = a - unknowns['a'][i]
                resids['ap'][i] = ap - unknowns['ap'][i]
            self.dR_dx[i] = dR_dx
            self.da_dx[i] = da_dx
            self.dap_dx[i] = dap_dx
    def linearize(self, params, unknowns, resids):
        # dx_dx = np.eye(9)
        # dcl_dx = params['dcl_dalpha']*params['dalpha_dx'] + params['dcl_dRe']*params['dRe_dx']
        # dcd_dx = params['dcd_dalpha']*params['dalpha_dx'] + params['dcd_dRe']*params['dRe_dx']
        #
        # if not (params['Omega'] != 0):
        #     dR_dx = np.zeros(9)
        #     dR_dx[0] = 1.0  # just to prevent divide by zero
        #     da_dx = np.zeros(9)
        #     dap_dx = np.zeros(9)
        #
        # else:
        #     # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------
        #     fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(params['r'][i], params['chord'][i], params['Rhub'], params['Rtip'],
        #          unknowns['phi'][i], params['cl'][i], params['cd'][i], params['B'], params['Vx'][i], params['Vy'][i], dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
        #          dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **params['bemoptions'])
        #
        #     resids['phi'] = fzero
        #     resids['a'] = a - unknowns['a']
        #     resids['ap'] = ap - unknowns['ap']
        #     # self.fzero = fzero
        #     # self.a = a
        #     # self.ap = ap

        # self.dR_dx = dR_dx
        # self.da_dx = da_dx
        # self.dap_dx = dap_dx
        J = {}
        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
        dR_dx = self.dR_dx
        da_dx = self.da_dx
        dap_dx = self.dap_dx

        J['phi', 'phi'] = np.diag(dR_dx[:, 0])
        J['phi', 'chord'] = np.diag(dR_dx[:, 1])
        # J['phi', 'theta'] = np.diag(dR_dx[:, 2])
        J['phi', 'Vx'] = np.diag(dR_dx[:, 3])
        J['phi', 'Vy'] = np.diag(dR_dx[:, 4])
        J['phi', 'r'] = np.diag(dR_dx[:, 5])
        J['phi', 'Rhub'] = dR_dx[:, 6]
        J['phi', 'Rtip'] = dR_dx[:, 7]
        # J['phi', 'pitch'] = np.diag(dR_dx[:, 8])

        # Vx = params['Vx']
        # Vy = params['Vy']
        # sigma_p = params['B']/2.0/pi*params['chord']/params['r']
        # cl = params['cl']
        # cd = params['cd']
        # cphi = cos(unknowns['phi'])
        # sphi = sin(unknowns['phi'])
        # B = params['B']
        # Rtip = params['Rtip']
        # r = params['r']
        # Rhub = params['Rhub']
        # factortip = B/2.0*(Rtip - r)/(r*abs(sphi))
        # Ftip = 2.0/pi*acos(exp(-factortip))
        #
        # factorhub = B/2.0*(r - Rhub)/(Rhub*abs(sphi))
        # Fhub = 2.0/pi*acos(exp(-factorhub))
        #
        # F = Ftip * Fhub
        #
        # # dR_dcl = 1/(Vy/Vx)*sigma_p/4.0/F
        # # dR_dcd = -1/(Vy/Vx)*sigma_p*cphi/4.0/F/sphi
        #
        # dR_dcl = cphi/(Vy/Vx)*(sigma_p*(sphi)/4.0/F/sphi/cphi) # 1/(Vy/Vx)*sigma_p/4.0/F
        # dR_dcd = cphi/(Vy/Vx)*(sigma_p*(-cphi)/4.0/F/sphi/cphi) # -1/(Vy/Vx)*sigma_p*cphi/4.0/F/sphi

        # J['phi', 'cl'] = dR_dcl
        # J['phi', 'cd'] = dR_dcd
        J['a', 'phi'] = np.diag(da_dx[:, 0])
        J['a', 'chord'] = np.diag(da_dx[:, 1])
        # J['a', 'theta'] = np.diag(da_dx[:, 2])
        J['a', 'Vx'] = np.diag(da_dx[:, 3])
        J['a', 'Vy'] = np.diag(da_dx[:, 4])
        J['a', 'r'] = np.diag(da_dx[:, 5])
        J['a', 'Rhub'] = da_dx[:, 6]
        J['a', 'Rtip'] = da_dx[:, 7]
        # J['a', 'pitch'] = np.diag(da_dx[:, 8])

        J['ap', 'phi'] = np.diag(dap_dx[:, 0])
        J['ap', 'chord'] = np.diag(dap_dx[:, 1])
        # J['ap', 'theta'] = np.diag(dap_dx[:, 2])
        J['ap', 'Vx'] = np.diag(dap_dx[:, 3])
        J['ap', 'Vy'] = np.diag(dap_dx[:, 4])
        J['ap', 'r'] = np.diag(dap_dx[:, 5])
        J['ap', 'Rhub'] = dap_dx[:, 6]
        J['ap', 'Rtip'] = dap_dx[:, 7]
        # J['ap', 'pitch'] = np.diag(dap_dx[:, 8])

        return J




class MUX(Component):
    """
    MUX - Combines all the sections into single variables of arrays

    Inputs: n - Number of sections analyzed across blade

    Outputs: phi, cl, cd, W

    """
    def __init__(self, n):
        super(MUX, self).__init__()
        for i in range(n):
            self.add_param('phi'+str(i+1), val=0.0)
            self.add_param('cl'+str(i+1), val=0.0)
            self.add_param('cd'+str(i+1), val=0.0)
            self.add_param('W'+str(i+1), val=0.0)

        self.add_output('phi', val=np.zeros(n))
        self.add_output('cl', val=np.zeros(n))
        self.add_output('cd', val=np.zeros(n))
        self.add_output('W', val=np.zeros(n))

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):
        for i in range(self.n):
            unknowns['phi'][i] = params['phi'+str(i+1)]
            unknowns['cl'][i] = params['cl'+str(i+1)]
            unknowns['cd'][i] = params['cd'+str(i+1)]
            unknowns['W'][i] = params['W'+str(i+1)]

    def linearize(self, params, unknowns, resids):
        n = self.n
        J = {}
        for i in range(n):
            zeros = np.zeros(n)
            zeros[i] = 1
            J['phi', 'phi'+str(i+1)] = zeros
            J['cl', 'cl'+str(i+1)] = zeros
            J['cd', 'cd'+str(i+1)] = zeros
            J['W', 'W'+str(i+1)] = zeros
        return J

class MUX_POWER(Component):
    """
    MUX_POWER - Combines all the sections into single variables of arrays
    Inputs: n2 - Number of Uinfs
    Outputs: CT, CQ, CP, T, Q, P

    """
    def __init__(self, n2):
        super(MUX_POWER, self).__init__()
        for i in range(n2):
            self.add_param('CT'+str(i+1), val=0.0)
            self.add_param('CQ'+str(i+1), val=0.0)
            self.add_param('CP'+str(i+1), val=0.0)
            self.add_param('T'+str(i+1), val=0.0)
            self.add_param('Q'+str(i+1), val=0.0)
            self.add_param('P'+str(i+1), val=0.0)

        self.add_output('CT', val=np.zeros(n2))
        self.add_output('CQ', val=np.zeros(n2))
        self.add_output('CP', val=np.zeros(n2))
        self.add_output('T', val=np.zeros(n2))
        self.add_output('Q', val=np.zeros(n2))
        self.add_output('P', val=np.zeros(n2))

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.n2 = n2

    def solve_nonlinear(self, params, unknowns, resids):
        for i in range(self.n2):
            unknowns['CT'][i] = params['CT'+str(i+1)]
            unknowns['CQ'][i] = params['CQ'+str(i+1)]
            unknowns['CP'][i] = params['CP'+str(i+1)]
            unknowns['T'][i] = params['T'+str(i+1)]
            unknowns['Q'][i] = params['Q'+str(i+1)]
            unknowns['P'][i] = params['P'+str(i+1)]

    def linearize(self, params, unknowns, resids):
        n2 = self.n2
        J = {}
        for i in range(n2):
            zeros = np.zeros(n2)
            zeros[i] = 1
            J['CT', 'CT'+str(i+1)] = zeros
            J['CQ', 'CQ'+str(i+1)] = zeros
            J['CP', 'CP'+str(i+1)] = zeros
            J['T', 'T'+str(i+1)] = zeros
            J['Q', 'Q'+str(i+1)] = zeros
            J['P', 'P'+str(i+1)] = zeros
        return J

class DistributedAeroLoads(Component):
    """
    DistributedAeroLoads

    Inputs: n - Number of sections analyzed across blade

    Outputs: Np, Tp

    """
    def __init__(self, n):
        super(DistributedAeroLoads, self).__init__()
        self.add_param('chord', shape=n)
        self.add_param('rho', shape=1)
        self.add_param('phi', val=np.zeros(n), units='rad')
        self.add_param('cl', val=np.zeros(n))
        self.add_param('cd', val=np.zeros(n))
        self.add_param('W', val=np.zeros(n))

        self.add_output('Np', shape=n)
        self.add_output('Tp', shape=n)

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):

        chord = params['chord']
        rho = params['rho']
        phi = params['phi']
        cl = params['cl']
        cd = params['cd']
        W = params['W']
        n = self.n
        Np = np.zeros(n)
        Tp = np.zeros(n)
        self.dNp_dcl, self.dTp_dcl, self.dNp_dcd, self.dTp_dcd, self.dNp_dphi, self.dTp_dphi, self.dNp_drho, self.dTp_drho, self.dNp_dW, self.dTp_dW, self.dNp_dchord, self.dTp_dchord = \
            np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        for i in range(n):
            cphi = cos(phi[i])
            sphi = sin(phi[i])

            cn = cl[i]*cphi + cd[i]*sphi  # these expressions should always contain drag
            ct = cl[i]*sphi - cd[i]*cphi
            q = 0.5*rho*W[i]**2
            Np[i] = cn*q*chord[i]
            Tp[i] = ct*q*chord[i]

            self.dNp_dcl[i] = cphi*q*chord[i]
            self.dTp_dcl[i] = sphi*q*chord[i]
            self.dNp_dcd[i] = sphi*q*chord[i]
            self.dTp_dcd[i] = -cphi*q*chord[i]
            self.dNp_dphi[i] = (-cl[i]*sphi + cd[i]*cphi)*q*chord[i]
            self.dTp_dphi[i] = (cl[i]*cphi + cd[i]*sphi)*q*chord[i]
            self.dNp_drho[i] = cn*q/rho*chord[i]
            self.dTp_drho[i] = ct*q/rho*chord[i]
            self.dNp_dW[i] = cn*0.5*rho*2*W[i]*chord[i]
            self.dTp_dW[i] = ct*0.5*rho*2*W[i]*chord[i]
            self.dNp_dchord[i] = cn*q
            self.dTp_dchord[i] = ct*q

        unknowns['Np'] = Np
        unknowns['Tp'] = Tp


    def linearize(self, params, unknowns, resids):

        J = {}
        # # add chain rule for conversion to radians
        ## TODO: Check radian conversion
        # ridx = [2, 6, 7, 9, 10, 13]
        # dNp_dz[ridx, :] *= pi/180.0
        # dTp_dz[ridx, :] *= pi/180.0
        J['Np', 'cl'] = np.diag(self.dNp_dcl)
        J['Tp', 'cl'] = np.diag(self.dTp_dcl)
        J['Np', 'cd'] = np.diag(self.dNp_dcd)
        J['Tp', 'cd'] = np.diag(self.dTp_dcd)
        J['Np', 'phi'] = np.diag(self.dNp_dphi)
        J['Tp', 'phi'] = np.diag(self.dTp_dphi)
        J['Np', 'rho'] = self.dNp_drho
        J['Tp', 'rho'] = self.dTp_drho
        J['Np', 'W'] = np.diag(self.dNp_dW) # rho*W*cn*chord
        J['Tp', 'W'] = np.diag(self.dTp_dW)
        J['Np', 'chord'] = np.diag(self.dNp_dchord)
        J['Tp', 'chord'] = np.diag(self.dTp_dchord)
        return J

class CCEvaluate(Component):
    """
    CCEvaluate
    Inputs: Uinf, Rtip, Omega, r, B, precurve, presweep, presweepTip, precurveTip, rho
    precone, Rhub, nSector, rotorR
    Outputs: CP, CT, CQ, P, T, Q
    """
    def __init__(self, n, nSector):
        super(CCEvaluate, self).__init__()

        self.add_param('Uinf', val=10.0)
        self.add_param('Rtip', val=63.)
        self.add_param('Omega', shape=1)
        self.add_param('r', shape=n)
        self.add_param('B', val=3, pass_by_obj=True)
        self.add_param('precurve', shape=n)
        self.add_param('presweep', shape=n)
        self.add_param('presweepTip', shape=1)
        self.add_param('precurveTip', shape=1)
        self.add_param('rho', shape=1)
        self.add_param('precone', shape=1, units='deg')
        self.add_param('Rhub', shape=1)
        self.add_param('nSector', val=4, pass_by_obj=True)
        self.add_param('rotorR', shape=1)
        for i in range(nSector):
            self.add_param('Np'+str(i+1), val=np.zeros(n))
            self.add_param('Tp'+str(i+1), val=np.zeros(n))

        self.add_output('P', shape=1)
        self.add_output('T', shape=1)
        self.add_output('Q', shape=1)
        self.add_output('CP', shape=1)
        self.add_output('CT', shape=1)
        self.add_output('CQ', shape=1)

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):

        r = params['r']
        precurve = params['precurve']
        presweep = params['presweep']
        precone = np.radians(params['precone'])
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        precurveTip = params['precurveTip']
        presweepTip = params['presweepTip']
        nSector = params['nSector']
        Uinf = params['Uinf']
        Omega = params['Omega']
        B = params['B']
        rho = params['rho']
        rotorR = params['rotorR']
        nsec = int(nSector)
        npts = 1 #len(Uinf)
        n = self.n
        Np = {}
        Tp = {}
        for i in range(nSector):
            Np['Np' + str(i+1)] = params['Np' + str(i+1)]
            Tp['Tp' + str(i+1)] = params['Tp' + str(i+1)]
        T = np.zeros(npts)
        Q = np.zeros(npts)
        self.dT_dr, self.dQ_dr, self.dP_dr, self.dT_dprecurve, self.dQ_dprecurve, self.dP_dprecurve, self.dT_dpresweep, self.dQ_dpresweep, self.dP_dpresweep, \
        self.dT_dprecone, self.dQ_dprecone, self.dP_dprecone, self.dT_dRhub, self.dQ_dRhub, self.dP_dRhub, self.dT_dRtip, self.dQ_dRtip, self.dP_dRtip, \
        self.dT_dprecurvetip, self.dQ_dprecurvetip, self.dP_dprecurvetip, self.dT_dpresweeptip, self.dQ_dpresweeptip, self.dP_dpresweeptip = \
            np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), \
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.


        self.dT_dNp_tot, self.dQ_dNp_tot, self.dP_dNp_tot, self.dT_dTp_tot, self.dQ_dTp_tot, self.dP_dTp_tot = {}, {}, {}, {}, {}, {}

        args = (r, precurve, presweep, precone, Rhub, Rtip, precurveTip, presweepTip)

        for i in range(npts):  # iterate across conditions

            for j in range(nsec):  # integrate across azimuth
                Np1 = Np['Np'+str(j+1)]
                Tp1 = Tp['Tp'+str(j+1)]

                Tsub, Qsub = _bem.thrusttorque(Np1, Tp1, *args)

                T[i] += B * Tsub / nsec
                Q[i] += B * Qsub / nsec

                Tb = np.array([1, 0])
                Qb = np.array([0, 1])
                Npb, Tpb, rb, precurveb, presweepb, preconeb, Rhubb, Rtipb, precurvetipb, presweeptipb = \
                    _bem.thrusttorque_bv(Np1, Tp1, r, precurve, presweep, precone, Rhub, Rtip, precurveTip, presweepTip, Tb, Qb)

                dT_dNp1 = Npb[0, :]
                dQ_dNp1 = Npb[1, :]
                dT_dTp1 = Tpb[0, :]
                dQ_dTp1 = Tpb[1, :]
                dP_dTp1 = Tpb[1, :] * Omega * pi / 30.0
                dP_dNp1 = Npb[1, :] * Omega * pi / 30.0
                dT_dr1 = rb[0, :]
                dQ_dr1 = rb[1, :]
                dP_dr1 = rb[1, :] * Omega * pi / 30.0
                dT_dprecurve1 = precurveb[0, :]
                dQ_dprecurve1 = precurveb[1, :]
                dP_dprecurve1 = precurveb[1, :] * Omega * pi / 30.0
                dT_dpresweep1 = presweepb[0, :]
                dQ_dpresweep1 = presweepb[1, :]
                dP_dpresweep1 = presweepb[1, :] * Omega * pi / 30.0
                dT_dprecone1 = preconeb[0]
                dQ_dprecone1 = preconeb[1]
                dP_dprecone1 = preconeb[1] * Omega * pi / 30.0
                dT_dRhub1 = Rhubb[0]
                dQ_dRhub1 = Rhubb[1]
                dP_dRhub1 = Rhubb[1] * Omega * pi / 30.0
                dT_dRtip1 = Rtipb[0]
                dQ_dRtip1 = Rtipb[1]
                dP_dRtip1 = Rtipb[1] * Omega * pi / 30.0
                dT_dprecurvetip1 = precurvetipb[0]
                dQ_dprecurvetip1 = precurvetipb[1]
                dP_dprecurvetip1 = precurvetipb[1] * Omega * pi / 30.0
                dT_dpresweeptip1 = presweeptipb[0]
                dQ_dpresweeptip1 = presweeptipb[1]
                dP_dpresweeptip1 = presweeptipb[1] * Omega * pi / 30.0

                dT_dNp, dT_dTp, dQ_dNp, dQ_dTp, dP_dNp, dP_dTp = np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n))

                dT_dNp += [x * B / nsec for x in dT_dNp1]
                self.dT_dNp_tot['Np'+str(j+1)] = dT_dNp
                dQ_dNp += [x * B / nsec for x in dQ_dNp1]
                self.dQ_dNp_tot['Np'+str(j+1)] = dQ_dNp
                dP_dNp += [x * B / nsec for x in dP_dNp1]
                self.dP_dNp_tot['Np'+str(j+1)] = dP_dNp
                dT_dTp += [x * B / nsec for x in dT_dTp1]
                self.dT_dTp_tot['Tp'+str(j+1)] = dT_dTp
                dQ_dTp += [x * B / nsec for x in dQ_dTp1]
                self.dQ_dTp_tot['Tp'+str(j+1)] = dQ_dTp
                dP_dTp += [x * B / nsec for x in dP_dTp1]
                self.dP_dTp_tot['Tp'+str(j+1)] = dP_dTp
                self.dT_dr += [x * B / nsec for x in dT_dr1]
                self.dQ_dr += [x * B / nsec for x in dQ_dr1]
                self.dP_dr += [x * B / nsec for x in dP_dr1]
                self.dT_dprecurve += [x * B / nsec for x in dT_dprecurve1]
                self.dQ_dprecurve += [x * B / nsec for x in dQ_dprecurve1]
                self.dP_dprecurve += [x * B / nsec for x in dP_dprecurve1]
                self.dT_dpresweep += [x * B / nsec for x in dT_dpresweep1]
                self.dQ_dpresweep += [x * B / nsec for x in dQ_dpresweep1]
                self.dP_dpresweep += [x * B / nsec for x in dP_dpresweep1]
                self.dT_dprecone += dT_dprecone1 * B / nsec
                self.dQ_dprecone += dQ_dprecone1 * B / nsec
                self.dP_dprecone += dP_dprecone1 * B / nsec
                self.dT_dRhub += dT_dRhub1 * B / nsec
                self.dQ_dRhub += dQ_dRhub1 * B / nsec
                self.dP_dRhub += dP_dRhub1 * B / nsec
                self.dT_dRtip += dT_dRtip1 * B / nsec
                self.dQ_dRtip += dQ_dRtip1 * B / nsec
                self.dP_dRtip += dP_dRtip1 * B / nsec
                self.dT_dprecurvetip += dT_dprecurvetip1 * B / nsec
                self.dQ_dprecurvetip += dQ_dprecurvetip1 * B / nsec
                self.dP_dprecurvetip += dP_dprecurvetip1 * B / nsec
                self.dT_dpresweeptip += dT_dpresweeptip1 * B / nsec
                self.dQ_dpresweeptip += dQ_dpresweeptip1 * B / nsec
                self.dP_dpresweeptip += dP_dpresweeptip1 * B / nsec

        # Power
        P = Q * Omega*pi/30.0  # RPM to rad/s

        # normalize if necessary
        q = 0.5 * rho * Uinf**2
        A = pi * rotorR**2
        CP = P / (q * A * Uinf)
        CT = T / (q * A)
        CQ = Q / (q * rotorR * A)
        unknowns['CP'] = CP[0]
        unknowns['CT'] = CT[0]
        unknowns['CQ'] = CQ[0]
        unknowns['P'] = P[0]
        unknowns['T'] = T[0]
        unknowns['Q'] = Q[0]

        self.dCP_drho = -CP / rho
        self.dCT_drho = -CT / rho
        self.dCQ_drho = -CQ / rho
        self.dCP_drotorR = -2 * P / (q * pi * rotorR**3 * Uinf)
        self.dCT_drotorR = -2 * T / (q * pi * rotorR**3)
        self.dCQ_drotorR = -3 * Q / (q * pi * rotorR**4)
        self.dCP_dUinf = -3 * P / (0.5 * rho * Uinf**4 * A)
        self.dCT_dUinf = -2 * T / (0.5 * rho * Uinf**3 * A)
        self.dCQ_dUinf = -2 * Q / (0.5 * rho * Uinf**3 * rotorR * A)
        self.dP_dOmega = Q * pi / 30.0

    def linearize(self, params, unknowns, resids):
        J = {}
        Uinf = params['Uinf']
        rotorR = params['rotorR']

        q = 0.5 * params['rho'] * Uinf**2
        A = pi * rotorR**2

        nSector = int(params['nSector'])

        for i in range(nSector):
            J['CP', 'Np'+str(i+1)] = self.dP_dNp_tot['Np'+str(i+1)] / (q * A * Uinf)
            J['CP', 'Tp'+str(i+1)] = self.dP_dTp_tot['Tp'+str(i+1)] / (q * A * Uinf)
            J['CT', 'Np'+str(i+1)] = self.dT_dNp_tot['Np'+str(i+1)] / (q * A)
            J['CT', 'Tp'+str(i+1)] = self.dT_dTp_tot['Tp'+str(i+1)] / (q * A)
            J['CQ', 'Np'+str(i+1)] = self.dQ_dNp_tot['Np'+str(i+1)] / (q * rotorR * A)
            J['CQ', 'Tp'+str(i+1)] = self.dQ_dTp_tot['Tp'+str(i+1)] / (q * rotorR * A)
            J['P', 'Np'+str(i+1)] = self.dP_dNp_tot['Np'+str(i+1)]
            J['P', 'Tp'+str(i+1)] = self.dP_dTp_tot['Tp'+str(i+1)]
            J['T', 'Np'+str(i+1)] = self.dT_dNp_tot['Np'+str(i+1)]
            J['T', 'Tp'+str(i+1)] = self.dT_dTp_tot['Tp'+str(i+1)]
            J['Q', 'Np'+str(i+1)] = self.dQ_dNp_tot['Np'+str(i+1)]
            J['Q', 'Tp'+str(i+1)] = self.dQ_dTp_tot['Tp'+str(i+1)]

        J['CP', 'Uinf'] = self.dCP_dUinf
        J['CP', 'Rtip'] = self.dP_dRtip / (q * A * Uinf)
        J['CP', 'Omega'] = self.dP_dOmega / (q * A * Uinf)
        J['CP', 'r'] = self.dP_dr / (q * A * Uinf)
        J['CP', 'precurve'] = self.dP_dprecurve / (q * A * Uinf)
        J['CP', 'presweep'] = self.dP_dpresweep / (q * A * Uinf)
        J['CP', 'presweepTip'] = self.dP_dpresweeptip / (q * A * Uinf)
        J['CP', 'precurveTip'] = self.dP_dprecurvetip / (q * A * Uinf)
        J['CP', 'precone'] = np.degrees(self.dP_dprecone / (q * A * Uinf))
        J['CP', 'rho'] = self.dCP_drho
        J['CP', 'Rhub'] = self.dP_dRhub / (q * A * Uinf)
        J['CP', 'rotorR'] = self.dCP_drotorR

        J['CT', 'Uinf'] = self.dCT_dUinf
        J['CT', 'Rtip'] = self.dT_dRtip / (q * A)
        J['CT', 'Omega'] = 0.0
        J['CT', 'r'] = (self.dT_dr / (q * A))
        J['CT', 'precurve'] = self.dT_dprecurve / (q * A)
        J['CT', 'presweep'] = self.dT_dpresweep / (q * A)
        J['CT', 'presweepTip'] = self.dT_dpresweeptip / (q * A)
        J['CT', 'precurveTip'] = self.dT_dprecurvetip / (q * A)
        J['CT', 'precone'] = np.degrees(self.dT_dprecone / (q * A))
        J['CT', 'rho'] = self.dCT_drho
        J['CT', 'Rhub'] = self.dT_dRhub / (q * A)
        J['CT', 'rotorR'] = self.dCT_drotorR

        J['CQ', 'Uinf'] = self.dCQ_dUinf
        J['CQ', 'Rtip'] = self.dQ_dRtip / (q * rotorR * A)
        J['CQ', 'Omega'] = 0.0
        J['CQ', 'r'] = self.dQ_dr / (q * rotorR * A)
        J['CQ', 'precurve'] = self.dQ_dprecurve / (q * rotorR * A)
        J['CQ', 'presweep'] = self.dQ_dpresweep/ (q * rotorR * A)
        J['CQ', 'presweepTip'] = self.dQ_dpresweeptip / (q * rotorR * A)
        J['CQ', 'precurveTip'] = self.dQ_dprecurvetip / (q * rotorR * A)
        J['CQ', 'precone'] = np.degrees(self.dQ_dprecone / (q * rotorR * A))
        J['CQ', 'rho'] = self.dCQ_drho
        J['CQ', 'Rhub'] = self.dQ_dRhub / (q * rotorR * A)
        J['CQ', 'rotorR'] = self.dCQ_drotorR

        J['P', 'Uinf'] = 0.0
        J['P', 'Rtip'] = self.dP_dRtip
        J['P', 'Omega'] = self.dP_dOmega
        J['P', 'r'] = self.dP_dr
        J['P', 'precurve'] = self.dP_dprecurve
        J['P', 'presweep'] = self.dP_dpresweep
        J['P', 'presweepTip'] = self.dP_dpresweeptip
        J['P', 'precurveTip'] = self.dP_dprecurvetip
        J['P', 'precone'] = np.degrees(self.dP_dprecone)
        J['P', 'rho'] = 0.0
        J['P', 'Rhub'] = self.dP_dRhub
        J['P', 'rotorR'] = 0.0

        J['T', 'Uinf'] = 0.0
        J['T', 'Rtip'] = self.dT_dRtip
        J['T', 'Omega'] = 0.0
        J['T', 'r'] = self.dT_dr
        J['T', 'precurve'] = self.dT_dprecurve
        J['T', 'presweep'] = self.dT_dpresweep
        J['T', 'presweepTip'] = self.dT_dpresweeptip
        J['T', 'precurveTip'] = self.dT_dprecurvetip
        J['T', 'precone'] = np.degrees(self.dT_dprecone)
        J['T', 'rho'] = 0.0
        J['T', 'Rhub'] = self.dT_dRhub
        J['T', 'rotorR'] = 0

        J['Q', 'Uinf'] = 0.0
        J['Q', 'Rtip'] = self.dQ_dRtip
        J['Q', 'Omega'] = 0.0
        J['Q', 'r'] = self.dQ_dr
        J['Q', 'precurve'] = self.dQ_dprecurve
        J['Q', 'presweep'] = self.dQ_dpresweep
        J['Q', 'presweepTip'] = self.dQ_dpresweeptip
        J['Q', 'precurveTip'] = self.dQ_dprecurvetip
        J['Q', 'precone'] = np.degrees(self.dQ_dprecone)
        J['Q', 'rho'] = 0.0
        J['Q', 'Rhub'] = self.dQ_dRhub
        J['Q', 'rotorR'] = 0
        return J

class BrentGroup(Group):
    def __init__(self, n):
        super(BrentGroup, self).__init__()
        epsilon = 1e-6
        phi_lower = epsilon
        phi_upper = pi/2
        # # if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # an uncommon but possible case
        # #
        # #         if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
        # #             phi_lower = -pi/4
        # #             phi_upper = -epsilon
        # #         else:
        # #             phi_lower = pi/2
        # #             phi_upper = pi - epsilon
        self.add('flow', FlowCondition(n), promotes=['*'])
        self.add('airfoils', AirfoilComp(n), promotes=['*'])
        self.add('bem', BEM(n), promotes=['*'])
        self.ln_solver = ScipyGMRES()
        self.nl_solver = Brent()
        self.nl_solver.options['lower_bound'] = phi_lower
        self.nl_solver.options['upper_bound'] = phi_upper
        self.nl_solver.options['state_var'] = 'phi'
    def list_deriv_vars(self):
        inputs = ('Vx', 'Vy', 'chord', 'theta', 'pitch', 'rho', 'mu')
        outputs = ('phi', 'a', 'ap')
        return inputs, outputs

class Loads(Group):
    def __init__(self, n):
        super(Loads, self).__init__()
        self.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        self.add('Uinf', IndepVarComp('Uinf', 0.0), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', 0.0), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0), promotes=['*'])
        self.add('azimuth', IndepVarComp('azimuth', 0.0), promotes=['*'])
        self.add('Omega', IndepVarComp('Omega', 0.0), promotes=['*'])
        self.add('r', IndepVarComp('r', val=np.zeros(n)), promotes=['*'])
        self.add('chord', IndepVarComp('chord', val=np.zeros(n)), promotes=['*'])
        self.add('theta', IndepVarComp('theta', val=np.zeros(n)), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', val=np.zeros(n)), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', val=np.zeros(n)), promotes=['*'])
        self.add('af', IndepVarComp('af', val=np.zeros(n), pass_by_obj=True), promotes=['*'])
        self.add('bemoptions', IndepVarComp('bemoptions', {}, pass_by_obj=True), promotes=['*'])
        self.add('init', CCInit(), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('mux', MUX(n), promotes=['*'])
        self.add('brent', BrentGroup(n), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'af', 'bemoptions'])
        self.add('loads', DistributedAeroLoads(n), promotes=['*'])
        self.add('obj_cmp', ExecComp('obj = -max(Np)', Np=np.zeros(n)), promotes=['*'])

class Sweep(Group):
    def __init__(self, n):
        super(Sweep, self).__init__()
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('brent', BrentGroup(n), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'bemoptions', 'af', 'theta', 'r', 'chord', 'phi', 'cl', 'cd', 'W', 'theta', 'Vx', 'Vy'])
        self.add('loads', DistributedAeroLoads(n), promotes=['chord', 'rho', 'phi', 'cl', 'cd', 'W'])

class SweepGroup(Group):
    def __init__(self, nSector, n):
        super(SweepGroup, self).__init__()
        self.add('init', CCInit(), promotes=['*'])
        for i in range(nSector):
            self.add('group'+str(i+1), Sweep(n), promotes=['r', 'Uinf', 'pitch', 'Rtip', 'Omega', 'chord', 'rho', 'mu', 'Rhub', 'hubHt', 'precurve', 'presweep', 'precone', 'tilt', 'yaw', 'shearExp', 'B', 'bemoptions', 'af', 'theta'])


class FlowSweep(Group):

    def __init__(self, nSector, n):
        super(FlowSweep, self).__init__()
        self.add('load_group', SweepGroup(nSector, n), promotes=['Uinf', 'Omega', 'pitch', 'Rtip', 'r', 'chord',  'rho', 'mu', 'Rhub', 'rotorR', 'precurve', 'presweep', 'precurveTip', 'precone', 'tilt', 'yaw', 'shearExp', 'hubHt', 'B', 'af', 'bemoptions', 'theta'])
        self.add('eval', CCEvaluate(n, nSector), promotes=['Uinf', 'Rtip', 'Omega', 'r', 'Rhub', 'B', 'precurve', 'presweep', 'presweepTip', 'precurveTip', 'precone', 'nSector', 'rotorR', 'rho', 'CP', 'CT', 'CQ', 'P', 'T', 'Q'])
        for i in range(nSector):
            self.connect('load_group.group' + str(i+1) + '.loads.Np', 'eval.Np' + str(i+1))
            self.connect('load_group.group' + str(i+1) + '.loads.Tp', 'eval.Tp' + str(i+1))

class CCBlade(Group):

    def __init__(self, nSector, n, n2):
        super(CCBlade, self).__init__()

        self.add('Uinf', IndepVarComp('Uinf', np.zeros(n2)), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', np.zeros(n2)), promotes=['*'])
        self.add('Omega', IndepVarComp('Omega', np.zeros(n2)), promotes=['*'])

        self.add('r', IndepVarComp('r', np.zeros(n)), promotes=['*'])
        self.add('chord', IndepVarComp('chord', np.zeros(n)), promotes=['*'])
        self.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        self.add('theta', IndepVarComp('theta', np.zeros(n)), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', np.zeros(n)), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', np.zeros(n)), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0), promotes=['*'])
        self.add('mu', IndepVarComp('mu', 0.0), promotes=['*'])
        self.add('rho', IndepVarComp('rho', 0.0), promotes=['*'])
        self.add('shearExp', IndepVarComp('shearExp', 0.0), promotes=['*'])
        self.add('af', IndepVarComp('af', np.zeros(n), pass_by_obj=True), promotes=['*'])
        self.add('B', IndepVarComp('B', 3, pass_by_obj=True), promotes=['*'])
        self.add('nSector', IndepVarComp('nSector', 4, pass_by_obj=True), promotes=['*'])
        self.add('bemoptions', IndepVarComp('bemoptions', {}, pass_by_obj=True), promotes=['*'])
        azimuth = np.zeros(nSector)
        for i in range(nSector):
            azimuth[i] = pi/180.0*360.0*float(i)/nSector
        self.add('azimuth', IndepVarComp('azimuth', azimuth), promotes=['*'])
        self.add('mux_power', MUX_POWER(n2), promotes=['*'])

        pg = self.add('parallel', ParallelGroup(), promotes=['*'])
        for i in range(n2):
            pg.add('results'+str(i), FlowSweep(nSector, n), promotes=['Rhub', 'Rtip', 'precone', 'tilt', 'hubHt', 'precurve', 'presweep', 'yaw', 'precurveTip', 'presweepTip', 'af', 'bemoptions', 'B', 'rho', 'mu', 'shearExp', 'nSector', 'r', 'chord', 'theta']) #, 'CP', 'CT', 'CQ', 'P', 'T', 'Q'])
            self.connect('Uinf', 'results'+str(i)+'.Uinf', src_indices=[i])
            self.connect('pitch', 'results'+str(i)+'.pitch', src_indices=[i])
            self.connect('Omega', 'results'+str(i)+'.Omega', src_indices=[i])
            self.connect('results'+str(i)+'.CT', 'CT'+str(i+1))
            self.connect('results'+str(i)+'.CQ', 'CQ'+str(i+1))
            self.connect('results'+str(i)+'.CP', 'CP'+str(i+1))
            self.connect('results'+str(i)+'.T', 'T'+str(i+1))
            self.connect('results'+str(i)+'.Q', 'Q'+str(i+1))
            self.connect('results'+str(i)+'.P', 'P'+str(i+1))
            for j in range(nSector):
                self.connect('azimuth', 'results'+str(i)+'.load_group.group'+str(j+1)+'.azimuth', src_indices=[j])
        self.add('obj_cmp', ExecComp('obj = -max(CP)', CP=np.zeros(n2)), promotes=['*'])

class Loads_for_RotorSE(Component):
    def __init__(self, n):
        super(Loads_for_RotorSE, self).__init__()
        self.add_param('Rhub', shape=1, units='m')
        self.add_param('r', val=np.zeros(n), units='m')
        self.add_param('Rtip', shape=1, units='m')
        self.add_param('Np', shape=n, units='N/m')
        self.add_param('Tp', shape=n, units='N/m')
        self.add_param('Uinf', shape=1, units='m/s')
        self.add_param('Omega', shape=1, units='rpm')
        self.add_param('pitch', shape=1, units='deg')
        self.add_param('azimuth', shape=1, units='deg')

        self.add_output('loads:r', shape=n+2, units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads:Px', shape=n+2, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads:Py', shape=n+2, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads:Pz', shape=n+2, units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_output('loads:V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_output('loads:Omega', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_output('loads:pitch', shape=1, units='deg', desc='pitch angle')
        self.add_output('loads:azimuth', shape=1, units='deg', desc='azimuthal angle')

        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['loads:r'] = np.concatenate([[params['Rhub']], params['r'], [params['Rtip']]])
        Np = np.concatenate([[0.0], params['Np'], [0.0]])
        Tp = np.concatenate([[0.0], params['Tp'], [0.0]])

        # conform to blade-aligned coordinate system
        unknowns['loads:Px'] = Np
        unknowns['loads:Py'] = -Tp
        unknowns['loads:Pz'] = 0*Np

        # return other outputs needed
        unknowns['loads:V'] = params['Uinf']
        unknowns['loads:Omega'] = params['Omega']
        unknowns['loads:pitch'] = params['pitch']
        unknowns['loads:azimuth'] = params['azimuth']

    def linearize(self, params, unknowns, resids):
        n = self.n

        dr_dr = np.vstack([np.zeros(n), np.eye(n), np.zeros(n)])
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

        J = {}
        zero = np.zeros(17)
        J['loads:r', 'r'] = dr_dr
        J['loads:r', 'Rhub'] = dr_dRhub
        J['loads:r', 'Rtip'] = dr_dRtip
        J['loads:Px', 'Np'] = np.vstack([zero, np.eye(n), zero])
        J['loads:Py', 'Tp'] = np.vstack([zero, -np.eye(n), zero])
        J['loads:V', 'Uinf'] = 1.0
        J['loads:Omega', 'Omega'] = 1.0
        J['loads:pitch', 'pitch'] = 1.0
        J['loads:azimuth', 'azimuth'] = 1.0

        return J
    
class PassThrough(Component):
    def __init__(self, n2):
        super(PassThrough, self).__init__()
        self.add_param('Uinf_in', shape=n2)
        self.add_param('pitch_in', shape=n2)
        self.add_param('Omega_in', shape=n2)

        self.add_output('Uinf', shape=n2)
        self.add_output('pitch', shape=n2)
        self.add_output('Omega', shape=n2)
        self.n2 = n2
    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['Uinf'] = params['Uinf_in']
        unknowns['pitch'] = params['pitch_in']
        unknowns['Omega'] = params['Omega_in']
    def linearize(self, params, unknowns, resids):
        J = {}
        if self.n2 == 1:
            J['Uinf', 'Uinf_in'] = 1.0
            J['pitch', 'pitch_in'] = 1.0
            J['Omega', 'Omega_in'] = 1.0
        else:
            J['Uinf', 'Uinf_in'] = np.diag(np.ones(len(params['Uinf_in'])))
            J['pitch', 'pitch_in'] = np.diag(np.ones(len(params['pitch_in'])))
            J['Omega', 'Omega_in'] = np.diag(np.ones(len(params['Omega_in'])))
        return J

class Loads_for_RotorSE(Component):
    def __init__(self, n):
        super(Loads_for_RotorSE, self).__init__()
        self.add_param('Rhub', shape=1, units='m')
        self.add_param('r', val=np.zeros(n), units='m')
        self.add_param('Rtip', shape=1, units='m')
        self.add_param('Np', shape=n, units='N/m')
        self.add_param('Tp', shape=n, units='N/m')
        self.add_param('Uinf', shape=1, units='m/s')
        self.add_param('Omega', shape=1, units='rpm')
        self.add_param('pitch', shape=1, units='deg')
        self.add_param('azimuth', shape=1, units='deg')

        self.add_output('loads:r', shape=n+2, units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads:Px', shape=n+2, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads:Py', shape=n+2, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads:Pz', shape=n+2, units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_output('loads:V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_output('loads:Omega', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_output('loads:pitch', shape=1, units='deg', desc='pitch angle')
        self.add_output('loads:azimuth', shape=1, units='deg', desc='azimuthal angle')

        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['loads:r'] = np.concatenate([[params['Rhub']], params['r'], [params['Rtip']]])
        Np = np.concatenate([[0.0], params['Np'], [0.0]])
        Tp = np.concatenate([[0.0], params['Tp'], [0.0]])

        # conform to blade-aligned coordinate system
        unknowns['loads:Px'] = Np
        unknowns['loads:Py'] = -Tp
        unknowns['loads:Pz'] = 0*Np

        # return other outputs needed
        unknowns['loads:V'] = params['Uinf']
        unknowns['loads:Omega'] = params['Omega']
        unknowns['loads:pitch'] = params['pitch']
        unknowns['loads:azimuth'] = params['azimuth']

    def linearize(self, params, unknowns, resids):
        n = self.n

        dr_dr = np.vstack([np.zeros(n), np.eye(n), np.zeros(n)])
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

        J = {}
        zero = np.zeros(17)
        J['loads:r', 'r'] = dr_dr
        J['loads:r', 'Rhub'] = dr_dRhub
        J['loads:r', 'Rtip'] = dr_dRtip
        J['loads:Px', 'Np'] = np.vstack([zero, np.eye(n), zero])
        J['loads:Py', 'Tp'] = np.vstack([zero, -np.eye(n), zero])
        J['loads:V', 'Uinf'] = 1.0
        J['loads:Omega', 'Omega'] = 1.0
        J['loads:pitch', 'pitch'] = 1.0
        J['loads:azimuth', 'azimuth'] = 1.0

        return J

class CCBlade_to_RotorSE_connection(Group):

    def __init__(self, run_case, nSector, n, n2):
        super(CCBlade_to_RotorSE_connection, self).__init__()
        self.add('pass_through', PassThrough(n2), promotes=['*'])
        if run_case == 'power':
            self.add('mux_power', MUX_POWER(n2), promotes=['*'])
            pg = self.add('parallel', ParallelGroup(), promotes=['*'])
            for i in range(n2):
                pg.add('results'+str(i), FlowSweep(nSector, n), promotes=['Rhub', 'Rtip', 'precone', 'tilt', 'hubHt', 'precurve', 'presweep', 'yaw', 'precurveTip', 'presweepTip', 'af', 'bemoptions', 'B', 'rho', 'mu', 'shearExp', 'nSector', 'r', 'chord', 'theta'])
                self.connect('Uinf', 'results'+str(i)+'.Uinf', src_indices=[i])
                self.connect('pitch', 'results'+str(i)+'.pitch', src_indices=[i])
                self.connect('Omega', 'results'+str(i)+'.Omega', src_indices=[i])
                self.connect('results'+str(i)+'.CT', 'CT'+str(i+1))
                self.connect('results'+str(i)+'.CQ', 'CQ'+str(i+1))
                self.connect('results'+str(i)+'.CP', 'CP'+str(i+1))
                self.connect('results'+str(i)+'.T', 'T'+str(i+1))
                self.connect('results'+str(i)+'.Q', 'Q'+str(i+1))
                self.connect('results'+str(i)+'.P', 'P'+str(i+1))
        elif run_case == 'loads':
            self.add('init', CCInit(), promotes=['*'])
            self.add('wind', WindComponents(n), promotes=['*'])
            self.add('brent', BrentGroup(n), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'bemoptions', 'af', 'theta', 'r', 'chord', 'phi', 'cl', 'cd', 'W', 'theta', 'Vx', 'Vy'])
            self.add('loads', DistributedAeroLoads(n), promotes=['*'])
            self.add('loads_rotor', Loads_for_RotorSE(n), promotes=['*'])
        else:
            print 'Warning: run_case needs to be either power or loads.'

class AirfoilInterface(Interface):
    """Interface for airfoil aerodynamic analysis."""

    def evaluate(alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        Any implementation can be used, but to keep the smooth properties
        of CCBlade, the implementation should be C1 continuous.

        """

class CCAirfoil:
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""
    implements(AirfoilInterface)


    def __init__(self, alpha, Re, cl, cd):
        """Setup CCAirfoil from raw airfoil data on a grid.

        Parameters
        ----------
        alpha : array_like (deg)
            angles of attack where airfoil data are defined
            (should be defined from -180 to +180 degrees)
        Re : array_like
            Reynolds numbers where airfoil data are defined
            (can be empty or of length one if not Reynolds number dependent)
        cl : array_like
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : array_like
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """

        alpha = np.radians(alpha)
        self.one_Re = False

        # special case if zero or one Reynolds number (need at least two for bivariate spline)
        if len(Re) < 2:
            Re = [1e1, 1e15]
            cl = np.c_[cl, cl]
            cd = np.c_[cd, cd]
            self.one_Re = True

        kx = min(len(alpha)-1, 3)
        ky = min(len(Re)-1, 3)

        # a small amount of smoothing is used to prevent spurious multiple solutions
        self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky, s=0.1)
        self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky, s=0.001)


    @classmethod
    def initFromAerodynFile(cls, aerodynFile):
        """convenience method for initializing with AeroDyn formatted files

        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file

        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object

        """

        af = Airfoil.initFromAerodynFile(aerodynFile)
        alpha, Re, cl, cd, cm = af.createDataGrid(useCm = True)
        return cls(alpha, Re, cl, cd)


    def evaluate(self, alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number.

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        This method uses a spline so that the output is continuously differentiable, and
        also uses a small amount of smoothing to help remove spurious multiple solutions.

        """

        cl = self.cl_spline.ev(alpha, Re)
        cd = self.cd_spline.ev(alpha, Re)

        return cl, cd


    def derivatives(self, alpha, Re):

        # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
        tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
        tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees

        dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
        dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

        if self.one_Re:
            dcl_dRe = 0.0
            dcd_dRe = 0.0
        else:
            dcl_dRe = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
            dcd_dRe = bisplev(alpha, Re, tck_cd, dx=0, dy=1)

        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe



if __name__ == "__main__":


    if MPI: # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl
    else:
        # if you didn't use `mpirun`, then use the numpy data passing
        from openmdao.api import BasicImpl as impl

    def mpi_print(prob, *args):
        """ helper function to only print on rank 0"""
        if prob.root.comm.rank == 0:
            print args


    # geometry
    Rhub = 1.5
    Rtip = 63.0

    r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
    chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    B = 3  # number of blades
    bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5

    import os
    afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
    basepath = '5MW_AFFiles' + os.path.sep

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
    airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
    airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
    airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
    airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
    airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    af = [0]*len(r)
    for i in range(len(r)):
        af[i] = airfoil_types[af_idx[i]]


    tilt = -5.0
    precone = 2.5
    yaw = 0.0
    shearExp = 0.2
    hubHt = 80.0
    nSector = 8
    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 90.
    n = len(r)

    #### Test LoadsGroup
    # loads = Problem(impl=impl)
    # root = loads.root = Loads(n)
    # loads.setup(check=False)
    #
    # loads['Rhub'] = Rhub
    # loads['Rtip'] = Rtip
    # loads['r'] = r
    # loads['chord'] = chord
    # loads['theta'] = np.radians(theta)
    # loads['rho'] = rho
    # loads['mu'] = mu
    # loads['tilt'] = np.radians(tilt)
    # loads['precone'] = np.radians(precone)
    # loads['yaw'] = np.radians(yaw)
    # loads['shearExp'] = shearExp
    # loads['hubHt'] = hubHt
    # loads['Uinf'] = Uinf
    # loads['Omega'] = Omega
    # loads['pitch'] = np.radians(pitch)
    # loads['azimuth'] = np.radians(azimuth)
    # loads['af'] = af
    # loads['bemoptions'] = bemoptions
    #
    # loads.run()
    #
    # print 'Np', loads['Np']
    # print 'Tp', loads['Tp']

    ##### Test CCBlade
    Uinf = np.array([10.0])  # Needs to be an array for CCBlade group
    tsr = 7.55
    pitch = np.array([0.0])
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM

    # tsr = np.linspace(2, 14, 20)
    # Omega = 10.0 * np.ones_like(tsr)
    # Uinf = Omega*pi/30.0 * Rtip/tsr
    # pitch = np.zeros_like(tsr)

    n2 = len(Uinf)

    ccblade = Problem(impl=impl)
    ccblade.root = CCBlade(nSector, n, n2)
    import time
    ### SETUP OPTIMIZATION
    # ccblade.driver = pyOptSparseDriver()
    # ccblade.driver.options['optimizer'] = 'SNOPT'
    # ccblade.driver.add_desvar('Omega', lower=1.5, upper=25.0)
    # ccblade.driver.add_objective('obj')
    # recorder = SqliteRecorder('recorder')
    # recorder.options['record_params'] = True
    # recorder.options['record_metadata'] = True
    # ccblade.driver.add_recorder(recorder)
    print "setup start"
    t0 = time.time()
    ccblade.setup(check=False)
    t = time.time()
    print t - t0
    print "setup finish"

    ccblade['Rhub'] = Rhub
    ccblade['Rtip'] = Rtip
    ccblade['r'] = r
    ccblade['chord'] = chord
    ccblade['theta'] = theta
    ccblade['B'] = B
    ccblade['rho'] = rho
    ccblade['mu'] = mu
    ccblade['tilt'] = tilt
    ccblade['precone'] = precone
    ccblade['yaw'] = yaw
    ccblade['shearExp'] = shearExp
    ccblade['hubHt'] = hubHt
    ccblade['nSector'] = nSector
    ccblade['Uinf'] = Uinf
    ccblade['Omega'] = Omega
    ccblade['pitch'] = pitch
    ccblade['af'] = af
    ccblade['bemoptions'] = bemoptions
    t0 = time.time()
    ccblade.run()
    t = time.time()
    print t - t0

    partial = open("partial.txt", 'w')
    # test = ccblade.check_partial_derivatives(out_stream=partial)
    test2 = ccblade.check_total_derivatives(out_stream=partial, unknown_list=['P'])

    print 'CP', ccblade['CP']
    print 'CT', ccblade['CT']
    print 'CQ', ccblade['CQ']