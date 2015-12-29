#!/usr/bin/env python
# encoding: utf-8
"""
environment.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

import math
import numpy as np
from scipy.optimize import brentq
from openmdao.core.component import Component
# from openmdao.main.api import Component
# from openmdao.main.datatypes.api import Float, Array

from utilities import hstack, vstack

# -----------------
#  Base Components
# -----------------


class WindBase(Component):
    def __init__(self):
        super(WindBase, self).__init__()

        """base component for wind speed/direction"""

        # TODO: if I put required=True here for Uref there is another bug

        # variables
        self.add_param('Uref', shape=1, units='m/s', desc='reference wind speed (usually at hub height)')
        self.add_param('zref', shape=1,  units='m', desc='corresponding reference height')
        self.add_param('z',  shape=1, units='m', desc='heights where wind speed should be computed')

        # parameters
        self.add_param('z0', val=0.0, units='m', desc='bottom of wind profile (height of ground/sea)')

        # out
        self.add_output('U', shape=1, units='m/s', desc='magnitude of wind speed at each z location')
        self.add_output('beta', shape=1, units='deg', desc='corresponding wind angles relative to inertial coordinate system')

        missing_deriv_policy = 'assume_zero'  # TODO: for now OpenMDAO issue


class WaveBase(Component):
    def __init__(self):
        super(WaveBase, self).__init__()
        """base component for wave speed/direction"""

        # variables
        self.add_param('z', shape=1, units='m', desc='heights where wave speed should be computed')
        self.add_param('z_surface', shape=1, units='m', desc='vertical location of water surface')
        self.add_param('z_floor', val=0.0, units='m', desc='vertical location of sea floor')

        # out
        self.add_output('U', shape=1, units='m/s', desc='magnitude of wave speed at each z location')
        self.add_output('A', shape=1, units='m/s**2', desc='magnitude of wave acceleration at each z location')
        self.add_output('beta', shape=1, units='deg', desc='corresponding wave angles relative to inertial coordinate system')
        self.add_output('U0', shape=1, units='m/s', desc='magnitude of wave speed at z=MSL')
        self.add_output('A0', shape=1, units='m/s**2', desc='magnitude of wave acceleration at z=MSL')
        self.add_output('beta0', shape=1, units='deg', desc='corresponding wave angles relative to inertial coordinate system at z=MSL')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):
        """default to no waves"""
        n = len(self.z)
        self.U = np.zeros(n)
        self.A = np.zeros(n)
        self.beta = np.zeros(n)
        self.U0 = 0.
        self.A0 = 0.
        self.beta0 = 0.



class SoilBase(Component):
    def __init__(self):
        super(SoilBase, self).__init__()
        """base component for soil stiffness"""

        # out
        self.add_output('k', shape=1, units='N/m', required=True, desc='spring stiffness. rigid directions should use \
            ``float(''inf'')``. order: (x, theta_x, y, theta_y, z, theta_z)')

        missing_deriv_policy = 'assume_zero'  # TODO: for now OpenMDAO issue


# -----------------------
#  Subclassed Components
# -----------------------


class PowerWind(Component):
    def __init__(self):
        super(PowerWind,  self).__init__()

        # variables
        self.add_param('Uref', shape=1, units='m/s', desc='reference wind speed (usually at hub height)')
        self.add_param('zref', shape=1,  units='m', desc='corresponding reference height')
        self.add_param('z',  val=90.0, units='m', desc='heights where wind speed should be computed')

        # parameters
        self.add_param('z0', val=0.0, units='m', desc='bottom of wind profile (height of ground/sea)')

        # out
        self.add_output('U', val=np.zeros(1), units='m/s', desc='magnitude of wind speed at each z location')
        self.add_output('beta', shape=1, units='deg', desc='corresponding wind angles relative to inertial coordinate system')

        """power-law profile wind.  any nodes must not cross z0, and if a node is at z0
        it must stay at that point.  otherwise gradients crossing the boundary will be wrong."""

        # parameters
        self.add_param('shearExp', val=0.2, desc='shear exponent')
        self.add_param('betaWind', val=0.0, units='deg', desc='wind angle relative to inertial coordinate system')

        missing_deriv_policy = 'assume_zero'


    def solve_nonlinear(self, params, unknowns, resids):

        # rename
        z = np.zeros(1)
        z[0] = params['z']
        zref = params['zref']
        z0 = params['z0']
        zref = 90.0
        # velocity
        idx = z > z0
        n = len(z)
        self.n = n
        unknowns['U'] = np.zeros(n)
        # unknowns['U'][idx] = params['Uref']*((z[idx] - z0)/(zref - z0))**params['shearExp']
        unknowns['U'][idx] = params['Uref']*((z[idx] - z0)/(zref - z0))**params['shearExp']
        unknowns['beta'] = params['betaWind']*np.ones_like(z)

        # # add small cubic spline to allow continuity in gradient
        # k = 0.01  # fraction of profile with cubic spline
        # zsmall = z0 + k*(zref - z0)

        # self.spline = CubicSpline(x1=z0, x2=zsmall, f1=0.0, f2=Uref*k**shearExp,
        #     g1=0.0, g2=Uref*k**shearExp*shearExp/(zsmall - z0))

        # idx = np.logical_and(z > z0, z < zsmall)
        # self.U[idx] = self.spline.eval(z[idx])

        # self.zsmall = zsmall
        # self.k = k


    def list_deriv_vars(self):

        inputs = ('Uref', 'z', 'zref')
        outputs = ('U',)

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        # rename
        z = params['z']
        zref = params['zref']
        z0 = params['z0']
        shearExp = params['shearExp']
        U = unknowns['U']
        Uref = params['Uref']

        # gradients
        n = self.n
        dU_dUref = np.zeros(n)
        dU_dz = np.zeros(n)
        dU_dzref = np.zeros(n)

        idx = z > z0
        dU_dUref[idx] = U[idx]/Uref
        dU_dz[idx] = U[idx]*shearExp/(z[idx] - z0)
        dU_dzref[idx] = -U[idx]*shearExp/(zref - z0)


        # # cubic spline region
        # idx = np.logical_and(z > z0, z < zsmall)

        # # d w.r.t z
        # dU_dz[idx] = self.spline.eval_deriv(z[idx])

        # # d w.r.t. Uref
        # df2_dUref = k**shearExp
        # dg2_dUref = k**shearExp*shearExp/(zsmall - z0)
        # dU_dUref[idx] = self.spline.eval_deriv_params(z[idx], 0.0, 0.0, 0.0, df2_dUref, 0.0, dg2_dUref)

        # # d w.r.t. zref
        # dx2_dzref = k
        # dg2_dzref = -Uref*k**shearExp*shearExp/k/(zref - z0)**2
        # dU_dzref[idx] = self.spline.eval_deriv_params(z[idx], 0.0, dx2_dzref, 0.0, 0.0, 0.0, dg2_dzref)

        J = hstack([dU_dUref, np.diag(dU_dz), dU_dzref])

        return J




class LogWind(WindBase):
    def __init__(self):
        super(LogWind, self).__init__()
        """logarithmic-profile wind"""

        # parameters
        self.add_param('z_roughness', val=10.0, units='mm', desc='surface roughness length')
        self.add_param('betaWind', val=0.0, units='deg', desc='wind angle relative to inertial coordinate system')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0
        z_roughness = self.z_roughness/1e3  # convert to m

        # find velocity
        idx = [z - z0 > z_roughness]
        self.U = np.zeros_like(z)
        self.U[idx] = self.Uref*np.log((z[idx] - z0)/z_roughness) / math.log((zref - z0)/z_roughness)
        self.beta = self.betaWind*np.ones_like(z)


    def list_deriv_vars(self):

        inputs = ('Uref', 'z', 'zref')
        outputs = ('U',)

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0
        z_roughness = self.z_roughness/1e3
        Uref = self.Uref

        n = len(z)

        dU_dUref = np.zeros(n)
        dU_dz_diag = np.zeros(n)
        dU_dzref = np.zeros(n)

        idx = [z - z0 > z_roughness]
        lt = np.log((z[idx] - z0)/z_roughness)
        lb = math.log((zref - z0)/z_roughness)
        dU_dUref[idx] = lt/lb
        dU_dz_diag[idx] = Uref/lb / (z[idx] - z0)
        dU_dzref[idx] = -Uref*lt / math.log((zref - z0)/z_roughness)**2 / (zref - z0)

        J = hstack([dU_dUref, np.diag(dU_dz_diag), dU_dzref])

        return J



class LinearWaves(WaveBase):
    def __init__(self):
        super(LinearWaves, self).__init__()
        """linear (Airy) wave theory"""

        # variables
        self.add_param('Uc', shape=1, units='m/s', desc='mean current speed')

        # parameters
        self.add_param('hs', shape=1, units='m', desc='significant wave height (crest-to-trough)')
        self.add_param('T', shape=1, units='s', desc='period of waves')
        self.add_param('g', shape=1, val=9.81, units='m/s**2', desc='acceleration of gravity')
        self.add_param('betaWave', val=0.0, units='deg', desc='wave angle relative to inertial coordinate system')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

        # water depth
        d = self.z_surface - self.z_floor

        # design wave height
        h = self.hs

        # circular frequency
        omega = 2.0*math.pi/self.T

        # compute wave number from dispersion relationship
        k = brentq(lambda k: omega**2 - self.g*k*math.tanh(d*k), 0, 10*omega**2/self.g)

        # zero at surface
        z_rel = self.z - self.z_surface

        # maximum velocity
        self.U = h/2.0*omega*np.cosh(k*(z_rel + d))/math.sinh(k*d) + self.Uc
        self.U0 = h/2.0*omega*np.cosh(k*(0. + d))/math.sinh(k*d) + self.Uc

        # check heights
        self.U[np.logical_or(self.z < self.z_floor, self.z > self.z_surface)] = 0.

        # acceleration
        self.A  = self.U * omega
        self.A0 = self.U0 * omega
        # angles
        self.beta = self.betaWave*np.ones_like(self.z)
        self.beta0 =self.betaWave

        # derivatives
        dU_dz = h/2.0*omega*np.sinh(k*(z_rel + d))/math.sinh(k*d)*k
        dU_dUc = np.ones_like(self.z)
        idx = np.logical_or(self.z < self.z_floor, self.z > self.z_surface)
        dU_dz[idx] = 0.0
        dU_dUc[idx] = 0.0
        dA_dz = omega*dU_dz
        dA_dUc = omega*dU_dUc

        dU0 = np.zeros(len(self.z) + 1)
        dU0[-1] = 1.0
        dA0 = omega * dU0

        self.J = vstack([hstack([np.diag(dU_dz), dU_dUc]), hstack([np.diag(dA_dz), dA_dUc]), np.transpose(dU0), np.transpose(dA0)])


    def list_deriv_vars(self):

        inputs = ('z', 'Uc')
        outputs = ('U', 'A', 'U0', 'A0')

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        return self.J

class TowerSoilK(SoilBase):
    def __init__(self):
        super(TowerSoilK, self).__init__()
        """Passthrough of Soil-Structure-INteraction equivalent spring constants used to bypass TowerSoil."""

        # variable
        self.add_param('kin',  shape=1, desc='spring stiffness. rigid directions should use \
            ``float(''inf'')``. order: (x, theta_x, y, theta_y, z, theta_z)')

        self.add_param('rigid', shape=1, dtype=np.bool, desc='directions that should be considered infinitely rigid\
            order is x, theta_x, y, theta_y, z, theta_z')

        missing_deriv_policy = 'assume_zero'


    def solve_nonlinear(self, params, unknowns, resids):
        self.k=self.kin
        self.k[self.rigid] = float('inf')

class TowerSoil(SoilBase):
    def __init__(self):
        super(TowerSoil, self).__init__()
        """textbook soil stiffness method"""

        # variable
        self.add_param('r0', val=1.0, units='m', desc='radius of base of tower')
        self.add_param('depth', val=1.0, units='m', desc='depth of foundation in the soil')

        # parameter
        self.add_param('G', val=140e6, units='Pa', desc='shear modulus of soil')
        self.add_param('nu', val=0.4, desc='Poisson''s ratio of soil')
        self.add_param('rigid', shape=1, dtype=np.bool, desc='directions that should be considered infinitely rigid\
            order is x, theta_x, y, theta_y, z, theta_z')

        missing_deriv_policy = 'assume_zero'


    def solve_nonlinear(self, params, unknowns, resids):

        G = self.G
        nu = self.nu
        h = self.depth
        r0 = self.r0

        # vertical
        eta = 1.0 + 0.6*(1.0-nu)*h/r0
        k_z = 4*G*r0*eta/(1.0-nu)

        # horizontal
        eta = 1.0 + 0.55*(2.0-nu)*h/r0
        k_x = 32.0*(1.0-nu)*G*r0*eta/(7.0-8.0*nu)

        # rocking
        eta = 1.0 + 1.2*(1.0-nu)*h/r0 + 0.2*(2.0-nu)*(h/r0)**3
        k_thetax = 8.0*G*r0**3*eta/(3.0*(1.0-nu))

        # torsional
        k_phi = 16.0*G*r0**3/3.0

        self.k = np.array([k_x, k_thetax, k_x, k_thetax, k_z, k_phi])
        self.k[self.rigid] = float('inf')


    def list_deriv_vars(self):

        inputs = ('r0', 'depth')
        outputs = ('k',)

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        G = self.G
        nu = self.nu
        h = self.depth
        r0 = self.r0

        # vertical
        eta = 1.0 + 0.6*(1.0-nu)*h/r0
        deta_dr0 = -0.6*(1.0-nu)*h/r0**2
        dkz_dr0 = 4*G/(1.0-nu)*(eta + r0*deta_dr0)

        deta_dh = 0.6*(1.0-nu)/r0
        dkz_dh = 4*G*r0/(1.0-nu)*deta_dh

        # horizontal
        eta = 1.0 + 0.55*(2.0-nu)*h/r0
        deta_dr0 = -0.55*(2.0-nu)*h/r0**2
        dkx_dr0 = 32.0*(1.0-nu)*G/(7.0-8.0*nu)*(eta + r0*deta_dr0)

        deta_dh = 0.55*(2.0-nu)/r0
        dkx_dh = 32.0*(1.0-nu)*G*r0/(7.0-8.0*nu)*deta_dh

        # rocking
        eta = 1.0 + 1.2*(1.0-nu)*h/r0 + 0.2*(2.0-nu)*(h/r0)**3
        deta_dr0 = -1.2*(1.0-nu)*h/r0**2 - 3*0.2*(2.0-nu)*(h/r0)**3/r0
        dkthetax_dr0 = 8.0*G/(3.0*(1.0-nu))*(3*r0**2*eta + r0**3*deta_dr0)

        deta_dh = 1.2*(1.0-nu)/r0 + 3*0.2*(2.0-nu)*(1.0/r0)**3*h**2
        dkthetax_dh = 8.0*G*r0**3/(3.0*(1.0-nu))*deta_dh

        # torsional
        dkphi_dr0 = 16.0*G*3*r0**2/3.0
        dkphi_dh = 0.0

        dk_dr0 = np.array([dkx_dr0, dkthetax_dr0, dkx_dr0, dkthetax_dr0, dkz_dr0, dkphi_dr0])
        dk_dr0[self.rigid] = 0.0
        dk_dh = np.array([dkx_dh, dkthetax_dh, dkx_dh, dkthetax_dh, dkz_dh, dkphi_dh])
        dk_dh[self.rigid] = 0.0

        J = hstack((dk_dr0, dk_dh))

        return J






if __name__ == '__main__':
    p = LogWind()
    p.Uref = 10.0
    p.zref = 100.0
    p.z0 = 1.0
    p.z = np.linspace(1.0, 5, 20)
    p.shearExp = 0.2
    p.betaWind = 0.0

    p.run()

    import matplotlib.pyplot as plt
    plt.plot(p.z, p.U)
    plt.show()
