#!/usr/bin/env python
# encoding: utf-8
"""
aerodefaults.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi
from openmdao.main.datatypes.api import Int, Float, Array, Str, List, Enum

from ccblade import CCAirfoil, CCBlade as CCBlade_PY
from openmdao.main.api import Component
from commonse.utilities import sind, cosd, smooth_abs, smooth_min, hstack, vstack, linspace_with_deriv
from rotoraero import GeomtrySetupBase, AeroBase, DrivetrainLossesBase, CDFBase
from akima import Akima


# ---------------------
# Map Design Variables to Discretization
# ---------------------



class GeometrySpline(Component):

    r_af = Array(iotype='in', units='m', desc='locations where airfoils are defined on unit radius')

    idx_cylinder = Int(iotype='in', desc='location where cylinder section ends on unit radius')
    r_max_chord = Float(iotype='in')

    Rhub = Float(iotype='in', units='m', desc='blade hub radius')
    Rtip = Float(iotype='in', units='m', desc='blade tip radius')

    chord_sub = Array(iotype='in', units='m', desc='chord at control points')
    theta_sub = Array(iotype='in', units='deg', desc='twist at control points')

    r = Array(iotype='out', units='m', desc='chord at airfoil locations')
    chord = Array(iotype='out', units='m', desc='chord at airfoil locations')
    theta = Array(iotype='out', units='deg', desc='twist at airfoil locations')
    r_af_spacing = Array(iotype='out')


    def execute(self):

        nc = len(self.chord_sub)
        nt = len(self.theta_sub)
        Rhub = self.Rhub
        Rtip = self.Rtip
        idxc = self.idx_cylinder
        r_max_chord = Rhub + (Rtip-Rhub)*self.r_max_chord
        r_cylinder = Rhub + (Rtip-Rhub)*self.r_af[idxc]

        # chord parameterization
        rc_outer, drc_drcmax, drc_drtip = linspace_with_deriv(r_max_chord, Rtip, nc-1)
        r_chord = np.concatenate([[Rhub], rc_outer])
        drc_drcmax = np.concatenate([[0.0], drc_drcmax])
        drc_drtip = np.concatenate([[0.0], drc_drtip])
        drc_drhub = np.concatenate([[1.0], np.zeros(nc-1)])

        # theta parameterization
        r_theta, drt_drcyl, drt_drtip = linspace_with_deriv(r_cylinder, Rtip, nt)

        # spline
        chord_spline = Akima(r_chord, self.chord_sub)
        theta_spline = Akima(r_theta, self.theta_sub)

        self.r = Rhub + (Rtip-Rhub)*self.r_af
        self.chord, dchord_dr, dchord_drchord, dchord_dchordsub = chord_spline.interp(self.r)
        theta_outer, dthetaouter_dr, dthetaouter_drtheta, dthetaouter_dthetasub = theta_spline.interp(self.r[idxc:])

        theta_inner = theta_outer[0] * np.ones(idxc)
        self.theta = np.concatenate([theta_inner, theta_outer])

        self.r_af_spacing = np.diff(self.r_af)

        # gradients (TODO: rethink these a bit or use Tapenade.)
        n = len(self.r_af)
        dr_draf = (Rtip-Rhub)*np.ones(n)
        dr_dRhub = 1.0 - self.r_af
        dr_dRtip = self.r_af
        dr = hstack([np.diag(dr_draf), np.zeros((n, 1)), dr_dRhub, dr_dRtip, np.zeros((n, nc+nt))])

        dchord_draf = dchord_dr * dr_draf
        dchord_drmaxchord0 = np.dot(dchord_drchord, drc_drcmax)
        dchord_drmaxchord = dchord_drmaxchord0 * (Rtip-Rhub)
        dchord_drhub = np.dot(dchord_drchord, drc_drhub) + dchord_drmaxchord0*(1.0 - self.r_max_chord) + dchord_dr*dr_dRhub
        dchord_drtip = np.dot(dchord_drchord, drc_drtip) + dchord_drmaxchord0*(self.r_max_chord) + dchord_dr*dr_dRtip
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
        dtheta_drhub += dtheta_dcyl*(1.0 - self.r_af[idxc])
        dtheta_drtip += dtheta_dcyl*self.r_af[idxc]

        dtheta = hstack([dtheta_draf, np.zeros((n, 1)), dtheta_drhub, dtheta_drtip, np.zeros((n, nc)), dtheta_dthetasub])

        drafs_dr = np.zeros((n-1, n))
        for i in range(n-1):
            drafs_dr[i, i] = -1.0
            drafs_dr[i, i+1] = 1.0
        drafs = hstack([drafs_dr, np.zeros((n-1, 3+nc+nt))])

        self.J = vstack([dr, dchord, dtheta, drafs])


    def list_deriv_vars(self):

        inputs = ('r_af', 'r_max_chord', 'Rhub', 'Rtip', 'chord_sub', 'theta_sub')
        outputs = ('r', 'chord', 'theta', 'r_af_spacing')

        return inputs, outputs


    def provideJ(self):

        return self.J




# ---------------------
# Default Implementations of Base Classes
# ---------------------


class CCBladeGeometry(GeomtrySetupBase):

    Rtip = Float(iotype='in', units='m', desc='tip radius')
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg')

    def execute(self):

        self.R = self.Rtip*cosd(self.precone)

    def list_deriv_vars(self):

        inputs = ('Rtip', 'precone')
        outputs = ('R',)

        return inputs, outputs

    def provideJ(self):

        J = np.array([[cosd(self.precone), -self.Rtip*sind(self.precone)*pi/180.0]])

        return J



class CCBlade(AeroBase):
    """blade element momentum code"""

    # (potential) variables
    r = Array(iotype='in', units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
    chord = Array(iotype='in', units='m', desc='chord length at each section')
    theta = Array(iotype='in', units='deg', desc='twist angle at each section (positive decreases angle of attack)')
    Rhub = Float(iotype='in', units='m', desc='hub radius')
    Rtip = Float(iotype='in', units='m', desc='tip radius')
    hubHt = Float(iotype='in', units='m')
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg')
    tilt = Float(0.0, iotype='in', desc='shaft tilt', units='deg')
    yaw = Float(0.0, iotype='in', desc='yaw error', units='deg')

    # parameters
    airfoil_files = List(Str, iotype='in', desc='names of airfoil file')
    B = Int(3, iotype='in', desc='number of blades')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    mu = Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air')
    shearExp = Float(0.2, iotype='in', desc='shear exponent')
    nSector = Int(4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power')

    missing_deriv_policy = 'assume_zero'


    def execute(self):

        # airfoil files
        n = len(self.airfoil_files)
        af = [0]*n
        afinit = CCAirfoil.initFromAerodynFile
        for i in range(n):
            af[i] = afinit(self.airfoil_files[i])

        self.ccblade = CCBlade_PY(self.r, self.chord, self.theta, af, self.Rhub, self.Rtip, self.B,
            self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubHt,
            self.nSector, derivatives=True)


        if self.run_case == 'power':

            # power, thrust, torque
            self.P, self.T, self.Q, self.dP, self.dT, self.dQ \
                = self.ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=False)


        elif self.run_case == 'loads':

            # distributed loads
            Np, Tp, self.dNp, self.dTp \
                = self.ccblade.distributedAeroLoads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)

            # concatenate loads at root/tip
            self.loads.r = np.concatenate([[self.Rhub], self.r, [self.Rtip]])
            Np = np.concatenate([[0.0], Np, [0.0]])
            Tp = np.concatenate([[0.0], Tp, [0.0]])

            # conform to blade-aligned coordinate system
            self.loads.Px = Np
            self.loads.Py = -Tp
            self.loads.Pz = 0*Np

            # return other outputs needed
            self.loads.V = self.V_load
            self.loads.Omega = self.Omega_load
            self.loads.pitch = self.pitch_load
            self.loads.azimuth = self.azimuth_load


    def list_deriv_vars(self):

        if self.run_case == 'power':
            inputs = ('precone', 'tilt', 'hubHt', 'Rhub', 'Rtip', 'yaw',
                'Uhub', 'Omega', 'pitch', 'r', 'chord', 'theta')
            outputs = ('P', 'T', 'Q')

        elif self.run_case == 'loads':

            inputs = ('r', 'chord', 'theta', 'Rhub', 'Rtip', 'hubHt', 'precone',
                'tilt', 'yaw', 'V_load', 'Omega_load', 'pitch_load', 'azimuth_load')
            outputs = ('loads.r', 'loads.Px', 'loads.Py', 'loads.Pz', 'loads.V',
                'loads.Omega', 'loads.pitch', 'loads.azimuth')

        return inputs, outputs


    def provideJ(self):

        if self.run_case == 'power':

            dP = self.dP
            dT = self.dT
            dQ = self.dQ

            jP = hstack([dP['dprecone'], dP['dtilt'], dP['dhubHt'], dP['dRhub'], dP['dRtip'],
                dP['dyaw'], dP['dUinf'], dP['dOmega'], dP['dpitch'], dP['dr'], dP['dchord'], dP['dtheta']])
            jT = hstack([dT['dprecone'], dT['dtilt'], dT['dhubHt'], dT['dRhub'], dT['dRtip'],
                dT['dyaw'], dT['dUinf'], dT['dOmega'], dT['dpitch'], dT['dr'], dT['dchord'], dT['dtheta']])
            jQ = hstack([dQ['dprecone'], dQ['dtilt'], dQ['dhubHt'], dQ['dRhub'], dQ['dRtip'],
                dQ['dyaw'], dQ['dUinf'], dQ['dOmega'], dQ['dpitch'], dQ['dr'], dQ['dchord'], dQ['dtheta']])

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
            dr = hstack([dr_dr, np.zeros((n+2, 2*n)), dr_dRhub, dr_dRtip, np.zeros((n+2, 8))])

            jNp = hstack([dNp['dr'], dNp['dchord'], dNp['dtheta'], dNp['dRhub'], dNp['dRtip'],
                dNp['dhubHt'], dNp['dprecone'], dNp['dtilt'], dNp['dyaw'], dNp['dUinf'],
                dNp['dOmega'], dNp['dpitch'], dNp['dazimuth']])
            jTp = hstack([dTp['dr'], dTp['dchord'], dTp['dtheta'], dTp['dRhub'], dTp['dRtip'],
                dTp['dhubHt'], dTp['dprecone'], dTp['dtilt'], dTp['dyaw'], dTp['dUinf'],
                dTp['dOmega'], dTp['dpitch'], dTp['dazimuth']])
            dPx = vstack([np.zeros(3*n+10), jNp, np.zeros(3*n+10)])
            dPy = vstack([np.zeros(3*n+10), -jTp, np.zeros(3*n+10)])
            dPz = np.zeros((n+2, 3*n+10))

            dV = np.zeros(3*n+10)
            dV[3*n+6] = 1.0
            dOmega = np.zeros(3*n+10)
            dOmega[3*n+7] = 1.0
            dpitch = np.zeros(3*n+10)
            dpitch[3*n+8] = 1.0
            dazimuth = np.zeros(3*n+10)
            dazimuth[3*n+9] = 1.0

            J = vstack([dr, dPx, dPy, dPz, dV, dOmega, dpitch, dazimuth])


        return J



class CSMDrivetrain(DrivetrainLossesBase):
    """drivetrain losses from NREL cost and scaling model"""

    drivetrainType = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')

    missing_deriv_policy = 'assume_zero'

    def execute(self):

        drivetrainType = self.drivetrainType
        aeroPower = self.aeroPower
        ratedPower = self.ratedPower


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
        Pbar, dPbar_dPbar1 = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant/Pbar + linear + quadratic*Pbar)

        self.power = aeroPower * eff


        # gradients
        dPbar_dPa = dPbar_dPbar1*dPbar1_dPbar0/ratedPower
        dPbar_dPr = -dPbar_dPbar1*dPbar1_dPbar0*aeroPower/ratedPower**2

        deff_dPa = dPbar_dPa*(constant/Pbar**2 - quadratic)
        deff_dPr = dPbar_dPr*(constant/Pbar**2 - quadratic)

        dP_dPa = eff + aeroPower*deff_dPa
        dP_dPr = aeroPower*deff_dPr

        self.J = hstack([np.diag(dP_dPa), dP_dPr])


    def list_deriv_vars(self):

        inputs = ('aeroPower', 'ratedPower')
        outputs = ('power',)

        return inputs, outputs

    def provideJ(self):

        return self.J




class WeibullCDF(CDFBase):
    """Weibull cumulative distribution function"""

    A = Float(iotype='in', desc='scale factor')
    k = Float(iotype='in', desc='shape or form factor')

    def execute(self):

        self.F = 1.0 - np.exp(-(self.x/self.A)**self.k)

    def list_deriv_vars(self):
        inputs = ('x',)
        outputs = ('F',)

        return inputs, outputs

    def provideJ(self):

        x = self.x
        A = self.A
        k = self.k
        J = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)

        return J



class RayleighCDF(CDFBase):
    """Rayleigh cumulative distribution function"""

    xbar = Float(iotype='in', desc='mean value of distribution')

    def execute(self):

        self.F = 1.0 - np.exp(-pi/4.0*(self.x/self.xbar)**2)

    def list_deriv_vars(self):

        inputs = ('x',)
        outputs = ('F',)

        return inputs, outputs

    def provideJ(self):

        x = self.x
        xbar = self.xbar
        J = np.diag(np.exp(-pi/4.0*(x/xbar)**2)*pi*x/(2*xbar**2))

        return J



if __name__ == '__main__':

    from rotoraero import RotorAeroVS

    # ---------- inputs ---------------

    # geometry
    r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
    chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    Rhub = 1.5
    Rtip = 63.0
    hubheight = 80.0
    precone = 2.5
    tilt = -5.0
    yaw = 0.0
    B = 3

    # airfoils
    basepath = '/Users/Andrew/Dropbox/NREL/5MW_files/5MW_AFFiles/'

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = basepath + 'Cylinder1.dat'
    airfoil_types[1] = basepath + 'Cylinder2.dat'
    airfoil_types[2] = basepath + 'DU40_A17.dat'
    airfoil_types[3] = basepath + 'DU35_A17.dat'
    airfoil_types[4] = basepath + 'DU30_A17.dat'
    airfoil_types[5] = basepath + 'DU25_A17.dat'
    airfoil_types[6] = basepath + 'DU21_A17.dat'
    airfoil_types[7] = basepath + 'NACA64_A17.dat'

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(r)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5
    shearExp = 0.2
    Ubar = 6.0

    # operational conditions
    Vin = 3.0
    Vout = 25.0
    ratedPower = 5e6
    minOmega = 0.0
    maxOmega = 12.0
    tsr_opt = 7.55
    pitch_opt = 0.0


    # options
    nSectors_power_integration = 4
    tsr_sweep_step_size = 0.25
    npts_power_curve = 200
    drivetrainType = 'geared'
    AEP_loss_factor = 1.0

    # -------------------------


    # OpenMDAO setup

    rotor = RotorAeroVS()
    rotor.replace('geom', CCBladeGeometry())
    rotor.replace('analysis', CCBlade())
    rotor.replace('dt', CSMDrivetrain())
    rotor.replace('cdf', RayleighCDF())

    # geometry
    rotor.geom.Rtip = Rtip
    rotor.geom.precone = precone

    # aero analysis

    rotor.analysis.r = r
    rotor.analysis.chord = chord
    rotor.analysis.theta = theta
    rotor.analysis.Rhub = Rhub
    rotor.analysis.Rtip = Rtip
    rotor.analysis.hubHt = hubheight
    rotor.analysis.airfoil_files = af
    rotor.analysis.precone = precone
    rotor.analysis.tilt = tilt
    rotor.analysis.yaw = yaw
    rotor.analysis.B = B
    rotor.analysis.rho = rho
    rotor.analysis.mu = mu
    rotor.analysis.shearExp = shearExp
    rotor.analysis.nSector = nSectors_power_integration


    # drivetrain efficiency
    rotor.dt.drivetrainType = drivetrainType

    # CDF
    rotor.cdf.xbar = Ubar

    # other parameters
    # rotor.rho = rho

    rotor.control.Vin = Vin
    rotor.control.Vout = Vout
    rotor.control.ratedPower = ratedPower
    rotor.control.minOmega = minOmega
    rotor.control.maxOmega = maxOmega
    rotor.control.tsr = tsr_opt
    rotor.control.pitch = pitch_opt

    rotor.run()

    print rotor.AEP

    # print len(rotor.V)

    rc = rotor.ratedConditions
    print rc.V
    print rc.Omega
    print rc.pitch
    print rc.T
    print rc.Q

    print rotor.diameter

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P/1e6)
    plt.show()


