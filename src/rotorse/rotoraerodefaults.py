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
from commonse import sind, cosd
from rotoraero import GeomtrySetupBase, AeroBase, DrivetrainLossesBase, CDFBase


# ---------------------
# Default Implementations of Base Classes
# ---------------------


class CCBladeGeometry(GeomtrySetupBase):

    Rtip = Float(iotype='in', units='m', desc='tip radius')
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg')

    def execute(self):

        self.R = self.Rtip*cosd(self.precone)

    def linearize(self):
        pass

    def provideJ(self):

        inputs = ('Rtip', 'precone')
        outputs = ('R',)

        J = np.array([[cosd(self.precone), -self.Rtip*sind(self.precone)*pi/180.0]])

        return inputs, outputs, J



class CCBlade(AeroBase):
    """blade element momentum code"""

    # variables
    r = Array(iotype='in', units='m', desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
    chord = Array(iotype='in', units='m', desc='chord length at each section')
    theta = Array(iotype='in', units='deg', desc='twist angle at each section (positive decreases angle of attack)')
    Rhub = Float(iotype='in', units='m', desc='hub radius')
    Rtip = Float(iotype='in', units='m', desc='tip radius')
    hubheight = Float(iotype='in', units='m')

    # parameters
    airfoil_files = List(Str, iotype='in', desc='names of airfoil file')
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg')
    tilt = Float(0.0, iotype='in', desc='shaft tilt', units='deg')
    yaw = Float(0.0, iotype='in', desc='yaw error', units='deg')
    B = Int(3, iotype='in', desc='number of blades')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    mu = Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air')
    shearExp = Float(0.2, iotype='in', desc='shear exponent')
    nSector = Int(4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power')


    def execute(self):

        # airfoil files
        n = len(self.airfoil_files)
        af = [0]*n
        afinit = CCAirfoil.initFromAerodynFile
        for i in range(n):
            af[i] = afinit(self.airfoil_files[i])

        ccblade = CCBlade_PY(self.r, self.chord, self.theta, af, self.Rhub, self.Rtip, self.B,
            self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubheight,
            self.nSector, derivatives=True)

        run_case = self.run_case

        if run_case == 'power':

            # power, thrust, torque
            P, T, Q, dP_ds, dT_ds, dQ_ds, dP_dv, dT_dv, dQ_dv \
                = ccblade.evaluate(self.Uhub, self.Omega, self.pitch, coefficient=False)

            self.P = P
            self.T = T
            self.Q = Q


        if run_case == 'loads':

            # distributed loads

            if self.Omega_load == 0.0:  # TODO: implement derivatives for this case
                Np, Tp = ccblade.distributedAeroLoads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)
            else:
                Np, Tp, dNp_dX, dTp_dX, dNp_dprecurve, dTp_dprecurve \
                    = ccblade.distributedAeroLoads(self.V_load, self.Omega_load, self.pitch_load, self.azimuth_load)

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
            self.loads.tilt = self.tilt




class CSMDrivetrain(DrivetrainLossesBase):
    """drivetrain losses from NREL cost and scaling model"""

    drivetrainType = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')


    def execute(self):

        drivetrainType = self.drivetrainType

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

        Pbar = self.aeroPower / self.ratedPower

        # TODO: think about these gradients.  may not be able to use abs and minimum

        # handle negative power case
        Pbar = np.abs(Pbar)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar = np.minimum(Pbar, 1.0)

        # compute efficiency
        eff = np.zeros_like(Pbar)
        idx = Pbar != 0

        eff[idx] = 1.0 - (constant/Pbar[idx] + linear + quadratic*Pbar[idx])

        self.power = self.aeroPower * eff



class WeibullCDF(CDFBase):
    """Weibull cumulative distribution function"""

    A = Float(iotype='in', desc='scale factor')
    k = Float(iotype='in', desc='shape or form factor')

    def execute(self):

        self.F = 1.0 - np.exp(-(self.x/self.A)**self.k)

    def linearize(self):
        pass

    def provideJ(self):

        inputs = ('x',)
        outputs = ('F',)

        x = self.x
        A = self.A
        k = self.k
        J = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)

        return inputs, outputs, J



class RayleighCDF(CDFBase):
    """Rayleigh cumulative distribution function"""

    xbar = Float(iotype='in', desc='mean value of distribution')

    def execute(self):

        self.F = 1.0 - np.exp(-pi/4.0*(self.x/self.xbar)**2)

    def linearize(self):
        pass

    def provideJ(self):

        inputs = ('x',)
        outputs = ('F',)

        x = self.x
        xbar = self.xbar
        J = np.diag(np.exp(-pi/4.0*(x/xbar)**2)*pi*x/(2*xbar**2))

        return inputs, outputs, J


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
    basepath = '/Users/sning/Dropbox/NREL/5MW_files/5MW_AFFiles/'

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
    rotor.analysis.hubheight = hubheight
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

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P/1e6)
    plt.show()


