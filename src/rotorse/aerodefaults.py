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
from commonse import cosd
from rotoraero import GeomtrySetupBase, AeroBase, DrivetrainLossesBase, CDFBase


# ---------------------
# Default Implementations of Base Classes
# ---------------------


class CCBladeGeometry(GeomtrySetupBase):

    Rtip = Float(iotype='in', units='m', desc='tip radius')
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg')

    def execute(self):

        self.R = self.Rtip*cosd(self.precone)



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


class RayleighCDF(CDFBase):
    """Rayleigh cumulative distribution function"""

    xbar = Float(iotype='in', desc='mean value of distribution')

    def execute(self):

        self.F = 1.0 - np.exp(-pi/4.0*(self.x/self.xbar)**2)



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

    from rotoraero import NewAssembly

    rotor = NewAssembly()
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

    print rotor.aep.AEP

    print len(rotor.powercurve.V)
    import matplotlib.pyplot as plt
    plt.plot(rotor.powercurve.V, rotor.powercurve.P)
    plt.show()



    exit()
    # -----------------------



    eta_Prated = 0.95
    R = Rtip
    RPM2RS = pi/30.0
    RS2RPM = 30.0/pi
    # maxOmega = 20.0
    from scipy.optimize import brentq
    from akima import Akima

    aero = CCBlade()
    aero.r = r
    aero.chord = chord
    aero.theta = theta
    aero.Rhub = Rhub
    aero.Rtip = Rtip
    aero.hubheight = hubheight
    aero.airfoil_files = af
    aero.precone = precone
    aero.tilt = tilt
    aero.yaw = yaw
    aero.B = B
    aero.rho = rho
    aero.mu = mu
    aero.shearExp = shearExp
    aero.nSector = nSectors_power_integration


    # the blind but general way

    from commonse.utilities import smooth_min

    # quick guess for rated power
    cpguess = 0.5
    Vr0 = (ratedPower/(cpguess*0.5*rho*pi*R**2))**(1.0/3)
    Vr0 *= 1.20
    print Vr0

    V1 = np.linspace(Vin, Vr0, 15)
    V2 = np.linspace(Vr0, Vout, 6)
    V = np.concatenate([V1, V2[1:]])

    # V = np.linspace(Vin, Vout, 20)
    aero.Uhub = V
    # aero.Omega = np.minimum(tsr_opt*V/R*RS2RPM, maxOmega)  # replace with smooth_min
    aero.Omega, dOmega_dOmegad = smooth_min(tsr_opt*V/R*RS2RPM, maxOmega, pct_offset=0.01)
    aero.pitch = pitch_opt*np.ones_like(V)
    aero.run_case = 'power'
    aero.run()

    P = aero.P*eta_Prated

    spline = Akima(V, P)

    VV = np.linspace(Vin, Vout, 200)
    PP, _, _, _ = spline.interp(VV)

    if True:  # replace with if var speed

        def error(VVV):

            PPP, _, _, _ = spline.interp(VVV)
            return PPP - ratedPower

        # Brent's method
        Vrated = brentq(error, Vin, Vout)

        print Vrated

        # PP = np.minimum(PP, ratedPower)  # replace with smooth min
        PP[VV >= Vrated] = ratedPower  # is this smooth?

    sigma = 6.0
    f = VV/sigma**2*np.exp(-VV**2/(2*sigma**2))
    F = 1 - np.exp(-VV**2/(2*sigma**2))


    print np.trapz(PP, F)

    import matplotlib.pyplot as plt
    plt.plot(VV, PP)
    plt.plot(V, P, 'o')
    plt.show()

    # exit()




    Vomega_max = maxOmega*RPM2RS*R/tsr_opt

    aero.Uhub = np.array([Vomega_max])
    aero.Omega = np.array([maxOmega])
    aero.pitch = np.array([pitch_opt])
    aero.run_case = 'power'
    aero.run()
    cp = aero.P / (0.5*rho*aero.Uhub**3 * pi*R**2)
    cp = cp[0]

    Vr0 = (ratedPower/(cp*0.5*rho*pi*R**2*eta_Prated))**(1.0/3)

    # print Vomega_max
    # print Vr0

    if Vomega_max < Vr0:

        def error(V):

            aero.Uhub = np.array([V])
            aero.run()
            return aero.P*eta_Prated - ratedPower

        # Brent's method
        Vrated = brentq(error, Vomega_max, Vr0+1)

        print Vrated

        V25 = np.linspace(Vomega_max, Vrated, 7)
        V25 = V25[1:-1]
        aero.Uhub = V25
        n = len(V25)
        aero.Omega = maxOmega*np.ones(n)
        aero.pitch = pitch_opt*np.ones(n)
        aero.run()
        P25 = aero.P*eta_Prated

        V2 = np.linspace(Vin, Vomega_max, 100)
        P2 = cp*0.5*rho*V2**3 * pi*R**2 * eta_Prated
        V3 = np.linspace(Vrated, Vout, 100)
        P3 = ratedPower*np.ones(100)
        V = np.concatenate([V2, V25, V3])
        P = np.concatenate([P2, P25, P3])


        import matplotlib.pyplot as plt
        plt.plot(V, P)
        plt.plot(V2, P2)
        plt.plot(V25, P25, 'o')
        plt.plot(V3, P3)
        plt.show()

    else:

        V2 = np.linspace(Vin, Vr0, 100)
        P2 = cp*0.5*rho*V2**3 * pi*R**2 * eta_Prated
        V3 = np.linspace(Vr0, Vout, 100)
        P3 = ratedPower*np.ones(100)
        V = np.concatenate([V2, V3[1:]])
        P = np.concatenate([P2, P3[1:]])

    sigma = 6.0
    f = V/sigma**2*np.exp(-V**2/(2*sigma**2))
    F = 1 - np.exp(-V**2/(2*sigma**2))

    # import matplotlib.pyplot as plt
    # plt.plot(V, f)
    # plt.plot(V, F)
    # plt.show()

    print np.trapz(P, F)
    # print np.trapz(P*f, V)

    exit()



    # ------ OpenMDAO setup -------------------

    rotor = RotorAeroVS()
    rotor.replace('geom', CCBladeGeometry())
    rotor.replace('analysis', CCBlade())
    rotor.replace('analysis2', CCBlade())
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


    rotor.analysis2.r = r
    rotor.analysis2.chord = chord
    rotor.analysis2.theta = theta
    rotor.analysis2.Rhub = Rhub
    rotor.analysis2.Rtip = Rtip
    rotor.analysis2.hubheight = hubheight
    rotor.analysis2.airfoil_files = af
    rotor.analysis2.precone = precone
    rotor.analysis2.tilt = tilt
    rotor.analysis2.yaw = yaw
    rotor.analysis2.B = B
    rotor.analysis2.rho = rho
    rotor.analysis2.mu = mu
    rotor.analysis2.shearExp = shearExp
    rotor.analysis2.nSector = nSectors_power_integration



    # drivetrain efficiency
    rotor.dt.drivetrainType = drivetrainType


    # CDF
    rotor.cdf.xbar = Ubar


    # other parameters
    rotor.rho = rho

    rotor.control.Vin = Vin
    rotor.control.Vout = Vout
    rotor.control.ratedPower = ratedPower
    rotor.control.minOmega = minOmega
    rotor.control.maxOmega = maxOmega
    rotor.control.tsr = tsr_opt
    rotor.control.pitch = pitch_opt

    # options
    rotor.tsr_sweep_step_size = tsr_sweep_step_size
    rotor.npts_power_curve = npts_power_curve
    rotor.AEP_loss_factor = AEP_loss_factor

    rotor.run()

    print rotor.AEP
    # print rotor.brent.xstar

    print rotor.rated.V
    print rotor.rated.Omega
    print rotor.rated.pitch
    print rotor.rated.T
    print rotor.rated.Q

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P/1e6)


    # plt.figure()
    # plt.plot(r, rotor.Np_rated)
    # plt.plot(r, rotor.Tp_rated)

    # plt.figure()
    # plt.plot(r, rotor.Np_extreme)
    # plt.plot(r, rotor.Tp_extreme)
    plt.show()




