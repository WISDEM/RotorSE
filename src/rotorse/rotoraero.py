#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function
import numpy as np
from math import pi
from openmdao.api import Component, Group, Brent, ScipyGMRES

from commonse.utilities import hstack, vstack, linspace_with_deriv, smooth_min, trapz_deriv
from akima import Akima
from enum import Enum

# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi

class VarSpeedMachine(Component):
    """variable speed machines"""
    def __init__(self):
        super(VarSpeedMachine, self).__init__()

        self.add_param('Vin', shape=1, units='m/s', desc='cut-in wind speed')
        self.add_param('Vout', shape=1, units='m/s', desc='cut-out wind speed')
        self.add_param('ratedPower', shape=1, units='W', desc='rated power')
        self.add_param('minOmega', shape=1, units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('tsr', shape=1, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

class FixedSpeedMachine(Component):
    """fixed speed machines"""
    def __init__(self):
        super(FixedSpeedMachine, self).__init__()
        self.add_param('Vin', shape=1, units='m/s', desc='cut-in wind speed')
        self.add_param('Vout', shape=1, units='m/s', desc='cut-out wind speed')
        self.add_param('ratedPower', shape=1, units='W', desc='rated power')
        self.add_param('Omega', shape=1, units='rpm', desc='fixed rotor rotation speed')
        self.add_param('maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

class RatedConditions(Component):
    """aerodynamic conditions at the rated wind speed"""
    def __init__(self):
        super(RatedConditions, self).__init__()
        self.add_param('V', shape=1, units='m/s', desc='rated wind speed')
        self.add_param('Omega', shape=1, units='rpm', desc='rotor rotation speed at rated')
        self.add_param('pitch', shape=1, units='deg', desc='pitch setting at rated')
        self.add_param('T', shape=1, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_param('T', shape=1, units='N*m', desc='rotor aerodynamic torque at rated')

class AeroLoads(Component):
    def __init__(self):
        super(AeroLoads, self).__init__()
        self.add_param('r', shape=1, units='m', desc='radial positions along blade going toward tip')
        self.add_param('Px', shape=1, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_param('Py', shape=1, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_param('Pz', shape=1, units='N/m', desc='distributed loads in blade-aligned z-direction')

        self.add_param('V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_param('Omega', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_param('pitch', shape=1, units='deg', desc='pitch angle')
        self.add_param('T', shape=1, units='deg', desc='azimuthal angle')


# ---------------------
# Base Components
# ---------------------

class GeometrySetupBase(Component):
    """a base component that computes the rotor radius from the geometry"""
    def __init__(self):
        super(GeometrySetupBase, self).__init__()
        self.add_output('R', shape=1, units='m', desc='rotor radius')


class AeroBase(Component):
    """A base component for a rotor aerodynamics code."""
    def __init__(self, naero, npower):
        super(AeroBase, self).__init__()

        # --- use these if (run_case == 'power') ---

        # inputs
        self.add_param('Uhub', shape=npower, units='m/s', desc='hub height wind speed')
        self.add_param('Omega', shape=npower, units='rpm', desc='rotor rotation speed')
        self.add_param('pitch', shape=npower, units='deg', desc='blade pitch setting')

        # outputs
        self.add_output('T', shape=npower, units='N', desc='rotor aerodynamic thrust')
        self.add_output('Q', shape=npower, units='N*m', desc='rotor aerodynamic torque')
        self.add_output('P', shape=npower, units='W', desc='rotor aerodynamic power')


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
        self.add_output('loads:r', shape=naero+2, units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads:Px', shape=naero+2, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads:Py', shape=naero+2, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads:Pz', shape=naero+2, units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_output('loads:V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_output('loads:Omega', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_output('loads:pitch', shape=1, units='deg', desc='pitch angle')
        self.add_output('loads:azimuth', shape=1, units='deg', desc='azimuthal angle')


class DrivetrainLossesBase(Component):
    """base component for drivetrain efficiency losses"""
    def __init__(self, npower):
        super(DrivetrainLossesBase, self).__init__()
        self.add_param('aeroPower', shape=npower, units='W', desc='aerodynamic power')
        self.add_param('aeroTorque', shape=npower, units='N*m', desc='aerodynamic torque')
        self.add_param('aeroThrust', shape=npower, units='N', desc='aerodynamic thrust')
        self.add_param('ratedPower', shape=1, units='W', desc='rated power')

        self.add_output('power', shape=npower, units='W', desc='total power after drivetrain losses')
        self.add_output('rpm', shape=npower, units='rpm', desc='rpm curve after drivetrain losses')


class PDFBase(Component):
    """probability distribution function"""
    def __init__(self, nspline):
        super(PDFBase, self).__init__()
        self.add_param('x', shape=nspline)

        self.add_output('f', shape=nspline)


class CDFBase(Component):
    """cumulative distribution function"""
    def __init__(self, nspline):
        super(CDFBase, self).__init__()

        self.add_param('x', shape=nspline,  units='m/s', desc='corresponding reference height')

        self.add_output('F', shape=nspline, units='m/s', desc='magnitude of wind speed at each z location')

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


# This Component is now no longer used, but I'll keep it around for now in case that changes.
class Coefficients(Component):
    def __init__(self, npower):
        super(Coefficients, self).__init__()
        """convert power, thrust, torque into nondimensional coefficient form"""

        # inputs
        self.add_param('V', shape=npower, units='m/s', desc='wind speed')
        self.add_param('T', shape=npower, units='N', desc='rotor aerodynamic thrust')
        self.add_param('Q', shape=npower, units='N*m', desc='rotor aerodynamic torque')
        self.add_param('P', shape=npower, units='W', desc='rotor aerodynamic power')

        # inputs used in normalization
        self.add_param('R', shape=1, units='m', desc='rotor radius')
        self.add_param('rho', shape=1, units='kg/m**3', desc='density of fluid')

        # outputs
        self.add_output('CT', shape=npower, desc='rotor aerodynamic thrust')
        self.add_output('CQ', shape=npower, desc='rotor aerodynamic torque')
        self.add_output('CP', shape=npower, desc='rotor aerodynamic power')

    def solve_nonlinear(self, params, unknowns, resids):

        q = 0.5 * params['rho'] * params['V']**2
        A = pi * params['R']**2
        unknowns['CP'] = params['P'] / (q * A * params['V'])
        unknowns['CT'] = params['T'] / (q * A)
        unknowns['CQ'] = params['Q'] / (q * params['R'] * A)

    def linearize(self, params, unknowns, resids):
        J = {}
        
        V = params['V']
        R = params['R']
        CT = unknowns['CT']
        CQ = unknowns['CQ']
        CP = unknowns['CP']
        n = len(params['V'])
        q = 0.5 * params['rho'] * V**2
        A = pi * R**2

        J['CT', 'V'] = np.diag(-2.0*CT/V)
        J['CT', 'T'] = np.diag(1.0/(q*A))
        J['CT', 'Q'] = np.zeros((n, n))
        J['CT', 'P'] = np.zeros((n, n))
        J['CT', 'R'] = -2.0*CT/R

        J['CQ', 'V'] = np.diag(-2.0*CQ/V)
        J['CQ', 'T'] = np.zeros((n, n))
        J['CQ', 'Q'] = np.diag(1.0/(q*R*A))
        J['CQ', 'P'] = np.zeros((n, n))
        J['CQ', 'R'] = -3.0*CQ/R

        J['CP', 'V'] = np.diag(-3.0*CP/V)
        J['CP', 'T'] = np.zeros((n, n))
        J['CP', 'Q'] = np.zeros((n, n))
        J['CP', 'P'] = np.diag(1.0/(q*A*V))
        J['CP', 'R'] = -2.0*CP/R

        return J



class SetupRunFixedSpeed(Component):
    def __init__(self):
        super(SetupRunFixedSpeed, self).__init__()
        """determines approriate conditions to run AeroBase code across the power curve"""

        self.add_param('control:Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control:Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control:ratedPower', units='W', desc='rated power')
        self.add_param('control:Omega', units='rpm', desc='fixed rotor rotation speed')
        self.add_param('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_param('control:npts', val=20, desc='number of points to evalute aero code to generate power curve', pass_by_obj=True)

        # outputs
        self.add_output('Uhub', units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', units='deg', desc='pitch angles to run')

        self.deriv_options['step_calc'] = 'absolute' #TODO
        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        n = params['control:npts']

        # velocity sweep
        V = np.linspace(params['control:Vin'], params['control:Vout'], n)

        # store values
        unknowns['Uhub'] = V
        unknowns['Omega'] = params['control:Omega']*np.ones_like(V)
        unknowns['pitch'] = params['control:pitch']*np.ones_like(V)

    def linearize(self, params, unknowns, resids):
        J = {}
        return J

class SetupRunVarSpeed(Component):
    def __init__(self, npower):
        super(SetupRunVarSpeed, self).__init__()
        """determines approriate conditions to run AeroBase code across the power curve"""

        self.add_param('control:Vin', shape=1, units='m/s', desc='cut-in wind speed')
        self.add_param('control:Vout', shape=1, units='m/s', desc='cut-out wind speed')
        self.add_param('control:ratedPower', shape=1, units='W', desc='rated power')
        self.add_param('control:minOmega', shape=1, units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control:maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control:tsr', shape=1, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control:pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_param('npts', val=200, desc='number of points for splined power curve', pass_by_obj=True)
        self.add_param('R', shape=1, units='m', desc='rotor radius')

        # outputs
        self.add_output('Uhub', shape=npower, units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', shape=npower, units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', shape=npower, units='deg', desc='pitch angles to run')

        self.deriv_options['type'] = 'fd' #TODO
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        n = params['npts']
        R = params['R']

        # # attempt to distribute points mostly before rated
        # cpguess = 0.5
        # Vr0 = (ctrl.ratedPower/(cpguess*0.5*rho*pi*R**2))**(1.0/3)
        # Vr0 *= 1.20

        # V1 = np.linspace(Vin, Vr0, 15)
        # V2 = np.linspace(Vr0, Vout, 6)
        # V = np.concatenate([V1, V2[1:]])

        # velocity sweep
        V = np.linspace(params['control:Vin'], params['control:Vout'], n)

        # corresponding rotation speed
        Omega_d = params['control:tsr']*V/R*RS2RPM
        Omega, dOmega_dOmegad, dOmega_dmaxOmega = smooth_min(Omega_d, params['control:maxOmega'], pct_offset=0.01)

        # store values
        unknowns['Uhub'] = V
        unknowns['Omega'] = Omega
        unknowns['pitch'] = params['control:pitch']*np.ones_like(V)

        # gradients
        J = {}
        J['Omega', 'control:tsr'] = dOmega_dOmegad * V/R*RS2RPM
        J['Omega', 'R'] = dOmega_dOmegad * -params['control:tsr']*V/R**2*RS2RPM
        J['Omega', 'control:maxOmega'] = dOmega_dmaxOmega

        self.J = J

    def linearize(self, params, unknowns, resids):

        return self.J





class UnregulatedPowerCurve(Component):
    def __init__(self, npower):
        super(UnregulatedPowerCurve, self).__init__()

        # inputs
        self.add_param('control:Vin', units='m/s', desc='cut-in wind speed')
        self.add_param('control:Vout', units='m/s', desc='cut-out wind speed')
        self.add_param('control:ratedPower', units='W', desc='rated power')
        self.add_param('control:Omega', units='rpm', desc='fixed rotor rotation speed')
        self.add_param('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_param('control:npts', val=npower, desc='number of points to evalute aero code to generate power curve', pass_by_obj=True)
        self.add_param('Vcoarse', shape=npower, units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', shape=npower, units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', shape=npower, units='N', desc='unregulated thrust curve')
        self.add_param('npts', val=200, desc='number of points for splined power curve', pass_by_obj=True)

        # outputs
        self.add_output('V', units='m/s', desc='wind speeds')
        self.add_output('P', units='W', desc='power')

    def solve_nonlinear(self, params, unknowns, resids):

        n = params['npts']

        # finer power curve
        V, _, _ = linspace_with_deriv(params['control:Vin'], params['control:Vout'], n)
        unknowns['V'] = V
        spline = Akima(params['Vcoarse'], params['Pcoarse'])
        P, dP_dV, dP_dVcoarse, dP_dPcoarse = spline.interp(unknowns['V'])
        unknowns['P'] = P

        J = {}
        J['P', 'Vcoarse'] = dP_dVcoarse
        J['P', 'Pcoarse'] = dP_dPcoarse
        self.J = J

    def linearize(self, params, unknowns, resids):

        return self.J


class RegulatedPowerCurve(Component): # Implicit COMPONENT
    def __init__(self, npower, nspline):
        super(RegulatedPowerCurve, self).__init__()

        """Fit a spline to the coarse sampled power curve (and thrust curve),
        find rated speed through a residual convergence strategy,
        then compute the regulated power curve and rated conditions"""

        # inputs
        self.add_param('control:Vin', val=0.0, units='m/s', desc='cut-in wind speed')
        self.add_param('control:Vout', shape=1, units='m/s', desc='cut-out wind speed')
        self.add_param('control:ratedPower', shape=1, units='W', desc='rated power')
        self.add_param('control:minOmega', shape=1, units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control:maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control:tsr', shape=1, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control:pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('Vcoarse', shape=npower, units='m/s', desc='wind speeds')
        self.add_param('Pcoarse', shape=npower, units='W', desc='unregulated power curve (but after drivetrain losses)')
        self.add_param('Tcoarse', shape=npower, units='N', desc='unregulated thrust curve')
        self.add_param('R', shape=1, units='m', desc='rotor radius')
        self.add_param('npts', val=nspline, desc='number of points for splined power curve', pass_by_obj=True)

        # state
        self.add_state('Vrated', val=11.0, units='m/s', desc='rated wind speed', lower=-1e-15, upper=1e15)

        # outputs
        self.add_output('V', shape=nspline, units='m/s', desc='wind speeds')
        self.add_output('P', shape=nspline, units='W', desc='power')

        self.add_output('ratedConditions:V', shape=1, units='m/s', desc='rated wind speed')
        self.add_output('ratedConditions:Omega', shape=1, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('ratedConditions:pitch', shape=1, units='deg', desc='pitch setting at rated')
        self.add_output('ratedConditions:T', shape=1, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('ratedConditions:Q', shape=1, units='N*m', desc='rotor aerodynamic torque at rated')
        self.add_output('azimuth', shape=1, units='deg', desc='azimuth load')

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def apply_nonlinear(self, params, unknowns, resids):
        n = params['npts']
        Vrated = unknowns['Vrated']
        # residual
        spline = Akima(params['Vcoarse'], params['Pcoarse'])
        P, dres_dVrated, dres_dVcoarse, dres_dPcoarse = spline.interp(Vrated)

        resids['Vrated'] = P - params['control:ratedPower']

        if True:
            P1, _, _, _ = spline.interp(params['control:Vin'])
            P2, _, _, _ = spline.interp(params['control:Vout'])
            resids1 = P1- params['control:ratedPower']
            resids2 = P2 - params['control:ratedPower']
            if ((resids1<0) == (resids2<0)):
                if Vrated == params['control:Vout']:
                    resids['Vrated'] = 10000
                elif Vrated != params['control:Vin']:
                    resids['Vrated'] = 0

        ## Test on

        # region 2
        V2, _, dV2_dVrated = linspace_with_deriv(params['control:Vin'], Vrated, n/2)
        P2, dP2_dV2, dP2_dVcoarse, dP2_dPcoarse = spline.interp(V2)

        # region 3
        V3, dV3_dVrated, _ = linspace_with_deriv(Vrated, params['control:Vout'], n/2+1)
        V3 = V3[1:]  # remove duplicate point
        dV3_dVrated = dV3_dVrated[1:]
        P3 = params['control:ratedPower']*np.ones_like(V3)

        # concatenate
        unknowns['V'] = np.concatenate([V2, V3])
        unknowns['P'] = np.concatenate([P2, P3])

        R = params['R']
        # rated speed conditions
        Omega_d = params['control:tsr']*Vrated/R*RS2RPM
        OmegaRated, dOmegaRated_dOmegad, dOmegaRated_dmaxOmega \
            = smooth_min(Omega_d, params['control:maxOmega'], pct_offset=0.01)

        splineT = Akima(params['Vcoarse'], params['Tcoarse'])
        Trated, dT_dVrated, dT_dVcoarse, dT_dTcoarse = splineT.interp(Vrated)

        unknowns['ratedConditions:V'] = Vrated
        unknowns['ratedConditions:Omega'] = OmegaRated
        unknowns['ratedConditions:pitch'] = params['control:pitch']
        unknowns['ratedConditions:T'] = Trated
        unknowns['ratedConditions:Q'] = params['control:ratedPower'] / (OmegaRated * RPM2RS)
        unknowns['azimuth'] = 180.0

        # gradients
        ncoarse = len(params['Vcoarse'])

        dV_dVrated = np.concatenate([dV2_dVrated, dV3_dVrated])

        dP_dVcoarse = vstack([dP2_dVcoarse, np.zeros((n/2, ncoarse))])
        dP_dPcoarse = vstack([dP2_dPcoarse, np.zeros((n/2, ncoarse))])
        dP_dVrated = np.concatenate([dP2_dV2*dV2_dVrated, np.zeros(n/2)])

        drOmega = np.concatenate([[dOmegaRated_dOmegad*Vrated/R*RS2RPM], np.zeros(3*ncoarse),
            [dOmegaRated_dOmegad*params['control:tsr']/R*RS2RPM, -dOmegaRated_dOmegad*params['control:tsr']*Vrated/R**2*RS2RPM,
            dOmegaRated_dmaxOmega]])
        drQ = -params['control:ratedPower'] / (OmegaRated**2 * RPM2RS) * drOmega

        J = {}
        J['Vrated', 'Vcoarse'] = np.reshape(dres_dVcoarse, (1, len(dres_dVcoarse)))
        J['Vrated', 'Pcoarse'] = np.reshape(dres_dPcoarse, (1, len(dres_dPcoarse)))
        J['Vrated', 'Vrated'] = dres_dVrated

        J['V', 'Vrated'] = dV_dVrated
        J['P', 'Vrated'] = dP_dVrated
        J['P', 'Vcoarse'] = dP_dVcoarse
        J['P', 'Pcoarse'] = dP_dPcoarse
        J['ratedConditions:V', 'Vrated'] = 1.0
        J['ratedConditions:Omega', 'control:tsr'] = dOmegaRated_dOmegad*Vrated/R*RS2RPM
        J['ratedConditions:Omega', 'Vrated'] = dOmegaRated_dOmegad*params['control:tsr']/R*RS2RPM
        J['ratedConditions:Omega', 'R'] = -dOmegaRated_dOmegad*params['control:tsr']*Vrated/R**2*RS2RPM
        J['ratedConditions:Omega', 'control:maxOmega'] = dOmegaRated_dmaxOmega
        J['ratedConditions:T', 'Vcoarse'] = np.reshape(dT_dVcoarse, (1, len(dT_dVcoarse)))
        J['ratedConditions:T', 'Tcoarse'] = np.reshape(dT_dTcoarse, (1, len(dT_dTcoarse)))
        J['ratedConditions:T', 'Vrated'] = dT_dVrated
        J['ratedConditions:Q', 'control:tsr'] = drQ[0]
        J['ratedConditions:Q', 'Vrated'] = drQ[-3]
        J['ratedConditions:Q', 'R'] = drQ[-2]
        J['ratedConditions:Q', 'control:maxOmega'] = drQ[-1]

        self.J = J


    def linearize(self, params, unknowns, resids):

        return self.J

class RegulatedPowerCurveGroup(Group):
    def __init__(self, npower, nspline):
        super(RegulatedPowerCurveGroup, self).__init__()
        self.add('powercurve_comp', RegulatedPowerCurve(npower, nspline), promotes=['*'])
        self.nl_solver = Brent()
        self.ln_solver = ScipyGMRES()
        self.nl_solver.options['var_lower_bound'] = 'powercurve.control:Vin'
        self.nl_solver.options['var_upper_bound'] = 'powercurve.control:Vout'
        self.nl_solver.options['state_var'] = 'Vrated'

        self.deriv_options['form'] = 'central'
        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

class AEP(Component):
    def __init__(self, nspline):
        super(AEP, self).__init__()
        """integrate to find annual energy production"""

        # inputs
        self.add_param('CDF_V', shape=nspline, units='m/s', desc='cumulative distribution function evaluated at each wind speed')
        self.add_param('P', shape=nspline, units='W', desc='power curve (power)')
        self.add_param('lossFactor', shape=1, desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', shape=1, units='kW*h', desc='annual energy production')

        self.deriv_options['step_size'] = 1.0

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['AEP'] = params['lossFactor']*np.trapz(params['P'], params['CDF_V'])/1e3*365.0*24.0  # in kWh

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

class COE(Component):
    def __init__(self):
        super(COE, self).__init__()
        """find cost of energy from mass and AEP (simplified model)"""

        # inputs
        self.add_param('mass_all_blades', shape=1, units='kg', desc='mass of all blades')
        self.add_param('AEP', shape=1, units='kW*h', desc='annual energy production')
        self.add_param('nBlades', val=3)

        # outputs
        self.add_output('COE', shape=1, units='$/kW*h', desc='cost of energy')

        self.deriv_options['step_size'] = 1.0

    def solve_nonlinear(self, params, unknowns, resids):
        # fixed cost assumptions from NREL 5-MW land-based turbine (update as needed)

        # compute the blade cost based on blade mass
        slope = 13.0
        intercept = 5813.9
        ppi_mat = 1.0465528035
        blade_cost = ((slope*params['mass_all_blades']/params['nBlades'] + intercept)*ppi_mat)

        # compute the parts costs
        hub_cost = 3.80 * 31644.5  # hub mass coefficient * hub mass
        rotor_cost = hub_cost + blade_cost*params['nBlades']
        nacelle_cost = 3495861.  # updated so that initial TCC matches tip-speed paper 1702*5e3
        tower_cost = 3.08 * 349644  # tower mass coefficient * tower mass
        parts_cost = (rotor_cost + nacelle_cost + tower_cost)

        # compute the turbine capital cost (TCC) with multipliers
        assemblyCostMultiplier = 0.30
        profitMultiplier = 0.20
        overheadCostMultiplier = 0.0
        transportMultiplier = 0.0
        turbine_multiplier = (1 + transportMultiplier + profitMultiplier) * (1 + overheadCostMultiplier + assemblyCostMultiplier)
        turbine_cost = turbine_multiplier * parts_cost

        # compute net annual energy production (AEP)
        array_losses = 0.059
        other_losses = 0.058  # used to make initial COE match tip-speed paper 6.22
        availability = 0.94
        losses = availability * (1-array_losses) * (1-other_losses)
        net_aep = (losses * params['AEP'])

        # compute cost of energy (COE)
        fixed_charge_rate = 0.095
        tax_deduction_rate = 0.4
        bos = 559. * 5e3  # $/kW * kW
        opex = 0.0122 * 19566000  # $/kWh * kWh
        unknowns['COE'] = (fixed_charge_rate*(turbine_cost+bos) + (1-tax_deduction_rate)*opex)/net_aep

        # analytic gradients
        self.dcoe_dmass_all_blades = fixed_charge_rate * turbine_multiplier * (slope/params['nBlades']*ppi_mat) / net_aep
        self.dcoe_dAEP = -fixed_charge_rate*(turbine_cost+bos) / (losses*(params['AEP']**2)) - ((1-tax_deduction_rate)*opex) / (losses*(params['AEP']**2))

    def linearize(self, params, unknowns, resids):

        J = {}
        J['COE', 'mass_all_blades'] = self.dcoe_dmass_all_blades
        J['COE', 'AEP'] = self.dcoe_dAEP

        return J




def common_io(group, varspeed, varpitch):

    regulated = varspeed or varpitch

    # add inputs
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


    # # add slots (must replace)
    # group.add('geom', Slot(GeomtrySetupBase))
    # group.add('analysis', Slot(AeroBase))
    # group.add('dt', Slot(DrivetrainLossesBase))
    # group.add('cdf', Slot(CDFBase))


    # add outputs
    group.add_output('AEP', units='kW*h', desc='annual energy production')
    group.add_output('V', units='m/s', desc='wind speeds (power curve)')
    group.add_output('P', units='W', desc='power (power curve)')
    group.add_output('diameter', units='m', desc='rotor diameter')
    if regulated:
        group.add_output('ratedConditions:V', units='m/s', desc='rated wind speed')
        group.add_output('ratedConditions:Omega', units='rpm', desc='rotor rotation speed at rated')
        group.add_output('ratedConditions:pitch', units='deg', desc='pitch setting at rated')
        group.add_output('ratedConditions:T', units='N', desc='rotor aerodynamic thrust at rated')
        group.add_output('ratedConditions:Q', units='N*m', desc='rotor aerodynamic torque at rated')


def common_configure(group, varspeed, varpitch, npower):

    regulated = varspeed or varpitch

    # add components
    group.add('geom', GeomtrySetupBase())

    if varspeed:
        group.add('setup', SetupRunVarSpeed(npower))
    else:
        group.add('setup', SetupRunFixedSpeed())

    group.add('analysis', AeroBase())
    group.add('dt', DrivetrainLossesBase())

    if varspeed or varpitch:
        group.add('powercurve', RegulatedPowerCurve(npower))
        group.add('brent', Brent())
        group.brent.workflow.add(['powercurve'])
    else:
        group.add('powercurve', UnregulatedPowerCurve(npower))

    group.add('cdf', CDFBase())
    group.add('aep', AEP())

    if regulated:
        group.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'brent', 'cdf', 'aep'])
    else:
        group.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'powercurve', 'cdf', 'aep'])


    # connections to setup
    group.connect('control', 'setup.control')
    group.connect('npts_coarse_power_curve', 'setup.npts')
    if varspeed:
        group.connect('geom.R', 'setup.R')


    # connections to analysis
    group.connect('setup.Uhub', 'analysis.Uhub')
    group.connect('setup.Omega', 'analysis.Omega')
    group.connect('setup.pitch', 'analysis.pitch')
    group.analysis.run_case = 'power'


    # connections to drivetrain
    group.connect('analysis.P', 'dt.aeroPower')
    group.connect('analysis.Q', 'dt.aeroTorque')
    group.connect('analysis.T', 'dt.aeroThrust')
    group.connect('control:ratedPower', 'dt.ratedPower')


    # connections to powercurve
    group.connect('control', 'powercurve.control')
    group.connect('setup.Uhub', 'powercurve.Vcoarse')
    group.connect('dt.power', 'powercurve.Pcoarse')
    group.connect('analysis.T', 'powercurve.Tcoarse')
    group.connect('npts_spline_power_curve', 'powercurve.npts')

    if regulated:
        group.connect('geom.R', 'powercurve.R')

        # setup Brent method to find rated speed
        group.connect('control:Vin', 'brent.lower_bound')
        group.connect('control:Vout', 'brent.upper_bound')
        group.brent.add_parameter('powercurve.Vrated', low=-1e-15, high=1e15)
        group.brent.add_constraint('powercurve.residual = 0')
        group.brent.invalid_bracket_return = 1.0


    # connections to cdf
    group.connect('powercurve.V', 'cdf.x')


    # connections to aep
    group.connect('cdf.F', 'aep.CDF_V')
    group.connect('powercurve.P', 'aep.P')
    group.connect('AEP_loss_factor', 'aep.lossFactor')


    # connections to outputs
    group.connect('powercurve.V', 'V')
    group.connect('powercurve.P', 'P')
    group.connect('aep.AEP', 'AEP')
    group.connect('2*geom.R', 'diameter')
    if regulated:
        group.connect('powercurve.ratedConditions', 'ratedConditions')



class RotorAeroVSVP(Group):

    def configure(self):
        varspeed = True
        varpitch = True
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroVSFP(Group):

    def configure(self):
        varspeed = True
        varpitch = False
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroFSVP(Group):

    def configure(self):
        varspeed = False
        varpitch = True
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)


class RotorAeroFSFP(Group):

    def configure(self):
        varspeed = False
        varpitch = False
        common_io(self, varspeed, varpitch)
        common_configure(self, varspeed, varpitch)





# class RotorAeroVS(Assembly):

#     # --- inputs ---
#     # rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
#     control = VarTree(VarSpeedMachine(), iotype='in')

#     # options
#     npts_coarse_power_curve = Int(20, iotype='in', desc='number of points to evaluate aero analysis at')
#     npts_spline_power_curve = Int(200, iotype='in', desc='number of points to use in fitting spline to power curve')
#     AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')

#     # slots (must replace)
#     geom = Slot(GeomtrySetupBase)
#     analysis = Slot(AeroBase)
#     dt = Slot(DrivetrainLossesBase)
#     cdf = Slot(CDFBase)

#     # --- outputs ---
#     AEP = Float(iotype='out', units='kW*h', desc='annual energy production')
#     V = Array(iotype='out', units='m/s', desc='wind speeds (power curve)')
#     P = Array(iotype='out', units='W', desc='power (power curve)')
#     ratedConditions = VarTree(RatedConditions(), iotype='out')
#     diameter = Float(iotype='out', units='m')


#     def configure(self):

#         self.add('geom', GeomtrySetupBase())
#         self.add('setup', SetupRun())
#         self.add('analysis', AeroBase())
#         self.add('dt', DrivetrainLossesBase())
#         self.add('powercurve', RegulatedPowerCurve())
#         self.add('brent', Brent())
#         self.add('cdf', CDFBase())
#         self.add('aep', AEP())

#         self.brent.workflow.add(['powercurve'])

#         self.driver.workflow.add(['geom', 'setup', 'analysis', 'dt', 'brent', 'cdf', 'aep'])

#         # connections to setup
#         self.connect('control', 'setup.control')
#         self.connect('geom.R', 'setup.R')
#         self.connect('npts_coarse_power_curve', 'setup.npts')

#         # connections to analysis
#         self.connect('setup.Uhub', 'analysis.Uhub')
#         self.connect('setup.Omega', 'analysis.Omega')
#         self.connect('setup.pitch', 'analysis.pitch')
#         self.analysis.run_case = 'power'

#         # connections to drivetrain
#         self.connect('analysis.P', 'dt.aeroPower')
#         self.connect('analysis.Q', 'dt.aeroTorque')
#         self.connect('analysis.T', 'dt.aeroThrust')
#         self.connect('control:ratedPower', 'dt.ratedPower')

#         # connections to powercurve
#         self.connect('control', 'powercurve.control')
#         self.connect('setup.Uhub', 'powercurve.Vcoarse')
#         self.connect('dt.power', 'powercurve.Pcoarse')
#         self.connect('analysis.T', 'powercurve.Tcoarse')
#         self.connect('geom.R', 'powercurve.R')
#         self.connect('npts_spline_power_curve', 'powercurve.npts')

#         # setup Brent method to find rated speed
#         self.connect('control:Vin', 'brent.lower_bound')
#         self.connect('control:Vout', 'brent.upper_bound')
#         self.brent.add_parameter('powercurve.Vrated', low=-1e-15, high=1e15)
#         self.brent.add_constraint('powercurve.residual = 0')

#         # connections to cdf
#         self.connect('powercurve.V', 'cdf.x')

#         # connections to aep
#         self.connect('cdf.F', 'aep.CDF_V')
#         self.connect('powercurve.P', 'aep.P')
#         self.connect('AEP_loss_factor', 'aep.lossFactor')


#         # connections to outputs
#         self.connect('powercurve.V', 'V')
#         self.connect('powercurve.P', 'P')
#         self.connect('aep.AEP', 'AEP')
#         self.connect('powercurve.ratedConditions', 'ratedConditions')
#         self.connect('2*geom.R', 'diameter')

