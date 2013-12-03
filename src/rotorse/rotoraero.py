#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2013-10-07.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
# from scipy.integrate import quad
from math import pi
from openmdao.main.api import VariableTree, Component, Assembly, Driver
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot, Enum
from openmdao.util.decorators import add_delegate
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasobjective import HasObjective
# from openmdao.main.hasobjectives import HasObjectives
# from openmdao.main.driver import Run_Once



# convert between rotations/minute and radians/second
RPM2RS = pi/30.0
RS2RPM = 30.0/pi


# ---------------------
# Variable Trees
# ---------------------


class VarSpeedMachine(VariableTree):
    """variable speed machines"""

    Vin = Float(units='m/s', desc='cut-in wind speed')
    Vout = Float(units='m/s', desc='cut-out wind speed')
    ratedPower = Float(units='W', desc='rated power')
    minOmega = Float(units='deg', desc='minimum allowed rotor rotation speed')
    maxOmega = Float(units='deg', desc='maximum allowed rotor rotation speed')
    tsr = Float(desc='tip-speed ratio in Region 2 (should be optimized externally)')
    pitch = Float(units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

    def update_outputs(one, two):  # TODO: remove this once vartree bug is fixed.
        pass


class FixedSpeedMachine(VariableTree):
    """fixed speed machines"""

    Vin = Float(units='m/s', desc='cut-in wind speed')
    Vout = Float(units='m/s', desc='cut-out wind speed')
    ratedPower = Float(units='W', desc='rated power')
    Omega = Float(units='rpm', desc='fixed rotor rotation speed')
    pitch = Float(units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')



class RatedConditions(VariableTree):
    """aerodynamic conditions at the rated wind speed"""

    V = Float(units='m/s', desc='rated wind speed')
    Omega = Float(units='rpm', desc='rotor rotation speed at rated')
    pitch = Float(units='deg', desc='pitch setting at rated')
    T = Float(units='N', desc='rotor aerodynamic thrust at rated')
    Q = Float(units='N*m', desc='rotor aerodynamic torque at rated')


class AeroLoads(VariableTree):

    r = Array()  # TODO: fill these out
    Px = Array()
    Py = Array()
    Pz = Array()

    V = Float()
    Omega = Float()
    pitch = Float()
    azimuth = Float()
    tilt = Float()

    def update_outputs(one, two):  # TODO: remove this once vartree bug is fixed.
        pass


# ---------------------
# Base Components
# ---------------------

class GeomtrySetupBase(Component):
    """a base component that computes the rotor radius from the geometry"""

    R = Float(iotype='out', units='m', desc='rotor radius')


class AeroBase(Component):
    """A base component for a rotor aerodynamics code."""

    run_case = Enum('power', ('power', 'loads'), iotype='in')


    # --- use these if (run_case == 'power') ---

    # inputs
    Uhub = Array(iotype='in', units='m/s', desc='hub height wind speed')
    Omega = Array(iotype='in', units='rpm', desc='rotor rotation speed')
    pitch = Array(iotype='in', units='deg', desc='blade pitch setting')

    # outputs
    T = Array(iotype='out', units='N', desc='rotor aerodynamic thrust')
    Q = Array(iotype='out', units='N*m', desc='rotor aerodynamic torque')
    P = Array(iotype='out', units='W', desc='rotor aerodynamic power')


    # --- use these if (run_case == 'loads') ---
    # if you only use rotoraero.py and not rotor.py
    # (i.e., only care about power curves, and not structural loads)
    # then these second set of inputs/outputs are not needed

    # inputs
    V_load = Float(iotype='in', units='m/s', desc='hub height wind speed')
    Omega_load = Float(iotype='in', units='rpm', desc='rotor rotation speed')
    pitch_load = Float(iotype='in', units='deg', desc='blade pitch setting')
    azimuth_load = Float(iotype='in', units='deg', desc='blade azimuthal location')

    # outputs
    loads = VarTree(AeroLoads(), iotype='out', desc='loads in blade-aligned coordinate system')




class DrivetrainLossesBase(Component):
    """base component for drivetrain efficiency losses"""

    aeroPower = Array(iotype='in', units='W', desc='aerodynamic power')
    aeroTorque = Array(iotype='in', units='N*m', desc='aerodynamic torque')
    aeroThrust = Array(iotype='in', units='N', desc='aerodynamic thrust')
    ratedPower = Float(iotype='in', units='W', desc='rated power')

    power = Array(iotype='out', units='W', desc='total power after drivetrain losses')


class PDFBase(Component):
    """probability distribution function"""

    x = Array(iotype='in')

    f = Array(iotype='out')


class CDFBase(Component):
    """cumulative distribution function"""

    x = Array(iotype='in')

    F = Array(iotype='out')





# ---------------------
# Components
# ---------------------

class Coefficients(Component):
    """convert power, thrust, torque into nondimensional coefficient form"""

    # inputs
    V = Array(iotype='in', units='m/s', desc='wind speed')
    T = Array(iotype='in', units='N', desc='rotor aerodynamic thrust')
    Q = Array(iotype='in', units='N*m', desc='rotor aerodynamic torque')
    P = Array(iotype='in', units='W', desc='rotor aerodynamic power')

    # inputs used in normalization
    R = Float(iotype='in', units='m', desc='rotor radius')
    rho = Float(iotype='in', units='kg/m**3', desc='density of fluid')


    # outputs
    CT = Array(iotype='out', desc='rotor aerodynamic thrust')
    CQ = Array(iotype='out', desc='rotor aerodynamic torque')
    CP = Array(iotype='out', desc='rotor aerodynamic power')


    def execute(self):

        q = 0.5 * self.rho * self.V**2
        A = pi * self.R**2
        self.CP = self.P / (q * A * self.V)
        self.CT = self.T / (q * A)
        self.CQ = self.Q / (q * self.R * A)




class SetupTSRFixedSpeed(Component):
    """find min and max tsr for a fixed speed machine"""

    # inputs
    R = Float(iotype='in', units='m', desc='rotor radius')
    control = VarTree(FixedSpeedMachine(), iotype='in')

    # outputs
    tsr_min = Float(iotype='out', desc='minimum tip speed ratio')
    tsr_max = Float(iotype='out', desc='maximum tip speed ratio')


    def execute(self):

        ctrl = self.control

        self.tsr_max = ctrl.Omega*RPM2RS*self.R/ctrl.Vin
        self.tsr_min = ctrl.Omega*RPM2RS*self.R/ctrl.Vout




class SetupTSRVarSpeed(Component):
    """find min and max tsr for a variable speed machine"""

    # inputs
    R = Float(iotype='in', units='m', desc='rotor radius')
    control = VarTree(VarSpeedMachine(), iotype='in')

    # outputs
    tsr_min = Float(iotype='out', desc='minimum tip speed ratio')
    tsr_max = Float(iotype='out', desc='maximum tip speed ratio')
    Omega_nominal = Float(iotype='out', units='rpm', desc='a nominal rotation speed to use in tsr sweep')


    def execute(self):

        ctrl = self.control
        R = self.R

        # at Vin
        tsr_low_Vin = ctrl.minOmega*RPM2RS*R/ctrl.Vin
        tsr_high_Vin = ctrl.maxOmega*RPM2RS*R/ctrl.Vin

        self.tsr_max = min(max(ctrl.tsr, tsr_low_Vin), tsr_high_Vin)

        # at Vout
        tsr_low_Vout = ctrl.minOmega*RPM2RS*R/ctrl.Vout
        tsr_high_Vout = ctrl.maxOmega*RPM2RS*R/ctrl.Vout

        self.tsr_min = max(min(ctrl.tsr, tsr_high_Vout), tsr_low_Vout)

        # a nominal rotation speed to use for this tip speed ratio (small Reynolds number effect)
        self.Omega_nominal = 0.5*(ctrl.maxOmega + ctrl.minOmega)





class SetupSweep(Component):
    """setup input conditions to AeroBase for a tsr sweep"""

    # inputs
    tsr_min = Float(iotype='in', desc='minimum tip speed ratio')
    tsr_max = Float(iotype='in', desc='maximum tip speed ratio')
    step_size = Float(0.25, iotype='in', desc='step size in tip-speed ratio for sweep (if necessary)')
    Omega_fixed = Float(iotype='in', units='rpm', desc='constant rotation speed used during tsr sweep')
    pitch_fixed = Float(iotype='in', units='deg', desc='set pitch angle')
    R = Float(iotype='in', units='m', desc='rotor radius')

    # outputs
    tsr = Array(iotype='out', desc='array of tip-speed ratios to run AeroBase at')
    Uinf = Array(iotype='out', units='m/s', desc='array of freestream velocities to run AeroBase at')
    Omega = Array(iotype='out', units='rpm', desc='array of rotation speeds to run AeroBase at')
    pitch = Array(iotype='out', units='deg', desc='array of pitch angles to run AeroBase at')


    def execute(self):

        # TODO: rethink this for gradients.  may need to use a fixed n even though some efficiency is lost

        n = int(round((self.tsr_max - self.tsr_min)/self.step_size))
        n = max(n, 1)

        # compute nominal power curve
        self.tsr = np.linspace(self.tsr_min, self.tsr_max, n)
        self.Omega = self.Omega_fixed*np.ones(n)
        self.pitch = self.pitch_fixed*np.ones(n)
        self.Uinf = self.Omega*RPM2RS*self.R/self.tsr




class AeroPowerCurveVarSpeed(Component):
    """create aerodynamic power curve for variable speed machines (no drivetrain losses, no regulation)"""

    # inputs
    tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
    CP_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
    CT_array = Array(iotype='in', desc='corresponding thrust coefficients')
    control = VarTree(VarSpeedMachine(), iotype='in')
    R = Float(iotype='in', units='m', desc='rotor radius')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    npts = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')

    # outputs
    V = Array(iotype='out', units='m/s', desc='aerodynamic power curve (wind speeds)')
    Omega = Array(iotype='out', units='rpm', desc='rotation speed curve')
    P = Array(iotype='out', units='W', desc='aerodynamic power curve (corresponding power)')
    T = Array(iotype='out', units='N', desc='thrust curve')
    Q = Array(iotype='out', units='N*m', desc='torque curve')


    def execute(self):

        R = self.R
        ctrl = self.control


        # evaluate from cut-in to cut-out
        V = np.linspace(ctrl.Vin, ctrl.Vout, self.npts)

        # evalaute nondimensional power curve using a spline
        if len(self.tsr_array) == 1:
            cp = self.CP_array[0]
            ct = self.CT_array[0]
            self.Omega = V*self.tsr_array[0]/R

        else:
            tsr = ctrl.tsr*np.ones(len(V))
            min_tsr = ctrl.minOmega*RPM2RS*R/V
            max_tsr = ctrl.maxOmega*RPM2RS*R/V
            tsr = np.minimum(np.maximum(tsr, min_tsr), max_tsr)

            self.Omega = V*tsr/R

            spline = interpolate.interp1d(self.tsr_array, self.CP_array, kind='cubic')
            cp = spline(tsr)

            spline = interpolate.interp1d(self.tsr_array, self.CT_array, kind='cubic')
            ct = spline(tsr)

        # convert to dimensional form
        self.V = V
        self.P = cp * 0.5*self.rho*V**3 * pi*R**2
        self.T = ct * 0.5*self.rho*V**2 * pi*R**2
        self.Q = self.P / self.Omega
        self.Omega *= RS2RPM




class AeroPowerCurveFixedSpeed(Component):
    """create aerodynamic power curve for fixed speed machines (no drivetrain losses, no regulation)"""

    # inputs
    tsr_array = Array(iotype='in', desc='tip speed ratios for nondimensional power curve')
    cp_array = Array(iotype='in', desc='corresponding power coefficients for nondimensional power curve')
    control = VarTree(FixedSpeedMachine(), iotype='in')
    R = Float(iotype='in', units='m', desc='rotor radius')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    npts = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')

    # outputs
    V = Array(iotype='out', units='m/s', desc='aerodynamic power curve (wind speeds)')
    P = Array(iotype='out', units='W', desc='aerodynamic power curve (corresponding power)')


    def execute(self):

        R = self.R
        ctrl = self.control

        # evaluate from cut-in to cut-out
        V = np.linspace(ctrl.Vin, ctrl.Vout, self.npts)
        tsr = ctrl.Omega*RPM2RS*R/V

        spline = interpolate.interp1d(self.tsr_array, self.cp_array, kind='cubic')
        cp = spline(tsr)

        # convert to dimensional form
        self.V = V
        self.P = cp * 0.5*self.rho*V**3 * pi*R**2



class RegulatedPowerCurve(Component):
    """power curve after drivetrain efficiency losses and control regulation"""

    # inputs
    V = Array(iotype='in', units='m/s', desc='wind speeds')
    P0 = Array(iotype='in', units='W', desc='unregulated power curve (but after drivetrain losses)')
    ratedPower = Float(iotype='in', units='W', desc='rated power')

    # outputs
    ratedSpeed = Float(iotype='out', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
    P = Array(iotype='out', units='W', desc='power curve (power)')

    def execute(self):

        self.P = np.copy(self.P0)

        # find rated speed
        idx = np.argmax(self.P)

        if (self.P[idx] <= self.ratedPower):  # check if rated power not reached
            self.ratedSpeed = self.V[idx]  # no rated speed, but provide max power as we use rated speed as a "worst case"

        else:  # apply control regulation
            self.ratedSpeed = np.interp(self.ratedPower, self.P[:idx], self.V[:idx])
            self.P[self.V > self.ratedSpeed] = self.ratedPower



class AEP(Component):
    """annual energy production"""

    # inputs
    CDF_V = Array(iotype='in')
    P = Array(iotype='in', units='W', desc='power curve (power)')
    lossFactor = Float(iotype='in', desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

    # outputs
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')


    def execute(self):

        self.AEP = self.lossFactor*np.trapz(self.P/1e3, self.CDF_V*365.0*24.0)  # in kWh



class SetupRatedConditionsVarSpeed(Component):
    """setup to run AeroBase at rated conditions for a variable speed machine"""

    # inputs
    ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')
    control = VarTree(VarSpeedMachine(), iotype='in')
    R = Float(iotype='in', units='m', desc='rotor radius')

    # outputs (1D arrays because AeroBase expects Array input)
    V_rated = Array(np.zeros(1), iotype='out', shape=((1,)), units='m/s', desc='rated speed')
    Omega_rated = Array(np.zeros(1), iotype='out', shape=((1,)), units='rpm', desc='rotation speed at rated')
    pitch_rated = Array(np.zeros(1), iotype='out', shape=((1,)), units='deg', desc='pitch setting at rated')


    def execute(self):

        ctrl = self.control

        self.V_rated = np.array([self.ratedSpeed])
        self.Omega_rated = np.array([min(ctrl.maxOmega, self.ratedSpeed*ctrl.tsr/self.R*RS2RPM)])
        self.pitch_rated = np.array([ctrl.pitch])




# ---------------------
# Assemblies
# ---------------------




class RotorAeroVS(Assembly):

    # --- inputs ---
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    control = VarTree(VarSpeedMachine(), iotype='in')

    # options
    tsr_sweep_step_size = Float(0.25, iotype='in', desc='step size in tip-speed ratio for sweep (if necessary)')
    npts_power_curve = Int(200, iotype='in', desc='number of points to generate power curve at (nondimensional power curve is sampled with a sample so a large number is not expensive)')
    AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')


    # --- slots (must replace) ---
    geom = Slot(GeomtrySetupBase)
    analysis = Slot(AeroBase)
    analysis2 = Slot(AeroBase)  # TODO: figure out how to replace both at same time
    dt = Slot(DrivetrainLossesBase)
    cdf = Slot(CDFBase)

    # --- outputs ---
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')
    V = Array(iotype='out', units='m/s', desc='wind speeds (power curve)')
    P = Array(iotype='out', units='W', desc='power (power curve)')
    rated = VarTree(RatedConditions(), iotype='out')



    def replace(self, name, obj):

        if name == 'analysis2':
            obj.T = np.zeros(1)
            obj.Q = np.zeros(1)

        super(RotorAeroVS, self).replace(name, obj)


    def configure(self):

        self.add('geom', GeomtrySetupBase())
        self.add('setuptsr', SetupTSRVarSpeed())
        self.add('sweep', SetupSweep())
        self.add('analysis', AeroBase())
        self.add('coeff', Coefficients())
        self.add('aeroPC', AeroPowerCurveVarSpeed())
        self.add('dt', DrivetrainLossesBase())
        self.add('powercurve', RegulatedPowerCurve())
        self.add('cdf', CDFBase())
        self.add('aep', AEP())
        self.add('rated_pre', SetupRatedConditionsVarSpeed())
        self.add('analysis2', AeroBase())

        self.driver.workflow.add(['geom', 'setuptsr', 'sweep', 'analysis', 'coeff',
            'aeroPC', 'dt', 'powercurve', 'cdf', 'aep', 'rated_pre', 'analysis2'])

        # connections to setuptsr
        self.connect('control', 'setuptsr.control')
        self.connect('geom.R', 'setuptsr.R')

        # connections to sweep
        self.connect('setuptsr.tsr_min', 'sweep.tsr_min')
        self.connect('setuptsr.tsr_max', 'sweep.tsr_max')
        self.connect('setuptsr.Omega_nominal', 'sweep.Omega_fixed')
        self.connect('control.pitch', 'sweep.pitch_fixed')
        self.connect('geom.R', 'sweep.R')
        self.connect('tsr_sweep_step_size', 'sweep.step_size')

        # connections to analysis
        self.connect('sweep.Uinf', 'analysis.Uhub')
        self.connect('sweep.Omega', 'analysis.Omega')
        self.connect('sweep.pitch', 'analysis.pitch')
        self.analysis.run_case = 'power'

        # connections to coefficients
        self.connect('sweep.Uinf', 'coeff.V')
        self.connect('analysis.T', 'coeff.T')
        self.connect('analysis.Q', 'coeff.Q')
        self.connect('analysis.P', 'coeff.P')
        self.connect('geom.R', 'coeff.R')
        self.connect('rho', 'coeff.rho')

        # connections to aeroPC
        self.connect('sweep.tsr', 'aeroPC.tsr_array')
        self.connect('coeff.CP', 'aeroPC.CP_array')
        self.connect('coeff.CT', 'aeroPC.CT_array')
        self.connect('control', 'aeroPC.control')
        self.connect('geom.R', 'aeroPC.R')
        self.connect('rho', 'aeroPC.rho')
        self.connect('npts_power_curve', 'aeroPC.npts')

        # connections to drivetrain
        self.connect('aeroPC.P', 'dt.aeroPower')
        self.connect('aeroPC.Q', 'dt.aeroTorque')
        self.connect('aeroPC.T', 'dt.aeroThrust')
        self.connect('control.ratedPower', 'dt.ratedPower')

        # connections to powercurve
        self.connect('aeroPC.V', 'powercurve.V')
        self.connect('dt.power', 'powercurve.P0')
        self.connect('control.ratedPower', 'powercurve.ratedPower')

        # connections to cdf
        self.connect('aeroPC.V', 'cdf.x')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to rated_pre
        self.connect('powercurve.ratedSpeed', 'rated_pre.ratedSpeed')
        self.connect('control', 'rated_pre.control')
        self.connect('geom.R', 'rated_pre.R')

        # connections to analysis2
        self.connect('rated_pre.V_rated', 'analysis2.Uhub')
        self.connect('rated_pre.Omega_rated', 'analysis2.Omega')
        self.connect('rated_pre.pitch_rated', 'analysis2.pitch')
        self.analysis2.run_case = 'power'
        self.analysis2.T = np.zeros(1)
        self.analysis2.Q = np.zeros(1)

        # connections to outputs
        self.connect('aeroPC.V', 'V')
        self.connect('powercurve.P', 'P')
        self.connect('aep.AEP', 'AEP')
        self.connect('rated_pre.V_rated[0]', 'rated.V')
        self.connect('rated_pre.Omega_rated[0]', 'rated.Omega')
        self.connect('rated_pre.pitch_rated[0]', 'rated.pitch')
        self.connect('analysis2.T[0]', 'rated.T')
        self.connect('analysis2.Q[0]', 'rated.Q')






# --------------------
# Not using right now
# ---------------------




class SetupPitchSearchVS(Component):

    control = VarTree(VarSpeedMachine(), iotype='in')

    R = Float(iotype='in', units='m', desc='rotor radius')
    ratedSpeed = Float(iotype='in', units='m/s', desc='rated speed (if regulated) otherwise speed for max power')

    Omega = Float(iotype='out')
    pitch_min = Float(iotype='out')
    pitch_max = Float(iotype='out')

    def execute(self):

        ctrl = self.control
        self.Omega = min(ctrl.maxOmega, self.ratedSpeed*ctrl.tsr/self.R*RS2RPM)

        # pitch to feather
        self.pitch_min = ctrl.pitch
        self.pitch_max = 40.0


@add_delegate(HasParameters, HasObjective)
class Brent(Driver):
    """root finding using Brent's method."""

    fstar = Float(0.0, iotype='in', desc='function value sought (i.e., find root of f-fstar)')
    xlow = Float(iotype='in')
    xhigh = Float(iotype='in')

    xstar = Float(iotype='out', desc='value sought.  find xstar s.t. f(xstar) = 0')


    def eval(self, x):
        """evaluate f(x)"""

        self.set_parameter_by_name('x', x)
        self.run_iteration()
        obj = self.get_objectives()
        f = obj['f'].evaluate(self.parent)

        return f


    def _error(self, x):
        """solve: f - fstar = 0"""

        f = self.eval(x)

        return f - self.fstar


    def execute(self):

        # bounds
        # params = self.get_parameters()
        # xlow = params['x'].low
        # xhigh = params['x'].high
        xlow = self.xlow
        xhigh = self.xhigh

        # TODO: better error handling.  ideally, if this failed we would attempt to find
        # bounds ourselves rather than throwing an error or returning a value at the bounds
        if self.eval(xlow)*self.eval(xhigh) > 0:
            raise Exception('bounds do not bracket a root')

        # Brent's method
        xstar = brentq(self._error, xlow, xhigh)

        # set result
        self.set_parameter_by_name('x', xstar)
        self.xstar = xstar




# class RotorAeroVSVP(RotorAeroVS):

#     V_load = Float(iotype='in')
#     azimuth_load = Float(iotype='in')

#     def configure(self):
#         super(RotorAeroVSVP, self).configure()

#         # --- residual workflow ---

#         self.add('setuppitch', SetupPitchSearchVS())
#         # self.add('analysis3', AeroBase())  # TODO: replace back later
#         # self.add('dt2', DrivetrainLossesBase())
#         self.add('analysis3', CCBlade())
#         self.add('dt2', CSMDrivetrain())
#         self.add('brent', Brent())

#         # connections to setuppitch
#         self.connect('control', 'setuppitch.control')
#         self.connect('R', 'setuppitch.R')
#         self.connect('powercurve.ratedSpeed', 'setuppitch.ratedSpeed')

#         # connections to analysis3
#         self.analysis3.Uhub = np.zeros(1)
#         self.analysis3.Omega = np.zeros(1)
#         self.analysis3.pitch = np.zeros(1)
#         self.analysis3.run_case = 'power'
#         self.connect('V_load', 'analysis3.Uhub[0]')
#         self.connect('setuppitch.Omega', 'analysis3.Omega[0]')
#         self.brent.add_parameter('analysis3.pitch[0]', name='x', low=0.0, high=40.0)

#         # connections to drivetrain
#         self.connect('analysis3.P', 'dt2.aeroPower')
#         self.connect('control.ratedPower', 'dt2.ratedPower')

#         # brent setup
#         self.brent.workflow.add(['setuppitch', 'analysis3', 'dt2'])
#         self.brent.add_objective('dt2.power[0]', name='f')
#         self.connect('control.ratedPower', 'brent.fstar')
#         self.connect('setuppitch.pitch_min', 'brent.xlow')
#         self.connect('setuppitch.pitch_max', 'brent.xhigh')


#         # --- functional workflow ---

#         # self.add('analysis4', AeroBase())  TODO
#         self.add('analysis4', CCBlade())

#         self.driver.workflow.add(['brent', 'analysis4'])

#         # connections to analysis4
#         self.analysis4.run_case = 'loads'
#         self.connect('V_load', 'analysis4.V_load')
#         self.connect('setuppitch.Omega', 'analysis4.Omega_load')
#         self.connect('brent.xstar', 'analysis4.pitch_load')
#         self.connect('azimuth_load', 'analysis4.azimuth_load')

#         self.create_passthrough('analysis4.Np')
#         self.create_passthrough('analysis4.Tp')




#         # TODO: remove later
#         self.connect('r', 'analysis3.r')
#         self.connect('chord', 'analysis3.chord')
#         self.connect('theta', 'analysis3.theta')
#         self.connect('Rhub', 'analysis3.Rhub')
#         self.connect('Rtip', 'analysis3.Rtip')
#         self.connect('hubheight', 'analysis3.hubheight')
#         self.connect('airfoil_files', 'analysis3.airfoil_files')
#         self.connect('precone', 'analysis3.precone')
#         self.connect('tilt', 'analysis3.tilt')
#         self.connect('yaw', 'analysis3.yaw')
#         self.connect('B', 'analysis3.B')
#         self.connect('rho', 'analysis3.rho')
#         self.connect('mu', 'analysis3.mu')
#         self.connect('shearExp', 'analysis3.shearExp')
#         self.connect('nSector', 'analysis3.nSector')
#         self.connect('drivetrainType', 'dt2.drivetrainType')
#         self.connect('r', 'analysis4.r')
#         self.connect('chord', 'analysis4.chord')
#         self.connect('theta', 'analysis4.theta')
#         self.connect('Rhub', 'analysis4.Rhub')
#         self.connect('Rtip', 'analysis4.Rtip')
#         self.connect('hubheight', 'analysis4.hubheight')
#         self.connect('airfoil_files', 'analysis4.airfoil_files')
#         self.connect('precone', 'analysis4.precone')
#         self.connect('tilt', 'analysis4.tilt')
#         self.connect('yaw', 'analysis4.yaw')
#         self.connect('B', 'analysis4.B')
#         self.connect('rho', 'analysis4.rho')
#         self.connect('mu', 'analysis4.mu')
#         self.connect('shearExp', 'analysis4.shearExp')
#         self.connect('nSector', 'analysis4.nSector')






