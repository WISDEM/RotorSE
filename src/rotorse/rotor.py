#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

# from __future__ import print_function
import numpy as np
import os
from openmdao.api import IndepVarComp, Component, Group, Problem, ExecComp
from rotor_aeropower import SetupRunVarSpeed, CSMDrivetrain, RegulatedPowerCurveGroup, AEP, OutputsAero
from rotor_structure import ResizeCompositeSection, BladeCurvature, CurveFEM, DamageLoads, TotalLoads, TipDeflection, BladeDeflection, RootMoment, MassProperties, ExtremeLoads, GustETM, SetupPCModVarSpeed, OutputsStructures, PreCompSections, RotorWithpBEAM

from ccblade.ccblade_component import CCBladeGeometry, CCBlade
from commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from commonse.environment import PowerWind
from precomp import Profile, Orthotropic2DMaterial, CompositeSection
from rotor_geometry import GridSetup, RGrid, GeometrySpline, TurbineClass

from rotorse import RPM2RS, RS2RPM, TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE


class RotorSE(Group):
    def __init__(self, naero=17, nstr=38, npts_coarse_power_curve=20, npts_spline_power_curve=200):
        super(RotorSE, self).__init__()
        """rotor model"""

        self.add('initial_aero_grid', IndepVarComp('initial_aero_grid', np.zeros(naero)), promotes=['*'])
        self.add('initial_str_grid', IndepVarComp('initial_str_grid', np.zeros(nstr)), promotes=['*'])
        self.add('idx_cylinder_aero', IndepVarComp('idx_cylinder_aero', 0, pass_by_obj=True), promotes=['*'])
        self.add('idx_cylinder_str', IndepVarComp('idx_cylinder_str', 0, pass_by_obj=True), promotes=['*'])
        self.add('hubFraction', IndepVarComp('hubFraction', 0.0), promotes=['*'])
        self.add('r_aero', IndepVarComp('r_aero', np.zeros(naero)), promotes=['*'])
        self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
        self.add('chord_sub', IndepVarComp('chord_sub', np.zeros(4),units='m'), promotes=['*'])
        self.add('theta_sub', IndepVarComp('theta_sub', np.zeros(4), units='deg'), promotes=['*'])
        self.add('precurve_sub', IndepVarComp('precurve_sub', np.zeros(3), units='m'), promotes=['*'])
        #self.add('delta_precurve_sub', IndepVarComp('delta_precurve_sub', np.zeros(3)), promotes=['*'])
        self.add('bladeLength', IndepVarComp('bladeLength', 0.0, units='m'), promotes=['*'])
	self.add('delta_bladeLength', IndepVarComp('delta_bladeLength', 0.0, units='m', desc='adjustment to blade length to account for curvature from loading'), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0, units='deg'), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0, units='deg'), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0, units='deg'), promotes=['*'])
        self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])
        self.add('airfoil_files', IndepVarComp('airfoil_files', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])
        self.add('rho', IndepVarComp('rho', val=1.225, units='kg/m**3', desc='density of air', pass_by_obj=True), promotes=['*'])
        self.add('mu', IndepVarComp('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air', pass_by_obj=True), promotes=['*'])
        self.add('shearExp', IndepVarComp('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', val=np.zeros(1), units='m', desc='hub height'), promotes=['*'])
        self.add('turbine_class', IndepVarComp('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
        self.add('turbulence_class', IndepVarComp('turbulence_class', val=TURBULENCE_CLASS['A'], desc='IEC turbulence class class', pass_by_obj=True), promotes=['*'])
        self.add('cdf_reference_height_wind_speed', IndepVarComp('cdf_reference_height_wind_speed', val=0.0, units='m', desc='reference hub height for IEC wind speed (used in CDF calculation)'), promotes=['*'])
        self.add('VfactorPC', IndepVarComp('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation'), promotes=['*'])

        # --- composite sections ---
        self.add('sparT', IndepVarComp('sparT', val=np.zeros(5), units='m', desc='spar cap thickness parameters'), promotes=['*'])
        self.add('teT', IndepVarComp('teT', val=np.zeros(5), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])
        self.add('chord_str_ref', IndepVarComp('chord_str_ref', val=np.zeros(nstr), units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)'), promotes=['*'])
        self.add('leLoc', IndepVarComp('leLoc', val=np.zeros(nstr), desc='array of leading-edge positions from a reference blade axis \
            (usually blade pitch axis). locations are normalized by the local chord length.  \
            e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
            positive in -x direction for airfoil-aligned coordinate system'), promotes=['*'])

        self.add('profile', IndepVarComp('profile', val=np.zeros(nstr), desc='airfoil shape at each radial position', pass_by_obj=True), promotes=['*'])
        self.add('materials', IndepVarComp('materials', val=np.zeros(6),
            desc='list of all Orthotropic2DMaterial objects used in defining the geometry', pass_by_obj=True), promotes=['*'])
        self.add('upperCS', IndepVarComp('upperCS', val=np.zeros(nstr),
            desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True), promotes=['*'])
        self.add('lowerCS', IndepVarComp('lowerCS', val=np.zeros(nstr),
            desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True), promotes=['*'])
        self.add('websCS', IndepVarComp('websCS', val=np.zeros(nstr),
            desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True), promotes=['*'])
        self.add('sector_idx_strain_spar', IndepVarComp('sector_idx_strain_spar', val=np.zeros(38,  dtype=np.int), desc='index of sector for spar (PreComp definition of sector)', pass_by_obj=True), promotes=['*'])
        self.add('sector_idx_strain_te', IndepVarComp('sector_idx_strain_te', val=np.zeros(38,  dtype=np.int), desc='index of sector for trailing-edge (PreComp definition of sector)', pass_by_obj=True), promotes=['*'])

        # --- control ---
        self.add('c_Vin', IndepVarComp('control:Vin', val=0.0, units='m/s', desc='cut-in wind speed'), promotes=['*'])
        self.add('c_Vout', IndepVarComp('control:Vout', val=0.0, units='m/s', desc='cut-out wind speed'), promotes=['*'])
        self.add('c_ratedPower', IndepVarComp('control:ratedPower', val=0.0,  units='W', desc='rated power'), promotes=['*'])
        self.add('c_minOmega', IndepVarComp('control:minOmega', val=0.0, units='rpm', desc='minimum allowed rotor rotation speed'), promotes=['*'])
        self.add('c_maxOmega', IndepVarComp('control:maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed'), promotes=['*'])
        self.add('c_tsr', IndepVarComp('control:tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)'), promotes=['*'])
        self.add('c_pitch', IndepVarComp('control:pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)'), promotes=['*'])
        self.add('pitch_extreme', IndepVarComp('pitch_extreme', val=0.0, units='deg', desc='worst-case pitch at survival wind condition'), promotes=['*'])
        self.add('pitch_extreme_full', IndepVarComp('pitch_extreme_full', val=np.array([0.0, 90.0]), units='deg', desc='worst-case pitch at survival wind condition'), promotes=['*'])
        self.add('azimuth_extreme', IndepVarComp('azimuth_extreme', val=0.0, units='deg', desc='worst-case azimuth at survival wind condition'), promotes=['*'])
        self.add('Omega_load', IndepVarComp('Omega_load', val=0.0, units='rpm', desc='worst-case azimuth at survival wind condition'), promotes=['*'])

        # --- drivetrain efficiency ---
        self.add('drivetrainType', IndepVarComp('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True), promotes=['*'])


        # --- fatigue ---
        self.add('rstar_damage', IndepVarComp('rstar_damage', val=np.zeros(naero+1), desc='nondimensional radial locations of damage equivalent moments'), promotes=['*'])
        self.add('Mxb_damage', IndepVarComp('Mxb_damage', val=np.zeros(naero+1), units='N*m', desc='damage equivalent moments about blade c.s. x-direction'), promotes=['*'])
        self.add('Myb_damage', IndepVarComp('Myb_damage', val=np.zeros(naero+1), units='N*m', desc='damage equivalent moments about blade c.s. y-direction'), promotes=['*'])
        self.add('strain_ult_spar', IndepVarComp('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap'), promotes=['*'])
        self.add('strain_ult_te', IndepVarComp('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels'), promotes=['*'])
        self.add('eta_damage', IndepVarComp('eta_damage', val=1.755, desc='safety factor for fatigue'), promotes=['*'])
        self.add('m_damage', IndepVarComp('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis'), promotes=['*'])
        self.add('N_damage', IndepVarComp('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis'), promotes=['*'])


        # --- options ---
        self.add('nSector', IndepVarComp('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True), promotes=['*'])
        self.add('npts_coarse_power_curve', IndepVarComp('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at', pass_by_obj=True), promotes=['*'])
        self.add('npts_spline_power_curve', IndepVarComp('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve', pass_by_obj=True), promotes=['*'])
        self.add('AEP_loss_factor', IndepVarComp('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)'), promotes=['*'])
        self.add('dynamic_amplication_tip_deflection', IndepVarComp('dynamic_amplication_tip_deflection', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation'), promotes=['*'])
        self.add('nF', IndepVarComp('nF', val=5, desc='number of natural frequencies to compute', pass_by_obj=True), promotes=['*'])

        #self.add('weibull_shape', IndepVarComp('weibull_shape', val=0.0), promotes=['*'])

        
        # --- Rotor Aero & Power ---
        self.add('turbineclass', TurbineClass())
        self.add('gridsetup', GridSetup(naero, nstr))
        self.add('grid', RGrid(naero, nstr))
        self.add('spline0', GeometrySpline(naero, nstr))
        self.add('spline', GeometrySpline(naero, nstr))
        self.add('geom', CCBladeGeometry())
        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRunVarSpeed(npts_coarse_power_curve))
        self.add('analysis', CCBlade('power', naero, npts_coarse_power_curve))
        self.add('dt', CSMDrivetrain(npts_coarse_power_curve))
        self.add('powercurve', RegulatedPowerCurveGroup(npts_coarse_power_curve, npts_spline_power_curve))
        self.add('wind', PowerWind(1))
        # self.add('cdf', WeibullWithMeanCDF(npts_spline_power_curve))
        self.add('cdf', RayleighCDF(npts_spline_power_curve))
        self.add('aep', AEP(npts_spline_power_curve))

        self.add('outputs_aero', OutputsAero(npts_spline_power_curve), promotes=['*'])


        # --- add structures ---
        self.add('curvature', BladeCurvature(nstr))
        self.add('resize', ResizeCompositeSection(nstr))
        self.add('gust', GustETM())
        self.add('setuppc',  SetupPCModVarSpeed())
        self.add('aero_rated', CCBlade('loads', naero, 1))
        self.add('aero_extrm', CCBlade('loads', naero,  1))
        self.add('aero_extrm_forces', CCBlade('power', naero, 2))
        self.add('aero_defl_powercurve', CCBlade('loads', naero,  1))
        self.add('beam', PreCompSections(nstr))
        self.add('loads_defl', TotalLoads(nstr))
        self.add('loads_pc_defl', TotalLoads(nstr))
        self.add('loads_strain', TotalLoads(nstr))
        self.add('damage', DamageLoads(nstr, naero))
        self.add('struc', RotorWithpBEAM(nstr))
        self.add('curvefem', CurveFEM(nstr))
        self.add('tip', TipDeflection())
        self.add('root_moment', RootMoment(nstr))
        self.add('mass', MassProperties())
        self.add('extreme', ExtremeLoads())
        self.add('blade_defl', BladeDeflection(nstr))
        self.add('aero_0', CCBlade('loads', naero,  1))
        self.add('aero_120', CCBlade('loads', naero,  1))
        self.add('aero_240', CCBlade('loads', naero,  1))
        self.add('root_moment_0', RootMoment(nstr))
        self.add('root_moment_120', RootMoment(nstr))
        self.add('root_moment_240', RootMoment(nstr))

        self.add('output_struc', OutputsStructures(nstr), promotes=['*'])
        
        # connections to turbineclass
        self.connect('turbine_class', 'turbineclass.turbine_class')

        # connections to gridsetup
        self.connect('initial_aero_grid', 'gridsetup.initial_aero_grid')
        self.connect('initial_str_grid', 'gridsetup.initial_str_grid')

        # connections to grid
        self.connect('r_aero', 'grid.r_aero')
        self.connect('gridsetup.fraction', 'grid.fraction')
        self.connect('gridsetup.idxj', 'grid.idxj')

        # connections to spline0
        self.connect('r_aero', 'spline0.r_aero_unit')
        self.connect('grid.r_str', 'spline0.r_str_unit')
        self.connect('r_max_chord', 'spline0.r_max_chord')
        self.connect('chord_sub', 'spline0.chord_sub')
        self.connect('theta_sub', 'spline0.theta_sub')
        self.connect('precurve_sub', 'spline0.precurve_sub')
        self.connect('bladeLength', 'spline0.bladeLength')
        self.connect('idx_cylinder_aero', 'spline0.idx_cylinder_aero')
        self.connect('idx_cylinder_str', 'spline0.idx_cylinder_str')
        self.connect('hubFraction', 'spline0.hubFraction')
        self.connect('sparT', 'spline0.sparT')
        self.connect('teT', 'spline0.teT')

        # connections to spline
        self.connect('r_aero', 'spline.r_aero_unit')
        self.connect('grid.r_str', 'spline.r_str_unit')
        self.connect('r_max_chord', 'spline.r_max_chord')
        self.connect('chord_sub', 'spline.chord_sub')
        self.connect('theta_sub', 'spline.theta_sub')
        self.connect('precurve_sub', 'spline.precurve_sub')
        self.connect('bladeLength', 'spline.bladeLength')
        self.connect('idx_cylinder_aero', 'spline.idx_cylinder_aero')
        self.connect('idx_cylinder_str', 'spline.idx_cylinder_str')
        self.connect('hubFraction', 'spline.hubFraction')
        self.connect('sparT', 'spline.sparT')
        self.connect('teT', 'spline.teT')

        # connections to geom
        # self.spline['precurve_str'] = np.zeros(1)
        self.connect('spline.Rtip', 'geom.Rtip')
        self.connect('precone', 'geom.precone')
        self.connect('spline.precurve_str', 'geom.precurveTip', src_indices=[naero-1])

        # # connectiosn to tipspeed
        # self.connect('geom.R', 'tipspeed.R')
        # self.connect('max_tip_speed', 'tipspeed.Vtip_max')
        # self.connect('tipspeed.Omega_max', 'control:maxOmega')

        # connections to setup
        # self.connect('control', 'setup.control')
        self.connect('control:Vin', 'setup.control:Vin')
        self.connect('control:Vout', 'setup.control:Vout')
        self.connect('control:maxOmega', 'setup.control:maxOmega')
        self.connect('control:minOmega', 'setup.control:minOmega')
        self.connect('control:pitch', 'setup.control:pitch')
        self.connect('control:ratedPower', 'setup.control:ratedPower')
        self.connect('control:tsr', 'setup.control:tsr')
        self.connect('geom.R', 'setup.R')
        self.connect('npts_coarse_power_curve', 'setup.npts')

        # connections to analysis
        self.connect('spline.r_aero', 'analysis.r')
        self.connect('spline.chord_aero', 'analysis.chord')
        self.connect('spline.theta_aero', 'analysis.theta')
        self.connect('spline.precurve_aero', 'analysis.precurve')
        self.connect('spline.precurve_str', 'analysis.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'analysis.Rhub')
        self.connect('spline.Rtip', 'analysis.Rtip')
        self.connect('hubHt', 'analysis.hubHt')
        self.connect('precone', 'analysis.precone')
        self.connect('tilt', 'analysis.tilt')
        self.connect('yaw', 'analysis.yaw')
        self.connect('airfoil_files', 'analysis.airfoil_files')
        self.connect('nBlades', 'analysis.B')
        self.connect('rho', 'analysis.rho')
        self.connect('mu', 'analysis.mu')
        self.connect('shearExp', 'analysis.shearExp')
        self.connect('nSector', 'analysis.nSector')
        self.connect('setup.Uhub', 'analysis.Uhub')
        self.connect('setup.Omega', 'analysis.Omega')
        self.connect('setup.pitch', 'analysis.pitch')

        # connections to drivetrain
        self.connect('analysis.P', 'dt.aeroPower')
        self.connect('analysis.Q', 'dt.aeroTorque')
        self.connect('analysis.T', 'dt.aeroThrust')
        self.connect('control:ratedPower', 'dt.ratedPower')
        self.connect('drivetrainType', 'dt.drivetrainType')

        # connections to powercurve
        self.connect('control:Vin', 'powercurve.control:Vin')
        self.connect('control:Vout', 'powercurve.control:Vout')
        self.connect('control:maxOmega', 'powercurve.control:maxOmega')
        self.connect('control:minOmega', 'powercurve.control:minOmega')
        self.connect('control:pitch', 'powercurve.control:pitch')
        self.connect('control:ratedPower', 'powercurve.control:ratedPower')
        self.connect('control:tsr', 'powercurve.control:tsr')
        self.connect('setup.Uhub', 'powercurve.Vcoarse')
        self.connect('dt.power', 'powercurve.Pcoarse')
        self.connect('analysis.T', 'powercurve.Tcoarse')
        self.connect('geom.R', 'powercurve.R')
        self.connect('npts_spline_power_curve', 'powercurve.npts')

        # # setup Brent method to find rated speed
        # self.connect('control:Vin', 'brent.lower_bound')
        # self.connect('control:Vout', 'brent.upper_bound')
        # self.brent.add_param('powercurve.Vrated', low=-1e-15, high=1e15)
        # self.brent.add_constraint('powercurve.residual = 0')
        # self.brent.invalid_bracket_return = 1.0

        # connections to wind
        #self.wind.z = np.zeros(1)
        #self.wind.U = np.zeros(1)
        # self.connect('cdf_reference_mean_wind_speed', 'wind.Uref')
        self.connect('turbineclass.V_mean', 'wind.Uref')
        self.connect('cdf_reference_height_wind_speed', 'wind.zref')
        self.connect('hubHt', 'wind.z', src_indices=[0])
        self.connect('shearExp', 'wind.shearExp')

        # connections to cdf
        self.connect('powercurve.V', 'cdf.x')
        self.connect('wind.U', 'cdf.xbar', src_indices=[0])
        # self.connect('weibull_shape', 'cdf.k') #TODO

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to outputs
        self.connect('powercurve.V', 'V_in')
        self.connect('powercurve.P', 'P_in')
        self.connect('aep.AEP', 'AEP_in')
        self.connect('powercurve.ratedConditions:V', 'ratedConditions:V_in')
        self.connect('powercurve.ratedConditions:Omega', 'ratedConditions:Omega_in')
        self.connect('powercurve.ratedConditions:pitch', 'ratedConditions:pitch_in')
        self.connect('powercurve.ratedConditions:T', 'ratedConditions:T_in')
        self.connect('powercurve.ratedConditions:Q', 'ratedConditions:Q_in')


        self.connect('spline.diameter', 'hub_diameter_in')
        self.connect('geom.diameter', 'diameter_in')


        
        # Structures connections
        
        
        # connections to curvature
        self.connect('spline.r_str', 'curvature.r')
        self.connect('spline.precurve_str', 'curvature.precurve')
        self.connect('spline.presweep_str', 'curvature.presweep')
        self.connect('precone', 'curvature.precone')

        # connections to resize
        self.connect('upperCS', 'resize.upperCSIn')
        self.connect('lowerCS', 'resize.lowerCSIn')
        self.connect('websCS', 'resize.websCSIn')
        self.connect('chord_str_ref', 'resize.chord_str_ref')
        self.connect('sector_idx_strain_spar', 'resize.sector_idx_strain_spar')
        self.connect('sector_idx_strain_te', 'resize.sector_idx_strain_te')
        self.connect('spline.chord_str', 'resize.chord_str')
        self.connect('spline.sparT_str', 'resize.sparT_str')
        self.connect('spline.teT_str', 'resize.teT_str')

        # connections to gust
        self.connect('turbulence_class', 'gust.turbulence_class')
        self.connect('turbineclass.V_mean', 'gust.V_mean')
        self.connect('powercurve.ratedConditions:V', 'gust.V_hub')

        # connections to setuppc
        self.connect('control:pitch', 'setuppc.control:pitch')
        self.connect('control:tsr', 'setuppc.control:tsr')
        self.connect('powercurve.ratedConditions:V', 'setuppc.Vrated')
        self.connect('geom.R', 'setuppc.R')
        self.connect('VfactorPC', 'setuppc.Vfactor')

        # connections to aero_rated (for max deflection)
        self.connect('spline.r_aero', 'aero_rated.r')
        self.connect('spline.chord_aero', 'aero_rated.chord')
        self.connect('spline.theta_aero', 'aero_rated.theta')
        self.connect('spline.precurve_aero', 'aero_rated.precurve')
        self.connect('spline.precurve_str', 'aero_rated.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_rated.Rhub')
        self.connect('spline.Rtip', 'aero_rated.Rtip')
        self.connect('hubHt', 'aero_rated.hubHt')
        self.connect('precone', 'aero_rated.precone')
        self.connect('tilt', 'aero_rated.tilt')
        self.connect('yaw', 'aero_rated.yaw')
        self.connect('airfoil_files', 'aero_rated.airfoil_files')
        self.connect('nBlades', 'aero_rated.B')
        self.connect('rho', 'aero_rated.rho')
        self.connect('mu', 'aero_rated.mu')
        self.connect('shearExp', 'aero_rated.shearExp')
        self.connect('nSector', 'aero_rated.nSector')
        # self.connect('powercurve.ratedConditions:V + 3*gust.sigma', 'aero_rated.V_load')  # OpenMDAO bug
        self.connect('gust.V_gust', 'aero_rated.V_load')
        self.connect('powercurve.ratedConditions:Omega', 'aero_rated.Omega_load')
        self.connect('powercurve.ratedConditions:pitch', 'aero_rated.pitch_load')
        self.connect('powercurve.azimuth', 'aero_rated.azimuth_load')
        self.aero_rated.azimuth_load = 180.0  # closest to tower

        # connections to aero_extrm (for max strain)
        self.connect('spline.r_aero', 'aero_extrm.r')
        self.connect('spline.chord_aero', 'aero_extrm.chord')
        self.connect('spline.theta_aero', 'aero_extrm.theta')
        self.connect('spline.precurve_aero', 'aero_extrm.precurve')
        self.connect('spline.precurve_str', 'aero_extrm.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_extrm.Rhub')
        self.connect('spline.Rtip', 'aero_extrm.Rtip')
        self.connect('hubHt', 'aero_extrm.hubHt')
        self.connect('precone', 'aero_extrm.precone')
        self.connect('tilt', 'aero_extrm.tilt')
        self.connect('yaw', 'aero_extrm.yaw')
        self.connect('airfoil_files', 'aero_extrm.airfoil_files')
        self.connect('nBlades', 'aero_extrm.B')
        self.connect('rho', 'aero_extrm.rho')
        self.connect('mu', 'aero_extrm.mu')
        self.connect('shearExp', 'aero_extrm.shearExp')
        self.connect('nSector', 'aero_extrm.nSector')
        self.connect('turbineclass.V_extreme', 'aero_extrm.V_load')
        self.connect('pitch_extreme', 'aero_extrm.pitch_load')
        self.connect('azimuth_extreme', 'aero_extrm.azimuth_load')
        self.connect('Omega_load', 'aero_extrm.Omega_load')
        self.aero_extrm.Omega_load = 0.0  # parked case

        # connections to aero_extrm_forces (for tower thrust)
        self.connect('spline.r_aero', 'aero_extrm_forces.r')
        self.connect('spline.chord_aero', 'aero_extrm_forces.chord')
        self.connect('spline.theta_aero', 'aero_extrm_forces.theta')
        self.connect('spline.precurve_aero', 'aero_extrm_forces.precurve')
        self.connect('spline.precurve_str', 'aero_extrm_forces.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_extrm_forces.Rhub')
        self.connect('spline.Rtip', 'aero_extrm_forces.Rtip')
        self.connect('hubHt', 'aero_extrm_forces.hubHt')
        self.connect('precone', 'aero_extrm_forces.precone')
        self.connect('tilt', 'aero_extrm_forces.tilt')
        self.connect('yaw', 'aero_extrm_forces.yaw')
        self.connect('airfoil_files', 'aero_extrm_forces.airfoil_files')
        self.connect('nBlades', 'aero_extrm_forces.B')
        self.connect('rho', 'aero_extrm_forces.rho')
        self.connect('mu', 'aero_extrm_forces.mu')
        self.connect('shearExp', 'aero_extrm_forces.shearExp')
        self.connect('nSector', 'aero_extrm_forces.nSector')
        self.aero_extrm_forces.Uhub = np.zeros(2)
        self.aero_extrm_forces.Omega = np.zeros(2)  # parked case
        self.aero_extrm_forces.pitch = np.zeros(2)
        self.connect('turbineclass.V_extreme_full', 'aero_extrm_forces.Uhub')
        self.connect('pitch_extreme_full', 'aero_extrm_forces.pitch')
        self.aero_extrm_forces.pitch[1] = 90  # feathered
        self.aero_extrm_forces.T = np.zeros(2)
        self.aero_extrm_forces.Q = np.zeros(2)

        # connections to aero_defl_powercurve (for gust reversal)
        self.connect('spline.r_aero', 'aero_defl_powercurve.r')
        self.connect('spline.chord_aero', 'aero_defl_powercurve.chord')
        self.connect('spline.theta_aero', 'aero_defl_powercurve.theta')
        self.connect('spline.precurve_aero', 'aero_defl_powercurve.precurve')
        self.connect('spline.precurve_str', 'aero_defl_powercurve.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_defl_powercurve.Rhub')
        self.connect('spline.Rtip', 'aero_defl_powercurve.Rtip')
        self.connect('hubHt', 'aero_defl_powercurve.hubHt')
        self.connect('precone', 'aero_defl_powercurve.precone')
        self.connect('tilt', 'aero_defl_powercurve.tilt')
        self.connect('yaw', 'aero_defl_powercurve.yaw')
        self.connect('airfoil_files', 'aero_defl_powercurve.airfoil_files')
        self.connect('nBlades', 'aero_defl_powercurve.B')
        self.connect('rho', 'aero_defl_powercurve.rho')
        self.connect('mu', 'aero_defl_powercurve.mu')
        self.connect('shearExp', 'aero_defl_powercurve.shearExp')
        self.connect('nSector', 'aero_defl_powercurve.nSector')
        self.connect('setuppc.Uhub', 'aero_defl_powercurve.V_load')
        self.connect('setuppc.Omega', 'aero_defl_powercurve.Omega_load')
        self.connect('setuppc.pitch', 'aero_defl_powercurve.pitch_load')
        self.connect('setuppc.azimuth', 'aero_defl_powercurve.azimuth_load')
        self.aero_defl_powercurve.azimuth_load = 0.0

        # connections to beam
        self.connect('spline.r_str', 'beam.r')
        self.connect('spline.chord_str', 'beam.chord')
        self.connect('spline.theta_str', 'beam.theta')
        self.connect('leLoc', 'beam.leLoc')
        self.connect('profile', 'beam.profile')
        self.connect('materials', 'beam.materials')
        self.connect('resize.upperCSOut', 'beam.upperCS')
        self.connect('resize.lowerCSOut', 'beam.lowerCS')
        self.connect('resize.websCSOut', 'beam.websCS')
        self.connect('sector_idx_strain_spar', 'beam.sector_idx_strain_spar')
        self.connect('sector_idx_strain_te', 'beam.sector_idx_strain_te')

        # connections to loads_defl
        self.connect('aero_rated.loads:Omega', 'loads_defl.aeroLoads:Omega')
        self.connect('aero_rated.loads:Px', 'loads_defl.aeroLoads:Px')
        self.connect('aero_rated.loads:Py', 'loads_defl.aeroLoads:Py')
        self.connect('aero_rated.loads:Pz', 'loads_defl.aeroLoads:Pz')
        self.connect('aero_rated.loads:azimuth', 'loads_defl.aeroLoads:azimuth')
        self.connect('aero_rated.loads:pitch', 'loads_defl.aeroLoads:pitch')
        self.connect('aero_rated.loads:r', 'loads_defl.aeroLoads:r')

        self.connect('beam.beam:z', 'loads_defl.r')
        self.connect('spline.theta_str', 'loads_defl.theta')
        self.connect('tilt', 'loads_defl.tilt')
        self.connect('curvature.totalCone', 'loads_defl.totalCone')
        self.connect('curvature.z_az', 'loads_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_defl.rhoA')

        # connections to loads_pc_defl
        self.connect('aero_defl_powercurve.loads:Omega', 'loads_pc_defl.aeroLoads:Omega')
        self.connect('aero_defl_powercurve.loads:Px', 'loads_pc_defl.aeroLoads:Px')
        self.connect('aero_defl_powercurve.loads:Py', 'loads_pc_defl.aeroLoads:Py')
        self.connect('aero_defl_powercurve.loads:Pz', 'loads_pc_defl.aeroLoads:Pz')
        self.connect('aero_defl_powercurve.loads:azimuth', 'loads_pc_defl.aeroLoads:azimuth')
        self.connect('aero_defl_powercurve.loads:pitch', 'loads_pc_defl.aeroLoads:pitch')
        self.connect('aero_defl_powercurve.loads:r', 'loads_pc_defl.aeroLoads:r')
        self.connect('beam.beam:z', 'loads_pc_defl.r')
        self.connect('spline.theta_str', 'loads_pc_defl.theta')
        self.connect('tilt', 'loads_pc_defl.tilt')
        self.connect('curvature.totalCone', 'loads_pc_defl.totalCone')
        self.connect('curvature.z_az', 'loads_pc_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_pc_defl.rhoA')


        # connections to loads_strain
        self.connect('aero_extrm.loads:Omega', 'loads_strain.aeroLoads:Omega')
        self.connect('aero_extrm.loads:Px', 'loads_strain.aeroLoads:Px')
        self.connect('aero_extrm.loads:Py', 'loads_strain.aeroLoads:Py')
        self.connect('aero_extrm.loads:Pz', 'loads_strain.aeroLoads:Pz')
        self.connect('aero_extrm.loads:azimuth', 'loads_strain.aeroLoads:azimuth')
        self.connect('aero_extrm.loads:pitch', 'loads_strain.aeroLoads:pitch')
        self.connect('aero_extrm.loads:r', 'loads_strain.aeroLoads:r')
        self.connect('beam.beam:z', 'loads_strain.r')
        self.connect('spline.theta_str', 'loads_strain.theta')
        self.connect('tilt', 'loads_strain.tilt')
        self.connect('curvature.totalCone', 'loads_strain.totalCone')
        self.connect('curvature.z_az', 'loads_strain.z_az')
        self.connect('beam.beam:rhoA', 'loads_strain.rhoA')


        # connections to damage
        self.connect('rstar_damage', 'damage.rstar')
        self.connect('Mxb_damage', 'damage.Mxb')
        self.connect('Myb_damage', 'damage.Myb')
        self.connect('spline.theta_str', 'damage.theta')
        self.connect('beam.beam:z', 'damage.r')


        # connections to struc
        self.connect('beam.beam:z', 'struc.beam:z')
        self.connect('beam.beam:EA', 'struc.beam:EA')
        self.connect('beam.beam:EIxx', 'struc.beam:EIxx')
        self.connect('beam.beam:EIyy', 'struc.beam:EIyy')
        self.connect('beam.beam:EIxy', 'struc.beam:EIxy')
        self.connect('beam.beam:GJ', 'struc.beam:GJ')
        self.connect('beam.beam:rhoA', 'struc.beam:rhoA')
        self.connect('beam.beam:rhoJ', 'struc.beam:rhoJ')
        self.connect('beam.beam:x_ec_str', 'struc.beam:x_ec_str')
        self.connect('beam.beam:y_ec_str', 'struc.beam:y_ec_str')
        self.connect('nF', 'struc.nF')
        self.connect('loads_defl.Px_af', 'struc.Px_defl')
        self.connect('loads_defl.Py_af', 'struc.Py_defl')
        self.connect('loads_defl.Pz_af', 'struc.Pz_defl')
        self.connect('loads_pc_defl.Px_af', 'struc.Px_pc_defl')
        self.connect('loads_pc_defl.Py_af', 'struc.Py_pc_defl')
        self.connect('loads_pc_defl.Pz_af', 'struc.Pz_pc_defl')
        self.connect('loads_strain.Px_af', 'struc.Px_strain')
        self.connect('loads_strain.Py_af', 'struc.Py_strain')
        self.connect('loads_strain.Pz_af', 'struc.Pz_strain')
        self.connect('beam.xu_strain_spar', 'struc.xu_strain_spar')
        self.connect('beam.xl_strain_spar', 'struc.xl_strain_spar')
        self.connect('beam.yu_strain_spar', 'struc.yu_strain_spar')
        self.connect('beam.yl_strain_spar', 'struc.yl_strain_spar')
        self.connect('beam.xu_strain_te', 'struc.xu_strain_te')
        self.connect('beam.xl_strain_te', 'struc.xl_strain_te')
        self.connect('beam.yu_strain_te', 'struc.yu_strain_te')
        self.connect('beam.yl_strain_te', 'struc.yl_strain_te')
        self.connect('damage.Mxa', 'struc.Mx_damage')
        self.connect('damage.Mya', 'struc.My_damage')
        self.connect('strain_ult_spar', 'struc.strain_ult_spar')
        self.connect('strain_ult_te', 'struc.strain_ult_te')
        self.connect('eta_damage', 'struc.eta_damage')
        self.connect('m_damage', 'struc.m_damage')
        self.connect('N_damage', 'struc.N_damage')

        # connections to curvefem
        self.connect('powercurve.ratedConditions:Omega', 'curvefem.Omega')
        self.connect('beam.beam:z', 'curvefem.beam:z')
        self.connect('beam.beam:EA', 'curvefem.beam:EA')
        self.connect('beam.beam:EIxx', 'curvefem.beam:EIxx')
        self.connect('beam.beam:EIyy', 'curvefem.beam:EIyy')
        self.connect('beam.beam:EIxy', 'curvefem.beam:EIxy')
        self.connect('beam.beam:GJ', 'curvefem.beam:GJ')
        self.connect('beam.beam:rhoA', 'curvefem.beam:rhoA')
        self.connect('beam.beam:rhoJ', 'curvefem.beam:rhoJ')
        self.connect('beam.beam:x_ec_str', 'curvefem.beam:x_ec_str')
        self.connect('beam.beam:y_ec_str', 'curvefem.beam:y_ec_str')
        self.connect('spline.theta_str', 'curvefem.theta_str')
        self.connect('spline.precurve_str', 'curvefem.precurve_str')
        self.connect('spline.presweep_str', 'curvefem.presweep_str')
        self.connect('nF', 'curvefem.nF')

        # connections to tip
        self.connect('struc.dx_defl', 'tip.dx', src_indices=[nstr-1])
        self.connect('struc.dy_defl', 'tip.dy', src_indices=[nstr-1])
        self.connect('struc.dz_defl', 'tip.dz', src_indices=[nstr-1])
        self.connect('spline.theta_str', 'tip.theta', src_indices=[nstr-1])
        self.connect('aero_rated.loads:pitch', 'tip.pitch')
        self.connect('aero_rated.loads:azimuth', 'tip.azimuth')
        self.connect('tilt', 'tip.tilt')
        self.connect('curvature.totalCone', 'tip.totalConeTip', src_indices=[nstr-1])
        self.connect('dynamic_amplication_tip_deflection', 'tip.dynamicFactor')


        # connections to root moment
        self.connect('spline.r_str', 'root_moment.r_str')
        self.connect('aero_rated.loads:Px', 'root_moment.aeroLoads:Px')
        self.connect('aero_rated.loads:Py', 'root_moment.aeroLoads:Py')
        self.connect('aero_rated.loads:Pz', 'root_moment.aeroLoads:Pz')
        self.connect('aero_rated.loads:r', 'root_moment.aeroLoads:r')
        self.connect('curvature.totalCone', 'root_moment.totalCone')
        self.connect('curvature.x_az', 'root_moment.x_az')
        self.connect('curvature.y_az', 'root_moment.y_az')
        self.connect('curvature.z_az', 'root_moment.z_az')
        self.connect('curvature.s', 'root_moment.s')


        # connections to mass
        self.connect('struc.blade_mass', 'mass.blade_mass')
        self.connect('struc.blade_moment_of_inertia', 'mass.blade_moment_of_inertia')
        self.connect('nBlades', 'mass.nBlades')
        self.connect('tilt', 'mass.tilt')

        # connectsion to extreme
        self.connect('aero_extrm_forces.T', 'extreme.T')
        self.connect('aero_extrm_forces.Q', 'extreme.Q')
        self.connect('nBlades', 'extreme.nBlades')

        # connections to blade_defl
        self.connect('struc.dx_pc_defl', 'blade_defl.dx')
        self.connect('struc.dy_pc_defl', 'blade_defl.dy')
        self.connect('struc.dz_pc_defl', 'blade_defl.dz')
        self.connect('aero_defl_powercurve.loads:pitch', 'blade_defl.pitch')
        self.connect('spline0.theta_str', 'blade_defl.theta_str')
        self.connect('spline0.r_sub_precurve', 'blade_defl.r_sub_precurve0')
        self.connect('spline0.Rhub', 'blade_defl.Rhub0')
        self.connect('spline0.r_str', 'blade_defl.r_str0')
        self.connect('spline0.precurve_str', 'blade_defl.precurve_str0')
        self.connect('bladeLength', 'blade_defl.bladeLength0')
        # self.connect('precurve_sub', 'blade_defl.precurve_sub0')

        # connect to outputs
        self.connect('turbineclass.V_extreme', 'V_extreme_in')
        self.connect('extreme.T_extreme', 'T_extreme_in')
        self.connect('extreme.Q_extreme', 'Q_extreme_in')
        self.connect('struc.blade_mass', 'mass_one_blade_in')
        self.connect('mass.mass_all_blades', 'mass_all_blades_in')
        self.connect('mass.I_all_blades', 'I_all_blades_in')
        self.connect('struc.freq', 'freq_in')
        self.connect('curvefem.freq', 'freq_curvefem_in')
        self.connect('tip.tip_deflection', 'tip_deflection_in')
        self.connect('struc.strainU_spar', 'strainU_spar_in')
        self.connect('struc.strainL_spar', 'strainL_spar_in')
        self.connect('struc.strainU_te', 'strainU_te_in')
        self.connect('struc.strainL_te', 'strainL_te_in')
        self.connect('root_moment.root_bending_moment', 'root_bending_moment_in')
        self.connect('beam.eps_crit_spar', 'eps_crit_spar_in')
        self.connect('beam.eps_crit_te', 'eps_crit_te_in')
        self.connect('struc.damageU_spar', 'damageU_spar_in')
        self.connect('struc.damageL_spar', 'damageL_spar_in')
        self.connect('struc.damageU_te', 'damageU_te_in')
        self.connect('struc.damageL_te', 'damageL_te_in')
        self.connect('blade_defl.delta_bladeLength', 'delta_bladeLength_out_in')
        self.connect('blade_defl.delta_precurve_sub', 'delta_precurve_sub_out_in')

        self.connect('spline.Rtip', 'Rtip_in')
        self.connect('spline.precurve_str', 'precurveTip_in', src_indices=[nstr-1])

        ### adding for the drivetrain root moment calculations:
        # TODO - number and value of azimuth angles should be arbitrary user inputs
        # connections to aero_0 (for rated loads at 0 azimuth angle)
        self.connect('spline.r_aero', ['aero_0.r','aero_120.r','aero_240.r'])
        self.connect('spline.chord_aero', ['aero_0.chord', 'aero_120.chord', 'aero_240.chord'])
        self.connect('spline.theta_aero', ['aero_0.theta', 'aero_120.theta', 'aero_240.theta'])
        self.connect('spline.precurve_aero', ['aero_0.precurve', 'aero_120.precurve', 'aero_240.precurve'])
        self.connect('spline.precurve_str', ['aero_0.precurveTip', 'aero_120.precurveTip', 'aero_240.precurveTip'], src_indices=[naero-1])
        self.connect('spline.Rhub', ['aero_0.Rhub', 'aero_120.Rhub', 'aero_240.Rhub'])
        self.connect('spline.Rtip', ['aero_0.Rtip', 'aero_120.Rtip', 'aero_240.Rtip'])
        self.connect('hubHt', ['aero_0.hubHt', 'aero_120.hubHt', 'aero_240.hubHt'])
        self.connect('precone', ['aero_0.precone', 'aero_120.precone', 'aero_240.precone'])
        self.connect('tilt', ['aero_0.tilt', 'aero_120.tilt', 'aero_240.tilt'])
	self.connect('airfoil_files', ['aero_0.airfoil_files', 'aero_120.airfoil_files', 'aero_240.airfoil_files'])
        self.connect('yaw', ['aero_0.yaw', 'aero_120.yaw', 'aero_240.yaw'])
        self.connect('nBlades', ['aero_0.B','aero_120.B', 'aero_240.B'])
        self.connect('rho', ['aero_0.rho', 'aero_120.rho', 'aero_240.rho'])
        self.connect('mu', ['aero_0.mu','aero_120.mu' ,'aero_240.mu'])
        self.connect('shearExp', ['aero_0.shearExp','aero_120.shearExp','aero_240.shearExp'])
        self.connect('nSector', ['aero_0.nSector','aero_120.nSector','aero_240.nSector'])
        # self.connect('powercurve.ratedConditions.V + 3*gust.sigma', 'aero_0.V_load')  # OpenMDAO bug
        self.connect('gust.V_gust', ['aero_0.V_load','aero_120.V_load','aero_240.V_load'])
        self.connect('powercurve.ratedConditions:Omega', ['aero_0.Omega_load','aero_120.Omega_load','aero_240.Omega_load'])
        self.add('pitch_load89', IndepVarComp('pitch_load89', val=89.0, units='deg'), promotes=['*'])
        self.add('azimuth_load0', IndepVarComp('azimuth_load0', val=0.0, units='deg'), promotes=['*'])
        self.add('azimuth_load120', IndepVarComp('azimuth_load120', val=120.0, units='deg'), promotes=['*'])
        self.add('azimuth_load240', IndepVarComp('azimuth_load240', val=240.0, units='deg'), promotes=['*'])
        self.connect('pitch_load89', 'aero_0.pitch_load')
        self.connect('pitch_load89', 'aero_120.pitch_load')
        self.connect('pitch_load89', 'aero_240.pitch_load')
        self.connect('azimuth_load0', 'aero_0.azimuth_load')
        self.connect('azimuth_load120', 'aero_120.azimuth_load')
        self.connect('azimuth_load240', 'aero_240.azimuth_load')

        # connections to root moment for drivetrain
        self.connect('spline.r_str', ['root_moment_0.r_str', 'root_moment_120.r_str', 'root_moment_240.r_str'])
        self.connect('aero_rated.loads:Px', ['root_moment_0.aeroLoads:Px', 'root_moment_120.aeroLoads:Px', 'root_moment_240.aeroLoads:Px'])
        self.connect('aero_rated.loads:Py', ['root_moment_0.aeroLoads:Py', 'root_moment_120.aeroLoads:Py', 'root_moment_240.aeroLoads:Py'])
        self.connect('aero_rated.loads:Pz', ['root_moment_0.aeroLoads:Pz', 'root_moment_120.aeroLoads:Pz', 'root_moment_240.aeroLoads:Pz'])
        self.connect('aero_rated.loads:r', ['root_moment_0.aeroLoads:r', 'root_moment_120.aeroLoads:r', 'root_moment_240.aeroLoads:r'])
        self.connect('curvature.totalCone', ['root_moment_0.totalCone', 'root_moment_120.totalCone', 'root_moment_240.totalCone'])
        self.connect('curvature.x_az', ['root_moment_0.x_az','root_moment_120.x_az','root_moment_240.x_az'])
        self.connect('curvature.y_az', ['root_moment_0.y_az','root_moment_120.y_az','root_moment_240.y_az'])
        self.connect('curvature.z_az', ['root_moment_0.z_az','root_moment_120.z_az','root_moment_240.z_az'])
        self.connect('curvature.s', ['root_moment_0.s','root_moment_120.s','root_moment_240.s'])

        # connections to root Mxyz outputs
        self.connect('root_moment_0.Mxyz','Mxyz_0_in')
        self.connect('root_moment_120.Mxyz','Mxyz_120_in')
        self.connect('root_moment_240.Mxyz','Mxyz_240_in')
        self.connect('curvature.totalCone','TotalCone_in', src_indices=[nstr-1])
        self.connect('aero_0.pitch_load','Pitch_in')
        self.connect('root_moment_0.Fxyz', 'Fxyz_0_in')
        self.connect('root_moment_120.Fxyz', 'Fxyz_120_in')
        self.connect('root_moment_240.Fxyz', 'Fxyz_240_in')
        #azimuths not passed. assumed 0,120,240 in drivese function

        self.add('obj_cmp', ExecComp('obj = -AEP', AEP=1000000.0), promotes=['*'])

if __name__ == '__main__':

	initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
	    0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
	    0.97777724])  # (Array): initial aerodynamic grid on unit radius
	initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
	    0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
	    0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
	    0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
	    0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
	    1.0])  # (Array): initial structural grid on unit radius


	rotor = Problem()
	naero = len(initial_aero_grid)
	nstr = len(initial_str_grid)
	npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
	npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve

	rotor.root = RotorSE(naero, nstr, npts_coarse_power_curve, npts_spline_power_curve)

	#rotor.setup(check=False)
	rotor.setup()

	# === blade grid ===
	rotor['initial_aero_grid'] = initial_aero_grid  # (Array): initial aerodynamic grid on unit radius
	rotor['initial_str_grid'] = initial_str_grid  # (Array): initial structural grid on unit radius
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
	rotor['hubHt'] = np.array([90.0])  # (Float, m): hub height
	rotor['turbine_class'] = TURBINE_CLASS['I']  # (Enum): IEC turbine class
	rotor['turbulence_class'] = TURBULENCE_CLASS['B']  # (Enum): IEC turbulence class class
	rotor['cdf_reference_height_wind_speed'] = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
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
	rotor['npts_coarse_power_curve'] = npts_coarse_power_curve  # (Int): number of points to evaluate aero analysis at
	rotor['npts_spline_power_curve'] = npts_spline_power_curve  # (Int): number of points to use in fitting spline to power curve
	rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
	rotor['drivetrainType'] = DRIVETRAIN_TYPE['GEARED']  # (Enum)
	rotor['nF'] = 5  # (Int): number of natural frequencies to compute
	rotor['dynamic_amplication_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
	# ----------------------

	# === materials and composite layup  ===
	basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PreCompFiles')

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
	print 'ratedConditions.V =', rotor['ratedConditions:V']
	print 'ratedConditions.Omega =', rotor['ratedConditions:Omega']
	print 'ratedConditions.pitch =', rotor['ratedConditions:pitch']
	print 'ratedConditions.T =', rotor['ratedConditions:T']
	print 'ratedConditions.Q =', rotor['ratedConditions:Q']
	print 'mass_one_blade =', rotor['mass_one_blade']
	print 'mass_all_blades =', rotor['mass_all_blades']
	print 'I_all_blades =', rotor['I_all_blades']
	print 'freq =', rotor['freq']
	print 'tip_deflection =', rotor['tip_deflection']
	print 'root_bending_moment =', rotor['root_bending_moment']
        #for io in rotor.root.unknowns:
        #    print(io + ' ' + str(rotor.root.unknowns[io]))



        import matplotlib.pyplot as plt
        plt.figure()
	plt.plot(rotor['V'], rotor['P']/1e6)
	plt.xlabel('wind speed (m/s)')
	plt.xlabel('power (W)')

	plt.figure()
	plt.plot(rotor['spline.r_str'], rotor['strainU_spar'], label='suction')
	plt.plot(rotor['spline.r_str'], rotor['strainL_spar'], label='pressure')
	plt.plot(rotor['spline.r_str'], rotor['eps_crit_spar'], label='critical')
	plt.ylim([-5e-3, 5e-3])
	plt.xlabel('r')
	plt.ylabel('strain')
	plt.legend()
	# plt.save('/Users/sning/Desktop/strain_spar.pdf')
	# plt.save('/Users/sning/Desktop/strain_spar.png')

	plt.figure()
	plt.plot(rotor['spline.r_str'], rotor['strainU_te'], label='suction')
	plt.plot(rotor['spline.r_str'], rotor['strainL_te'], label='pressure')
	plt.plot(rotor['spline.r_str'], rotor['eps_crit_te'], label='critical')
	plt.ylim([-5e-3, 5e-3])
	plt.xlabel('r')
	plt.ylabel('strain')
	plt.legend()
	# plt.save('/Users/sning/Desktop/strain_te.pdf')
	# plt.save('/Users/sning/Desktop/strain_te.png')

	plt.show()
	# ----------------
