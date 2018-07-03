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
from rotor_aeropower import SetupRunVarSpeed, CSMDrivetrain, RegulatedPowerCurve, AEP, OutputsAero
from rotor_structure import ResizeCompositeSection, BladeCurvature, CurveFEM, DamageLoads, TotalLoads, TipDeflection, \
    BladeDeflection, RootMoment, MassProperties, ExtremeLoads, GustETM, SetupPCModVarSpeed, OutputsStructures, \
    PreCompSections, RotorWithpBEAM, ConstraintsStructures

from ccblade.ccblade_component import CCBladeGeometry, CCBladePower, CCBladeLoads
from commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from commonse.environment import PowerWind
from precomp import Profile, Orthotropic2DMaterial, CompositeSection
from rotor_geometry import RotorGeometry, NREL5MW, DTU10MW, NINPUT

from rotorse import RPM2RS, RS2RPM, TURBULENCE_CLASS, DRIVETRAIN_TYPE


class RotorSE(Group):
    def __init__(self, RefBlade, npts_coarse_power_curve=20, npts_spline_power_curve=200):
        super(RotorSE, self).__init__()
        """rotor model"""

        NPTS = RefBlade.npts
        
        self.add('turbulence_class', IndepVarComp('turbulence_class', val=TURBULENCE_CLASS['A'], desc='IEC turbulence class class', pass_by_obj=True), promotes=['*'])
        self.add('gust_stddev', IndepVarComp('gust_stddev', val=3, pass_by_obj=True), promotes=['*'])
        #self.add('cdf_reference_height_wind_speed', IndepVarComp('cdf_reference_height_wind_speed', val=0.0, units='m', desc='reference hub height for IEC wind speed (used in CDF calculation)'), promotes=['*'])
        self.add('VfactorPC', IndepVarComp('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation'), promotes=['*'])

        # --- control ---
        self.add('c_Vin', IndepVarComp('control_Vin', val=0.0, units='m/s', desc='cut-in wind speed'), promotes=['*'])
        self.add('c_Vout', IndepVarComp('control_Vout', val=0.0, units='m/s', desc='cut-out wind speed'), promotes=['*'])
        self.add('machine_rating', IndepVarComp('machine_rating', val=0.0,  units='W', desc='rated power'), promotes=['*'])
        self.add('c_minOmega', IndepVarComp('control_minOmega', val=0.0, units='rpm', desc='minimum allowed rotor rotation speed'), promotes=['*'])
        self.add('c_maxOmega', IndepVarComp('control_maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed'), promotes=['*'])
        self.add('c_tsr', IndepVarComp('control_tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)'), promotes=['*'])
        self.add('c_pitch', IndepVarComp('control_pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)'), promotes=['*'])
        self.add('pitch_extreme', IndepVarComp('pitch_extreme', val=0.0, units='deg', desc='worst-case pitch at survival wind condition'), promotes=['*'])
        self.add('azimuth_extreme', IndepVarComp('azimuth_extreme', val=0.0, units='deg', desc='worst-case azimuth at survival wind condition'), promotes=['*'])

        # --- drivetrain efficiency ---
        self.add('drivetrainType', IndepVarComp('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True), promotes=['*'])


        # --- fatigue ---
        self.add('rstar_damage', IndepVarComp('rstar_damage', val=np.zeros(NPTS+1), desc='nondimensional radial locations of damage equivalent moments'), promotes=['*'])
        self.add('Mxb_damage', IndepVarComp('Mxb_damage', val=np.zeros(NPTS+1), units='N*m', desc='damage equivalent moments about blade c.s. x-direction'), promotes=['*'])
        self.add('Myb_damage', IndepVarComp('Myb_damage', val=np.zeros(NPTS+1), units='N*m', desc='damage equivalent moments about blade c.s. y-direction'), promotes=['*'])
        self.add('strain_ult_spar', IndepVarComp('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap'), promotes=['*'])
        self.add('strain_ult_te', IndepVarComp('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels'), promotes=['*'])
        self.add('m_damage', IndepVarComp('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis'), promotes=['*'])
        #self.add('lifetime', IndepVarComp('lifetime', val=20.0, units='year', desc='project lifetime for fatigue analysis'), promotes=['*'])


        # --- options ---
        self.add('nSector', IndepVarComp('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True), promotes=['*'])
        self.add('tiploss', IndepVarComp('tiploss', True, pass_by_obj=True), promotes=['*'])
        self.add('hubloss', IndepVarComp('hubloss', True, pass_by_obj=True), promotes=['*'])
        self.add('wakerotation', IndepVarComp('wakerotation', True, pass_by_obj=True), promotes=['*'])
        self.add('usecd', IndepVarComp('usecd', True, pass_by_obj=True), promotes=['*'])
        self.add('AEP_loss_factor', IndepVarComp('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)'), promotes=['*'])
        self.add('dynamic_amplication_tip_deflection', IndepVarComp('dynamic_amplication_tip_deflection', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation'), promotes=['*'])
        self.add('shape_parameter', IndepVarComp('shape_parameter', val=0.0), promotes=['*'])
        
        # --- Rotor Aero & Power ---
        self.add('rotorGeometry', RotorGeometry(RefBlade), promotes=['*'])

        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRunVarSpeed(npts_coarse_power_curve))
        self.add('analysis', CCBladePower(NPTS, npts_coarse_power_curve))
        self.add('dt', CSMDrivetrain(npts_coarse_power_curve))
        self.add('powercurve', RegulatedPowerCurve(npts_coarse_power_curve, npts_spline_power_curve))
        self.add('wind', PowerWind(1))
        # self.add('cdf', WeibullWithMeanCDF(npts_spline_power_curve))
        self.add('cdf', RayleighCDF(npts_spline_power_curve))
        self.add('aep', AEP(npts_spline_power_curve))

        self.add('outputs_aero', OutputsAero(npts_spline_power_curve), promotes=['*'])


        # --- add structures ---
        self.add('curvature', BladeCurvature(NPTS))
        self.add('resize', ResizeCompositeSection(NPTS))
        self.add('gust', GustETM())
        self.add('setuppc',  SetupPCModVarSpeed())
        self.add('beam', PreCompSections(NPTS))

        self.add('aero_rated', CCBladeLoads(NPTS, 1))
        self.add('aero_extrm', CCBladeLoads(NPTS,  1))
        self.add('aero_extrm_forces', CCBladePower(NPTS, 2))
        self.add('aero_defl_powercurve', CCBladeLoads(NPTS,  1))

        self.add('loads_defl', TotalLoads(NPTS))
        self.add('loads_pc_defl', TotalLoads(NPTS))
        self.add('loads_strain', TotalLoads(NPTS))

        self.add('damage', DamageLoads(NPTS))
        self.add('struc', RotorWithpBEAM(NPTS), promotes=['gamma_fatigue'])
        self.add('curvefem', CurveFEM(NPTS))
        self.add('tip', TipDeflection())
        self.add('root_moment', RootMoment(NPTS))
        self.add('mass', MassProperties())
        self.add('extreme', ExtremeLoads())
        self.add('blade_defl', BladeDeflection(NPTS))

        self.add('aero_0', CCBladeLoads(NPTS,  1))
        self.add('aero_120', CCBladeLoads(NPTS,  1))
        self.add('aero_240', CCBladeLoads(NPTS,  1))
        self.add('root_moment_0', RootMoment(NPTS))
        self.add('root_moment_120', RootMoment(NPTS))
        self.add('root_moment_240', RootMoment(NPTS))

        self.add('output_struc', OutputsStructures(NPTS), promotes=['*'])
        self.add('constraints', ConstraintsStructures(NPTS), promotes=['*'])

        # # connectiosn to tipspeed
        # self.connect('geom.R', 'tipspeed.R')
        # self.connect('max_tip_speed', 'tipspeed.Vtip_max')
        # self.connect('tipspeed.Omega_max', 'control_maxOmega')

        # connections to setup
        self.connect('control_Vin', 'setup.control_Vin')
        self.connect('control_Vout', 'setup.control_Vout')
        self.connect('control_maxOmega', 'setup.control_maxOmega')
        self.connect('control_pitch', 'setup.control_pitch')
        self.connect('control_tsr', 'setup.control_tsr')
        self.connect('geom.R', 'setup.R')

        # connections to analysis
        self.connect('r_pts', 'analysis.r')
        self.connect('chord', 'analysis.chord')
        self.connect('theta', 'analysis.theta')
        self.connect('precurve', 'analysis.precurve')
        self.connect('precurve_tip', 'analysis.precurveTip')
        self.connect('Rhub', 'analysis.Rhub')
        self.connect('Rtip', 'analysis.Rtip')
        self.connect('precone', 'analysis.precone')
        self.connect('tilt', 'analysis.tilt')
        self.connect('yaw', 'analysis.yaw')
        self.connect('airfoil_files', 'analysis.airfoil_files')
        self.connect('nBlades', 'analysis.B')
        self.connect('nSector', 'analysis.nSector')
        self.connect('setup.Uhub', 'analysis.Uhub')
        self.connect('setup.Omega', 'analysis.Omega')
        self.connect('setup.pitch', 'analysis.pitch')

        # Connections from external modules
        self.connect('hub_height', ['analysis.hubHt','aero_0.hubHt','aero_120.hubHt','aero_240.hubHt','aero_defl_powercurve.hubHt','aero_extrm_forces.hubHt','aero_extrm.hubHt','aero_rated.hubHt'])
        self.connect('analysis.rho', ['aero_0.rho','aero_120.rho','aero_240.rho','aero_defl_powercurve.rho','aero_extrm_forces.rho','aero_extrm.rho','aero_rated.rho'])
        self.connect('analysis.mu', ['aero_0.mu','aero_120.mu','aero_240.mu','aero_defl_powercurve.mu','aero_extrm_forces.mu','aero_extrm.mu','aero_rated.mu'])
        self.connect('wind.shearExp', ['analysis.shearExp', 'aero_0.shearExp','aero_120.shearExp','aero_240.shearExp','aero_defl_powercurve.shearExp','aero_extrm_forces.shearExp','aero_extrm.shearExp','aero_rated.shearExp'])
        #self.connect('analysis.shearExp', ['aero_0.shearExp','aero_120.shearExp','aero_240.shearExp','aero_defl_powercurve.shearExp','aero_extrm_forces.shearExp','aero_extrm.shearExp','aero_rated.shearExp'])
        
        # connections to drivetrain
        self.connect('analysis.P', 'dt.aeroPower')
        self.connect('analysis.Q', 'dt.aeroTorque')
        self.connect('analysis.T', 'dt.aeroThrust')
        self.connect('machine_rating', 'dt.ratedPower')
        self.connect('drivetrainType', 'dt.drivetrainType')

        # connections to powercurve
        self.connect('control_Vin', 'powercurve.control_Vin')
        self.connect('control_Vout', 'powercurve.control_Vout')
        self.connect('control_maxOmega', 'powercurve.control_maxOmega')
        self.connect('control_minOmega', 'powercurve.control_minOmega')
        self.connect('control_pitch', 'powercurve.control_pitch')
        self.connect('machine_rating', 'powercurve.control_ratedPower')
        self.connect('control_tsr', 'powercurve.control_tsr')
        self.connect('setup.Uhub', 'powercurve.Vcoarse')
        self.connect('dt.power', 'powercurve.Pcoarse')
        self.connect('analysis.T', 'powercurve.Tcoarse')
        self.connect('geom.R', 'powercurve.R')

        # connections to wind
        # self.connect('cdf_reference_mean_wind_speed', 'wind.Uref')
        #self.connect('cdf_reference_height_wind_speed', 'wind.zref')
        self.connect('wind_zvec', 'wind.z')
        self.connect('turbineclass.V_mean', 'wind.Uref')

        # connections to cdf
        self.connect('powercurve.V', 'cdf.x')
        self.connect('wind.U', 'cdf.xbar', src_indices=[0])
        self.connect('shape_parameter', 'cdf.k')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to outputs
        self.connect('powercurve.V', 'V_in')
        self.connect('powercurve.P', 'P_in')
        self.connect('aep.AEP', 'AEP_in')
        self.connect('powercurve.rated_V', 'rated_V_in')
        self.connect('powercurve.rated_Omega', 'rated_Omega_in')
        self.connect('powercurve.rated_pitch', 'rated_pitch_in')
        self.connect('powercurve.rated_T', 'rated_T_in')
        self.connect('powercurve.rated_Q', 'rated_Q_in')
        self.connect('geom.diameter', 'diameter_in')
        self.connect('presweep_tip', 'presweepTip_in')

        
        # Structures connections
        
        
        # connections to curvature
        self.connect('r_pts', 'curvature.r')
        self.connect('precurve', 'curvature.precurve')
        self.connect('presweep', 'curvature.presweep')
        self.connect('precone', 'curvature.precone')

        # connections to resize
        self.connect('chord', 'resize.chord')
        self.connect('sparT', 'resize.sparT')
        self.connect('teT', 'resize.teT')
        self.connect('upperCS', 'resize.upperCS_in')
        self.connect('lowerCS', 'resize.lowerCS_in')
        self.connect('websCS', 'resize.websCS_in')
        self.connect('chord_ref', 'resize.chord_ref')
        self.connect('sector_idx_strain_spar', ['resize.sector_idx_strain_spar','beam.sector_idx_strain_spar'])
        self.connect('sector_idx_strain_te', ['resize.sector_idx_strain_te','beam.sector_idx_strain_te'])

        # connections to gust
        self.connect('turbulence_class', 'gust.turbulence_class')
        self.connect('turbineclass.V_mean', 'gust.V_mean')
        self.connect('powercurve.rated_V', 'gust.V_hub')
        self.connect('gust_stddev', 'gust.std')

        # connections to setuppc
        self.connect('control_pitch', 'setuppc.control_pitch')
        self.connect('control_tsr', 'setuppc.control_tsr')
        self.connect('powercurve.rated_V', 'setuppc.Vrated')
        self.connect('geom.R', 'setuppc.R')
        self.connect('VfactorPC', 'setuppc.Vfactor')

        # connections to aero_rated (for max deflection)
        self.connect('r_pts', 'aero_rated.r')
        self.connect('chord', 'aero_rated.chord')
        self.connect('theta', 'aero_rated.theta')
        self.connect('precurve', 'aero_rated.precurve')
        self.connect('precurve_tip', 'aero_rated.precurveTip')
        self.connect('Rhub', 'aero_rated.Rhub')
        self.connect('Rtip', 'aero_rated.Rtip')
        self.connect('precone', 'aero_rated.precone')
        self.connect('tilt', 'aero_rated.tilt')
        self.connect('yaw', 'aero_rated.yaw')
        self.connect('airfoil_files', 'aero_rated.airfoil_files')
        self.connect('nBlades', 'aero_rated.B')
        self.connect('nSector', 'aero_rated.nSector')
        self.connect('gust.V_gust', 'aero_rated.V_load')
        self.connect('powercurve.rated_Omega', 'aero_rated.Omega_load')
        self.connect('powercurve.rated_pitch', 'aero_rated.pitch_load')
        self.connect('powercurve.azimuth', 'aero_rated.azimuth_load')
        self.aero_rated.azimuth_load = 180.0  # closest to tower

        # connections to aero_extrm (for max strain)
        self.connect('r_pts', 'aero_extrm.r')
        self.connect('chord', 'aero_extrm.chord')
        self.connect('theta', 'aero_extrm.theta')
        self.connect('precurve', 'aero_extrm.precurve')
        self.connect('precurve_tip', 'aero_extrm.precurveTip')
        self.connect('Rhub', 'aero_extrm.Rhub')
        self.connect('Rtip', 'aero_extrm.Rtip')
        self.connect('precone', 'aero_extrm.precone')
        self.connect('tilt', 'aero_extrm.tilt')
        self.connect('yaw', 'aero_extrm.yaw')
        self.connect('airfoil_files', 'aero_extrm.airfoil_files')
        self.connect('nBlades', 'aero_extrm.B')
        self.connect('nSector', 'aero_extrm.nSector')
        self.connect('turbineclass.V_extreme', 'aero_extrm.V_load')
        self.connect('pitch_extreme', 'aero_extrm.pitch_load')
        self.connect('azimuth_extreme', 'aero_extrm.azimuth_load')
        self.aero_extrm.Omega_load = 0.0  # parked case

        # connections to aero_extrm_forces (for tower thrust)
        self.connect('r_pts', 'aero_extrm_forces.r')
        self.connect('chord', 'aero_extrm_forces.chord')
        self.connect('theta', 'aero_extrm_forces.theta')
        self.connect('precurve', 'aero_extrm_forces.precurve')
        self.connect('precurve_tip', 'aero_extrm_forces.precurveTip')
        self.connect('Rhub', 'aero_extrm_forces.Rhub')
        self.connect('Rtip', 'aero_extrm_forces.Rtip')
        self.connect('precone', 'aero_extrm_forces.precone')
        self.connect('tilt', 'aero_extrm_forces.tilt')
        self.connect('yaw', 'aero_extrm_forces.yaw')
        self.connect('airfoil_files', 'aero_extrm_forces.airfoil_files')
        self.connect('nBlades', 'aero_extrm_forces.B')
        self.connect('nSector', 'aero_extrm_forces.nSector')
        self.aero_extrm_forces.Uhub = np.zeros(2)
        self.aero_extrm_forces.Omega = np.zeros(2)  # parked case
        self.aero_extrm_forces.pitch = np.zeros(2)
        self.connect('turbineclass.V_extreme_full', 'aero_extrm_forces.Uhub')
        self.aero_extrm_forces.pitch = np.array([0.0, 90.0])  # feathered
        self.aero_extrm_forces.T = np.zeros(2)
        self.aero_extrm_forces.Q = np.zeros(2)

        # connections to aero_defl_powercurve (for gust reversal)
        self.connect('r_pts', 'aero_defl_powercurve.r')
        self.connect('chord', 'aero_defl_powercurve.chord')
        self.connect('theta', 'aero_defl_powercurve.theta')
        self.connect('precurve', 'aero_defl_powercurve.precurve')
        self.connect('precurve_tip', 'aero_defl_powercurve.precurveTip')
        self.connect('Rhub', 'aero_defl_powercurve.Rhub')
        self.connect('Rtip', 'aero_defl_powercurve.Rtip')
        self.connect('precone', 'aero_defl_powercurve.precone')
        self.connect('tilt', 'aero_defl_powercurve.tilt')
        self.connect('yaw', 'aero_defl_powercurve.yaw')
        self.connect('airfoil_files', 'aero_defl_powercurve.airfoil_files')
        self.connect('nBlades', 'aero_defl_powercurve.B')
        self.connect('nSector', 'aero_defl_powercurve.nSector')
        self.connect('setuppc.Uhub', 'aero_defl_powercurve.V_load')
        self.connect('setuppc.Omega', 'aero_defl_powercurve.Omega_load')
        self.connect('setuppc.pitch', 'aero_defl_powercurve.pitch_load')
        self.connect('setuppc.azimuth', 'aero_defl_powercurve.azimuth_load')
        self.aero_defl_powercurve.azimuth_load = 0.0

        # connections to beam
        self.connect('r_pts', 'beam.r')
        self.connect('chord', 'beam.chord')
        self.connect('theta', 'beam.theta')
        self.connect('resize.upperCS', 'beam.upperCS')
        self.connect('resize.lowerCS', 'beam.lowerCS')
        self.connect('resize.websCS', 'beam.websCS')
        self.connect('profile', 'beam.profile')
        self.connect('le_location', 'beam.le_location')
        self.connect('materials', 'beam.materials')

        # connections to loads_defl
        self.connect('aero_rated.loads_Omega', 'loads_defl.aeroloads_Omega')
        self.connect('aero_rated.loads_Px', 'loads_defl.aeroloads_Px')
        self.connect('aero_rated.loads_Py', 'loads_defl.aeroloads_Py')
        self.connect('aero_rated.loads_Pz', 'loads_defl.aeroloads_Pz')
        self.connect('aero_rated.loads_azimuth', 'loads_defl.aeroloads_azimuth')
        self.connect('aero_rated.loads_pitch', 'loads_defl.aeroloads_pitch')
        self.connect('aero_rated.loads_r', 'loads_defl.aeroloads_r')

        self.connect('beam.beam:z', 'loads_defl.r')
        self.connect('theta', 'loads_defl.theta')
        self.connect('tilt', 'loads_defl.tilt')
        self.connect('curvature.totalCone', 'loads_defl.totalCone')
        self.connect('curvature.z_az', 'loads_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_defl.rhoA')

        # connections to loads_pc_defl
        self.connect('aero_defl_powercurve.loads_Omega', 'loads_pc_defl.aeroloads_Omega')
        self.connect('aero_defl_powercurve.loads_Px', 'loads_pc_defl.aeroloads_Px')
        self.connect('aero_defl_powercurve.loads_Py', 'loads_pc_defl.aeroloads_Py')
        self.connect('aero_defl_powercurve.loads_Pz', 'loads_pc_defl.aeroloads_Pz')
        self.connect('aero_defl_powercurve.loads_azimuth', 'loads_pc_defl.aeroloads_azimuth')
        self.connect('aero_defl_powercurve.loads_pitch', 'loads_pc_defl.aeroloads_pitch')
        self.connect('aero_defl_powercurve.loads_r', 'loads_pc_defl.aeroloads_r')
        self.connect('beam.beam:z', 'loads_pc_defl.r')
        self.connect('theta', 'loads_pc_defl.theta')
        self.connect('tilt', 'loads_pc_defl.tilt')
        self.connect('curvature.totalCone', 'loads_pc_defl.totalCone')
        self.connect('curvature.z_az', 'loads_pc_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_pc_defl.rhoA')


        # connections to loads_strain
        self.connect('aero_extrm.loads_Omega', 'loads_strain.aeroloads_Omega')
        self.connect('aero_extrm.loads_Px', 'loads_strain.aeroloads_Px')
        self.connect('aero_extrm.loads_Py', 'loads_strain.aeroloads_Py')
        self.connect('aero_extrm.loads_Pz', 'loads_strain.aeroloads_Pz')
        self.connect('aero_extrm.loads_azimuth', 'loads_strain.aeroloads_azimuth')
        self.connect('aero_extrm.loads_pitch', 'loads_strain.aeroloads_pitch')
        self.connect('aero_extrm.loads_r', 'loads_strain.aeroloads_r')
        self.connect('beam.beam:z', 'loads_strain.r')
        self.connect('theta', 'loads_strain.theta')
        self.connect('tilt', 'loads_strain.tilt')
        self.connect('curvature.totalCone', 'loads_strain.totalCone')
        self.connect('curvature.z_az', 'loads_strain.z_az')
        self.connect('beam.beam:rhoA', 'loads_strain.rhoA')


        # connections to damage
        self.connect('rstar_damage', 'damage.rstar')
        self.connect('Mxb_damage', 'damage.Mxb')
        self.connect('Myb_damage', 'damage.Myb')
        self.connect('theta', 'damage.theta')
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
        self.connect('beam.beam:x_ec', 'struc.beam:x_ec')
        self.connect('beam.beam:y_ec', 'struc.beam:y_ec')
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
        self.connect('m_damage', 'struc.m_damage')
        #self.connect('lifetime', 'struc.lifetime')

        # connections to curvefem
        self.connect('powercurve.rated_Omega', 'curvefem.Omega')
        self.connect('beam.beam:z', 'curvefem.beam:z')
        self.connect('beam.beam:EA', 'curvefem.beam:EA')
        self.connect('beam.beam:EIxx', 'curvefem.beam:EIxx')
        self.connect('beam.beam:EIyy', 'curvefem.beam:EIyy')
        self.connect('beam.beam:EIxy', 'curvefem.beam:EIxy')
        self.connect('beam.beam:GJ', 'curvefem.beam:GJ')
        self.connect('beam.beam:rhoA', 'curvefem.beam:rhoA')
        self.connect('beam.beam:rhoJ', 'curvefem.beam:rhoJ')
        self.connect('beam.beam:x_ec', 'curvefem.beam:x_ec')
        self.connect('beam.beam:y_ec', 'curvefem.beam:y_ec')
        self.connect('theta', 'curvefem.theta')
        self.connect('precurve', 'curvefem.precurve')
        self.connect('presweep', 'curvefem.presweep')

        # connections to tip
        self.connect('struc.dx_defl', 'tip.dx', src_indices=[NPTS-1])
        self.connect('struc.dy_defl', 'tip.dy', src_indices=[NPTS-1])
        self.connect('struc.dz_defl', 'tip.dz', src_indices=[NPTS-1])
        self.connect('theta', 'tip.theta', src_indices=[NPTS-1])
        self.connect('aero_rated.loads_pitch', 'tip.pitch')
        self.connect('aero_rated.loads_azimuth', 'tip.azimuth')
        self.connect('tilt', 'tip.tilt')
        self.connect('curvature.totalCone', 'tip.totalConeTip', src_indices=[NPTS-1])
        self.connect('dynamic_amplication_tip_deflection', 'tip.dynamicFactor')

        # connections to root moment

        self.connect('r_pts', 'root_moment.r_pts')
        self.connect('aero_rated.loads_Px', 'root_moment.aeroloads_Px')
        self.connect('aero_rated.loads_Py', 'root_moment.aeroloads_Py')
        self.connect('aero_rated.loads_Pz', 'root_moment.aeroloads_Pz')
        self.connect('aero_rated.loads_r', 'root_moment.aeroloads_r')
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
        self.connect('aero_defl_powercurve.loads_pitch', 'blade_defl.pitch')
        self.connect('theta', 'blade_defl.theta')
        self.connect('r_in', 'blade_defl.r_in0')
        self.connect('Rhub', 'blade_defl.Rhub0')
        self.connect('r_pts', 'blade_defl.r_pts0')
        self.connect('precurve', 'blade_defl.precurve0')
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

        self.connect('precurve_tip', 'precurveTip_in')

        ### adding for the drivetrain root moment calculations:
        # TODO - number and value of azimuth angles should be arbitrary user inputs
        # connections to aero_0 (for rated loads at 0 azimuth angle)
        self.add('pitch_load89', IndepVarComp('pitch_load89', val=89.0, units='deg'), promotes=['*'])
        self.add('azimuth_load0', IndepVarComp('azimuth_load0', val=0.0, units='deg'), promotes=['*'])
        self.add('azimuth_load120', IndepVarComp('azimuth_load120', val=120.0, units='deg'), promotes=['*'])
        self.add('azimuth_load240', IndepVarComp('azimuth_load240', val=240.0, units='deg'), promotes=['*'])
        self.connect('r_pts', ['aero_0.r','aero_120.r','aero_240.r'])
        self.connect('chord', ['aero_0.chord', 'aero_120.chord', 'aero_240.chord'])
        self.connect('theta', ['aero_0.theta', 'aero_120.theta', 'aero_240.theta'])
        self.connect('precurve', ['aero_0.precurve', 'aero_120.precurve', 'aero_240.precurve'])
        self.connect('precurve_tip', ['aero_0.precurveTip', 'aero_120.precurveTip', 'aero_240.precurveTip'])
        self.connect('Rhub', ['aero_0.Rhub', 'aero_120.Rhub', 'aero_240.Rhub'])
        self.connect('Rtip', ['aero_0.Rtip', 'aero_120.Rtip', 'aero_240.Rtip'])
        self.connect('precone', ['aero_0.precone', 'aero_120.precone', 'aero_240.precone'])
        self.connect('tilt', ['aero_0.tilt', 'aero_120.tilt', 'aero_240.tilt'])
	self.connect('airfoil_files', ['aero_0.airfoil_files', 'aero_120.airfoil_files', 'aero_240.airfoil_files'])
        self.connect('yaw', ['aero_0.yaw', 'aero_120.yaw', 'aero_240.yaw'])
        self.connect('nBlades', ['aero_0.B','aero_120.B', 'aero_240.B'])
        self.connect('nSector', ['aero_0.nSector','aero_120.nSector','aero_240.nSector'])
        self.connect('gust.V_gust', ['aero_0.V_load','aero_120.V_load','aero_240.V_load'])
        self.connect('powercurve.rated_Omega', ['aero_0.Omega_load','aero_120.Omega_load','aero_240.Omega_load'])
        self.connect('pitch_load89', ['aero_0.pitch_load','aero_120.pitch_load','aero_240.pitch_load'])
        self.connect('azimuth_load0', 'aero_0.azimuth_load')
        self.connect('azimuth_load120', 'aero_120.azimuth_load')
        self.connect('azimuth_load240', 'aero_240.azimuth_load')

        self.connect('tiploss', ['analysis.tiploss', 'aero_0.tiploss','aero_120.tiploss','aero_240.tiploss','aero_defl_powercurve.tiploss','aero_extrm_forces.tiploss','aero_extrm.tiploss','aero_rated.tiploss'])
        self.connect('hubloss', ['analysis.hubloss', 'aero_0.hubloss','aero_120.hubloss','aero_240.hubloss','aero_defl_powercurve.hubloss','aero_extrm_forces.hubloss','aero_extrm.hubloss','aero_rated.hubloss'])
        self.connect('wakerotation', ['analysis.wakerotation', 'aero_0.wakerotation','aero_120.wakerotation','aero_240.wakerotation','aero_defl_powercurve.wakerotation','aero_extrm_forces.wakerotation','aero_extrm.wakerotation','aero_rated.wakerotation'])
        self.connect('usecd', ['analysis.usecd', 'aero_0.usecd','aero_120.usecd','aero_240.usecd','aero_defl_powercurve.usecd','aero_extrm_forces.usecd','aero_extrm.usecd','aero_rated.usecd'])

        # connections to root moment for drivetrain

        self.connect('r_pts', ['root_moment_0.r_pts', 'root_moment_120.r_pts', 'root_moment_240.r_pts'])
        self.connect('aero_rated.loads_Px', ['root_moment_0.aeroloads_Px', 'root_moment_120.aeroloads_Px', 'root_moment_240.aeroloads_Px'])
        self.connect('aero_rated.loads_Py', ['root_moment_0.aeroloads_Py', 'root_moment_120.aeroloads_Py', 'root_moment_240.aeroloads_Py'])
        self.connect('aero_rated.loads_Pz', ['root_moment_0.aeroloads_Pz', 'root_moment_120.aeroloads_Pz', 'root_moment_240.aeroloads_Pz'])
        self.connect('aero_rated.loads_r', ['root_moment_0.aeroloads_r', 'root_moment_120.aeroloads_r', 'root_moment_240.aeroloads_r'])
        self.connect('curvature.totalCone', ['root_moment_0.totalCone', 'root_moment_120.totalCone', 'root_moment_240.totalCone'])
        self.connect('curvature.x_az', ['root_moment_0.x_az','root_moment_120.x_az','root_moment_240.x_az'])
        self.connect('curvature.y_az', ['root_moment_0.y_az','root_moment_120.y_az','root_moment_240.y_az'])
        self.connect('curvature.z_az', ['root_moment_0.z_az','root_moment_120.z_az','root_moment_240.z_az'])
        self.connect('curvature.s', ['root_moment_0.s','root_moment_120.s','root_moment_240.s'])

        # connections to root Mxyz outputs
        self.connect('root_moment_0.Mxyz','Mxyz_1_in')
        self.connect('root_moment_120.Mxyz','Mxyz_2_in')
        self.connect('root_moment_240.Mxyz','Mxyz_3_in')
        self.connect('curvature.totalCone','TotalCone_in', src_indices=[NPTS-1])
        self.connect('aero_0.pitch_load','Pitch_in')
        self.connect('root_moment_0.Fxyz', 'Fxyz_1_in')
        self.connect('root_moment_120.Fxyz', 'Fxyz_2_in')
        self.connect('root_moment_240.Fxyz', 'Fxyz_3_in')
        #azimuths not passed. assumed 0,120,240 in drivese function

        # Connections to constraints not accounted for by promotes=*
        self.connect('aero_rated.Omega_load', 'Omega')

        self.add('obj_cmp', ExecComp('obj = -AEP', AEP=1000000.0), promotes=['*'])



        
if __name__ == '__main__':
    myref = DTU10MW() #NREL5MW() 

    rotor = Problem()
    npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve

    rotor.root = RotorSE(myref, npts_coarse_power_curve, npts_spline_power_curve)
    rotor.setup()

    # === blade grid ===
    rotor['hubFraction'] = myref.hubFraction #0.025  # (Float): hub location as fraction of radius
    rotor['bladeLength'] = myref.bladeLength #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    # rotor['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    rotor['precone'] = myref.precone #2.5  # (Float, deg): precone angle
    rotor['tilt'] = myref.tilt #5.0  # (Float, deg): shaft tilt
    rotor['yaw'] = 0.0  # (Float, deg): yaw error
    rotor['nBlades'] = myref.nBlades #3  # (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    rotor['r_max_chord'] =  myref.r_max_chord  # 0.23577 #(Float): location of max chord on unit radius
    rotor['chord_in'] = myref.chord # np.array([3.2612, 4.3254, 4.5709, 3.7355, 2.69923333, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in'] = myref.theta # np.array([0.0, 13.2783, 12.30514836,  6.95106536,  2.72696309, -0.0878099]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in'] = myref.precurve #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in'] = myref.presweep #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    # rotor['delta_precurve_in'] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
    rotor['sparT_in'] = myref.spar_thickness # np.array([0.0, 0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in'] = myref.te_thickness # np.array([0.0, 0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # ------------------

    # === atmosphere ===
    rotor['analysis.rho'] = 1.225  # (Float, kg/m**3): density of air
    rotor['analysis.mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['wind.shearExp'] = 0.25  # (Float): shear exponent
    rotor['hub_height'] = myref.hub_height #90.0  # (Float, m): hub height
    rotor['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['turbulence_class'] = TURBULENCE_CLASS['B']  # (Enum): IEC turbulence class class
    rotor['wind.zref'] = myref.hub_height #90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    rotor['gust_stddev'] = 3
    # ----------------------

    # === control ===
    rotor['control_Vin'] = myref.control_Vin #3.0  # (Float, m/s): cut-in wind speed
    rotor['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
    rotor['control_minOmega'] = myref.control_minOmega #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control_maxOmega'] = myref.control_maxOmega #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control_tsr'] = myref.control_tsr #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control_pitch'] = myref.control_pitch #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor['machine_rating'] = myref.rating #5e6  # (Float, W): rated power
    rotor['pitch_extreme'] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    rotor['azimuth_extreme'] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    rotor['VfactorPC'] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType'] = myref.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    rotor['dynamic_amplication_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------


    # === fatigue ===
    r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
	               0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
	               0.97777724])  # (Array): new aerodynamic grid on unit radius
    rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    xp = np.r_[0.0, r_aero]
    xx = np.r_[0.0, myref.r]
    rotor['rstar_damage'] = np.interp(xx, xp, rstar_damage)
    rotor['Mxb_damage'] = np.interp(xx, xp, Mxb_damage)
    rotor['Myb_damage'] = np.interp(xx, xp, Myb_damage)
    rotor['strain_ult_spar'] = 1.0e-2  # (Float): ultimate strain in spar cap
    rotor['strain_ult_te'] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    rotor['gamma_fatigue'] = 1.755 # (Float): safety factor for fatigue
    rotor['gamma_f'] = 1.35 # (Float): safety factor for loads/stresses
    rotor['gamma_m'] = 1.1 # (Float): safety factor for materials
    rotor['gamma_freq'] = 1.1 # (Float): safety factor for resonant frequencies
    rotor['m_damage'] = 10.0  # (Float): slope of S-N curve for fatigue analysis
    rotor['struc.lifetime'] = 20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------

    # from myutilities import plt

    # === run and outputs ===
    rotor.run()

    print 'AEP =', rotor['AEP']
    print 'diameter =', rotor['diameter']
    print 'ratedConditions.V =', rotor['rated_V']
    print 'ratedConditions.Omega =', rotor['rated_Omega']
    print 'ratedConditions.pitch =', rotor['rated_pitch']
    print 'ratedConditions.T =', rotor['rated_T']
    print 'ratedConditions.Q =', rotor['rated_Q']
    print 'mass_one_blade =', rotor['mass_one_blade']
    print 'mass_all_blades =', rotor['mass_all_blades']
    print 'I_all_blades =', rotor['I_all_blades']
    print 'freq =', rotor['freq']
    print 'tip_deflection =', rotor['tip_deflection']
    print 'root_bending_moment =', rotor['root_bending_moment']
    #for io in rotor.root.unknowns:
    #    print(io + ' ' + str(rotor.root.unknowns[io]))
    '''
    print 'Pn_margin', rotor[ 'Pn_margin']
    print 'P1_margin', rotor[ 'P1_margin']
    print 'Pn_margin_cfem', rotor[ 'Pn_margin_cfem']
    print 'P1_margin_cfem', rotor[ 'P1_margin_cfem']
    print 'rotor_strain_sparU', rotor[ 'rotor_strain_sparU']
    print 'rotor_strain_sparL', rotor[ 'rotor_strain_sparL']
    print 'rotor_strain_teU', rotor[ 'rotor_strain_teU']
    print 'rotor_strain_teL', rotor[ 'rotor_strain_teL']
    print 'eps_crit_spar', rotor['eps_crit_spar']
    print 'strain_ult_spar', rotor['strain_ult_spar']
    print 'eps_crit_te', rotor['eps_crit_te']
    print 'strain_ult_te', rotor['strain_ult_te']
    print 'rotor_buckling_sparU', rotor[ 'rotor_buckling_sparU']
    print 'rotor_buckling_sparL', rotor[ 'rotor_buckling_sparL']
    print 'rotor_buckling_teU', rotor[ 'rotor_buckling_teU']
    print 'rotor_buckling_teL', rotor[ 'rotor_buckling_teL']
    print 'rotor_damage_sparU', rotor[ 'rotor_damage_sparU']
    print 'rotor_damage_sparL', rotor[ 'rotor_damage_sparL']
    print 'rotor_damage_teU', rotor[ 'rotor_damage_teU']
    print 'rotor_damage_teL', rotor[ 'rotor_damage_teL']
    '''

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rotor['V'], rotor['P']/1e6)
    plt.xlabel('wind speed (m/s)')
    plt.xlabel('power (W)')

    plt.figure()

    plt.plot(rotor['r_pts'], rotor['strainU_spar'], label='suction')
    plt.plot(rotor['r_pts'], rotor['strainL_spar'], label='pressure')
    plt.plot(rotor['r_pts'], rotor['eps_crit_spar'], label='critical')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r')
    plt.ylabel('strain')
    plt.legend()
    # plt.savefig('/Users/sning/Desktop/strain_spar.pdf')
    # plt.savefig('/Users/sning/Desktop/strain_spar.png')

    plt.figure()

    plt.plot(rotor['r_pts'], rotor['strainU_te'], label='suction')
    plt.plot(rotor['r_pts'], rotor['strainL_te'], label='pressure')
    plt.plot(rotor['r_pts'], rotor['eps_crit_te'], label='critical')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r')
    plt.ylabel('strain')
    plt.legend()
    # plt.savefig('/Users/sning/Desktop/strain_te.pdf')
    # plt.savefig('/Users/sning/Desktop/strain_te.png')

    plt.show()
    # ----------------
