# from __future__ import print_function

from openmdao.api import Problem, SqliteRecorder
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
import numpy as np
from rotor import RotorSE
import os
import matplotlib.pyplot as plt
from precomp import Orthotropic2DMaterial, CompositeSection, Profile
from ccblade import CCAirfoil
import time
from airfoilprep import AirfoilAnalysis

rotor = Problem()

# === airfoil parameters ===
# airfoilOptions = dict(AnalysisMethod='CFD', AirfoilParameterization='CST', DirectSpline=True,
#                                 CFDOptions=dict(iterations=5000, processors=64, configFile='discrete_check.cfg', computeAirfoilsInParallel=True, CFDGradType='AutoDiff'),
#                                 GradientOptions=dict(ComputeGradient=True, ComputeAirfoilGradients=False),
#                                 SplineOptions=dict(AnalysisMethod='XFOIL', maxDirectAoA=180, alphas=np.linspace(-5, 15, 10), Re=1e6, cd_max=1.5,
#                                                    correction3D=False, r_over_R=0.5, chord_over_r=0.15, tsr=7.55),
#                                 PrecomputationalOptions=dict(AirfoilParameterization='Blended', numAirfoilsToCompute=10, tcMax=0.42, tcMin=0.13))

airfoilOptions = dict(AnalysisMethod='XFOIL', AirfoilParameterization='Precomputational', DirectSpline=True,
                                CFDOptions=dict(iterations=5000, processors=64, configFile='discrete_check.cfg', computeAirfoilsInParallel=True, CFDGradType='AutoDiff'),
                                GradientOptions=dict(ComputeGradient=True, ComputeAirfoilGradients=True),
                                SplineOptions=dict(AnalysisMethod='XFOIL', maxDirectAoA=0, alphas=np.linspace(-5, 15, 10), Re=1e6, cd_max=1.5,
                                                   correction3D=False, r_over_R=0.5, chord_over_r=0.15, tsr=7.55),
                                PrecomputationalOptions=dict(AirfoilParameterization='Blended', numAirfoilsToCompute=10, tcMax=0.42, tcMin=0.13))



########### Airfoil Options ###########
# AnalysisMethod = None, 'XFOIL', 'CFD', 'Files', AirfoilParameterization = None, 'CST', 'Precomputational', 'NACA'
# CFDOptions: iterations = max iterations of CFD, processors = number of processors available to use, configFile = SU2 config file in AirfoilAnalysisFiles, computeAirfoilsInParallel = whether or not to compute multiple airfoils in parallel
# GradientOptions: ComputeGradient = whether or not to compute gradients in CCBlade, ComputeAirfoilGradients = whether or not to calculate airfoil parameterization gradients, fd_step = finite difference step size, cs_step = complex step size
# SplineOptions: AnalysisMethod: 'XFOIL', 'CFD', maxDirectAoA: deg at which spline takes over, alphas: alphas to use to compute spline, Re: Reynolds number to compute spline
# PrecomputationalOptions: AirfoilParameterization = 'TC' (thickness-to-chord ratio), 'Blended' (thickness-to-chord ratio with blended airfoil families)
# CFDGradType='AutoDiff', 'FiniteDiff', 'ContAdjoint'

# Create name for optimization files
if airfoilOptions['GradientOptions']['ComputeAirfoilGradients']:
    opt_type ='FreeForm'
else:
    opt_type = 'Conventional'
if airfoilOptions['AirfoilParameterization'] == 'Precomputational':
        airfoil_param = airfoilOptions['AirfoilParameterization'] + '_' + airfoilOptions['PrecomputationalOptions']['AirfoilParameterization']
else:
        airfoil_param = airfoilOptions['AirfoilParameterization']


other = '6_16_5'

if airfoilOptions['AnalysisMethod'] is not None:
    description = airfoilOptions['AnalysisMethod'] + '_' + airfoil_param + '_' + opt_type + '_' + other
else:
    description = ''

## SETUP OPTIMIZATION
# rotor.driver = pyOptSparseDriver()
# rotor.driver.options['optimizer'] = 'SNOPT'
# rotor.driver.opt_settings['Print file'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SNOPToutputs') + os.sep + 'SNOPT_print_' + description +'.out'
# rotor.driver.opt_settings['Summary file'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SNOPToutputs') + os.sep + 'SNOPT_summary_' + description +'.out'
# rotor.driver.opt_settings['Major optimality tolerance'] = 5e-4
# rotor.driver.opt_settings['Verify level'] = -1 # 3
# recorder = SqliteRecorder(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SNOPToutputs') + os.sep + 'recorder_' + description +'.sql')
# recorder.options['record_params'] = True
# recorder.options['record_metadata'] = True
# rotor.driver.add_recorder(recorder)

## Setup design variables and objective
rotor.driver.add_objective('obj')

rotor.driver.add_desvar('r_max_chord', lower=0.1, upper=0.5)#, scaler=0.1113567)
rotor.driver.add_desvar('chord_sub', lower=1.3, upper=5.3 , scaler=0.25)#, scaler=np.asarray([0.02234462, 0.002056 ,  0.02063628 , 0.00910857])) # scaler=np.asarray([0.1, 0.005, 0.1, 0.1]))
rotor.driver.add_desvar('theta_sub', lower=-10.0, upper=19.0, scaler=0.25)#, scaler=np.asarray([0.00304422,  0.02875508,  0.01657161,  0.00333825]))#, scaler=0.1)
rotor.driver.add_desvar('control:tsr', lower=3.0, upper=9.0, scaler=0.1)#, scaler=0.00071386*100.) #scaler=0.01)
rotor.driver.add_desvar('sparT', lower=0.005, upper=0.2)#, scaler=dsparT)
rotor.driver.add_desvar('teT', lower=0.005, upper=0.2)#, scaler=dteT)

### Setup free form optimization
num_airfoils = 6
if airfoilOptions['AirfoilParameterization'] == 'CST':
    airfoils_dof = 8
    lower = np.ones((num_airfoils,airfoils_dof))*[[-0.6, -0.76, -0.4, -0.25, 0.13, 0.16, 0.13, 0.1],[-0.6, -0.76, -0.4, -0.25, 0.13, 0.16, 0.13, 0.1],[-0.6, -0.76, -0.4, -0.25, 0.13, 0.16, 0.13, 0.1],
                            [-0.6, -0.76, -0.4, -0.25, 0.13, 0.16, 0.13, 0.1],[-0.6, -0.76, -0.4, -0.25, 0.13, 0.16, 0.13, 0.1],[-0.3, -0.36, -0.3, -0.25, 0.13, 0.16, 0.13, 0.1]]
    upper = np.ones((num_airfoils,airfoils_dof))*[[-0.13, -0.16, -0.13, 0.15, 0.55, 0.55, 0.4, 0.4],[-0.13, -0.16, -0.13, 0.15, 0.55, 0.55, 0.4, 0.4],[-0.13, -0.16, -0.13, 0.15, 0.55, 0.55, 0.4, 0.4],
                            [-0.13, -0.16, -0.13, 0.28, 0.55, 0.55, 0.4, 0.4],[-0.13, -0.16, -0.13, 0.20, 0.55, 0.55, 0.4, 0.4],[-0.10, -0.13, -0.10, 0.2, 0.4, 0.45, 0.4, 0.4]]
    scaler_airfoilparam = 20 #np.ones((num_airfoils,airfoils_dof))
    # scaler_airfoilparam[1][4],scaler_airfoilparam[1][5],scaler_airfoilparam[1][7],scaler_airfoilparam[2][3], scaler_airfoilparam[2][1] = 20, 20, 20, 0.1, 10
elif airfoilOptions['AirfoilParameterization'] == 'Precomputational':
    tcMax = airfoilOptions['PrecomputationalOptions']['tcMax']
    tcMin = airfoilOptions['PrecomputationalOptions']['tcMin']
    tcMin = 0.1775
    if airfoilOptions['PrecomputationalOptions']['AirfoilParameterization'] == 'Blended':
        airfoils_dof = 2

        lower = np.ones((num_airfoils,airfoils_dof))*[[tcMin, 0.], [tcMin, 0.], [tcMin, 0.], [tcMin, 0.], [tcMin, 0.], [tcMin, 0.]]
        upper = np.ones((num_airfoils,airfoils_dof))*[[tcMax, 1.], [tcMax, 1.], [tcMax, 1.], [tcMax, 1.], [tcMax, 1.], [tcMax, 1.]]
        scaler_airfoilparam = np.asarray(np.ones((num_airfoils,airfoils_dof))*np.asarray([5, 1])).flatten()

        scaler_airfoilparam = 10 #np.ones(12)
        #for w in range(6):
        #    scaler_airfoilparam[2*w] = 10
        #    scaler_airfoilparam[2*w+1] = 0.1

    else:
        airfoils_dof = 1
        lower = tcMin
        upper = tcMax
        scaler_airfoilparam = 5
else:
    print "Setting default airfoil parameterization bounds"
    lower = None
    upper = None
    scaler_airfoilparam = 1
# scaler_airfoilparam = 10 #np.ones(48)
rotor.driver.add_desvar('airfoil_parameterization', lower=lower, upper=upper, scaler=scaler_airfoilparam)

## Setup constraints
rotor.driver.add_constraint('con_strain_spar', lower=-1.0, upper=1.0)  # rotor strain sparL
rotor.driver.add_constraint('con_strainU_te', lower=-1.0, upper=1.0)  # rotor strain teL
rotor.driver.add_constraint('con_strainL_te', upper=1.0)
rotor.driver.add_constraint('con_eps_spar', upper=0.0)  # rotor buckling spar
rotor.driver.add_constraint('con_eps_te', upper=0.0)  # rotor buckling te
#rotor.driver.add_constraint('con_freq', lower=0.0)  # flap/edge freq
if airfoilOptions['AirfoilParameterization'] == 'CST':
    rotor.driver.add_constraint('con_afp', lower=0.03)  # To prevent overlapping airfoils shapes
elif airfoilOptions['AirfoilParameterization'] == 'Precomputational':
    rotor.driver.add_constraint('con_afp', lower=0.005)#175)#75)  # To prevent overlapping airfoils shapes
rotor.driver.add_constraint('obj', lower=5) # To insure that COE does not go negative with negative AEP
rotor.driver.add_constraint('con_power', lower=0.0)#, scaler=1e-6)
rotor.driver.add_constraint('con_thrust', upper=0.73)#37) # Constrain to scaled initial thrust so that other components of turbine remain constant

print "Setting up RotorSE...\n"
initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
    0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
    0.97777724])  # (Array): initial aerodynamic grid on unit radius
initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
    0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
    0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
    0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
    0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
    1.0])  # (Array): initial structural grid on unit radius
naero = len(initial_aero_grid)
nstr = len(initial_str_grid)
npower = 5  # 20
rotor.root = RotorSE(naero, nstr, npower, num_airfoils, airfoils_dof)
rotor.setup(check=False)

# === blade grid ===
rotor['initial_aero_grid'] = initial_aero_grid
rotor['initial_str_grid'] = initial_str_grid
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
rotor['tilt'] = 0.0 # 5.0  # (Float, deg): shaft tilt
rotor['yaw'] = 0.0  # (Float, deg): yaw error
rotor['nBlades'] = 3  # (Int): number of blades
# ------------------

af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
af_str_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]
rotor['af_idx'] = np.asarray(af_idx)
rotor['af_str_idx'] = np.asarray(af_str_idx)
basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_AFFiles/')

if airfoilOptions['AnalysisMethod'] == None:
    # === airfoil files ===
    airfoil_types = [0]*8
    # airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    # airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    # airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    # airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    # airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    # airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    # airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    # airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17')

    # place at appropriate radial stations
    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]
    airfoil_parameterization = np.zeros((num_airfoils, airfoils_dof))
else:
    if airfoilOptions['AirfoilParameterization'] == 'Precomputational':
        # Settings for precomputational methods
        if airfoilOptions['AnalysisMethod'] == 'Files':
            # Airfoil data files
            # Corresponding to blended airfoil family factor of 0.0
            baseAirfoilsData0 = [0]*5
            baseAirfoilsData0[0] = os.path.join(basepath, 'DU40_A17')
            baseAirfoilsData0[1] = os.path.join(basepath, 'DU35_A17')
            baseAirfoilsData0[2] = os.path.join(basepath, 'DU30_A17')
            baseAirfoilsData0[3] = os.path.join(basepath, 'DU25_A17')
            baseAirfoilsData0[4] = os.path.join(basepath, 'DU21_A17')
            # Corresponding to blended airfoil family factor of 1.0
            baseAirfoilsData1 = [0]*1
            baseAirfoilsData1[0] = os.path.join(basepath, 'NACA64_A17')
            airfoilOptions['PrecomputationalOptions']['BaseAirfoilsData0'] = baseAirfoilsData0
            airfoilOptions['PrecomputationalOptions']['BaseAirfoilsData1'] = baseAirfoilsData1
            # airfoilOptions['PrecomputationalOptions']['precomp_idx'] = [0.405, 0.35, 0.30, 0.25, 0.21, 0.18]
            # Corresponding to blended airfoil family factor of 0.0
            baseAirfoilsCoordindates0 = [0]*5
            baseAirfoilsCoordindates0[0] = os.path.join(basepath, 'DU40.dat')
            baseAirfoilsCoordindates0[1] = os.path.join(basepath, 'DU35.dat')
            baseAirfoilsCoordindates0[2] = os.path.join(basepath, 'DU30.dat')
            baseAirfoilsCoordindates0[3] = os.path.join(basepath, 'DU25.dat')
            baseAirfoilsCoordindates0[4] = os.path.join(basepath, 'DU21.dat')
            # Corresponding to blended airfoil family factor of 1.0
            baseAirfoilsCoordindates1 = [0]*1
            baseAirfoilsCoordindates1[0] = os.path.join(basepath, 'NACA64.dat')
            airfoilOptions['PrecomputationalOptions']['BaseAirfoilsCoordinates0'] = baseAirfoilsCoordindates0
            airfoilOptions['PrecomputationalOptions']['BaseAirfoilsCoordinates1'] = baseAirfoilsCoordindates1

        else:
            # Airfoil coordinate files
            # Corresponding to blended airfoil family factor of 0.0
            baseAirfoilsCoordindates0 = [0]*5
            baseAirfoilsCoordindates0[0] = os.path.join(basepath, 'DU40.dat')
            baseAirfoilsCoordindates0[1] = os.path.join(basepath, 'DU35.dat')
            baseAirfoilsCoordindates0[2] = os.path.join(basepath, 'DU30.dat')
            baseAirfoilsCoordindates0[3] = os.path.join(basepath, 'DU25.dat')
            baseAirfoilsCoordindates0[4] = os.path.join(basepath, 'DU21.dat')
            # Corresponding to blended airfoil family factor of 1.0
            baseAirfoilsCoordindates1 = [0]*1
            baseAirfoilsCoordindates1[0] = os.path.join(basepath, 'NACA64.dat')
            airfoilOptions['PrecomputationalOptions']['BaseAirfoilsCoordinates0'] = baseAirfoilsCoordindates0
            airfoilOptions['PrecomputationalOptions']['BaseAirfoilsCoordinates1'] = baseAirfoilsCoordindates1
        if airfoilOptions['PrecomputationalOptions']['AirfoilParameterization'] == 'Blended':
            # Specify precomputational airfoil shapes starting point
            airfoil_parameterization = np.asarray([[0.404458, 0.0], [0.349012, 0.0], [0.29892, 0.0], [0.251105, 0.0], [0.211299, 0.0], [0.179338, 1.0]])

            # airfoil_parameterization = np.asarray([[ 0.39654312,  0.        ],
            #                          [ 0.29664731,  0.        ],
            #                          [ 0.26664731,  0.        ],
            #                          [ 0.23664731,  0.        ],
            #                          [ 0.20664731 , 0.        ],
            #                          [ 0.17664731 , 1.0]])
            #airfoil_parameterization = np.asarray([[0.404458, 1.0], [0.349012, 1.0], [0.29892, 1.0], [0.251105, 1.0], [0.211299, 1.0], [0.179338, 1.0]])
            #airfoil_parameterization = np.asarray([[0.419, 0.5], [0.38, 0.5], [0.35, 0.5], [0.32, 0.5], [0.30, 0.5], [0.24, 0.5]])

        else:
            airfoil_parameterization = np.asarray([[0.404458], [0.349012], [0.29892], [0.251105], [0.211299], [0.179338]])
            # airfoil_parameterization = np.asarray([[ 0.39654312],
            #              [ 0.29664731 ],
            #              [ 0.26664731  ],
            #              [ 0.23664731       ],
            #              [ 0.20664731      ],
            #              [ 0.17664731]])
            airfoilOptions['PrecomputationalOptions']['AirfoilFamilySpecification'] = [0., 0., 0., 0., 0., 1.]
        # Generate precomputational model
        print "Generating precomputational model"
        time0 = time.time()
        afanalysis = AirfoilAnalysis([0.25, 0.0], airfoilOptions)
        print "Precomputational model generation complete in ", time.time() - time0, " seconds."
        af_precomp_init = CCAirfoil.initFromPrecomputational

    elif airfoilOptions['AirfoilParameterization'] == 'NACA':
        af_freeform_init = CCAirfoil.initFromFreeForm
        airfoil_parameterization = np.asarray([[2440], [2435], [2430], [2425], [2421], [2418]])

    elif airfoilOptions['AirfoilParameterization'] == 'CST':
        af_freeform_init = CCAirfoil.initFromFreeForm
        airfoil_parameterization = np.asarray([[-0.49209940079930325, -0.72861624849999296, -0.38147646962813714, 0.13679205926397994, 0.50396496117640877, 0.54798355691567613, 0.37642896917099616, 0.37017796580840234],
                                           [-0.38027535114760153, -0.75920832612723133, -0.21834261746205941, 0.086359012110824224, 0.38364567865371835, 0.48445264573011815, 0.26999944648962521, 0.34675843509167931],
                                           [-0.29817561716727448, -0.67909473119918973, -0.15737231648880162, 0.12798260780188203, 0.2842322211249545, 0.46026650967959087, 0.21705062978922526, 0.33758303223369945],
                                           [-0.27413320446357803, -0.40701949670950271, -0.29237424992338562, 0.27867844397438357, 0.23582783854698663, 0.43718573158380936, 0.25389099250498309, 0.31090780344061775],
                                           [-0.19600050454371795, -0.28861738331958697, -0.20594891135118523, 0.19143138186871009, 0.22876347660120994, 0.39940768357615447, 0.28896745336793572, 0.29519782561050112],
                                           [-0.17200255338600826, -0.13744743777735921, -0.24288986290945222, 0.15085289615063024, 0.20650016452789369, 0.35540642522188848, 0.32797634888819488, 0.2592276816645861]])
    else:
        raise ValueError("Error. Please specify AirfoilParameterization parameter.")

    # load all airfoils
    af_nonairfoil_init = CCAirfoil.initFromInput
    non_airfoils_idx = 2
    airfoil_types = [0]*(num_airfoils+non_airfoils_idx)
    non_airfoils_alphas, non_airfoils_cls, non_airfoils_cds = [-180.0, 0.0, 180.0], [0.0, 0.0, 0.0], [[0.5, 0.5, 0.5],[0.35, 0.35, 0.35]]

    print "Generating initial airfoil data..."
    for i in range(len(airfoil_types)):
        if i < non_airfoils_idx:
            airfoil_types[i] = af_nonairfoil_init(non_airfoils_alphas, airfoilOptions['SplineOptions']['Re'], non_airfoils_cls, non_airfoils_cds[i], non_airfoils_cls)
        else:
            if airfoilOptions['AirfoilParameterization'] != 'Precomputational':
                time0 = time.time()
                airfoil_types[i] = af_freeform_init(airfoil_parameterization[i-2], airfoilOptions, airfoilNum=i-2)
                print "Airfoil ", str(i+1-2), " data generation complete in ", time.time() - time0, " seconds."
            else:
                airfoil_types[i] = af_precomp_init(airfoil_parameterization[i-2], airfoilOptions, afanalysis, airfoilNum=i-2)
    print "Finished generating initial airfoil data.\n"

    af = [0]*naero
    for i in range(len(af)):
        af[i] = airfoil_types[af_idx[i]]

if airfoil_parameterization.shape[0] != num_airfoils and airfoil_parameterization.shape[1] != airfoils_dof:
    raise ValueError("Error in airfoil number specification or degrees of freedom for airfoil parameterization.")
rotor['airfoil_parameterization'] = airfoil_parameterization
rotor['airfoil_files'] = np.array(af) # (List): names of airfoil file
rotor['airfoilOptions'] = airfoilOptions  # (List): names of airfoil file
# ----------------------

# === atmosphere ===
rotor['rho'] = 1.225  # (Float, kg/m**3): density of air
rotor['mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
rotor['shearExp'] = 0.0 #0.2  # (Float): shear exponent
rotor['hubHt'] = np.array([90.0])  # (Float, m): hub height
rotor['turbine_class'] = 'I'  # (Enum): IEC turbine class
rotor['turbulence_class'] = 'B'  # (Enum): IEC turbulence class class
rotor['cdf_reference_height_wind_speed'] = np.array([90.0])  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
rotor['g'] = 9.81  # (Float, m/s**2): acceleration of gravity
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
rotor['nSector'] = 1 #4 # (Int): number of sectors to divide rotor face into in computing thrust and power
rotor['npts_coarse_power_curve'] = npower #20  # (Int): number of points to evaluate aero analysis at
rotor['npts_spline_power_curve'] = 200  # (Int): number of points to use in fitting spline to power curve
rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
rotor['drivetrainType'] = 'geared'  # (Enum)
rotor['nF'] = 5  # (Int): number of natural frequencies to compute
rotor['dynamic_amplication_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
# ----------------------
rotor['weibull_shape'] = 2.0
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
     2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c not true for this case)
rotor['thick_str_ref'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0.404457084248, 0.404457084248,
                                   0.349012780126, 0.349012780126, 0.349012780126, 0.349012780126, 0.29892003076, 0.29892003076, 0.25110545018, 0.25110545018, 0.25110545018, 0.25110545018,
                                   0.211298863564, 0.211298863564, 0.211298863564, 0.211298863564, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591,
                                   0.17933792591, 0.17933792591])  # (Array, m): airfoil thickness distribution for reference section, thickness of structural layup scaled with reference thickness

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

## XFOIL SEQUENTIAL PRECOMP
# rotor['r_max_chord'] = 0.30594864483
# rotor['chord_sub'] = np.asarray([2.64431191,  4.68831876,  2.93181727,  1.4663922])
# rotor['theta_sub'] =np.asarray([11.22756732,   4.07292633,   2.07397116 , -1.26310705])
# rotor['control:tsr'] = 7.56231791713
# rotor['sparT'] = np.asarray([0.0465602 ,  0.04067914,  0.0424077 ,  0.03058118 , 0.005 ])
# rotor['teT'] = np.asarray([0.06578571,  0.03063708 , 0.01774828, 0.00851841 , 0.005     ])
# rotor['airfoil_parameterization'] = np.asarray([[0.4165], [0.3285], [0.3269], [0.2704], [0.2607], [0.2060]])

## XFOIL SEQUENTIAL FREE-FORM
# rotor['r_max_chord'] = 0.305853750637
# rotor['chord_sub'] = np.asarray([2.65368979,  4.7019722,   2.9144189 ,  1.43591315])
# rotor['theta_sub'] =np.asarray([11.16407188,   4.08841314,   2.03440192 , -1.59038968])
# rotor['control:tsr'] = 7.56454894469
# rotor['sparT'] = np.asarray([0.04356289,  0.04073581 , 0.04169466,  0.0305305,   0.005 ])
# rotor['teT'] = np.asarray([ 0.06508523,  0.03107021,  0.01814533,  0.00902979 , 0.005     ])
# rotor['airfoil_parameterization'] = np.asarray([[-0.48411868, -0.74006757, -0.37450258,  0.14374694,  0.49325893,  0.53275051,   0.37720118,  0.36419747],
#                                                 [-0.37307419, -0.75547518, -0.22526599,  0.12909754,  0.3552338 ,  0.4861 ,  0.24750146,  0.32033204],
#                                                 [-0.30113695, -0.67333271, -0.15302167,  0.13397681,  0.28470387,  0.47014058,   0.22346286,  0.33281683],
#                                                 [-0.26801205, -0.40230151, -0.29072191,  0.26597296,  0.22577472,  0.44711499,   0.24778359,  0.31924774],
#                                                 [-0.18452911, -0.28555648, -0.20338616,  0.1905786 ,  0.1976176 ,  0.41002144,   0.29075454,  0.29180838],
#                                                 [-0.14143763, -0.1395226 , -0.2235385 ,  0.171756  ,  0.19729459,  0.39172983,   0.34186528,  0.24913183]])

## WIND TUNNEL SEQUENTIAL PRECOMP
# rotor['r_max_chord'] = 0.362730389616
# rotor['chord_sub'] = np.asarray([ 2.48038535,  4.21313897  ,2.39517919  ,1.3       ])
# rotor['theta_sub'] =np.asarray([ 10.11568347 ,  6.24530378  , 1.65649693,  -1.66688645])
# rotor['control:tsr'] = 8.72078746507
# rotor['sparT'] = np.asarray([ 0.08258899,  0.03849649 , 0.0433979 ,  0.02711884 , 0.005     ])
# rotor['teT'] = np.asarray([ 0.04901721,  0.02797307 , 0.01577557  ,0.0087453 ,  0.005     ])
# rotor['airfoil_parameterization'] = np.asarray([[0.416], [0.376], [0.291], [0.22], [0.177], [0.170]])

# rotor['r_max_chord'] = 0.353237296134
# rotor['chord_sub'] = np.asarray([ 2.43425837,  4.42881131  ,2.36417173  ,1.3       ])
# rotor['theta_sub'] =np.asarray([ 10.4970535 ,  6.30041292  , 1.48065974,  -1.61284556])
# rotor['control:tsr'] = 8.76189536732
# rotor['sparT'] = np.asarray([ 0.05814816,  0.03737911 , 0.04152867 ,  0.02746216 , 0.005     ])
# rotor['teT'] = np.asarray([ 0.07149582,  0.02691555 , 0.0171385  ,0.00771193 ,  0.005     ])

## XFOIL BLENDED
# rotor['r_max_chord'] = 0.397
# rotor['chord_sub'] = np.asarray([2.50,  3.47,  2.50,  1.30])
# rotor['theta_sub'] =np.asarray([10.01, 2.43, -0.453, -3.45])
# rotor['control:tsr'] = 7.17
# rotor['sparT'] = np.asarray([ 0.1314, 0.0437, 0.0414,0.0253, 0.005])
# rotor['teT'] = np.asarray([0.0122, 0.0313, 0.0121,0.0101, 0.005])

# rotor['r_max_chord'] = 0.304442536224
# rotor['chord_sub'] = np.asarray([ 2.51298944 , 4.38093478 , 2.40454857 , 1.3000004 ])
# rotor['theta_sub'] =np.asarray([ 12.96943982 ,  6.4020822 ,   1.38278901,  -0.5616445 ])
# rotor['control:tsr'] = 8.81741325805
# rotor['sparT'] = np.asarray([ 0.03893594 , 0.03769856 , 0.03876112 , 0.02845868,  0.005 ])
# rotor['teT'] = np.asarray([0.05935936,  0.0257054,   0.0166151 ,  0.00828156,  0.005   ])
# rotor['airfoil_parameterization'] = np.asarray(
#                     [[ 0.40286791,  0.        ],
#                      [ 0.34616276,  0.        ],
#                      [ 0.27907581,  0.        ],
#                      [ 0.27846536,  0.        ],
#                      [ 0.21039297,  0.        ],
#                      [ 0.18198916,  1.        ]])


## CFD SEQUENTIAL PRECOMP*
# rotor['r_max_chord'] = 0.337403328198
# rotor['chord_sub'] = np.asarray([ 2.62783842,  5.3 ,        3.17906231,  1.3       ])
# rotor['theta_sub'] =np.asarray([ 13.84094714 ,  7.89939298,   4.38008742,   2.67636968])
# rotor['control:tsr'] = 6.92194720297
# rotor['sparT'] = np.asarray([ 0.0280493 ,  0.04275217 , 0.03990218 , 0.03125918,  0.005    ])
# rotor['teT'] = np.asarray([ 0.10398376  ,0.02872634 , 0.02068796 , 0.00780592  , 0.005     ])


#rotor['r_max_chord'] = 0.337403328198
#rotor['chord_sub'] = np.asarray([ 2.62783842,  5.3 ,        3.17906231,  1.3       ])
#rotor['theta_sub'] =np.asarray([ 13.84094714 ,  7.89939298,   4.38008742,   2.67636968])
#rotor['control:tsr'] = 6.92194720297
#rotor['sparT'] = np.asarray([ 0.0280493 ,  0.04275217 , 0.03990218 , 0.03125918,  0.005    ])
#rotor['teT'] = np.asarray([ 0.10398376  ,0.02872634 , 0.02068796 , 0.00780592  , 0.005     ])

# rotor['r_max_chord'] = 0.337403328198
# rotor['chord_sub'] = np.asarray([ 2.62783842,  5.3 ,        3.17906231,  1.3       ])
# rotor['theta_sub'] =np.asarray([ 13.84094714 ,  7.89939298,   4.38008742,   2.67636968])
# rotor['control:tsr'] = 6.92194720297
# rotor['sparT'] = np.asarray([ 0.0280493 ,  0.04275217 , 0.03990218 , 0.03125918,  0.005    ])
# rotor['teT'] = np.asarray([ 0.10398376  ,0.02872634 , 0.02068796 , 0.00780592  , 0.005     ])


# r_max_chord = 0.5
# chord_sub = [ 2.85014334  5.3         2.8970185   1.3       ]
# theta_sub = [ 16.19903195   4.497619     3.61676888   1.82262644]
# control:tsr = 8.38634480806
# sparT = [ 0.07715639  0.04509293  0.03503564  0.0473798   0.005     ]
# teT = [ 0.0402252   0.03742472  0.01091087  0.00711263  0.00817813]
# airfoil_parameterization =  [[ 0.36402902  0.00127083]
#  [ 0.32085978  0.00471985]
#  [ 0.38057649  0.03870216]
#  [ 0.38319624  0.28605152]
#  [ 0.3966964   0.56574417]
#  [ 0.13        0.82064758]]


## RANS CFD Freeform sequential
#rotor['r_max_chord'] = 0.305867746519
#rotor['chord_sub'] = np.asarray([ 2.72848913,  4.62947959,  3.73599201,  1.40079366])
#rotor['theta_sub'] =np.asarray([ 18.79599459,   6.15752196,   4.22456476,   2.27658322])
#rotor['control:tsr'] = 6.51664361838
#rotor['sparT'] = np.asarray([ 0.09238801,  0.0428983 ,  0.04932103,  0.03016356,  0.005     ])
#rotor['teT'] = np.asarray([ 0.0054388682,  0.03394868,  0.01738666 , 0.01053027,  0.005     ])


## RANS CFD Precomp Blended
# rotor['r_max_chord'] = 0.348442998143
# rotor['chord_sub'] = np.asarray([ 2.79646745,  5.12141766,  4.096272,    1.3 ])
# rotor['theta_sub'] =np.asarray([ 12.99911322,   7.38757736,   2.96244579,   0.8966654])
# rotor['control:tsr'] = 5.52412038766
# rotor['sparT'] = np.asarray([ 0.09354558 , 0.04740579 , 0.05440516,  0.04242511,  0.006303     ])
# rotor['teT'] = np.asarray([0.05508666,  0.03272339,  0.02220209 , 0.01134766,  0.00584235])
# rotor['airfoil_parameterization'] = np.asarray([[-0.35509472, -0.28900029 ,-0.41783446,  0.28699514 , 0.34542002,  0.69218962 , 0.58466902 , 0.56369997],
#         [-0.32626775, -0.43328654, -0.27722026,  0.2977187 ,  0.22543824,  0.49591305,  0.26701317,  0.38116127],
#         [-0.24552364, -0.32279951, -0.20617259,  0.21572266,  0.23123506,  0.47645063,  0.31987228,  0.38500504],
#         [-0.21504009, -0.28269067, -0.18059485,  0.18893535,  0.20252879,  0.41729999,  0.28018349,  0.33720541],
#         [-0.18659439, -0.1530313 , -0.2194988 ,  0.15892766,  0.18343914,  0.36992634 , 0.30966212,  0.30159353],
#         [-0.15265034, -0.20033559, -0.12843173,  0.13410369,  0.14379709,  0.29626216,  0.19913759,  0.2394092]])

#rotor['airfoil_parameterization'] = np.asarray([[-0.48353824, -0.83350169 ,-0.20464719,  0.10133592,  0.39013416,  0.59639011,  0.28181401 , 0.45055916],
#[-0.35475002, -0.4711439,  -0.30140156,  0.32372274,  0.24509285,  0.53915757 , 0.29025907 , 0.41439808],
#[-0.31887863, -0.42348871, -0.27093018,  0.29099452,  0.22030111,  0.48464265 , 0.26089089 , 0.3725061 ],
#[-0.24046286 ,-0.31617119, -0.2018998 ,  0.21127403,  0.22646517,  0.46662295 , 0.31325703 , 0.37706029],
#[-0.2099751  ,-0.27609713 ,-0.17629817 , 0.18448575,  0.19775332,  0.40747299 , 0.27353164  ,0.32926456],
#[-0.18190136 ,-0.13012688 ,-0.22694908,  0.15389686,  0.18041638 , 0.36224118  ,0.31561678  ,0.29586921]])


# === run and outputs ===
"Running RotorSE..."
time0 = time.time()
rotor.run()
time1 = time.time() - time0
print
print "================== RotorSE Outputs =================="
print "Time to run: ", time1
print "COE: ", rotor['COE']*100., 'cents/kWh'
print "mass / AEP: ", (rotor['mass_all_blades'] + 589154) / rotor['AEP']
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
print
print "================== Optimization Design Variables =================="
print 'r_max_chord =', rotor['r_max_chord']
print 'chord_sub =', rotor['chord_sub']
print 'theta_sub =', rotor['theta_sub']
print 'control:tsr =', rotor['control:tsr']
print 'sparT =', rotor['sparT']
print 'teT =', rotor['teT']
print 'airfoil_parameterization = ', rotor['airfoil_parameterization']



#### Check specific gradients
# grad = rotor.calc_gradient(['curvefem.beam:GJ', 'curvefem.beam:rhoA', 'curvefem.beam:rhoJ', 'curvefem.beam:x_ec_str', 'curvefem.beam:y_ec_str', 'curvefem.theta_str', 'curvefem.precurve_str', 'curvefem.presweep_str'], ['curvefem.freq'], mode='auto')
# grad = grad.flatten()
# gradfd = rotor.calc_gradient(['curvefem.beam:GJ', 'curvefem.beam:rhoA', 'curvefem.beam:rhoJ', 'curvefem.beam:x_ec_str', 'curvefem.beam:y_ec_str', 'curvefem.theta_str', 'curvefem.precurve_str', 'curvefem.presweep_str'], ['curvefem.freq'], mode='fd')
# gradfd = gradfd.flatten()

# grad = rotor.calc_gradient(['curvefem.theta_str', 'curvefem.precurve_str', 'curvefem.presweep_str'], ['curvefem.freq'], mode='auto')
# grad = grad.flatten()
# gradfd = rotor.calc_gradient(['curvefem.theta_str', 'curvefem.precurve_str', 'curvefem.presweep_str'], ['curvefem.freq'], mode='fd')
# gradfd = gradfd.flatten()
# grad = rotor.calc_gradient(['r_max_chord'], ['beam.beam:z', 'beam.beam:EIxx', 'beam.beam:EIyy', 'beam.beam:EIxy', 'beam.beam:EA', 'beam.beam:rhoJ', 'beam.beam:x_ec_str', 'beam.beam:y_ec_str', 'spline.theta_str'], mode='auto')
# grad = grad.flatten()
# gradfd = rotor.calc_gradient(['r_max_chord'], ['beam.beam:z', 'beam.beam:EIxx', 'beam.beam:EIyy', 'beam.beam:EIxy', 'beam.beam:EA', 'beam.beam:rhoJ', 'beam.beam:x_ec_str', 'beam.beam:y_ec_str', 'spline.theta_str'], mode='fd')
# gradfd = gradfd.flatten()
# """grad = rotor.calc_gradient(['airfoil_parameterization'], ['obj'], mode='auto')
# grad = grad.flatten()
# rotor['airfoilOptions']['GradientOptions']['ComputeGradient'] = False
# gradfd = rotor.calc_gradient(['airfoil_parameterization'], ['obj'], mode='fd')
# gradfd = gradfd.flatten()
# print 'ad', grad
# print 'fd', gradfd
# plt.figure()
# plt.semilogy(range(len(grad)), abs(grad))
# plt.semilogy(range(len(gradfd)), abs(gradfd))
# plt.show()"""


#print grad
# airfoilOptions['ComputeGradient'] = False
# rotor['airfoilOptions'] = airfoilOptions
# gradfd = rotor.calc_gradient(['control:tsr', 'chord_sub', 'r_max_chord'], ['obj'], mode='fd')


#### Check total derivatives (design variables, obj, and cons)
# total = open('total_derivatives_' + description + '.txt', 'w')
# rotor.check_total_derivatives(out_stream=total)
# # total = open('partial_derivatives_' + description + '.txt', 'w')
# rotor.check_partial_derivatives(out_stream=total)
# total.close()
#
# #### Check scaled gradients
# w = rotor.calc_gradient(list(rotor.driver.get_desvars().keys()),
#                       list(rotor.driver.get_objectives().keys()),# + list(rotor.driver.get_constraints().keys()),
#                       dv_scale=rotor.driver.dv_conversions,
#                       cn_scale=rotor.driver.fn_conversions)
# print w
#
# w = rotor.calc_gradient(list(rotor.driver.get_desvars().keys()),
#                       list(rotor.driver.get_constraints().keys()),
#                       dv_scale=rotor.driver.dv_conversions,
#                       cn_scale=rotor.driver.fn_conversions)
# print w

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
print "RotorSE Complete"
plt.show()
# ----------------

