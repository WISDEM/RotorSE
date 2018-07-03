#!/usr/bin/env python
# encoding: utf-8
"""
test_rotor_gradients.py

Created by Andrew Ning on 2013-01-28.
Copyright (c) NREL. All rights reserved.
"""

import pytest
import numpy as np
#from commonse.utilities import check_gradient_unit_test, check_for_missing_unit_tests
from rotorse.rotor_structure import TotalLoads, RootMoment, MassProperties, TipDeflection, \
    ExtremeLoads, GustETM, BladeCurvature, SetupPCModVarSpeed, BladeDeflection, DamageLoads, NREL5MW
from rotorse import TURBULENCE_CLASS
from openmdao.api import IndepVarComp, Problem, Group
from enum import Enum

from commonse_testing import check_gradient_unit_test, init_IndepVar_add, init_IndepVar_set# <- TODO give this a permanent home

##### Input Fixtures #####

@pytest.fixture
def inputs_str():
    data = {}
    data['r'] = np.array([1.575, 1.87769653853, 1.9760701684, 2.07444379828, 2.17896578003, 2.27733940991, 2.37571303979, 2.940033033, 3.07662515267, 3.17499878255, 5.67000020475, 7.0730538665, 8.39996676225, 11.8743527705, 13.86, 14.7396807928, 15.9074997953, 18.5704718586, 20.0025, 22.0688840711, 24.0975002047, 26.1698347667, 28.1924997953, 32.2875, 33.5663020632, 36.3825002048, 38.5649121314, 40.4774997953, 40.6864988295, 40.7861021746, 40.887, 42.6792760302, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
    data['theta'] = np.array([13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 12.9342402117, 12.4835185594, 11.4807962375, 10.9555376235, 10.2141732458, 9.50474414552, 8.79808349002, 8.12523177814, 6.8138304713, 6.42067815056, 5.58414310075, 4.96394649167, 4.44089107951, 4.38490319227, 4.35830230526, 4.3314093512, 3.86273855446, 3.38640148153, 1.57771432025, 0.953398905137, 0.504982546655, 0.0995167038088, -0.0878099])
    data['tilt'] = 5.0
    data['rhoA'] = np.array([1086.31387923, 1102.59977206, 1120.90516514, 1126.20434689, 1131.813843, 1137.07339365, 1086.16660785, 882.217532971, 894.901622709, 899.719853943, 731.023082747, 608.342843886, 542.222718332, 341.332102119, 336.952507859, 333.167490612, 330.70966121, 326.000136659, 323.264256247, 317.941928823, 310.869166296, 273.899186484, 262.928994775, 245.05275646, 236.749218603, 217.487633686, 205.107910133, 194.259756942, 167.463953528, 166.764129206, 165.810677408, 155.69651911, 146.647497247, 100.919341248, 82.5679535343, 67.796593467, 46.2981454714, 31.2090805766])
    data['totalCone'] = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    data['z_az'] = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])
    return data

@pytest.fixture
def inputs_aeroloads_operating():
    data = {}
    data['aeroLoads_r'] = np.array([1.575, 2.940033033, 5.67000020475, 8.39996676225, 13.86, 15.9074997953, 20.0025, 24.0975002047, 28.1924997953, 32.2875, 36.3825002048, 40.4774997953, 40.887, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
    data['aeroLoads_Px'] = np.array([0.0, 394.495335464, 488.816521749, 403.91526216, 3498.00637403, 3269.97046313, 4043.59672475, 4501.44255062, 5033.93200097, 5718.97291507, 6320.75067512, 7109.17757276, 8059.89616754, 8525.76224891, 8922.31561839, 8701.63574206, 8270.06152072, 7295.11706029, 0.0])
    data['aeroLoads_Py'] = np.array([-0, 70.1691953218, 168.88301855, 208.278564575, -1235.05334966, -1093.35619071, -1356.19151969, -1524.64653804, -1637.39245978, -1682.20028084, -1815.96503432, -1854.41936505, -2010.09761627, -1958.48701857, -1702.10190133, -1523.95755076, -1325.12331629, -935.412289309, -0])
    data['aeroLoads_Pz'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data['aeroLoads_Omega'] = 12.0
    data['aeroLoads_pitch'] = 0.0
    data['aeroLoads_azimuth'] = 180.0
    return data

@pytest.fixture
def inputs_aeroloads_extreme():
    data = {}
    data['aeroLoads_r'] = np.array([1.575, 2.940033033, 5.67000020475, 8.39996676225, 13.86, 15.9074997953, 20.0025, 24.0975002047, 28.1924997953, 32.2875, 36.3825002048, 40.4774997953, 40.887, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
    data['aeroLoads_Px'] = np.array([0.0, 5284.01751742, 5957.44357892, 4576.24416488, 24983.1752719, 23185.4068553, 22772.6584971, 20742.2633915, 19438.9406889, 18477.9925607, 17798.8416165, 16587.2148132, 16307.1178682, 15198.3868957, 12397.4212608, 11065.3344497, 9918.01519381, 8690.90719256, 0.0])
    data['aeroLoads_Py'] = np.array([-0, 3.23552756967e-13, 3.64788210501e-13, 2.80214138432e-13, -8241.69218443, -6959.18655392, -6216.4717769, -5238.79147412, -3965.60018365, -3301.4647433, -2434.41539477, -1921.45833684, -1784.90503315, -1413.83527174, -766.483242904, -565.612693573, -430.805328036, -317.225077607, -0])
    data['aeroLoads_Pz'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data['aeroLoads_Omega'] = 0.0
    data['aeroLoads_pitch'] = 0.0
    data['aeroLoads_azimuth'] = 0.0
    return data


##### Tests #####

@pytest.mark.parametrize("inputs_str,inputs_aeroloads", [
    (inputs_str(), inputs_aeroloads_operating()),
    (inputs_str(), inputs_aeroloads_extreme())
])
def test_TotalLoads(inputs_str, inputs_aeroloads):
    npts = np.size(inputs_str['r'])

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', TotalLoads(npts), promotes=['*'])
    prob = init_IndepVar_add(prob, inputs_str)
    prob = init_IndepVar_add(prob, inputs_aeroloads)
    prob.setup(check=False)

    prob = init_IndepVar_set(prob, inputs_str)
    prob = init_IndepVar_set(prob, inputs_aeroloads)

    check_gradient_unit_test(prob, tol=.003, display=False)

def test_TestRootMoment():

    data_aero = inputs_aeroloads_operating()
    data_str = inputs_str()
    npts = np.size(data_str['z_az'])
    
    data = {}
    data['aeroLoads_r'] = data_aero['aeroLoads_r']
    data['aeroLoads_Px'] = data_aero['aeroLoads_Px']
    data['aeroLoads_Py'] = data_aero['aeroLoads_Py']
    data['aeroLoads_Pz'] = data_aero['aeroLoads_Pz']

    data['r_pts'] = data_str['r'] + 0.1
    data['totalCone'] = data_str['totalCone']
    data['s'] = data_str['z_az']
    data['x_az'] = np.zeros_like(data['s'])
    data['y_az'] = np.zeros_like(data['s'])
    data['z_az'] = data_str['z_az']
    
    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', RootMoment(npts), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.root.comp.deriv_options['form'] = 'central'
    prob.root.comp.deriv_options['check_form'] = 'central'
    prob.root.comp.deriv_options['step_calc'] = 'relative'   
    prob.root.comp.deriv_options['check_step_calc'] = 'relative'
    prob.setup(check=False)

    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob, tol=5e-3)
    # check_gradient_unit_test(prob, tol=0.1)


def test_MassProperties():

    data = {}
    data['blade_mass'] = 17288.717087
    data['blade_moment_of_inertia'] = 11634376.0531
    data['tilt'] = 5.0
    data['nBlades'] = 3

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', MassProperties(), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.root.comp.deriv_options['check_form'] = 'central'
    prob.root.comp.deriv_options['check_step_calc'] = 'relative'
    prob.setup(check=False)

    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob)


def test_TipDeflection():
    data = {}
    data['dx'] = 4.27242809591
    data['dy'] = -0.371550675139
    data['dz'] = 0.0400553989266
    data['theta'] = -0.0878099
    data['pitch'] = 0.0
    data['azimuth'] = 180.0
    data['tilt'] = 5.0
    data['totalConeTip'] = 2.5
    data['dynamicFactor'] = 1.2

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', TipDeflection(), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.setup(check=False)
    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob)


def test_ExtremeLoads():

    data = {}
    data['T'] = np.array([2414072.40260361, 188461.59444074])
    data['Q']= np.array([10926313.24295958, -8041330.51312603])
    data['nBlades'] = 3

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', ExtremeLoads(), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.setup(check=False)
    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob, tol=1.5e-4)



def test_GustETM():

    data = {}
    data['V_mean'] = 10.0
    data['V_hub'] = 11.7733866478
    data['std'] = 3
    data['turbulence_class'] = TURBULENCE_CLASS['B']

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', GustETM(), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.setup(check=False)
    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob)


def test_BladeCurvature():

    data = {}
    data['r'] = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])
    data['precurve'] = np.array([0.0, 0.043324361025, 0.0573893371698, 0.0714469497372, 0.0863751069858, 0.100417566593, 0.114452695996, 0.194824077331, 0.214241777459, 0.228217752953, 0.580295739194, 0.776308800624, 0.960411633829, 1.4368012564, 1.7055214864, 1.823777005, 1.98003324362, 2.32762426752, 2.50856911855, 2.76432512112, 3.0113656418, 3.26199245912, 3.50723775206, 4.0150233695, 4.17901272929, 4.55356019347, 4.85962948702, 5.14086873143, 5.17214747287, 5.18708601127, 5.20223968442, 5.47491847385, 5.77007321175, 7.12818875977, 7.7314427824, 8.22913789456, 8.73985955154, 9.0])
    data['presweep'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data['precone'] = 2.5
    npts = np.size(data['r'])

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', BladeCurvature(npts), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.setup(check=False)
    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob, tol=5e-5)


def test_SetupPCModVarSpeed():

    data = {}
    data['control_tsr'] = 8.0
    data['control_pitch'] = 1.0
    data['Vrated'] = 12.0
    data['R'] = 63.0
    data['Vfactor'] = 0.7

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', SetupPCModVarSpeed(), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.setup(check=False)
    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob)


def test_BladeDeflection():
    NINPUT = 5

    data = {}
    data['dx'] = np.array([0.0, 1.39292987639e-05, 2.42964362361e-05, 3.73611415086e-05, 5.41276445811e-05, 7.25773833888e-05, 9.35928069108e-05, 0.000268596883961, 0.000326596195925, 0.000372259193056, 0.00253285667923, 0.00455502505195, 0.00706548815047, 0.017530829505, 0.0272151750396, 0.0326222002275, 0.0410034329532, 0.0655343595834, 0.0818962168404, 0.109404026845, 0.140873579829, 0.177728457961, 0.218834564559, 0.318921553007, 0.354759106888, 0.442199496033, 0.51925808593, 0.593774567144, 0.602307030873, 0.606403310701, 0.610574803156, 0.688594372351, 0.779039709944, 1.2575016857, 1.49348304065, 1.69400834707, 1.90049599542, 2.00437890947])
    data['dy'] = np.array([0.0, -9.11673273998e-07, -1.59100573731e-06, -2.44958708331e-06, -3.55417484862e-06, -4.77185559999e-06, -6.16078184177e-06, -1.77741315033e-05, -2.16387627714e-05, -2.46907649535e-05, -0.000177142122562, -0.00032886393339, -0.000523351402122, -0.00136667950935, -0.00215626107629, -0.00258384925494, -0.00322230578732, -0.00497009730837, -0.00607094164543, -0.00784913963514, -0.00980572470543, -0.0120149227037, -0.0143773305481, -0.0197932419212, -0.0216584171821, -0.0260612290834, -0.0297556792352, -0.0331954636074, -0.0335827012855, -0.0337680683508, -0.0339563913295, -0.0373950218981, -0.0412189884014, -0.0597108654077, -0.0681209435104, -0.0750137913709, -0.0819682184214, -0.085450339495])
    data['dz'] = np.array([0.0, 0.000190681386865, 0.000249594803444, 0.000305808888278, 0.000363037181592, 0.000414559803159, 0.000465006717172, 0.00079021115147, 0.000878140045153, 0.000937905765842, 0.00236994444878, 0.003141201122, 0.00385188728597, 0.0059090995974, 0.00721749042255, 0.00775193614485, 0.00838246793108, 0.00965065210961, 0.010256099836, 0.0110346081014, 0.0117147556349, 0.0123373351129, 0.0128834601832, 0.0138847141125, 0.014168689932, 0.0147331399477, 0.0151335663707, 0.015450772371, 0.015480903206, 0.0154926033582, 0.0155017716157, 0.0156218668894, 0.0157089720273, 0.0159512118376, 0.0160321965202, 0.0160695719649, 0.0160814363339, 0.0160814363339])
    data['pitch'] = 0.0
    data['theta'] = np.array([13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 12.9342399734, 12.4835173655, 11.4807910924, 10.9555299481, 10.2141623046, 9.50473135323, 8.7980712345, 8.12522342051, 6.81383731475, 6.42068577363, 5.58414615907, 4.96394315694, 4.44088253161, 4.38489418276, 4.35829308634, 4.33139992746, 3.86272702658, 3.38639207628, 1.57773054352, 0.953410121155, 0.504987738102, 0.0995174088527, -0.0878099])
    data['r_in0'] = np.array([1.5375, 13.2620872, 17.35321975, 40.19535987, 63.0375])
    data['Rhub0'] = 1.5375
    data['r_pts0'] = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])
    data['precurve0'] = np.zeros_like(data['dx']) #np.linspace(0.0, 5.0, np.size(data['dx']))
    data['bladeLength0'] = 61.5
    npts = np.size(data['dx'])

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', BladeDeflection(npts), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.setup(check=False)
    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob, tol=1e-4)


def test_DamageLoads():

    data = {}
    data['r'] = np.array([2.8667, 5.6, 8.3333, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25, 36.35, 40.45, 44.55, 48.65, 52.75, 56.1667, 58.9, 61.6333])  # (Array): new aerodynamic grid on unit radius
    data['rstar'] = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    data['Mxb'] = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    data['Myb'] = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    data['theta'] = np.array([13.308, 13.308, 13.308, 13.308, 11.48, 10.162, 9.011, 7.795, 6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.37, 0.106])
    npts = np.size(data['r'])

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', DamageLoads(npts), promotes=['*'])
    prob = init_IndepVar_add(prob, data)
    prob.root.comp.deriv_options['check_form'] = 'central'
    prob.root.comp.deriv_options['check_step_calc'] = 'relative'   
    prob.setup(check=False)
    prob = init_IndepVar_set(prob, data)

    check_gradient_unit_test(prob, tol=5e-5)

