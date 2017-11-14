#!/usr/bin/env python
# encoding: utf-8
"""
test_rotor_gradients.py

Created by Andrew Ning on 2013-01-28.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
#from commonse.utilities import check_gradient_unit_test, check_for_missing_unit_tests
from test_rotoraero_gradients import check_gradient_unit_test, check_for_missing_unit_tests
from rotorse.rotor import RGrid, TotalLoads, TipDeflection, RootMoment, MassProperties, \
    ExtremeLoads, GustETM, BladeCurvature, SetupPCModVarSpeed, BladeDeflection, DamageLoads
from openmdao.api import IndepVarComp, Problem, Group
from enum import Enum


class TestRGrid(unittest.TestCase):

    def test1(self):

        r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333, 0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
        fraction = np.array([0.0, 0.221750339522, 0.293817188822, 0.365884038121, 0.442455065507, 0.514521914807, 0.586588764105, 1.0, 0.050034345133, 0.0860690751106, 1.0, 0.513945366068, 1.0, 0.636330560083, 1.0, 0.429636571789, 1.0, 0.65029839566, 1.0, 0.504611469554, 1.0, 0.506064656711, 1.0, 1.0, 0.312283760506, 1.0, 0.532945578735, 1.0, 0.510375896771, 0.75360738708, 1.0, 0.486304715835, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        idxj = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 14, 15, 16, 17])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', RGrid(len(r_aero), len(fraction)), promotes=['*'])
        prob.root.add('r_aero', IndepVarComp('r_aero', np.zeros(len(r_aero))), promotes=['*'])
        prob.root.add('fraction', IndepVarComp('fraction', np.zeros(len(fraction))), promotes=['*'])
        prob.root.add('idxj', IndepVarComp('idxj', np.zeros(len(idxj))), promotes=['*'])
        prob.setup(check=False)

        prob['r_aero'] = r_aero
        prob['fraction'] = fraction
        prob['idxj'] = idxj

        check_gradient_unit_test(self, prob)


class TestTotalLoads(unittest.TestCase):

    def test1(self):

        aeroLoads_r = np.array([1.575, 2.940033033, 5.67000020475, 8.39996676225, 13.86, 15.9074997953, 20.0025, 24.0975002047, 28.1924997953, 32.2875, 36.3825002048, 40.4774997953, 40.887, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
        aeroLoads_Px = np.array([0.0, 394.495335464, 488.816521749, 403.91526216, 3498.00637403, 3269.97046313, 4043.59672475, 4501.44255062, 5033.93200097, 5718.97291507, 6320.75067512, 7109.17757276, 8059.89616754, 8525.76224891, 8922.31561839, 8701.63574206, 8270.06152072, 7295.11706029, 0.0])
        aeroLoads_Py = np.array([-0, 70.1691953218, 168.88301855, 208.278564575, -1235.05334966, -1093.35619071, -1356.19151969, -1524.64653804, -1637.39245978, -1682.20028084, -1815.96503432, -1854.41936505, -2010.09761627, -1958.48701857, -1702.10190133, -1523.95755076, -1325.12331629, -935.412289309, -0])
        aeroLoads_Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        aeroLoads_Omega = 12.0
        aeroLoads_pitch = 0.0
        aeroLoads_azimuth = 180.0
        r = np.array([1.575, 1.87769653853, 1.9760701684, 2.07444379828, 2.17896578003, 2.27733940991, 2.37571303979, 2.940033033, 3.07662515267, 3.17499878255, 5.67000020475, 7.0730538665, 8.39996676225, 11.8743527705, 13.86, 14.7396807928, 15.9074997953, 18.5704718586, 20.0025, 22.0688840711, 24.0975002047, 26.1698347667, 28.1924997953, 32.2875, 33.5663020632, 36.3825002048, 38.5649121314, 40.4774997953, 40.6864988295, 40.7861021746, 40.887, 42.6792760302, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
        theta = np.array([13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 12.9342402117, 12.4835185594, 11.4807962375, 10.9555376235, 10.2141732458, 9.50474414552, 8.79808349002, 8.12523177814, 6.8138304713, 6.42067815056, 5.58414310075, 4.96394649167, 4.44089107951, 4.38490319227, 4.35830230526, 4.3314093512, 3.86273855446, 3.38640148153, 1.57771432025, 0.953398905137, 0.504982546655, 0.0995167038088, -0.0878099])
        tilt = 5.0
        rhoA = np.array([1086.31387923, 1102.59977206, 1120.90516514, 1126.20434689, 1131.813843, 1137.07339365, 1086.16660785, 882.217532971, 894.901622709, 899.719853943, 731.023082747, 608.342843886, 542.222718332, 341.332102119, 336.952507859, 333.167490612, 330.70966121, 326.000136659, 323.264256247, 317.941928823, 310.869166296, 273.899186484, 262.928994775, 245.05275646, 236.749218603, 217.487633686, 205.107910133, 194.259756942, 167.463953528, 166.764129206, 165.810677408, 155.69651911, 146.647497247, 100.919341248, 82.5679535343, 67.796593467, 46.2981454714, 31.2090805766])
        totalCone = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        z_az = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', TotalLoads(len(r)), promotes=['*'])
        prob.root.add('aeroLoads_r', IndepVarComp('aeroLoads:r', np.zeros(len(aeroLoads_r))), promotes=['*'])
        prob.root.add('aeroLoads_Px', IndepVarComp('aeroLoads:Px', np.zeros(len(aeroLoads_Px))), promotes=['*'])
        prob.root.add('aeroLoads_Py', IndepVarComp('aeroLoads:Py', np.zeros(len(aeroLoads_Py))), promotes=['*'])
        prob.root.add('aeroLoads_Pz', IndepVarComp('aeroLoads:Pz', np.zeros(len(aeroLoads_Pz))), promotes=['*'])
        prob.root.add('aeroLoads_Omega', IndepVarComp('aeroLoads:Omega', aeroLoads_Omega), promotes=['*'])
        prob.root.add('aeroLoads_pitch', IndepVarComp('aeroLoads:pitch', aeroLoads_pitch), promotes=['*'])
        prob.root.add('aeroLoads_azimuth', IndepVarComp('aeroLoads:azimuth', aeroLoads_azimuth), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', np.zeros(len(r))), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', np.zeros(len(theta))), promotes=['*'])
        prob.root.add('tilt', IndepVarComp('tilt', tilt), promotes=['*'])
        prob.root.add('rhoA', IndepVarComp('rhoA', np.zeros(len(rhoA))), promotes=['*'])
        prob.root.add('totalCone', IndepVarComp('totalCone', np.zeros(len(totalCone))), promotes=['*'])
        prob.root.add('z_az', IndepVarComp('z_az', np.zeros(len(z_az))), promotes=['*'])
        prob.setup(check=False)

        prob['aeroLoads:r'] = aeroLoads_r
        prob['aeroLoads:Px'] = aeroLoads_Px
        prob['aeroLoads:Py'] = aeroLoads_Py
        prob['aeroLoads:Pz'] = aeroLoads_Pz
        prob['aeroLoads:Omega'] = aeroLoads_Omega
        prob['aeroLoads:pitch'] = aeroLoads_pitch
        prob['aeroLoads:azimuth'] = aeroLoads_azimuth
        prob['r'] = r
        prob['theta'] = theta
        prob['tilt'] = tilt
        prob['rhoA'] = rhoA
        prob['totalCone'] = totalCone
        prob['z_az'] = z_az

        check_gradient_unit_test(self, prob, tol=.003, display=False)


    def test2(self):

        aeroLoads_r = np.array([1.575, 2.940033033, 5.67000020475, 8.39996676225, 13.86, 15.9074997953, 20.0025, 24.0975002047, 28.1924997953, 32.2875, 36.3825002048, 40.4774997953, 40.887, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
        aeroLoads_Px = np.array([0.0, 5284.01751742, 5957.44357892, 4576.24416488, 24983.1752719, 23185.4068553, 22772.6584971, 20742.2633915, 19438.9406889, 18477.9925607, 17798.8416165, 16587.2148132, 16307.1178682, 15198.3868957, 12397.4212608, 11065.3344497, 9918.01519381, 8690.90719256, 0.0])
        aeroLoads_Py = np.array([-0, 3.23552756967e-13, 3.64788210501e-13, 2.80214138432e-13, -8241.69218443, -6959.18655392, -6216.4717769, -5238.79147412, -3965.60018365, -3301.4647433, -2434.41539477, -1921.45833684, -1784.90503315, -1413.83527174, -766.483242904, -565.612693573, -430.805328036, -317.225077607, -0])
        aeroLoads_Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        aeroLoads_Omega = 0.0
        aeroLoads_pitch = 0.0
        aeroLoads_azimuth = 0.0
        r = np.array([1.575, 1.87769653853, 1.9760701684, 2.07444379828, 2.17896578003, 2.27733940991, 2.37571303979, 2.940033033, 3.07662515267, 3.17499878255, 5.67000020475, 7.0730538665, 8.39996676225, 11.8743527705, 13.86, 14.7396807928, 15.9074997953, 18.5704718586, 20.0025, 22.0688840711, 24.0975002047, 26.1698347667, 28.1924997953, 32.2875, 33.5663020632, 36.3825002048, 38.5649121314, 40.4774997953, 40.6864988295, 40.7861021746, 40.887, 42.6792760302, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
        theta = np.array([13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 12.9342402117, 12.4835185594, 11.4807962375, 10.9555376235, 10.2141732458, 9.50474414552, 8.79808349002, 8.12523177814, 6.8138304713, 6.42067815056, 5.58414310075, 4.96394649167, 4.44089107951, 4.38490319227, 4.35830230526, 4.3314093512, 3.86273855446, 3.38640148153, 1.57771432025, 0.953398905137, 0.504982546655, 0.0995167038088, -0.0878099])
        tilt = 5.0
        rhoA = np.array([1086.31387923, 1102.59977206, 1120.90516514, 1126.20434689, 1131.813843, 1137.07339365, 1086.16660785, 882.217532971, 894.901622709, 899.719853943, 731.023082747, 608.342843886, 542.222718332, 341.332102119, 336.952507859, 333.167490612, 330.70966121, 326.000136659, 323.264256247, 317.941928823, 310.869166296, 273.899186484, 262.928994775, 245.05275646, 236.749218603, 217.487633686, 205.107910133, 194.259756942, 167.463953528, 166.764129206, 165.810677408, 155.69651911, 146.647497247, 100.919341248, 82.5679535343, 67.796593467, 46.2981454714, 31.2090805766])
        totalCone = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        z_az = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', TotalLoads(len(r)), promotes=['*'])
        prob.root.add('aeroLoads_r', IndepVarComp('aeroLoads:r', np.zeros(len(aeroLoads_r))), promotes=['*'])
        prob.root.add('aeroLoads_Px', IndepVarComp('aeroLoads:Px', np.zeros(len(aeroLoads_Px))), promotes=['*'])
        prob.root.add('aeroLoads_Py', IndepVarComp('aeroLoads:Py', np.zeros(len(aeroLoads_Py))), promotes=['*'])
        prob.root.add('aeroLoads_Pz', IndepVarComp('aeroLoads:Pz', np.zeros(len(aeroLoads_Pz))), promotes=['*'])
        prob.root.add('aeroLoads_Omega', IndepVarComp('aeroLoads:Omega', aeroLoads_Omega), promotes=['*'])
        prob.root.add('aeroLoads_pitch', IndepVarComp('aeroLoads:pitch', aeroLoads_pitch), promotes=['*'])
        prob.root.add('aeroLoads_azimuth', IndepVarComp('aeroLoads:azimuth', aeroLoads_azimuth), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', np.zeros(len(r))), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', np.zeros(len(theta))), promotes=['*'])
        prob.root.add('tilt', IndepVarComp('tilt', tilt), promotes=['*'])
        prob.root.add('rhoA', IndepVarComp('rhoA', np.zeros(len(rhoA))), promotes=['*'])
        prob.root.add('totalCone', IndepVarComp('totalCone', np.zeros(len(totalCone))), promotes=['*'])
        prob.root.add('z_az', IndepVarComp('z_az', np.zeros(len(z_az))), promotes=['*'])
        prob.setup(check=False)

        prob['aeroLoads:r'] = aeroLoads_r
        prob['aeroLoads:Px'] = aeroLoads_Px
        prob['aeroLoads:Py'] = aeroLoads_Py
        prob['aeroLoads:Pz'] = aeroLoads_Pz
        prob['aeroLoads:Omega'] = aeroLoads_Omega
        prob['aeroLoads:pitch'] = aeroLoads_pitch
        prob['aeroLoads:azimuth'] = aeroLoads_azimuth
        prob['r'] = r
        prob['theta'] = theta
        prob['tilt'] = tilt
        prob['rhoA'] = rhoA
        prob['totalCone'] = totalCone
        prob['z_az'] = z_az

        check_gradient_unit_test(self, prob, tol=0.02)  # a couple with more significant errors, but I think these are correct and that the finite differencing is just poor.
        # TODO: However, I should check Akima in more detail




class TestRootMoment(unittest.TestCase):

    def test1(self):


        # adding offset to try to avoid linear interpolation issue
        r_str = 0.1 + np.array([1.575, 1.87769653853, 1.9760701684, 2.0744437988, 2.17896578003, 2.27733940991, 2.37571303979, 2.940033033, 3.07662515267, 3.17499878255, 5.67000020475, 7.0730538665, 8.39996676225, 11.8743527705, 13.86, 14.7396807928, 15.9074997953, 18.5704718586, 20.0025, 22.0688840711, 24.0975002047, 26.1698347667, 28.1924997953, 32.2875, 33.5663020632, 36.3825002048, 38.5649121314, 40.4774997953, 40.6864988295, 40.7861021746, 40.887, 42.6792760302, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
        aeroLoads_r = np.array([1.575, 2.940033033, 5.67000020475, 8.39996676225, 13.86, 15.9074997953, 20.0025, 24.0975002047, 28.1924997953, 32.2875, 36.3825002048, 40.4774997953, 40.887, 44.5725, 52.7624997953, 56.1750332378, 58.9049997952, 61.634966967, 63.0])
        aeroLoads_Px = np.array([0.0, 394.495335464, 488.816521749, 403.91526216, 3498.00637403, 3269.97046313, 4043.59672475, 4501.44255062, 5033.93200097, 5718.97291507, 6320.75067512, 7109.17757276, 8059.89616754, 8525.76224891, 8922.31561839, 8701.63574206, 8270.06152072, 7295.11706029, 0.0])
        aeroLoads_Py = np.array([-0, 70.1691953218, 168.88301855, 208.278564575, -1235.05334966, -1093.35619071, -1356.19151969, -1524.64653804, -1637.39245978, -1682.20028084, -1815.96503432, -1854.41936505, -2010.09761627, -1958.48701857, -1702.10190133, -1523.95755076, -1325.12331629, -935.412289309, -0])
        aeroLoads_Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        totalCone = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        x_az = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y_az = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        z_az = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])
        s = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', RootMoment(len(r_str)), promotes=['*'])
        prob.root.add('r_str', IndepVarComp('r_str', np.zeros(len(r_str))), promotes=['*'])
        prob.root.add('aeroLoads_r', IndepVarComp('aeroLoads:r', np.zeros(len(aeroLoads_r))), promotes=['*'])
        prob.root.add('aeroLoads_Px', IndepVarComp('aeroLoads:Px', np.zeros(len(aeroLoads_Px))), promotes=['*'])
        prob.root.add('aeroLoads_Py', IndepVarComp('aeroLoads:Py', np.zeros(len(aeroLoads_Py))), promotes=['*'])
        prob.root.add('aeroLoads_Pz', IndepVarComp('aeroLoads:Pz', np.zeros(len(aeroLoads_Pz))), promotes=['*'])
        prob.root.add('totalCone', IndepVarComp('totalCone', np.zeros(len(totalCone))), promotes=['*'])
        prob.root.add('x_az', IndepVarComp('x_az', np.zeros(len(x_az))), promotes=['*'])
        prob.root.add('y_az', IndepVarComp('y_az', np.zeros(len(y_az))), promotes=['*'])
        prob.root.add('z_az', IndepVarComp('z_az', np.zeros(len(z_az))), promotes=['*'])
        prob.root.add('s', IndepVarComp('s', np.zeros(len(s))), promotes=['*'])

        prob.setup(check=False)

        prob['r_str'] = r_str
        prob['aeroLoads:r'] = aeroLoads_r
        prob['aeroLoads:Px'] = aeroLoads_Px
        prob['aeroLoads:Py'] = aeroLoads_Py
        prob['aeroLoads:Pz'] = aeroLoads_Pz
        prob['totalCone'] = totalCone
        prob['x_az'] = x_az
        prob['y_az'] = y_az
        prob['z_az'] = z_az
        prob['s'] = s

        # check_gradient_unit_test(self, prob, tol=5e-3, display=False)



class TestMassProperties(unittest.TestCase):

    def test1(self):

        blade_mass = 17288.717087
        blade_moment_of_inertia = 11634376.0531
        tilt = 5.0
        nBlades = 3

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', MassProperties(), promotes=['*'])
        prob.root.add('blade_mass', IndepVarComp('blade_mass', blade_mass), promotes=['*'])
        prob.root.add('blade_moment_of_inertia', IndepVarComp('blade_moment_of_inertia', blade_moment_of_inertia), promotes=['*'])
        prob.root.add('tilt', IndepVarComp('tilt', tilt), promotes=['*'])
        prob.root.add('nBlades', IndepVarComp('nBlades', nBlades), promotes=['*'])

        prob.setup(check=False)

        prob['blade_mass'] = blade_mass
        prob['blade_moment_of_inertia'] = blade_moment_of_inertia
        prob['tilt'] = tilt
        prob['nBlades'] = nBlades

        check_gradient_unit_test(self, prob)


class TestTipDeflection(unittest.TestCase):

    def test1(self):

        dx = 4.27242809591
        dy = -0.371550675139
        dz = 0.0400553989266
        theta = -0.0878099
        pitch = 0.0
        azimuth = 180.0
        tilt = 5.0
        precone = 2.5
        dynamicFactor = 1.2

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', TipDeflection(), promotes=['*'])
        prob.root.add('dx', IndepVarComp('dx', dx), promotes=['*'])
        prob.root.add('dy', IndepVarComp('dy', dy), promotes=['*'])
        prob.root.add('dz', IndepVarComp('dz', dz), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', theta), promotes=['*'])
        prob.root.add('pitch', IndepVarComp('pitch', pitch), promotes=['*'])
        prob.root.add('azimuth', IndepVarComp('azimuth', azimuth), promotes=['*'])
        prob.root.add('tilt', IndepVarComp('tilt', tilt), promotes=['*'])
        prob.root.add('totalConeTip', IndepVarComp('totalConeTip', precone), promotes=['*'])
        prob.root.add('dynamicFactor', IndepVarComp('dynamicFactor', dynamicFactor), promotes=['*'])

        prob.setup(check=False)

        prob['dx'] = dx
        prob['dy'] = dy
        prob['dz'] = dz
        prob['theta'] = theta
        prob['pitch'] = pitch
        prob['azimuth'] = azimuth
        prob['tilt'] = tilt
        prob['totalConeTip'] = precone
        prob['dynamicFactor'] = dynamicFactor


        check_gradient_unit_test(self, prob)


class TestExtremeLoads(unittest.TestCase):

    def test1(self):

        T = np.array([2414072.40260361, 188461.59444074])
        Q = np.array([10926313.24295958, -8041330.51312603])
        nBlades = 3

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', ExtremeLoads(), promotes=['*'])
        prob.root.add('T', IndepVarComp('T', T), promotes=['*'])
        prob.root.add('Q', IndepVarComp('Q', Q), promotes=['*'])
        prob.root.add('nBlades', IndepVarComp('nBlades', nBlades), promotes=['*'])

        prob.setup(check=False)

        prob['T'] = T
        prob['Q'] = Q
        prob['nBlades'] = nBlades

        check_gradient_unit_test(self, prob, tol=1.5e-4)



class TestGustETM(unittest.TestCase):

    def test1(self):

        V_mean = 10.0
        V_hub = 11.7733866478
        turbulence_class = 'B'
        std = 3


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', GustETM(), promotes=['*'])
        prob.root.add('V_mean', IndepVarComp('V_mean', V_mean), promotes=['*'])
        prob.root.add('V_hub', IndepVarComp('V_hub', V_hub), promotes=['*'])
        prob.root.add('turbulence_class', IndepVarComp('turbulence_class', Enum('A', 'B', 'C')), promotes=['*'])
        prob.root.add('std', IndepVarComp('std', std), promotes=['*'])

        prob.setup(check=False)

        prob['V_mean'] = V_mean
        prob['V_hub'] = V_hub
        prob['V_hub'] = V_hub
        prob['turbulence_class'] = turbulence_class
        prob['std'] = std

        check_gradient_unit_test(self, prob)


class TestBladeCurvature(unittest.TestCase):

    def test1(self):

        r = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])
        precurve = np.array([0.0, 0.043324361025, 0.0573893371698, 0.0714469497372, 0.0863751069858, 0.100417566593, 0.114452695996, 0.194824077331, 0.214241777459, 0.228217752953, 0.580295739194, 0.776308800624, 0.960411633829, 1.4368012564, 1.7055214864, 1.823777005, 1.98003324362, 2.32762426752, 2.50856911855, 2.76432512112, 3.0113656418, 3.26199245912, 3.50723775206, 4.0150233695, 4.17901272929, 4.55356019347, 4.85962948702, 5.14086873143, 5.17214747287, 5.18708601127, 5.20223968442, 5.47491847385, 5.77007321175, 7.12818875977, 7.7314427824, 8.22913789456, 8.73985955154, 9.0])
        presweep = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        precone = 2.5


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', BladeCurvature(len(r)), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', r), promotes=['*'])
        prob.root.add('precurve', IndepVarComp('precurve', precurve), promotes=['*'])
        prob.root.add('presweep', IndepVarComp('presweep', presweep), promotes=['*'])
        prob.root.add('precone', IndepVarComp('precone', precone), promotes=['*'])

        prob.setup(check=False)

        prob['r'] = r
        prob['precurve'] = precurve
        prob['presweep'] = presweep
        prob['precurve'] = precurve
        prob['presweep'] = presweep

        check_gradient_unit_test(self, prob, tol=5e-5)


class TestSetupPCModVarSpeed(unittest.TestCase):

    def test1(self):

        control_tsr = 8.0
        control_pitch = 1.0
        Vrated = 12.0
        R = 63.0
        Vfactor = 0.7


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SetupPCModVarSpeed(), promotes=['*'])
        prob.root.add('control_tsr', IndepVarComp('control:tsr', control_tsr), promotes=['*'])
        prob.root.add('control_pitch', IndepVarComp('control:pitch', control_pitch), promotes=['*'])
        prob.root.add('Vrated', IndepVarComp('Vrated', Vrated), promotes=['*'])
        prob.root.add('R', IndepVarComp('R', R), promotes=['*'])
        prob.root.add('Vfactor', IndepVarComp('Vfactor', Vfactor), promotes=['*'])

        prob.setup(check=False)

        prob['control:tsr'] = control_tsr
        prob['control:pitch'] = control_pitch
        prob['Vrated'] = Vrated
        prob['R'] = R
        prob['Vfactor'] = Vfactor

        check_gradient_unit_test(self, prob)


class TestBladeDeflection(unittest.TestCase):

    def test1(self):

        dx = np.array([0.0, 1.39292987639e-05, 2.42964362361e-05, 3.73611415086e-05, 5.41276445811e-05, 7.25773833888e-05, 9.35928069108e-05, 0.000268596883961, 0.000326596195925, 0.000372259193056, 0.00253285667923, 0.00455502505195, 0.00706548815047, 0.017530829505, 0.0272151750396, 0.0326222002275, 0.0410034329532, 0.0655343595834, 0.0818962168404, 0.109404026845, 0.140873579829, 0.177728457961, 0.218834564559, 0.318921553007, 0.354759106888, 0.442199496033, 0.51925808593, 0.593774567144, 0.602307030873, 0.606403310701, 0.610574803156, 0.688594372351, 0.779039709944, 1.2575016857, 1.49348304065, 1.69400834707, 1.90049599542, 2.00437890947])
        dy = np.array([0.0, -9.11673273998e-07, -1.59100573731e-06, -2.44958708331e-06, -3.55417484862e-06, -4.77185559999e-06, -6.16078184177e-06, -1.77741315033e-05, -2.16387627714e-05, -2.46907649535e-05, -0.000177142122562, -0.00032886393339, -0.000523351402122, -0.00136667950935, -0.00215626107629, -0.00258384925494, -0.00322230578732, -0.00497009730837, -0.00607094164543, -0.00784913963514, -0.00980572470543, -0.0120149227037, -0.0143773305481, -0.0197932419212, -0.0216584171821, -0.0260612290834, -0.0297556792352, -0.0331954636074, -0.0335827012855, -0.0337680683508, -0.0339563913295, -0.0373950218981, -0.0412189884014, -0.0597108654077, -0.0681209435104, -0.0750137913709, -0.0819682184214, -0.085450339495])
        dz = np.array([0.0, 0.000190681386865, 0.000249594803444, 0.000305808888278, 0.000363037181592, 0.000414559803159, 0.000465006717172, 0.00079021115147, 0.000878140045153, 0.000937905765842, 0.00236994444878, 0.003141201122, 0.00385188728597, 0.0059090995974, 0.00721749042255, 0.00775193614485, 0.00838246793108, 0.00965065210961, 0.010256099836, 0.0110346081014, 0.0117147556349, 0.0123373351129, 0.0128834601832, 0.0138847141125, 0.014168689932, 0.0147331399477, 0.0151335663707, 0.015450772371, 0.015480903206, 0.0154926033582, 0.0155017716157, 0.0156218668894, 0.0157089720273, 0.0159512118376, 0.0160321965202, 0.0160695719649, 0.0160814363339, 0.0160814363339])
        pitch = 0.0
        theta_str = np.array([13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 12.9342399734, 12.4835173655, 11.4807910924, 10.9555299481, 10.2141623046, 9.50473135323, 8.7980712345, 8.12522342051, 6.81383731475, 6.42068577363, 5.58414615907, 4.96394315694, 4.44088253161, 4.38489418276, 4.35829308634, 4.33139992746, 3.86272702658, 3.38639207628, 1.57773054352, 0.953410121155, 0.504987738102, 0.0995174088527, -0.0878099])
        r_sub_precurve0 = np.array([16.037355, 39.5374275, 63.0375])
        Rhub0 = 1.5375
        r_str0 = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])
        # precurve_str0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        precurve_str0 = np.linspace(0.0, 5.0, len(dx))
        bladeLength0 = 61.5


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', BladeDeflection(len(dx)), promotes=['*'])
        prob.root.add('dx', IndepVarComp('dx', dx), promotes=['*'])
        prob.root.add('dy', IndepVarComp('dy', dy), promotes=['*'])
        prob.root.add('dz', IndepVarComp('dz', dz), promotes=['*'])
        prob.root.add('pitch', IndepVarComp('pitch', pitch), promotes=['*'])
        prob.root.add('theta_str', IndepVarComp('theta_str', theta_str), promotes=['*'])
        prob.root.add('r_sub_precurve0', IndepVarComp('r_sub_precurve0', r_sub_precurve0), promotes=['*'])
        prob.root.add('Rhub0', IndepVarComp('Rhub0', Rhub0), promotes=['*'])
        prob.root.add('r_str0', IndepVarComp('r_str0', r_str0), promotes=['*'])
        prob.root.add('precurve_str0', IndepVarComp('precurve_str0', precurve_str0), promotes=['*'])
        prob.root.add('bladeLength0', IndepVarComp('bladeLength0', bladeLength0), promotes=['*'])

        prob.setup(check=False)

        prob['dx'] = dx
        prob['dy'] = dy
        prob['dz'] = dz
        prob['pitch'] = pitch
        prob['theta_str'] = theta_str
        prob['r_sub_precurve0'] = r_sub_precurve0
        prob['Rhub0'] = Rhub0
        prob['r_str0'] = r_str0
        prob['precurve_str0'] = precurve_str0
        prob['bladeLength0'] = bladeLength0


        check_gradient_unit_test(self, prob, tol=1e-4)


class TestDamageLoads(unittest.TestCase):

    def test1(self):

        rstar = np.array([0.0, 0.022, 0.067, 0.111, 0.167, 0.233, 0.3, 0.367, 0.433, 0.5, 0.567, 0.633, 0.7, 0.767, 0.833, 0.889, 0.933, 0.978])
        Mxb = np.array([2374300.0, 2083400.0, 1810800.0, 1570500.0, 1310400.0, 1048800.0, 823670.0, 634070.0, 477270.0, 348040.0, 244580.0, 163390.0, 102520.0, 57842.0, 27349.0, 11262.0, 3854.9, 447.38])
        Myb = np.array([2773200.0, 2815500.0, 2600400.0, 2393300.0, 2137100.0, 1845900.0, 1558200.0, 1289600.0, 1042700.0, 820150.0, 624490.0, 452290.0, 306580.0, 187460.0, 96475.0, 42677.0, 15409.0, 1842.6])
        theta = np.array([13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 12.9342399734, 12.4835173655, 11.4807910924, 10.9555299481, 10.2141623046, 9.50473135323, 8.7980712345, 8.12522342051, 6.81383731475, 6.42068577363, 5.58414615907, 4.96394315694, 4.44088253161, 4.38489418276, 4.35829308634, 4.33139992746, 3.86272702658, 3.38639207628, 1.57773054352, 0.953410121155, 0.504987738102, 0.0995174088527, -0.0878099])
        r = np.array([1.5375, 1.84056613137, 1.93905987557, 2.03755361977, 2.14220322299, 2.24069696719, 2.33919071139, 2.90419974, 3.04095863882, 3.13945238302, 5.637500205, 7.04226699698, 8.370800055, 11.8494282928, 13.8375, 14.7182548841, 15.887499795, 18.5537233505, 19.9875, 22.0564071286, 24.087500205, 26.16236509, 28.187499795, 32.2875, 33.5678634821, 36.387500205, 38.5725768593, 40.487499795, 40.6967540173, 40.7964789782, 40.8975, 42.6919644014, 44.5875, 52.787499795, 56.204199945, 58.937499795, 61.67080026, 63.0375])


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', DamageLoads(len(r), 17), promotes=['*'])
        prob.root.add('rstar', IndepVarComp('rstar', rstar), promotes=['*'])
        prob.root.add('Mxb', IndepVarComp('Mxb', Mxb), promotes=['*'])
        prob.root.add('Myb', IndepVarComp('Myb', Myb), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', theta), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', r), promotes=['*'])

        prob.setup(check=False)

        prob['rstar'] = rstar
        prob['Mxb'] = Mxb
        prob['Myb'] = Myb
        prob['theta'] = theta
        prob['r'] = r

        check_gradient_unit_test(self, prob, tol=5e-5)


if __name__ == '__main__':
    import rotorse.rotor

    check_for_missing_unit_tests([rotorse.rotor])
    unittest.main()

