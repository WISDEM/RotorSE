#!/usr/bin/env python
# encoding: utf-8
"""
test_rotoraero_gradients.py

Created by Andrew Ning on 2013-12-30.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from commonse.utilities import check_gradient_unit_test, check_for_missing_unit_tests
from rotorse.rotor_aeropower import SetupRunVarSpeed, RegulatedPowerCurve, AEP, CSMDrivetrain, OutputsAero
from ccblade.ccblade_component import CCBladePower, CCBladeLoads, CCBladeGeometry
from commonse.distribution import WeibullCDF, WeibullWithMeanCDF, RayleighCDF

from enum import Enum
from openmdao.api import IndepVarComp, Component, Problem, Group, SqliteRecorder, BaseRecorder
import os

from rotorse import DRIVETRAIN_TYPE

class TestSetupRunVarSpeed(unittest.TestCase):

    def test1(self):

        control_Vin = 3.0
        control_Vout = 25.0
        control_tsr = 7.55
        control_maxOmega = 12.0
        control_pitch = 0.0
        R = 62.9400379597

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SetupRunVarSpeed(20), promotes=['*'])
        prob.root.add('c_Vin', IndepVarComp('control:Vin', control_Vin), promotes=['*'])
        prob.root.add('c_Vout', IndepVarComp('control:Vout', control_Vout), promotes=['*'])
        prob.root.add('c_tsr', IndepVarComp('control:tsr', control_tsr), promotes=['*'])
        prob.root.add('c_maxOmega', IndepVarComp('control:maxOmega', control_maxOmega), promotes=['*'])
        prob.root.add('c_pitch', IndepVarComp('control:pitch', control_pitch), promotes=['*'])
        prob.root.add('R', IndepVarComp('R', 0.0), promotes=['*'])
        prob.root.add('rho', IndepVarComp('rho', 0.0), promotes=['*'])
        prob.setup(check=False)

        prob['control:Vin'] = control_Vin
        prob['control:Vout'] = control_Vout
        prob['control:tsr'] = control_tsr
        prob['control:maxOmega'] = control_maxOmega
        prob['R'] = R

        check_gradient_unit_test(self, prob)


# @unittest.skip("RegulatedPowerCurve")
class TestRegulatedPowerCurve(unittest.TestCase):

    def test1(self):
        control_Vin = 3.0
        control_Vout = 25.0
        control_ratedPower = 5e6
        control_tsr = 7.5
        control_pitch = 0.0
        control_maxOmega = 12.0
        Vcoarse = np.array([3.0, 4.15789473684, 5.31578947368, 6.47368421053, 7.63157894737, 8.78947368421, 9.94736842105, 11.1052631579, 12.2631578947, 13.4210526316, 14.5789473684, 15.7368421053, 16.8947368421, 18.0526315789, 19.2105263158, 20.3684210526, 21.5263157895, 22.6842105263, 23.8421052632, 25.0])
        Pcoarse = np.array([22025.3984542, 165773.30206, 416646.421046, 804477.093872, 1359097.659, 2110340.45488, 3088037.81999, 4297554.655, 5634839.03406, 7014620.69097, 8243314.7876, 9151515.42199, 9805082.34125, 10369558.6255, 10906183.6245, 11403861.5527, 11836495.4076, 12179487.9734, 12422864.4516, 12564556.3923])
        Tcoarse = np.array([52447.090344, 100745.549657, 164669.981102, 244220.384676, 339396.760382, 450199.108219, 576627.428186, 690511.171188, 779190.745114, 856112.80582, 912563.587402, 948902.381907, 976338.470922, 1006515.00082, 1040084.63496, 1074759.69165, 1108268.14897, 1138947.1794, 1166201.26418, 1190149.76946])
        Vrated = 12.0
        R = 62.9400379597

        prob = Problem()
        prob.root = Group()
        prob.root.add('powercurve', RegulatedPowerCurve(len(Vcoarse), 200))
        # prob.root.add('powercurve', RegulatedPowerCurveGroup(len(Vcoarse), 200))
        prob.root.add('c_Vin', IndepVarComp('control:Vin', control_Vin), promotes=['*'])
        prob.root.add('c_Vout', IndepVarComp('control:Vout', control_Vout), promotes=['*'])
        prob.root.add('c_ratedPower', IndepVarComp('control:ratedPower', control_ratedPower), promotes=['*'])
        prob.root.add('c_tsr', IndepVarComp('control:tsr', control_tsr), promotes=['*'])
        prob.root.add('c_pitch', IndepVarComp('control:pitch', control_pitch), promotes=['*'])
        prob.root.add('c_maxOmega', IndepVarComp('control:maxOmega', control_maxOmega), promotes=['*'])
        prob.root.add('Vcoarse', IndepVarComp('Vcoarse', np.zeros(len(Vcoarse))), promotes=['*'])
        prob.root.add('Pcoarse', IndepVarComp('Pcoarse', np.zeros(len(Pcoarse))), promotes=['*'])
        prob.root.add('Tcoarse', IndepVarComp('Tcoarse', np.zeros(len(Tcoarse))), promotes=['*'])
        prob.root.add('Vrated', IndepVarComp('Vrated', Vrated))
        prob.root.add('R', IndepVarComp('R', 0.0), promotes=['*'])

        prob.root.connect('control:Vin', 'powercurve.control:Vin')
        prob.root.connect('control:Vout', 'powercurve.control:Vout')
        prob.root.connect('control:maxOmega', 'powercurve.control:maxOmega')
        prob.root.connect('control:pitch', 'powercurve.control:pitch')
        prob.root.connect('control:ratedPower', 'powercurve.control:ratedPower')
        prob.root.connect('control:tsr', 'powercurve.control:tsr')
        prob.root.connect('Vcoarse', 'powercurve.Vcoarse')
        prob.root.connect('Pcoarse', 'powercurve.Pcoarse')
        prob.root.connect('Tcoarse', 'powercurve.Tcoarse')
        prob.root.connect('R', 'powercurve.R')

        prob.setup(check=False)

        prob['control:Vin'] = control_Vin
        prob['control:Vout'] = control_Vout
        prob['control:ratedPower'] = control_ratedPower
        prob['control:tsr'] = control_tsr
        prob['control:pitch'] = control_pitch
        prob['control:maxOmega'] = control_maxOmega
        prob['Vcoarse'] = Vcoarse
        prob['Pcoarse'] = Pcoarse
        prob['Tcoarse'] = Tcoarse
        prob['R'] = R

        check_gradient_unit_test(self, prob, tol=3e-5, display=False, comp=prob.root.powercurve)



class TestAEP(unittest.TestCase):

    def test1(self):

        CDF_V = np.array([0.178275041966, 0.190296788962, 0.202568028297, 0.215071993446, 0.227791807017, 0.240710518196, 0.253811139789, 0.267076684744, 0.280490202059, 0.294034811963, 0.307693740274, 0.32145035184, 0.335288182978, 0.349190972822, 0.363142693506, 0.377127579106, 0.391130153286, 0.405135255571, 0.419128066214, 0.43309412959, 0.447019376098, 0.460890142513, 0.474693190784, 0.488415725243, 0.502045408211, 0.515570374001, 0.528979241305, 0.54226112398, 0.555405640233, 0.568402920224, 0.581243612108, 0.593918886545, 0.606420439704, 0.618740494789, 0.630871802149, 0.642807637993, 0.654541801767, 0.666068612246, 0.67738290239, 0.68848001302, 0.699355785386, 0.710006552662, 0.720429130464, 0.730620806428, 0.740579328925, 0.750302894983, 0.759790137479, 0.76904011166, 0.778052281079, 0.786826502994, 0.795363013307, 0.803662411103, 0.811725642861, 0.819553986387, 0.827149034543, 0.834512678823, 0.841647092838, 0.848554715765, 0.855238235811, 0.861700573747, 0.867944866556, 0.873974451244, 0.879792848857, 0.885403748745, 0.890810993109, 0.896018561866, 0.90103055787, 0.90585119251, 0.91048477172, 0.914935682413, 0.919208379383, 0.923307372661, 0.927237215373, 0.931002492094, 0.934607807711, 0.93805777681, 0.941357013592, 0.944510122313, 0.947521688261, 0.950396269264, 0.95313838773, 0.955752523204, 0.95824310546, 0.960614508091, 0.962871042612, 0.965016953057, 0.967056411054, 0.968993511377, 0.970832267946, 0.972576610282, 0.974230380379, 0.975797329995, 0.977281118336, 0.978685310119, 0.980013374, 0.981268681344, 0.982454505324, 0.98357402033, 0.984630301666, 0.985626325531, 0.986564969246, 0.987449011729, 0.988281134189, 0.98906392103, 0.989799860938, 0.990491348148, 0.991140683869, 0.991750077849, 0.992321650072, 0.992857432566, 0.993359371317, 0.99382932827, 0.994269083401, 0.994680336859, 0.995064711163, 0.995423753433, 0.995758937661, 0.996071667003, 0.996363276084, 0.996635033314, 0.996888143197, 0.997123748634, 0.997342933215, 0.997546723483, 0.997736091178, 0.997911955448, 0.998075185027, 0.998226600365, 0.998366975728, 0.998497041241, 0.998617484891, 0.99872895447, 0.998832059475, 0.998927372946, 0.999015433252, 0.999096745819, 0.999171784802, 0.999240994698, 0.999304791902, 0.999363566204, 0.999417682229, 0.999467480822, 0.999513280372, 0.999555378083, 0.999594051191, 0.999629558123, 0.999662139611, 0.999692019741, 0.999719406969, 0.999744495073, 0.999767464065, 0.999788481055, 0.99980770107, 0.999825267834, 0.999841314496, 0.999855964334, 0.999869331401, 0.999881521151, 0.999892631024, 0.999902750988, 0.999911964064, 0.999920346811, 0.999927969778, 0.99993489794, 0.999941191092, 0.999946904234, 0.999952087915, 0.999956788569, 0.99996104882, 0.999964907768, 0.99996840126, 0.999971562138, 0.999974420473, 0.999977003779, 0.999979337216, 0.999981443776, 0.999983344458, 0.999985058426, 0.999986603162, 0.999987994599, 0.999989247253, 0.999990374339, 0.999991387879, 0.999992298805, 0.999993117053, 0.999993851641, 0.999994510758, 0.999995101829, 0.999995631585, 0.999996106123, 0.999996530963, 0.999996911098, 0.999997251045, 0.999997554883, 0.999997826298, 0.999998068616, 0.999998284835, 0.999998477661, 0.999998649529, 0.999998802632])
        P = np.array([22025.3984542, 31942.1185436, 42589.4002304, 53993.2085473, 66179.5085267, 79174.2652013, 93003.4436034, 107693.008766, 123268.925721, 139757.159501, 157183.67514, 175574.437668, 194955.41212, 215352.563527, 236791.856922, 259299.257337, 282900.729806, 307622.23936, 333489.751032, 360529.229855, 388766.640862, 418227.949084, 448939.119554, 480926.117306, 514214.907371, 548831.454781, 584801.724571, 622151.681771, 660907.291415, 701094.518535, 742739.328164, 785867.685334, 830505.555078, 876678.902429, 924413.692418, 973735.890078, 1024671.46044, 1077246.36854, 1131486.57941, 1187418.05809, 1245066.76959, 1304458.67896, 1365619.75124, 1428575.95144, 1493353.24461, 1559977.59577, 1628474.96996, 1698871.33222, 1771192.64757, 1845464.88105, 1921713.99769, 1999965.96251, 2080246.74057, 2162582.29688, 2246998.59648, 2333521.6044, 2422177.28568, 2512991.60535, 2605990.52843, 2701200.01997, 2798646.04499, 2898354.56853, 3000351.55562, 3104662.97129, 3211314.78058, 3320332.94852, 3431743.44013, 3545572.22046, 3661284.29229, 3777883.06101, 3895967.17288, 4015492.83297, 4136416.74759, 4258697.48598, 4382294.05017, 4507165.36302, 4631531.67845, 4757024.71876, 4883638.63072, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0])
        lossFactor = 0.95

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', AEP(len(P)), promotes=['*'])
        prob.root.add('CDF_V', IndepVarComp('CDF_V', np.zeros(len(CDF_V))), promotes=['*'])
        prob.root.add('P', IndepVarComp('P', np.zeros(len(P))), promotes=['*'])
        prob.root.add('lossFactor', IndepVarComp('lossFactor', 0.0), promotes=['*'])
        prob.root.comp.deriv_options['check_step_size'] = 1.
        prob.root.comp.deriv_options['check_form'] = 'central'
        prob.root.comp.deriv_options['check_step_calc'] = 'relative'    
        prob.setup(check=False)

        prob['CDF_V'] = CDF_V
        prob['P'] = P
        prob['lossFactor'] = lossFactor


        check_gradient_unit_test(self, prob, step_size=1, display=True)  # larger step size b.c. AEP is big value


class TestCCBladeGeometry(unittest.TestCase): # EMG: CCBladeGeometry moved from rotoraerodefaults to ccblade.ccblade_component

    def test1(self):

        Rtip = 63.0
        precone = 5.0
        precurveTip = 0.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CCBladeGeometry(), promotes=['*'])
        prob.root.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        prob.root.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        prob.root.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        prob.setup(check=False)

        prob['Rtip'] = Rtip
        prob['precone'] = precone
        prob['precurveTip'] = precurveTip

        check_gradient_unit_test(self, prob)


@unittest.skip("CCBlade test takes a long time")
class TestCCBlade(unittest.TestCase): # EMG: move this to CCBlade unit testing?

    def test1(self):

        r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
        chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
        theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
        Rhub = 1.5
        Rtip = 63.0
        hubHt = 80.0
        precone = 2.5
        tilt = -5.0
        yaw = 0.0
        B = 3
        rho = 1.225
        mu = 1.81206e-5
        shearExp = 0.2
        nSector = 4
        precurve = np.zeros(len(r))
        precurveTip = 0.0

        # airfoils
        basepath = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'), '5MW_AFFiles/')

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

        airfoil_files = af

        # run_case = 'power'
        Uhub = np.array([3.0, 4.15789473684, 5.31578947368, 6.47368421053, 7.63157894737, 8.78947368421, 9.94736842105, 11.1052631579, 12.2631578947, 13.4210526316, 14.5789473684, 15.7368421053, 16.8947368421, 18.0526315789, 19.2105263158, 20.3684210526, 21.5263157895, 22.6842105263, 23.8421052632, 25.0])
        Omega = np.array([3.43647024491, 4.76282718154, 6.08918411817, 7.41554105481, 8.74189799144, 10.0682549281, 11.3946118647, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
        pitch = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        n2 = len(Uhub)

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CCBladePower(n, n2), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', np.zeros(len(r))), promotes=['*'])
        prob.root.add('chord', IndepVarComp('chord', np.zeros(len(chord))), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', np.zeros(len(theta))), promotes=['*'])
        prob.root.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        prob.root.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        prob.root.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        prob.root.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        prob.root.add('precurve', IndepVarComp('precurve', precurve), promotes=['*'])
        prob.root.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        prob.root.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        prob.root.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        prob.root.add('B', IndepVarComp('B', 0), promotes=['*'])
        prob.root.add('rho', IndepVarComp('rho', 0.0), promotes=['*'])
        prob.root.add('mu', IndepVarComp('mu', 0.0), promotes=['*'])
        prob.root.add('shearExp', IndepVarComp('shearExp', 0.0, pass_by_obj=True), promotes=['*'])
        prob.root.add('nSector', IndepVarComp('nSector', nSector), promotes=['*'])
        prob.root.add('airfoil_files', IndepVarComp('airfoil_files', airfoil_files), promotes=['*'])
        prob.root.add('Uhub', IndepVarComp('Uhub', np.zeros(len(Uhub))), promotes=['*'])
        prob.root.add('Omega', IndepVarComp('Omega', np.zeros(len(Omega))), promotes=['*'])
        prob.root.add('pitch', IndepVarComp('pitch', np.zeros(len(pitch))), promotes=['*'])

        prob.root.comp.deriv_options['check_step_calc'] = 'relative' 
        prob.root.comp.deriv_options['check_form'] = 'central'
        prob.setup(check=False)

        prob['r'] = r
        prob['chord'] = chord
        prob['theta'] = theta
        prob['Rhub'] = Rhub
        prob['Rtip'] = Rtip
        prob['hubHt'] = hubHt
        prob['precone'] = precone
        prob['tilt'] = tilt
        prob['yaw'] = yaw
        prob['B'] = B
        prob['rho'] = rho
        prob['mu'] = mu
        prob['shearExp'] = shearExp
        prob['nSector'] = nSector
        prob['airfoil_files'] = airfoil_files
        prob['Uhub'] = Uhub
        prob['Omega'] = Omega
        prob['pitch'] = pitch
        prob['precurve'] = precurve
        prob['precurveTip'] = precurveTip

        check_gradient_unit_test(self, prob, tol=1e-3, display=True)

    def test2(self):

        r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
        chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
        theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
        Rhub = 1.5
        Rtip = 63.0
        hubHt = 80.0
        precone = 2.5
        tilt = -5.0
        yaw = 0.0
        B = 3
        rho = 1.225
        mu = 1.81206e-5
        shearExp = 0.2
        nSector = 4
        precurve = np.zeros(len(r))

        # airfoils
        basepath = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'), '5MW_AFFiles/')

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

        airfoil_files = af

        # run_case = 'loads'
        V_load = 12.0
        Omega_load = 10.0
        pitch_load = 0.0
        azimuth_load = 180.0
        n2 = 1

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CCBladeLoads(n, n2), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', np.zeros(len(r))), promotes=['*'])
        prob.root.add('chord', IndepVarComp('chord', np.zeros(len(chord))), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', np.zeros(len(theta))), promotes=['*'])
        prob.root.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        prob.root.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        prob.root.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        prob.root.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        prob.root.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        prob.root.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        prob.root.add('B', IndepVarComp('B', 0), promotes=['*'])
        prob.root.add('rho', IndepVarComp('rho', 0.0), promotes=['*'])
        prob.root.add('mu', IndepVarComp('mu', 0.0), promotes=['*'])
        prob.root.add('shearExp', IndepVarComp('shearExp', 0.0, pass_by_obj=True), promotes=['*'])
        prob.root.add('precurve', IndepVarComp('precurve', precurve), promotes=['*'])
        prob.root.add('nSector', IndepVarComp('nSector', nSector), promotes=['*'])
        prob.root.add('airfoil_files', IndepVarComp('airfoil_files', airfoil_files), promotes=['*'])
        prob.root.add('V_load', IndepVarComp('V_load', 0.0), promotes=['*'])
        prob.root.add('Omega_load', IndepVarComp('Omega_load', 0.0), promotes=['*'])
        prob.root.add('pitch_load', IndepVarComp('pitch_load', 0.0), promotes=['*'])
        prob.root.add('azimuth_load', IndepVarComp('azimuth_load', 0.0), promotes=['*'])

        prob.root.comp.deriv_options['check_form'] = 'central'
        prob.root.comp.deriv_options['check_step_calc'] = 'relative' 
        prob.setup(check=False)

        prob['r'] = r
        prob['chord'] = chord
        prob['theta'] = theta
        prob['Rhub'] = Rhub
        prob['Rtip'] = Rtip
        prob['hubHt'] = hubHt
        prob['precone'] = precone
        prob['tilt'] = tilt
        prob['yaw'] = yaw
        prob['B'] = B
        prob['rho'] = rho
        prob['mu'] = mu
        prob['shearExp'] = shearExp
        prob['nSector'] = nSector
        prob['airfoil_files'] = airfoil_files
        prob['V_load'] = V_load
        prob['Omega_load'] = Omega_load
        prob['pitch_load'] = pitch_load
        prob['azimuth_load'] = azimuth_load
        prob['precurve'] = precurve

        check_gradient_unit_test(self, prob, tol=1e-3, display=True)



class TestCSMDrivetrain(unittest.TestCase):

    def test1(self):

        aeroPower = np.array([94518.9621316, 251637.667571, 525845.9078, 949750.895039, 1555959.84151, 2377079.95943, 3445718.46102, 4767739.26659, 6246980.6699, 7776655.12685, 9138828.60234, 10145691.7573, 10870259.0229, 11496057.2782, 12090978.6194, 12642721.8686, 13122354.9713, 13502608.5891, 13772424.3097, 13929508.977])
        ratedPower = 5000000.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CSMDrivetrain(len(aeroPower)), promotes=['*'])
        prob.root.add('aeroPower', IndepVarComp('aeroPower', np.zeros(len(aeroPower))), promotes=['*'])
        prob.root.add('ratedPower', IndepVarComp('ratedPower', 0.0), promotes=['*'])
        prob.root.add('drivetrainType', IndepVarComp('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True), promotes=['*'])
        # prob.root.add('drivetrainType', IndepVarComp('drivetrainType', Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True), promotes=['*']) # EMG: changed to reflect Enum used in Aeropower
        prob.root.comp.deriv_options['check_form'] = 'central'
        prob.root.comp.deriv_options['check_step_calc'] = 'relative'    
        prob.setup(check=False)

        prob['aeroPower'] = aeroPower
        prob['ratedPower'] = ratedPower
        prob['drivetrainType'] = DRIVETRAIN_TYPE['GEARED']
        # prob['drivetrainType'] = 'geared'

        check_gradient_unit_test(self, prob, tol=6e-4)


    def test2(self):

        aeroPower = np.linspace(0.0, 10.0, 50)
        ratedPower = 5.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CSMDrivetrain(len(aeroPower)), promotes=['*'])
        prob.root.add('aeroPower', IndepVarComp('aeroPower', np.zeros(len(aeroPower))), promotes=['*'])
        prob.root.add('ratedPower', IndepVarComp('ratedPower', 0.0), promotes=['*'])
        prob.root.add('drivetrainType', IndepVarComp('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True), promotes=['*'])
        # prob.root.add('drivetrainType', IndepVarComp('drivetrainType', Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True), promotes=['*'])
        prob.setup(check=False)

        prob['aeroPower'] = aeroPower
        prob['ratedPower'] = ratedPower
        prob['drivetrainType'] = DRIVETRAIN_TYPE['GEARED']
        # prob['drivetrainType'] = 'geared'

        check_gradient_unit_test(self, prob)


    def test3(self):

        aeroPower = np.linspace(-10.0, 10.0, 50)
        ratedPower = 5.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CSMDrivetrain(len(aeroPower)), promotes=['*'])
        prob.root.add('aeroPower', IndepVarComp('aeroPower', np.zeros(len(aeroPower))), promotes=['*'])
        prob.root.add('ratedPower', IndepVarComp('ratedPower', 0.0), promotes=['*'])
        prob.root.add('drivetrainType', IndepVarComp('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True), promotes=['*'])
        # prob.root.add('drivetrainType', IndepVarComp('drivetrainType', Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True), promotes=['*'])
        prob.setup(check=False)

        prob['aeroPower'] = aeroPower
        prob['ratedPower'] = ratedPower
        prob['drivetrainType'] = DRIVETRAIN_TYPE['GEARED']
        # prob['drivetrainType'] = 'geared'

        check_gradient_unit_test(self, prob)


    def test4(self):

        aeroPower = np.linspace(-10.0, 30.0, 50)
        ratedPower = 5.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CSMDrivetrain(len(aeroPower)), promotes=['*'])
        prob.root.add('aeroPower', IndepVarComp('aeroPower', np.zeros(len(aeroPower))), promotes=['*'])
        prob.root.add('ratedPower', IndepVarComp('ratedPower', 0.0), promotes=['*'])
        prob.root.add('drivetrainType', IndepVarComp('drivetrainType', val=DRIVETRAIN_TYPE['GEARED'], pass_by_obj=True), promotes=['*'])
        # prob.root.add('drivetrainType', IndepVarComp('drivetrainType', Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True), promotes=['*'])
        prob.setup(check=False)

        prob['aeroPower'] = aeroPower
        prob['ratedPower'] = ratedPower
        prob['drivetrainType'] = DRIVETRAIN_TYPE['GEARED']
        # prob['drivetrainType'] = 'geared'

        check_gradient_unit_test(self, prob)


class TestOutputsAero(unittest.TestCase):

    def test1(self):

        AEP = 23490703.70603071
        V = np.array([0.178275041966, 0.190296788962, 0.202568028297, 0.215071993446, 0.227791807017, 0.240710518196, 0.253811139789, 0.267076684744, 0.280490202059, 0.294034811963, 0.307693740274, 0.32145035184, 0.335288182978, 0.349190972822, 0.363142693506, 0.377127579106, 0.391130153286, 0.405135255571, 0.419128066214, 0.43309412959, 0.447019376098, 0.460890142513, 0.474693190784, 0.488415725243, 0.502045408211, 0.515570374001, 0.528979241305, 0.54226112398, 0.555405640233, 0.568402920224, 0.581243612108, 0.593918886545, 0.606420439704, 0.618740494789, 0.630871802149, 0.642807637993, 0.654541801767, 0.666068612246, 0.67738290239, 0.68848001302, 0.699355785386, 0.710006552662, 0.720429130464, 0.730620806428, 0.740579328925, 0.750302894983, 0.759790137479, 0.76904011166, 0.778052281079, 0.786826502994, 0.795363013307, 0.803662411103, 0.811725642861, 0.819553986387, 0.827149034543, 0.834512678823, 0.841647092838, 0.848554715765, 0.855238235811, 0.861700573747, 0.867944866556, 0.873974451244, 0.879792848857, 0.885403748745, 0.890810993109, 0.896018561866, 0.90103055787, 0.90585119251, 0.91048477172, 0.914935682413, 0.919208379383, 0.923307372661, 0.927237215373, 0.931002492094, 0.934607807711, 0.93805777681, 0.941357013592, 0.944510122313, 0.947521688261, 0.950396269264, 0.95313838773, 0.955752523204, 0.95824310546, 0.960614508091, 0.962871042612, 0.965016953057, 0.967056411054, 0.968993511377, 0.970832267946, 0.972576610282, 0.974230380379, 0.975797329995, 0.977281118336, 0.978685310119, 0.980013374, 0.981268681344, 0.982454505324, 0.98357402033, 0.984630301666, 0.985626325531, 0.986564969246, 0.987449011729, 0.988281134189, 0.98906392103, 0.989799860938, 0.990491348148, 0.991140683869, 0.991750077849, 0.992321650072, 0.992857432566, 0.993359371317, 0.99382932827, 0.994269083401, 0.994680336859, 0.995064711163, 0.995423753433, 0.995758937661, 0.996071667003, 0.996363276084, 0.996635033314, 0.996888143197, 0.997123748634, 0.997342933215, 0.997546723483, 0.997736091178, 0.997911955448, 0.998075185027, 0.998226600365, 0.998366975728, 0.998497041241, 0.998617484891, 0.99872895447, 0.998832059475, 0.998927372946, 0.999015433252, 0.999096745819, 0.999171784802, 0.999240994698, 0.999304791902, 0.999363566204, 0.999417682229, 0.999467480822, 0.999513280372, 0.999555378083, 0.999594051191, 0.999629558123, 0.999662139611, 0.999692019741, 0.999719406969, 0.999744495073, 0.999767464065, 0.999788481055, 0.99980770107, 0.999825267834, 0.999841314496, 0.999855964334, 0.999869331401, 0.999881521151, 0.999892631024, 0.999902750988, 0.999911964064, 0.999920346811, 0.999927969778, 0.99993489794, 0.999941191092, 0.999946904234, 0.999952087915, 0.999956788569, 0.99996104882, 0.999964907768, 0.99996840126, 0.999971562138, 0.999974420473, 0.999977003779, 0.999979337216, 0.999981443776, 0.999983344458, 0.999985058426, 0.999986603162, 0.999987994599, 0.999989247253, 0.999990374339, 0.999991387879, 0.999992298805, 0.999993117053, 0.999993851641, 0.999994510758, 0.999995101829, 0.999995631585, 0.999996106123, 0.999996530963, 0.999996911098, 0.999997251045, 0.999997554883, 0.999997826298, 0.999998068616, 0.999998284835, 0.999998477661, 0.999998649529, 0.999998802632])
        P = np.array([22025.3984542, 31942.1185436, 42589.4002304, 53993.2085473, 66179.5085267, 79174.2652013, 93003.4436034, 107693.008766, 123268.925721, 139757.159501, 157183.67514, 175574.437668, 194955.41212, 215352.563527, 236791.856922, 259299.257337, 282900.729806, 307622.23936, 333489.751032, 360529.229855, 388766.640862, 418227.949084, 448939.119554, 480926.117306, 514214.907371, 548831.454781, 584801.724571, 622151.681771, 660907.291415, 701094.518535, 742739.328164, 785867.685334, 830505.555078, 876678.902429, 924413.692418, 973735.890078, 1024671.46044, 1077246.36854, 1131486.57941, 1187418.05809, 1245066.76959, 1304458.67896, 1365619.75124, 1428575.95144, 1493353.24461, 1559977.59577, 1628474.96996, 1698871.33222, 1771192.64757, 1845464.88105, 1921713.99769, 1999965.96251, 2080246.74057, 2162582.29688, 2246998.59648, 2333521.6044, 2422177.28568, 2512991.60535, 2605990.52843, 2701200.01997, 2798646.04499, 2898354.56853, 3000351.55562, 3104662.97129, 3211314.78058, 3320332.94852, 3431743.44013, 3545572.22046, 3661284.29229, 3777883.06101, 3895967.17288, 4015492.83297, 4136416.74759, 4258697.48598, 4382294.05017, 4507165.36302, 4631531.67845, 4757024.71876, 4883638.63072, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0, 5000000.0])
        rated_V = 11.738606532616528
        rated_Omega = 12.
        rated_pitch = 0.0
        rated_T = 714206.5321080858
        rated_Q = 3978873.5772973835
        hub_diameter = 3.075
        diameter = 125.95500453593273
        max_chord = 3.0459289459935825
        V_extreme = 70.0
        T_extreme = 0.0
        Q_extreme = 0.0
        Rtip = 63.0375
        precurveTip = 0.0
        precsweepTip = 0.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', OutputsAero(len(P)), promotes=['*'])
        prob.root.add('AEP_in', IndepVarComp('AEP_in', 0.0), promotes=['*'])
        prob.root.add('V_in', IndepVarComp('V_in', np.zeros(len(V))), promotes=['*'])
        prob.root.add('P_in', IndepVarComp('P_in', np.zeros(len(P))), promotes=['*'])
        prob.root.add('r_V_in', IndepVarComp('ratedConditions:V_in', 0.0), promotes=['*'])
        prob.root.add('r_Omega_in', IndepVarComp('ratedConditions:Omega_in', 0.0), promotes=['*'])
        prob.root.add('r_pitch_in', IndepVarComp('ratedConditions:pitch_in', 0.0), promotes=['*'])
        prob.root.add('r_T_in', IndepVarComp('ratedConditions:T_in', 0.0), promotes=['*'])
        prob.root.add('r_Q_in', IndepVarComp('ratedConditions:Q_in', 0.0), promotes=['*'])
        prob.root.add('hub_diameter_in', IndepVarComp('hub_diameter_in', 0.0), promotes=['*'])
        prob.root.add('diameter_in', IndepVarComp('diameter_in', 0.0), promotes=['*'])
        prob.root.add('max_chord_in', IndepVarComp('max_chord_in', 0.0), promotes=['*'])
        prob.root.add('V_extreme_in', IndepVarComp('V_extreme_in', 0.0), promotes=['*'])
        prob.root.add('T_extreme_in', IndepVarComp('T_extreme_in', 0.0), promotes=['*'])
        prob.root.add('Q_extreme_in', IndepVarComp('Q_extreme_in', 0.0), promotes=['*'])
        prob.root.add('Rtip_in', IndepVarComp('Rtip_in', 0.0), promotes=['*'])
        prob.root.add('precurveTip_in', IndepVarComp('precurveTip_in', 0.0), promotes=['*'])
        prob.root.add('presweepTip_in', IndepVarComp('precsweepTip_in', 0.0), promotes=['*'])
        prob.root.comp.deriv_options['check_step_calc'] = 'relative' 
        prob.setup(check=False)

        prob['AEP_in'] = AEP
        prob['V_in'] = V
        prob['P_in'] = P
        prob['ratedConditions:V_in'] = rated_V
        prob['ratedConditions:Omega_in'] = rated_Omega
        prob['ratedConditions:pitch_in'] = rated_pitch
        prob['ratedConditions:T_in'] = rated_T
        prob['ratedConditions:Q_in'] = rated_Q
        prob['hub_diameter_in'] = hub_diameter
        prob['diameter_in'] = diameter
        prob['max_chord_in'] = max_chord
        prob['V_extreme_in'] = V_extreme
        prob['T_extreme_in'] = T_extreme
        prob['Q_extreme_in'] = Q_extreme
        prob['Rtip_in'] = Rtip
        prob['precurveTip_in'] = precurveTip
        prob['precsweepTip_in'] = precsweepTip

        check_gradient_unit_test(self, prob)



class TestWeibullCDF(unittest.TestCase): # EMG: moved to commonse.distribution

    def test1(self):

        A = 5.0
        k = 2.2
        x = np.linspace(1.0, 15.0, 50)
        n = len(x)

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', WeibullCDF(n), promotes=['*'])
        prob.root.add('A', IndepVarComp('A', A), promotes=['*'])
        prob.root.add('k', IndepVarComp('k', k), promotes=['*'])
        prob.root.add('x', IndepVarComp('x', x), promotes=['*'])
        prob.root.comp.deriv_options['check_step_calc'] = 'relative' 
        prob.root.comp.deriv_options['check_form'] = 'central'
        prob.setup(check=False)

        prob['A'] = A
        prob['k'] = k
        prob['x'] = x

        check_gradient_unit_test(self, prob)

class TestWeibullWithMeanCDF(unittest.TestCase): # EMG: moved to commonse.distribution

    def test1(self):

        xbar = 8.0
        k = 2.2
        x = np.linspace(1.0, 15.0, 50)
        n = len(x)

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', WeibullWithMeanCDF(n), promotes=['*'])
        prob.root.add('xbar', IndepVarComp('xbar', xbar), promotes=['*'])
        prob.root.add('k', IndepVarComp('k', k), promotes=['*'])
        prob.root.add('x', IndepVarComp('x', x), promotes=['*'])
        prob.root.comp.deriv_options['check_step_calc'] = 'relative' 
        prob.root.comp.deriv_options['check_form'] = 'central'
        prob.setup(check=False)

        prob['xbar'] = xbar
        prob['k'] = k
        prob['x'] = x

        check_gradient_unit_test(self, prob)



class TestRayleighCDF(unittest.TestCase): # EMG: moved to commonse.distribution

    def test1(self):

        xbar = 6.0
        x = np.array([3.0, 3.08805770406, 3.17611540812, 3.26417311218, 3.35223081624, 3.4402885203, 3.52834622436, 3.61640392842, 3.70446163248, 3.79251933654, 3.8805770406, 3.96863474466, 4.05669244872, 4.14475015278, 4.23280785684, 4.3208655609, 4.40892326496, 4.49698096902, 4.58503867308, 4.67309637714, 4.7611540812, 4.84921178526, 4.93726948932, 5.02532719338, 5.11338489744, 5.2014426015, 5.28950030556, 5.37755800962, 5.46561571369, 5.55367341775, 5.64173112181, 5.72978882587, 5.81784652993, 5.90590423399, 5.99396193805, 6.08201964211, 6.17007734617, 6.25813505023, 6.34619275429, 6.43425045835, 6.52230816241, 6.61036586647, 6.69842357053, 6.78648127459, 6.87453897865, 6.96259668271, 7.05065438677, 7.13871209083, 7.22676979489, 7.31482749895, 7.40288520301, 7.49094290707, 7.57900061113, 7.66705831519, 7.75511601925, 7.84317372331, 7.93123142737, 8.01928913143, 8.10734683549, 8.19540453955, 8.28346224361, 8.37151994767, 8.45957765173, 8.54763535579, 8.63569305985, 8.72375076391, 8.81180846797, 8.89986617203, 8.98792387609, 9.07598158015, 9.16403928421, 9.25209698827, 9.34015469233, 9.42821239639, 9.51627010045, 9.60432780451, 9.69238550857, 9.78044321263, 9.86850091669, 9.95655862075, 10.0446163248, 10.1326740289, 10.2207317329, 10.308789437, 10.3968471411, 10.4849048451, 10.5729625492, 10.6610202532, 10.7490779573, 10.8371356614, 10.9251933654, 11.0132510695, 11.1013087735, 11.1893664776, 11.2774241817, 11.3654818857, 11.4535395898, 11.5415972938, 11.6296549979, 11.717712702, 11.8505355749, 11.9833584479, 12.1161813209, 12.2490041939, 12.3818270669, 12.5146499398, 12.6474728128, 12.7802956858, 12.9131185588, 13.0459414318, 13.1787643047, 13.3115871777, 13.4444100507, 13.5772329237, 13.7100557967, 13.8428786696, 13.9757015426, 14.1085244156, 14.2413472886, 14.3741701616, 14.5069930345, 14.6398159075, 14.7726387805, 14.9054616535, 15.0382845265, 15.1711073994, 15.3039302724, 15.4367531454, 15.5695760184, 15.7023988914, 15.8352217644, 15.9680446373, 16.1008675103, 16.2336903833, 16.3665132563, 16.4993361293, 16.6321590022, 16.7649818752, 16.8978047482, 17.0306276212, 17.1634504942, 17.2962733671, 17.4290962401, 17.5619191131, 17.6947419861, 17.8275648591, 17.960387732, 18.093210605, 18.226033478, 18.358856351, 18.491679224, 18.6245020969, 18.7573249699, 18.8901478429, 19.0229707159, 19.1557935889, 19.2886164618, 19.4214393348, 19.5542622078, 19.6870850808, 19.8199079538, 19.9527308267, 20.0855536997, 20.2183765727, 20.3511994457, 20.4840223187, 20.6168451916, 20.7496680646, 20.8824909376, 21.0153138106, 21.1481366836, 21.2809595565, 21.4137824295, 21.5466053025, 21.6794281755, 21.8122510485, 21.9450739215, 22.0778967944, 22.2107196674, 22.3435425404, 22.4763654134, 22.6091882864, 22.7420111593, 22.8748340323, 23.0076569053, 23.1404797783, 23.2733026513, 23.4061255242, 23.5389483972, 23.6717712702, 23.8045941432, 23.9374170162, 24.0702398891, 24.2030627621, 24.3358856351, 24.4687085081, 24.6015313811, 24.734354254, 24.867177127, 25.0])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', RayleighCDF(len(x)), promotes=['*'])
        prob.root.add('xbar', IndepVarComp('xbar', 0.0), promotes=['*'])
        prob.root.add('x', IndepVarComp('x', np.zeros(len(x))), promotes=['*'])
        prob.root.comp.deriv_options['check_step_calc'] = 'relative' 
        prob.root.comp.deriv_options['check_form'] = 'central'
        prob.setup(check=False)

        prob['xbar'] = xbar
        prob['x'] = x

        check_gradient_unit_test(self, prob, tol=2e-6)

def check_for_missing_unit_tests(modules):
    """A heuristic check to find components that don't have a corresonding unit test
    for its gradients.
    Parameters
    ----------
    modules : list(str)
        a list of modules to check for missin ggradients
    """
    import sys
    import inspect
    import ast

    thisfilemembers = inspect.getmembers(sys.modules['__main__'], lambda member: inspect.isclass(member) and member.__module__ == '__main__')
    tests = [name for name, classname in thisfilemembers]

    totest = []
    tomod = []
    reserved = ['Assembly', 'Slot', 'ImplicitComponent']
    for mod in modules:
        modulemembers = inspect.getmembers(mod, inspect.isclass)
        fname = mod.__file__
        if fname[-1] == 'c':
            fname = fname[:-1]
        f = open(fname, 'r')
        p = ast.parse(f.read())
        f.close()
        nonimported = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
        for name, classname in modulemembers:
            bases = classname.__bases__
            if 'Base' not in name and name not in reserved and name in nonimported and len(bases) > 0:
                base = bases[0].__name__
                if base == 'Component' or 'Base' in base:
                    totest.append(name)
                    tomod.append(mod.__name__)

    for mod, test in zip(tomod, totest):
        if 'Test'+test not in tests:
            print '!!! There does not appear to be a unit test for:', mod + '.' + test


def check_gradient_unit_test(unittest, prob, fd='central', step_size=1e-6, tol=1e-6, display=False,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6, comp=None):
    """compare provided analytic gradients to finite-difference gradients with unit testing.
    Same as check_gradient, but provides a unit test for each gradient for convenience.
    the unit tests checks that the error for each gradient is less than tol.
    Parameters
    ----------
    comp : obj
        An OpenMDAO component that provides analytic gradients through provideJ()
    fd : str
        the type of finite difference to use.  options are central or forward
    step_size : float
        step size to use in finite differencing
    tol : float
        tolerance for how close the gradients should agree to
    display : boolean
        if True, display results for each gradient
    show_missing_warnings: boolean
        if True, warn for gradients that were not provided
        (they may be ones that are unecessary, but may be ones that were accidentally skipped)
    show_scaling_warnings: boolean
        if True, warn for gradients that are either very small or very large which may lead
        to challenges in solving the full linear system
    min_grad/max_grad : float
        quantifies what "very small" or "very large" means when using show_scaling_warnings
    """

    J_fd, J_fwd, J_rev = check_gradient(prob, fd, step_size, tol, display, show_missing_warnings,
        show_scaling_warnings, min_grad, max_grad)
    if comp == None:
        comp = prob.root.comp
    if "list_deriv_vars" in dir(comp): #  callable(getattr(comp, 'list_deriv_vars')):
        inputs, outputs = comp.list_deriv_vars()
        for output in outputs:
            for input in inputs:
                J = J_fwd[output, input]
                JFD = J_fd[output, input]
                m, n = J.shape
                # print '---------', output, input, '---------'
                # print 'J', J
                # print 'JFD', JFD
                for i in range(m):
                    for j in range(n):
                        if np.abs(J[i, j]) <= tol:
                            errortype = 'absolute'
                            error = J[i, j] - JFD[i, j]
                        else:
                            errortype = 'relative'
                            error = 1.0 - JFD[i, j]/J[i, j]
                        error = np.abs(error)

                        # # display
                        # if error > tol:
                        #     star = ' ***** '
                        # else:
                        #     star = ''
                        #
                        # if display:
                        #     output = '{}{:<20} ({}) {}: ({}, {})'.format(star, error, errortype, name, J[i, j], JFD[i, j])
                        #     print output
                        #
                        # if show_scaling_warnings and J[i, j] != 0 and np.abs(J[i, j]) < min_grad:
                        #     print '*** Warning: The following analytic gradient is very small and may need to be scaled:'
                        #     print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
                        #
                        # if show_scaling_warnings and np.abs(J[i, j]) > max_grad:
                        #     print '*** Warning: The following analytic gradient is very large and may need to be scaled:'
                        #     print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
                        #

                        try:
                            unittest.assertLessEqual(error, tol)
                        except AssertionError, e:
                            print '*** error in:', "\n\tOutput: ", output, "\n\tInput: ", input, "\n\tPosition: ", i, j
                            print JFD[i, j], J[i, j]
                            raise e
    else:
        for key, value in J_fd.iteritems():
                J = J_fwd[key]
                JFD = J_fd[key]
                m, n = J.shape
                for i in range(m):
                    for j in range(n):
                        if np.abs(J[i, j]) <= tol:
                            errortype = 'absolute'
                            error = J[i, j] - JFD[i, j]
                        else:
                            errortype = 'relative'
                            error = 1.0 - JFD[i, j]/J[i, j]
                        error = np.abs(error)
                        try:
                            unittest.assertLessEqual(error, tol)
                        except AssertionError, e:
                            print '*** error in:', "\n\tKey: ", key, "\n\tPosition: ", i, j
                            raise e


def check_gradient(prob, fd='central', step_size=1e-6, tol=1e-6, display=False,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6):
    """compare provided analytic gradients to finite-difference gradients
    Parameters
    ----------
    comp : obj
        An OpenMDAO component that provides analytic gradients through provideJ()
    fd : str
        the type of finite difference to use.  options are central or forward
    step_size : float
        step size to use in finite differencing
    tol : float
        tolerance for how close the gradients should agree to
    display : boolean
        if True, display results for each gradient
    show_missing_warnings: boolean
        if True, warn for gradients that were not provided
        (they may be ones that are unecessary, but may be ones that were accidentally skipped)
    show_scaling_warnings: boolean
        if True, warn for gradients that are either very small or very large which may lead
        to challenges in solving the full linear system
    min_grad/max_grad : float
        quantifies what "very small" or "very large" means when using show_scaling_warnings
    Returns
    -------
    names : array(str)
        list of the names of all the gradients
    errorvec : array(float)
        list of all the errors for the gradients.  If the magnitude of the gradient is less than
        tol, then an absolute error is used, otherwise a relative error is used.
    """
    # inputs = comp.list_deriv_vars
    # inputs, outputs = comp.list_deriv_vars()

    # show_missing_warnings = False

    # if show_missing_warnings:
    #     all_inputs = _explodeall(comp, vtype='inputs')
    #     all_outputs = _explodeall(comp, vtype='outputs')
    #     reserved_inputs = ['missing_deriv_policy', 'directory', 'force_fd', 'force_execute', 'eval_only']
    #     reserved_outputs = ['derivative_exec_count', 'itername', 'exec_count']
    #     potential_missed_inputs = list(set(all_inputs) - set(reserved_inputs) - set(inputs))
    #     potential_missed_outputs = list(set(all_outputs) - set(reserved_outputs) - set(outputs))
    #
    #     if len(potential_missed_inputs) > 0 or len(potential_missed_outputs) > 0:
    #         print
    #         print '*** Warning: ' + comp.__class__.__name__ + ' does not supply derivatives for the following'
    #         print '\tinputs:', potential_missed_inputs
    #         print '\toutputs:', potential_missed_outputs
    #         print

    # prob = Problem()
    # prob.root = Group()
    # prob.root.add('comp', comp, promotes=['*'])
    # prob.setup()
    #
    # for i in range(len(inputs)):
    #     prob[inputs[i]] = comp

    prob.run()
    root = prob.root

    # Linearize the model
    root._sys_linearize(root.params, root.unknowns, root.resids)

    data = {}

    # Derivatives should just be checked without parallel adjoint for now.
    voi = None

    jac_fwd = {}
    jac_rev = {}
    jac_fd = {}

    # Check derivative calculations for all comps at every level of the
    # system hierarchy.
    for comp in root.components(recurse=True):
        cname = comp.pathname

        # No need to check comps that don't have any derivs.
        if comp.deriv_options['type'] == 'fd':
            continue

        # IndepVarComps are just clutter too.
        if isinstance(comp, IndepVarComp):
            continue

        data[cname] = {}
        jac_fwd = {}
        jac_rev = {}
        jac_fd = {}

        # try:
        #     params, unknowns = comp.list_deriv_vars()
        # except:
        #     pass
        params = comp.params
        unknowns = comp.unknowns
        resids = comp.resids
        dparams = comp.dpmat[voi]
        dunknowns = comp.dumat[voi]
        dresids = comp.drmat[voi]

        # Skip if all of our inputs are unconnected.
        # if len(dparams) == 0:
        #     continue

        # if out_stream is not None:
        #     out_stream.write('-'*(len(cname)+15) + '\n')
        #     out_stream.write("Component: '%s'\n" % cname)
        #     out_stream.write('-'*(len(cname)+15) + '\n')

        states = comp.states

        param_list = [item for item in dparams if not \
                      dparams.metadata(item).get('pass_by_obj')]
        param_list.extend(states)

        # Create all our keys and allocate Jacs
        for p_name in param_list:

            dinputs = dunknowns if p_name in states else dparams
            p_size = np.size(dinputs[p_name])

            # Check dimensions of user-supplied Jacobian
            for u_name in unknowns:

                u_size = np.size(dunknowns[u_name])
                if comp._jacobian_cache:

                    # We can perform some additional helpful checks.
                    if (u_name, p_name) in comp._jacobian_cache:

                        user = comp._jacobian_cache[(u_name, p_name)].shape

                        # User may use floats for scalar jacobians
                        if len(user) < 2:
                            user = (user[0], 1)

                        if user[0] != u_size or user[1] != p_size:
                            msg = "derivative in component '{}' of '{}' wrt '{}' is the wrong size. " + \
                                  "It should be {}, but got {}"
                            msg = msg.format(cname, u_name, p_name, (u_size, p_size), user)
                            raise ValueError(msg)

                jac_fwd[(u_name, p_name)] = np.zeros((u_size, p_size))
                jac_rev[(u_name, p_name)] = np.zeros((u_size, p_size))

        # Reverse derivatives first
        for u_name in dresids:
            u_size = np.size(dunknowns[u_name])

            # Send columns of identity
            for idx in range(u_size):
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0

                dresids._dat[u_name].val[idx] = 1.0
                try:
                    comp.apply_linear(params, unknowns, dparams,
                                      dunknowns, dresids, 'rev')
                finally:
                    dparams._apply_unit_derivatives()

                for p_name in param_list:

                    dinputs = dunknowns if p_name in states else dparams
                    # try:
                    jac_rev[(u_name, p_name)][idx, :] = dinputs._dat[p_name].val
                    # except:
                    #     pass
        # Forward derivatives second
        for p_name in param_list:

            dinputs = dunknowns if p_name in states else dparams
            p_size = np.size(dinputs[p_name])

            # Send columns of identity
            for idx in range(p_size):
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0

                dinputs._dat[p_name].val[idx] = 1.0
                dparams._apply_unit_derivatives()
                comp.apply_linear(params, unknowns, dparams,
                                  dunknowns, dresids, 'fwd')

                for u_name, u_val in dresids.vec_val_iter():
                    jac_fwd[(u_name, p_name)][:, idx] = u_val

        # Finite Difference goes last
        dresids.vec[:] = 0.0
        root.clear_dparams()
        dunknowns.vec[:] = 0.0

        # Component can request to use complex step.
        if comp.deriv_options['form'] == 'complex_step':
            fd_func = comp.complex_step_jacobian
        else:
            fd_func = comp.fd_jacobian
        jac_fd = fd_func(params, unknowns, resids, option_overrides=comp.deriv_options) #EMG: derv_options were not being passed

        # # Assemble and Return all metrics.
        # _assemble_deriv_data(chain(dparams, states), resids, data[cname],
        #                      jac_fwd, jac_rev, jac_fd, out_stream,
        #                      c_name=cname)

    return jac_fd, jac_fwd, jac_rev



if __name__ == '__main__':
    # import rotorse.rotoraero
    # import rotorse.rotoraerodefaults
    import rotorse.rotor_aeropower

    # check_for_missing_unit_tests([rotorse.rotoraero, rotorse.rotoraerodefaults])
    check_for_missing_unit_tests([rotorse.rotor_aeropower])

    unittest.main()

    # from unittest import TestSuite
    # blah = TestSuite()
    # blah.addTest(TestRegulatedPowerCurve('test1'))
    # unittest.TextTestRunner().run(blah)
