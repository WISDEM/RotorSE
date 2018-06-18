

import unittest
import numpy as np
from openmdao.api import IndepVarComp, Problem, Group

from test_rotor_aeropower_gradients import check_gradient_unit_test, check_for_missing_unit_tests
from rotorse.rotor_geometry import GeometrySpline, Location, TurbineClass
from rotorse import TURBINE_CLASS

class TestGeometrySpline(unittest.TestCase):

    def test1(self):

        idx_cylinder = 3
        r_max_chord = 0.22

        chord_sub = np.array([3.2612, 4.5709, 3.3178, 1.4621])
        theta_sub = np.array([13.2783, 7.46036, 2.89317, -0.0878099])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', GeometrySpline(), promotes=['*'])
        prob.root.add('idx_cylinder', IndepVarComp('idx_cylinder', 0), promotes=['*'])
        prob.root.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
        prob.root.add('chord_sub', IndepVarComp('chord_sub', np.zeros(len(chord_sub))), promotes=['*'])
        prob.root.add('theta_sub', IndepVarComp('theta_sub', np.zeros(len(theta_sub))), promotes=['*'])
        prob.setup(check=False)

        prob['idx_cylinder'] = idx_cylinder
        prob['r_max_chord'] = r_max_chord
        prob['chord_sub'] = chord_sub
        prob['theta_sub'] = theta_sub

        check_gradient_unit_test(self, prob, tol=5e-5)

class TestLocation(unittest.TestCase):
    def test1(self):

        hub_height = 90.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', Location(), promotes=['*'])
        prob.root.add('hub_height', IndepVarComp('hub_height', 0.), promotes=['*'])
        prob.setup(check=False)

        prob['hub_height'] = hub_height

        check_gradient_unit_test(self, prob)

class TestTurbineClass(unittest.TestCase):
    def test1(self):

        # turbine_class = 90.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', TurbineClass(), promotes=['*'])
        prob.root.add('turbine_class', IndepVarComp('turbine_class', val=TURBINE_CLASS['I'], pass_by_obj=True), promotes=['*'])
        prob.setup(check=False)

        prob['turbine_class'] = TURBINE_CLASS['I']

        check_gradient_unit_test(self, prob)


if __name__ == '__main__':
    import rotorse.rotor

    check_for_missing_unit_tests([rotorse.rotor_geometry])
    unittest.main()