import numpy as np
from openmdao.api import Component, Group, IndepVarComp
from akima import Akima, akima_interp_with_derivs
from rotorse import TURBINE_CLASS, r_aero, r_str
from ccblade.ccblade_component import CCBladeGeometry

naero = len(r_aero)
nstr = len(r_str)

class Location(Component):
    def __init__(self):
        super(Location, self).__init__()
        self.add_param('hub_height', val=0.0, units='m', desc='Tower top hub height')
        self.add_output('wind_zvec', val=np.zeros(1), units='m', desc='Tower top hub height as vector')
    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['wind_zvec'] = np.array([ params['hub_height'] ])
        
class TurbineClass(Component):
    def __init__(self):
        super(TurbineClass, self).__init__()
        # parameters
        self.add_param('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbine class', pass_by_obj=True)

        # outputs should be constant
        self.add_output('V_mean', shape=1, units='m/s', desc='IEC mean wind speed for Rayleigh distribution')
        self.add_output('V_extreme', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_output('V_extreme_full', shape=2, units='m/s', desc='IEC extreme wind speed at hub height')
        
	self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        self.turbine_class = params['turbine_class']

        if self.turbine_class == TURBINE_CLASS['I']:
            Vref = 50.0
        elif self.turbine_class == TURBINE_CLASS['II']:
            Vref = 42.5
        elif self.turbine_class == TURBINE_CLASS['III']:
            Vref = 37.5
        elif self.turbine_class == TURBINE_CLASS['IV']:
            Vref = 30.0

        unknowns['V_mean'] = 0.2*Vref
        unknowns['V_extreme'] = 1.4*Vref
        unknowns['V_extreme_full'][0] = 1.4*Vref # for extreme cases TODO: check if other way to do
        unknowns['V_extreme_full'][1] = 1.4*Vref



class GeometrySpline(Component):
    def __init__(self):
        super(GeometrySpline, self).__init__()
        # variables
        self.add_param('r_max_chord', shape=1, desc='location of max chord on unit radius')
        self.add_param('chord_sub', shape=4, units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
        self.add_param('theta_sub', shape=4, units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
        self.add_param('precurve_sub', shape=3, units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_param('bladeLength', shape=1, units='m', desc='blade length (if not precurved or swept) otherwise length of blade before curvature')
        self.add_param('sparT', shape=5, units='m', desc='thickness values of spar cap')
        self.add_param('teT', shape=5, units='m', desc='thickness values of trailing edge panels')

        # parameters
        self.add_param('idx_cylinder_aero', val=0, shape=1, desc='first idx in r_aero_unit of non-cylindrical section', pass_by_obj=True)  # constant twist inboard of here
        self.add_param('idx_cylinder_str', val=0, shape=1, desc='first idx in r_str_unit of non-cylindrical section', pass_by_obj=True)
        self.add_param('hubFraction', shape=1, desc='hub location as fraction of radius')

        # out
        self.add_output('Rhub', shape=1, units='m', desc='dimensional radius of hub')
        self.add_output('Rtip', shape=1, units='m', desc='dimensional radius of tip')
        self.add_output('r_aero', shape=naero, units='m', desc='dimensional aerodynamic grid')
        self.add_output('r_str', shape=nstr, units='m', desc='dimensional structural grid')
        self.add_output('max_chord', shape=1, units='m', desc='maximum chord length')
        self.add_output('chord_aero', shape=naero, units='m', desc='chord at airfoil locations')
        self.add_output('chord_str', shape=nstr, units='m', desc='chord at structural locations')
        self.add_output('theta_aero', shape=naero, units='deg', desc='twist at airfoil locations')
        self.add_output('theta_str', shape=nstr, units='deg', desc='twist at structural locations')
        self.add_output('precurve_aero', shape=naero, units='m', desc='precurve at airfoil locations')
        self.add_output('precurve_str', shape=nstr, units='m', desc='precurve at structural locations')
        self.add_output('presweep_str', shape=nstr, units='m', desc='presweep at structural locations')
        self.add_output('sparT_str', shape=nstr, units='m', desc='dimensional spar cap thickness distribution')
        self.add_output('teT_str', shape=nstr, units='m', desc='dimensional trailing-edge panel thickness distribution')
        self.add_output('r_sub_precurve', shape=3, desc='precurve locations (used internally)')

        self.add_output('diameter', shape=1, units='m')
        
        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_size'] = 1e-5

    def solve_nonlinear(self, params, unknowns, resids):

        Rhub = params['hubFraction'] * params['bladeLength']
        Rtip = Rhub + params['bladeLength']

        # setup chord parmeterization
        nc = len(params['chord_sub'])
        r_max_chord = Rhub + (Rtip-Rhub)*params['r_max_chord']
        rc = np.linspace(r_max_chord, Rtip, nc-1)
        rc = np.concatenate([[Rhub], rc])
        chord_spline = Akima(rc, params['chord_sub'])

        # setup theta parmeterization
        nt = len(params['theta_sub'])
        idxc_aero = params['idx_cylinder_aero']
        idxc_str = params['idx_cylinder_str']
        r_cylinder = Rhub + (Rtip-Rhub)*r_aero[idxc_aero]
        rt = np.linspace(r_cylinder, Rtip, nt)
        theta_spline = Akima(rt, params['theta_sub'])

        # setup precurve parmeterization
        precurve_spline = Akima(rc, np.concatenate([[0.0], params['precurve_sub']]))
        unknowns['r_sub_precurve'] = rc[1:]


        # make dimensional and evaluate splines
        unknowns['Rhub'] = Rhub
        unknowns['Rtip'] = Rtip
        unknowns['diameter'] = 2.0*Rhub
        unknowns['r_aero'] = Rhub + (Rtip-Rhub)*r_aero
        unknowns['r_str'] = Rhub + (Rtip-Rhub)*r_str
        unknowns['max_chord'], _, _, _ = chord_spline.interp(params['r_max_chord'])
        unknowns['chord_aero'], _, _, _ = chord_spline.interp(unknowns['r_aero'])
        unknowns['chord_str'], _, _, _ = chord_spline.interp(unknowns['r_str'])
        theta_outer_aero, _, _, _ = theta_spline.interp(unknowns['r_aero'][idxc_aero:])
        theta_outer_str, _, _, _ = theta_spline.interp(unknowns['r_str'][idxc_str:])
        unknowns['theta_aero'] = np.concatenate([theta_outer_aero[0]*np.ones(idxc_aero), theta_outer_aero])
        unknowns['theta_str'] = np.concatenate([theta_outer_str[0]*np.ones(idxc_str), theta_outer_str])
        unknowns['precurve_aero'], _, _, _ = precurve_spline.interp(unknowns['r_aero'])
        unknowns['precurve_str'], _, _, _ = precurve_spline.interp(unknowns['r_str'])
        unknowns['presweep_str'] = np.zeros_like(unknowns['precurve_str'])  # TODO: for now
        
        # setup sparT parameterization
        nt = len(params['sparT'])
        rt = np.linspace(0.0, Rtip, nt)
        sparT_spline = Akima(rt, params['sparT'])
        teT_spline = Akima(rt, params['teT'])

        unknowns['sparT_str'], _, _, _ = sparT_spline.interp(unknowns['r_str'])
        unknowns['teT_str'], _, _, _ = teT_spline.interp(unknowns['r_str'])


class RotorGeometry(Group):
    def __init__(self):
        super(RotorGeometry, self).__init__()
        """rotor model"""

        self.add('idx_cylinder_aero', IndepVarComp('idx_cylinder_aero', 0, pass_by_obj=True), promotes=['*'])
        self.add('idx_cylinder_str', IndepVarComp('idx_cylinder_str', 0, pass_by_obj=True), promotes=['*'])
        self.add('hubFraction', IndepVarComp('hubFraction', 0.0), promotes=['*'])
        self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
        self.add('chord_sub', IndepVarComp('chord_sub', np.zeros(4),units='m'), promotes=['*'])
        self.add('theta_sub', IndepVarComp('theta_sub', np.zeros(4), units='deg'), promotes=['*'])
        self.add('precurve_sub', IndepVarComp('precurve_sub', np.zeros(3), units='m'), promotes=['*'])
        self.add('bladeLength', IndepVarComp('bladeLength', 0.0, units='m'), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0, units='deg'), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0, units='deg'), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0, units='deg'), promotes=['*'])
        self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])
        self.add('turbine_class', IndepVarComp('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
        
        # --- composite sections ---
        self.add('sparT', IndepVarComp('sparT', val=np.zeros(5), units='m', desc='spar cap thickness parameters'), promotes=['*'])
        self.add('teT', IndepVarComp('teT', val=np.zeros(5), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])
        
        # --- Rotor Definition ---
        self.add('loc', Location(), promotes=['*'])
        self.add('turbineclass', TurbineClass())
        self.add('spline0', GeometrySpline())
        self.add('spline', GeometrySpline())
        self.add('geom', CCBladeGeometry())
        
        # connections to turbineclass
        self.connect('turbine_class', 'turbineclass.turbine_class')

        # connections to spline0
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
