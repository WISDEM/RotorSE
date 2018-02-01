import numpy as np
from openmdao.api import Component
from akima import Akima, akima_interp_with_derivs
from rotorse import TURBINE_CLASS



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

        
class GridSetup(Component):
    def __init__(self, naero, nstr):
        super(GridSetup, self).__init__()

        """preprocessing step.  inputs and outputs should not change during optimization"""

        # should be constant
        self.add_param('initial_aero_grid', shape=naero, desc='initial aerodynamic grid on unit radius')
        self.add_param('initial_str_grid', shape=nstr, desc='initial structural grid on unit radius')

        # outputs are also constant during optimization
        self.add_output('fraction', shape=nstr, desc='fractional location of structural grid on aero grid')
        self.add_output('idxj', val=np.zeros(nstr, dtype=np.int), desc='index of augmented aero grid corresponding to structural index', pass_by_obj=True)

        self.naero = naero
        self.nstr = nstr
        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        r_aero = params['initial_aero_grid']
        r_str = params['initial_str_grid']
        r_aug = np.concatenate([[0.0], r_aero, [1.0]])

        nstr = len(r_str)
        naug = len(r_aug)

        # find idx in augmented aero array that brackets the structural index
        # then find the fraction the structural value is between the two bounding indices
        unknowns['fraction'] = np.zeros(nstr)
        unknowns['idxj'] = np.zeros(nstr, dtype=np.int)

        for i in range(nstr):
            for j in range(1, naug):
                if r_aug[j] >= r_str[i]:
                    j -= 1
                    break
            unknowns['idxj'][i] = j
            unknowns['fraction'][i] = (r_str[i] - r_aug[j]) / (r_aug[j+1] - r_aug[j])


class RGrid(Component):
    def __init__(self, naero, nstr):
        super(RGrid, self).__init__()
        # variables
        self.add_param('r_aero', shape=naero, desc='new aerodynamic grid on unit radius')

        # parameters
        self.add_param('fraction', shape=nstr, desc='fractional location of structural grid on aero grid')
        self.add_param('idxj', shape=nstr, dtype=np.int, desc='index of augmented aero grid corresponding to structural index')

        # outputs
        self.add_output('r_str', shape=nstr, desc='corresponding structural grid corresponding to new aerodynamic grid')
	
	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        r_aug = np.concatenate([[0.0], params['r_aero'], [1.0]])

        nstr = len(params['fraction'])
        unknowns['r_str'] = np.zeros(nstr)
        for i in range(nstr):
            j = params['idxj'][i]
            unknowns['r_str'][i] = r_aug[j] + params['fraction'][i]*(r_aug[j+1] - r_aug[j])


    def list_deriv_vars(self):

        inputs = ('r_aero',)
        outputs = ('r_str', )

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        J = {}
        nstr = len(params['fraction'])
        naero = len(params['r_aero'])
        J_sub = np.zeros((nstr, naero))

        for i in range(nstr):
            j = params['idxj'][i]
            if j > 0 and j < naero+1:
                J_sub[i, j-1] = 1 - params['fraction'][i]
            if j > -1 and j < naero:
                J_sub[i, j] = params['fraction'][i]
        J['r_str', 'r_aero'] = J_sub

        return J



class GeometrySpline(Component):
    def __init__(self, naero, nstr):
        super(GeometrySpline, self).__init__()
        # variables
        self.add_param('r_aero_unit', shape=naero, desc='locations where airfoils are defined on unit radius')
        self.add_param('r_str_unit', shape=nstr, desc='locations where airfoils are defined on unit radius')
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
        r_cylinder = Rhub + (Rtip-Rhub)*params['r_aero_unit'][idxc_aero]
        rt = np.linspace(r_cylinder, Rtip, nt)
        theta_spline = Akima(rt, params['theta_sub'])

        # setup precurve parmeterization
        precurve_spline = Akima(rc, np.concatenate([[0.0], params['precurve_sub']]))
        unknowns['r_sub_precurve'] = rc[1:]


        # make dimensional and evaluate splines
        unknowns['Rhub'] = Rhub
        unknowns['Rtip'] = Rtip
        unknowns['diameter'] = 2.0*Rhub
        unknowns['r_aero'] = Rhub + (Rtip-Rhub)*params['r_aero_unit']
        unknowns['r_str'] = Rhub + (Rtip-Rhub)*params['r_str_unit']
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

	# below is not generalized and was for a specific study
        # nt = len(self.sparT) - 1
        # rt = np.linspace(r_cylinder, Rtip, nt)
        # sparT_spline = Akima(rt, self.sparT[1:])

        # self.sparT_cylinder = np.array([0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05457, 0.03859, 0.03812, 0.03906, 0.04799, 0.05363, 0.05833])
        # self.teT_cylinder = np.array([0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05457, 0.03859, 0.05765, 0.05765, 0.04731, 0.04167])

        # sparT_str_in = self.sparT[0]*self.sparT_cylinder
        # sparT_str_out, _, _, _ = sparT_spline.interp(self.r_str[idxc_str:])
        # self.sparT_str = np.concatenate([sparT_str_in, [sparT_str_out[0]], sparT_str_out])

        # # trailing edge thickness
        # teT_str_in = self.teT[0]*self.teT_cylinder
        # self.teT_str = np.concatenate((teT_str_in, self.teT[1]*np.ones(9), self.teT[2]*np.ones(7), self.teT[3]*np.ones(8), self.teT[4]*np.ones(2)))

    # def linearize(self, params, unknowns, resids):
    #     J = {}
    #     naero = len(self.r_aero_unit)
    #     nstr = len(self.r_str_unit)
    #     ncs = len(self.chord_sub)
    #     nts = len(self.theta_sub)
    #     nst = len(self.sparT)
    #     ntt = len(self.teT)
    #
    #     n = naero + nstr + ncs + nts + nst + ntt + 2
    #
    #     dRtip = np.zeros(n)
    #     dRhub = np.zeros(n)
    #     dRtip[naero + nstr + 1 + ncs + nts] = 1.0
    #     dRhub[naero + nstr + 1 + ncs + nts] = self.hubFraction
    #
    #     draero = np.zeros((naero, n))
    #     draero[:, naero + nstr + 1 + ncs + nts] = (1.0 - self.r_aero_unit)*self.hubFraction + self.r_aero_unit
    #     draero[:, :naero] = Rtip-Rhub
    #
    #     drstr = np.zeros((nstr, n))
    #     drstr[:, naero + nstr + 1 + ncs + nts] = (1.0 - self.r_str_unit)*self.hubFraction + self.r_str_unit
    #     drstr[:, naero:nstr] = Rtip-Rhub
    #     TODO: do with Tapenade
    #     return J
'''
class GeometrySpline(Component):
    def __init__(self, naero):
        super(GeometrySpline, self).__init__()
        self.add_param('r_af', shape=naero, units='m', desc='locations where airfoils are defined on unit radius')

        self.add_param('idx_cylinder', val=0, desc='location where cylinder section ends on unit radius')
        self.add_param('r_max_chord', shape=1, desc='position of max chord on unit radius')

        self.add_param('Rhub', shape=1, units='m', desc='blade hub radius')
        self.add_param('Rtip', shape=1, units='m', desc='blade tip radius')

        self.add_param('chord_sub', shape=4, units='m', desc='chord at control points')
        self.add_param('theta_sub', shape=4, units='deg', desc='twist at control points')

        self.add_output('r', shape=naero, units='m', desc='chord at airfoil locations')
        self.add_output('chord', shape=naero, units='m', desc='chord at airfoil locations')
        self.add_output('theta', shape=naero, units='deg', desc='twist at airfoil locations')
        self.add_output('precurve', shape=naero, units='m', desc='precurve at airfoil locations')
        # self.add_output('r_af_spacing', shape=16)  # deprecated: not used anymore

        
    def solve_nonlinear(self, params, unknowns, resids):

        chord_sub = params['chord_sub']
        theta_sub = params['theta_sub']
        r_max_chord = params['r_max_chord']

        nc = len(chord_sub)
        nt = len(theta_sub)
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        idxc = params['idx_cylinder']
        r_max_chord = Rhub + (Rtip-Rhub)*r_max_chord
        r_af = params['r_af']
        r_cylinder = Rhub + (Rtip-Rhub)*r_af[idxc]

        # chord parameterization
        rc_outer, drc_drcmax, drc_drtip = linspace_with_deriv(r_max_chord, Rtip, nc-1)
        r_chord = np.concatenate([[Rhub], rc_outer])
        drc_drcmax = np.concatenate([[0.0], drc_drcmax])
        drc_drtip = np.concatenate([[0.0], drc_drtip])
        drc_drhub = np.concatenate([[1.0], np.zeros(nc-1)])

        # theta parameterization
        r_theta, drt_drcyl, drt_drtip = linspace_with_deriv(r_cylinder, Rtip, nt)

        # spline
        chord_spline = Akima(r_chord, chord_sub)
        theta_spline = Akima(r_theta, theta_sub)

        r = Rhub + (Rtip-Rhub)*r_af
        unknowns['r'] = r
        chord, dchord_dr, dchord_drchord, dchord_dchordsub = chord_spline.interp(r)
        theta_outer, dthetaouter_dr, dthetaouter_drtheta, dthetaouter_dthetasub = theta_spline.interp(r[idxc:])
        unknowns['chord'] = chord

        theta_inner = theta_outer[0] * np.ones(idxc)
        unknowns['theta'] = np.concatenate([theta_inner, theta_outer])

        # unknowns['r_af_spacing'] = np.diff(r_af)

        unknowns['precurve'] = np.zeros_like(unknowns['chord'])  # TODO: for now I'm forcing this to zero, just for backwards compatibility

        # gradients (TODO: rethink these a bit or use Tapenade.)
        n = len(r_af)
        dr_draf = (Rtip-Rhub)*np.ones(n)
        dr_dRhub = 1.0 - r_af
        dr_dRtip = r_af
        # dr = hstack([np.diag(dr_draf), np.zeros((n, 1)), dr_dRhub, dr_dRtip, np.zeros((n, nc+nt))])

        dchord_draf = dchord_dr * dr_draf
        dchord_drmaxchord0 = np.dot(dchord_drchord, drc_drcmax)
        dchord_drmaxchord = dchord_drmaxchord0 * (Rtip-Rhub)
        dchord_drhub = np.dot(dchord_drchord, drc_drhub) + dchord_drmaxchord0*(1.0 - params['r_max_chord']) + dchord_dr*dr_dRhub
        dchord_drtip = np.dot(dchord_drchord, drc_drtip) + dchord_drmaxchord0*(params['r_max_chord']) + dchord_dr*dr_dRtip

        dthetaouter_dcyl = np.dot(dthetaouter_drtheta, drt_drcyl)
        dthetaouter_draf = dthetaouter_dr*dr_draf[idxc:]
        dthetaouter_drhub = dthetaouter_dr*dr_dRhub[idxc:]
        dthetaouter_drtip = dthetaouter_dr*dr_dRtip[idxc:] + np.dot(dthetaouter_drtheta, drt_drtip)

        dtheta_draf = np.concatenate([np.zeros(idxc), dthetaouter_draf])
        dtheta_drhub = np.concatenate([dthetaouter_drhub[0]*np.ones(idxc), dthetaouter_drhub])
        dtheta_drtip = np.concatenate([dthetaouter_drtip[0]*np.ones(idxc), dthetaouter_drtip])
        sub = dthetaouter_dthetasub[0, :]
        dtheta_dthetasub = vstack([np.dot(np.ones((idxc, 1)), sub[np.newaxis, :]), dthetaouter_dthetasub])

        dtheta_draf = np.diag(dtheta_draf)
        dtheta_dcyl = np.concatenate([dthetaouter_dcyl[0]*np.ones(idxc), dthetaouter_dcyl])
        dtheta_draf[idxc:, idxc] += dthetaouter_dcyl*(Rtip-Rhub)
        dtheta_drhub += dtheta_dcyl*(1.0 - r_af[idxc])
        dtheta_drtip += dtheta_dcyl*r_af[idxc]

        drafs_dr = np.zeros((n-1, n))
        for i in range(n-1):
            drafs_dr[i, i] = -1.0
            drafs_dr[i, i+1] = 1.0

        J = {}
        J['r', 'r_af'] = np.diag(dr_draf)
        J['r', 'Rhub'] = dr_dRhub
        J['r', 'Rtip'] = dr_dRtip
        J['chord', 'r_af'] = np.diag(dchord_draf)
        J['chord', 'r_max_chord'] = dchord_drmaxchord
        J['chord', 'Rhub'] = dchord_drhub
        J['chord', 'Rtip'] = dchord_drtip
        J['chord', 'chord_sub'] =dchord_dchordsub
        J['theta', 'r_af'] = dtheta_draf
        J['theta', 'Rhub'] =dtheta_drhub
        J['theta', 'Rtip'] =dtheta_drtip
        J['theta', 'theta_sub'] =dtheta_dthetasub
        # J['r_af_spacing', 'r_af'] = drafs_dr

        self.J = J


    def list_deriv_vars(self):

        inputs = ('r_af', 'r_max_chord', 'Rhub', 'Rtip', 'chord_sub', 'theta_sub')
        outputs = ('r', 'chord', 'theta', 'precurve')

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        return self.J
'''
