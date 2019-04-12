import numpy as np
from openmdao.api import Component
from bladecostse import blade_cost_model


# Class to initiate the blade cost model
class blade_cost_model_mdao(Component):
    def __init__(self, NPTS, name='', options={}):
        super(blade_cost_model_mdao, self).__init__()

        self.NPTS = NPTS
        self.ref_name = name

        if options == {}:
            self.options = {}
            self.options['verbosity']        = False
            self.options['tex_table']        = False
            self.options['generate_plots']   = False
            self.options['show_plots']       = False
            self.options['show_warnings']    = False
            self.options['discrete']         = False
        else:
            self.options = options
        
        # These parameters will come from outside
        self.add_param('materials',     val=np.zeros(NPTS), desc='material properties of composite materials', pass_by_obj=True)
        self.add_param('upperCS',       val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_param('lowerCS',       val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_param('websCS',        val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)
        self.add_param('profile',       val=np.zeros(NPTS), desc='list of CompositeSection profiles', pass_by_obj=True)
        
        self.add_param('Rtip',          val=0.0,            units='m', desc='rotor radius')
        self.add_param('Rhub',          val=0.0,            units='m', desc='hub radius')
        self.add_param('bladeLength',   val=0.0,            units='m', desc='blade length')
        self.add_param('r_pts',         val=np.zeros(NPTS), units='m', desc='blade radial locations, expressed in the rotor system')
        self.add_param('chord',         val=np.zeros(NPTS), desc='Chord distribution')
        self.add_param('le_location',   val=np.zeros(NPTS), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')

        # outputs
        self.add_output('total_blade_cost', val=0.0, units='USD', desc='total blade cost')
        self.add_output('total_blade_mass', val=0.0, units='USD', desc='total blade cost')

    def solve_nonlinear(self, params, unknowns, resids):
        bcm             = blade_cost_model(options=self.options)
        bcm.name        = self.ref_name
        bcm.materials   = params['materials']
        bcm.upperCS     = params['upperCS']
        bcm.lowerCS     = params['lowerCS']
        bcm.websCS      = params['websCS']
        bcm.profile     = params['profile']
        bcm.chord       = params['chord']
                
        bcm.r           = (params['r_pts'] - params['Rhub'])/(params['Rtip'] - params['Rhub']) * float(params['bladeLength'])
        bcm.bladeLength = float(params['bladeLength'])
        
        bcm.le_location              = params['le_location']
        blade_cost, blade_mass       = bcm.execute_blade_cost_model()
        
        unknowns['total_blade_cost'] = blade_cost
        unknowns['total_blade_mass'] = blade_mass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        