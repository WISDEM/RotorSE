
import os, sys, copy, time
import csv
import operator

from ruamel import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, interp1d
import numpy as np
import jsonschema as json

# WISDEM
from openmdao.api import Component, Group, IndepVarComp, Problem
import commonse
from akima import Akima, akima_interp_with_derivs
from ccblade.ccblade_component import CCBladeGeometry
from ccblade import CCAirfoil
from airfoilprep.airfoilprep import Airfoil, Polar
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp, PreCompWriter

from rotorse.geometry_tools.geometry import AirfoilShape, Curve


TURBULENCE_CLASS = commonse.enum.Enum('A B C')
TURBINE_CLASS = commonse.enum.Enum('I II III')
DRIVETRAIN_TYPE = commonse.enum.Enum('geared single_stage multi_drive pm_direct_drive')

def remap2grid(x_ref, y_ref, x, spline=PchipInterpolator):
    spline_y = spline(x_ref, y_ref)

    # error handling for x[-1] - x_ref[-1] > 0 and x[-1]~x_ref[-1]
    try:
        _ = iter(x)
        if x[-1]>max(x_ref) and np.isclose(x[-1], x_ref[-1]):
            x[-1]=x_ref[-1]
    except:
        if x>max(x_ref) and np.isclose(x, x_ref[-1]):
            x=x_ref[-1]

    return spline_y(x)

def arc_length(x, y):
    npts = len(x)
    arc = np.array([0.]*npts)
    for k in range(1, npts):
        arc[k] = arc[k-1] + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)

    return arc

class ReferenceBlade(object):
    def __init__(self):

        # Validate input file against JSON schema
        self.validate        = False       # (bool) run IEA turbine ontology JSON validation
        self.fname_schema    = ''          # IEA turbine ontology JSON schema file

        # Grid sizes
        self.NINPUT          = 5
        self.NPTS            = 50
        self.NPTS_AfProfile  = 200
        self.NPTS_AfPolar    = 100

        # 
        self.analysis_level  = 0           # 0: Precomp, (1: FAST/ElastoDyn, 2: FAST/BeamDyn)
        self.verbose         = False

        # Precomp analyis
        self.spar_var        = ''          # name of composite layer for RotorSE spar cap buckling analysis
        self.te_var          = ''          # name of composite layer for RotorSE trailing edge buckling analysis


        

    def initialize(self, fname):
        if self.verbose:
            t0 = time.time()
            print 'Running initialization: %s' % fname

        # Load input
        self.fname_input = fname
        self.wt_ref = self.load_ontology(self.fname_input, validate=self.validate, fname_schema=self.fname_schema)

        # Renaming and converting lists to dicts for simplicity
        blade_ref = copy.deepcopy(self.wt_ref['components']['blade'])
        af_ref    = {}
        for afi in self.wt_ref['airfoils']:
            af_ref[afi['name']] = afi

        if self.verbose:
            print 'Complete: Load Input File: \t%f s'%(time.time()-t0)
            t1 = time.time()

        # build blade
        blade = {}
        blade = self.set_configuration(blade, self.wt_ref)
        blade = self.remap_composites(blade, blade_ref)
        blade = self.remap_planform(blade, blade_ref, af_ref)
        blade = self.remap_profiles(blade, blade_ref, af_ref)
        blade = self.remap_polars(blade, blade_ref, af_ref)
        blade = self.calc_composite_bounds(blade)
        blade = self.calc_control_points(blade)
        
        blade['analysis_level'] = self.analysis_level

        if self.verbose:
            print 'Complete: Geometry Analysis: \t%f s'%(time.time()-t1)
            
        # Conversion
        if self.analysis_level == 0:
            t2 = time.time()
            blade = self.convert_precomp(blade, self.wt_ref['materials'])
            if self.verbose:
                print 'Complete: Precomp Conversion: \t%f s'%(time.time()-t2)
        elif self.analysis_level == 1:
            # sonata/ anba

            # meshing with sonata

            # 
            pass

        return blade

    def update(self, blade):
        # 
        t1 = time.time()
        blade = self.update_planform(blade)
        blade = self.calc_composite_bounds(blade)

        if self.verbose:
            print 'Complete: Geometry Update: \t%f s'%(time.time()-t1)

        # Conversion
        if self.analysis_level == 0:
            t2 = time.time()
            blade = self.convert_precomp(blade)
            if self.verbose:
                print 'Complete: Precomp Conversion: \t%f s'%(time.time()-t2)


        return blade

    def load_ontology(self, fname_input, validate=False, fname_schema=''):
        # """ Load inputs IEA turbine ontology yaml inputs, optional validation """
        # # Read IEA turbine ontology yaml input file
        # with open(fname_input, 'r') as myfile:
        #     inputs = myfile.read()

        # # Validate the turbine input with the IEA turbine ontology schema
        # if validate:
        #     with open(fname_schema, 'r') as myfile:
        #         schema = myfile.read()
        #     json.validate(yaml.load(inputs), yaml.load(schema))

        # return yaml.load(inputs)
        with open(fname_input, 'r') as myfile:
            inputs = myfile.read()
        yaml = YAML()
        return yaml.load(inputs)

    def write_ontology(self, fname, blade, wt_out):

        ### this works for dictionaries, but not what ever ordered dictionary nonsenes is coming out of ruamel
        # def format_dict_for_yaml(out):
        # # recursively loop through output dictionary, convert numpy objects to base python
        #     def get_dict(vartree, branch):
        #         return reduce(operator.getitem, branch, vartree)
        #     def loop_dict(vartree_full, vartree, branch):
        #         for var in vartree.keys():
        #             branch_i = copy.copy(branch)
        #             branch_i.append(var)
        #             if type(vartree[var]) in [dict, CommentedMap]:
        #                 loop_dict(vartree_full, vartree[var], branch_i)
        #             else:
        #                 if type(get_dict(vartree_full, branch_i[:-1])[branch_i[-1]]) is np.ndarray:
        #                     get_dict(vartree_full, branch_i[:-1])[branch_i[-1]] = get_dict(vartree_full, branch_i[:-1])[branch_i[-1]].tolist()
        #                 elif type(get_dict(vartree_full, branch_i[:-1])[branch_i[-1]]) is np.float64:
        #                     get_dict(vartree_full, branch_i[:-1])[branch_i[-1]] = float(get_dict(vartree_full, branch_i[:-1])[branch_i[-1]])
        #                 elif type(get_dict(vartree_full, branch_i[:-1])[branch_i[-1]]) in [tuple, list, CommentedSeq]:
        #                     get_dict(vartree_full, branch_i[:-1])[branch_i[-1]] = [loop_dict(obji, obji, []) for obji in get_dict(vartree_full, branch_i[:-1])[branch_i[-1]] if type(obji) in [dict, CommentedMap]]


        #     loop_dict(out, out, [])
        #     return out

        # dict_out = format_dict_for_yaml(dict_out)


        #### Build Output dictionary

        # Composite layups
        st = blade['st']
        for idx_sec, sec in enumerate(st['sections']):
            for var in st['sections'][idx_sec].keys():
                try:
                    _ = st['sections'][idx_sec][var].keys()
                    st['sections'][idx_sec][var]['grid'] = [r for val, r in zip(st['sections'][idx_sec][var]['values'], st['sections'][idx_sec][var]['grid']) if val != None]
                    st['sections'][idx_sec][var]['values'] = [val for val in st['sections'][idx_sec][var]['values'] if val != None]
                except:
                    pass
        wt_out['components']['blade']['2d_fem'] = st


        f = open(fname, "w")
        yaml=YAML()
        yaml.default_flow_style = None
        yaml.width = float("inf")
        yaml.indent(mapping=4, sequence=6, offset=3)
        yaml.dump(wt_out, f)

    def calc_spanwise_grid(self, st):

        ### Spanwise grid
        # Finds spanwise composite start and end points, creates a linear distribution with the addition of the explist starts and ends
        r_points = []
        for idx_sec, sec in enumerate(st['sections']):
            for var in sec.keys():
                if type(sec[var]) not in [str, bool]:
                    if 'grid' in sec[var].keys():
                        if len(sec[var]['grid']) > 0.:

                            # remove approximate duplicates
                            r0 = sec[var]['grid'][0]
                            r1 = sec[var]['grid'][-1]

                            if r0 != 0:
                                r0_close = np.isclose(r0,r_points)
                                if len(r0_close)>0 and any(r0_close):
                                    st['sections'][idx_sec][var]['grid'][0] = r_points[np.argmax(r0_close)]
                                else:
                                    r_points.append(r0)

                            if r1 != 1:
                                r1_close = np.isclose(r1,r_points)
                                if any(r1_close):
                                    st['sections'][idx_sec][var]['grid'][-1] = r_points[np.argmax(r1_close)]
                                else:
                                    r_points.append(r1)

        self.s = list(sorted(set(np.linspace(0,1,num=self.NPTS-len(r_points)).tolist() + r_points)))

        # error handling for 1 or more composite section start/end point falling on the linspace grid
        if self.NPTS - len(self.s) != 0:
            add_s = np.linspace(self.s[-2],1,num=self.NPTS-len(self.s)+2).tolist()[1:-1]
            self.s = list(sorted(set(np.linspace(0,1,num=self.NPTS-len(r_points)).tolist() + r_points + add_s)))

        return st

    
    def set_configuration(self, blade, wt_ref):

        blade['config'] = {}

        for var in wt_ref['assembly']['global']:
            blade['config'][var] = wt_ref['assembly']['global'][var]
        for var in wt_ref['assembly']['control']:
            blade['config'][var] = wt_ref['assembly']['control'][var]

        return blade

    def remap_planform(self, blade, blade_ref, af_ref):

        blade['pf'] = {}

        blade['pf']['s']        = self.s
        blade['pf']['chord']    = remap2grid(blade_ref['bem_aero']['chord']['grid'], blade_ref['bem_aero']['chord']['values'], self.s)
        blade['pf']['theta']    = remap2grid(blade_ref['bem_aero']['twist']['grid'], blade_ref['bem_aero']['twist']['values'], self.s)
        blade['pf']['p_le']     = remap2grid(blade_ref['bem_aero']['pitch_axis']['grid'], blade_ref['bem_aero']['pitch_axis']['values'], self.s)
        blade['pf']['r']        = remap2grid(blade_ref['bem_aero']['coordinates']['x']['grid'], blade_ref['bem_aero']['coordinates']['x']['values'], self.s)
        blade['pf']['precurve'] = remap2grid(blade_ref['bem_aero']['coordinates']['y']['grid'], blade_ref['bem_aero']['coordinates']['y']['values'], self.s)
        blade['pf']['presweep'] = remap2grid(blade_ref['bem_aero']['coordinates']['z']['grid'], blade_ref['bem_aero']['coordinates']['z']['values'], self.s)

        thk_ref = [af_ref[af]['relative_thickness'] for af in blade_ref['bem_aero']['airfoil_position']['labels']]
        blade['pf']['rthick']   = remap2grid(blade_ref['bem_aero']['airfoil_position']['grid'], thk_ref, self.s)
        return blade

    def remap_profiles(self, blade, blade_ref, AFref, spline=PchipInterpolator):

        # Get airfoil thicknesses in decending order and cooresponding airfoil names
        AFref_thk = [AFref[af]['relative_thickness'] for af in blade_ref['bem_aero']['airfoil_position']['labels']]

        af_thk_dict = {}
        for afi in blade_ref['bem_aero']['airfoil_position']['labels']:
            afi_thk = AFref[afi]['relative_thickness']
            if afi_thk not in af_thk_dict.keys():
                af_thk_dict[afi_thk] = afi

        af_thk = sorted(af_thk_dict.keys())
        af_labels = [af_thk_dict[afi] for afi in af_thk]
        
        # Build array of reference airfoil coordinates, remapped
        AFref_n  = len(af_labels)
        AFref_xy = np.zeros((self.NPTS_AfProfile, 2, AFref_n))

        for afi, af_label in enumerate(af_labels):
            points = np.column_stack((AFref[af_label]['coordinates']['x'], AFref[af_label]['coordinates']['y']))
            af = AirfoilShape(points=points)
            af.redistribute(self.NPTS_AfProfile, dLE=True)

            af_points = af.points
            af_points[:,0] -= af.LE[0]
            af_points[:,1] -= af.LE[1]
            c = max(af_points[:,0])-min(af_points[:,0])
            af_points[:,:] /= c

            AFref_xy[:,:,afi] = af_points

        # Spanwise thickness interpolation
        profile_spline = spline(af_thk, AFref_xy, axis=2)
        blade['profile'] = profile_spline(blade['pf']['rthick'])
        blade['profile_spline'] = profile_spline


        for i in range(self.NPTS):
            af_le = blade['profile'][np.argmin(blade['profile'][:,0,i]),:,i]
            blade['profile'][:,0,i] -= af_le[0]
            blade['profile'][:,1,i] -= af_le[1]
            c = max(blade['profile'][:,0,i]) - min(blade['profile'][:,0,i])
            blade['profile'][:,:,i] /= c

        return blade

    def remap_polars(self, blade, blade_ref, AFref, spline=PchipInterpolator):
        # TODO: does not support multiple polars at different Re, takes the first polar from list

        ## Set angle of attack grid for airfoil resampling
        # assume grid for last airfoil is sufficient
        alpha = np.array(AFref[blade_ref['bem_aero']['airfoil_position']['labels'][-1]]['polars'][0]['c_l']['grid'])
        Re    = [AFref[blade_ref['bem_aero']['airfoil_position']['labels'][-1]]['polars'][0]['re']]

        # get reference airfoil polars
        af_ref = []
        for afi in blade_ref['bem_aero']['airfoil_position']['labels']:
            if afi not in af_ref:
                af_ref.append(afi)

        n_af_ref  = len(af_ref)
        n_aoa     = len(alpha)
        n_span    = self.NPTS

        cl_ref = np.zeros((n_aoa, n_af_ref))
        cd_ref = np.zeros((n_aoa, n_af_ref))
        cm_ref = np.zeros((n_aoa, n_af_ref))
        for i, af in enumerate(af_ref[::-1]):
            cl_ref[:,i] = remap2grid(np.array(AFref[af]['polars'][0]['c_l']['grid']), np.array(AFref[af]['polars'][0]['c_l']['values']), alpha)
            cd_ref[:,i] = remap2grid(np.array(AFref[af]['polars'][0]['c_d']['grid']), np.array(AFref[af]['polars'][0]['c_d']['values']), alpha)
            cm_ref[:,i] = remap2grid(np.array(AFref[af]['polars'][0]['c_m']['grid']), np.array(AFref[af]['polars'][0]['c_m']['values']), alpha)

        # reference airfoil and spanwise thicknesses
        thk_span  = blade['pf']['rthick']
        thk_afref = [AFref[af]['relative_thickness'] for af in af_ref[::-1]]
        # error handling for spanwise thickness greater/less than the max/min airfoil thicknesses
        np.place(thk_span, thk_span>max(thk_afref), max(thk_afref))
        np.place(thk_span, thk_span<min(thk_afref), min(thk_afref))

        # interpolate
        spline_cl = spline(thk_afref, cl_ref, axis=1)
        spline_cd = spline(thk_afref, cd_ref, axis=1)
        spline_cm = spline(thk_afref, cm_ref, axis=1)
        cl = spline_cl(thk_span)
        cd = spline_cd(thk_span)
        cm = spline_cm(thk_span)

        # CCBlade airfoil class instances
        airfoils = [None]*n_span
        for i in range(n_span):
            airfoils[i] = CCAirfoil(np.degrees(alpha), Re, cl[:,i], cd[:,i], cm[:,i])

        blade['airfoils'] = airfoils

        return blade


    def remap_composites(self, blade, blade_ref):
        # Remap composite sections to a common grid
        t = time.time()
        
        st = copy.deepcopy(blade_ref['2d_fem'])
        st = self.calc_spanwise_grid(st)

        # remap
        for idx_sec, sec in enumerate(st['sections']):
            for var in sec.keys():
                if type(sec[var]) not in [str, bool]:
                    if 'grid' in sec[var].keys():
                        if len(sec[var]['grid']) > 0.:
                            # if section is only for part of the blade, find start and end of new grid
                            if sec[var]['grid'][0] > 0.:
                                idx_s = np.argmax(np.array(self.s)>=sec[var]['grid'][0])
                            else:
                                idx_s = 0
                            if sec[var]['grid'][-1] < 1.:
                                idx_e = np.argmax(np.array(self.s)>sec[var]['grid'][-1])
                            else:
                                idx_e = -1

                            # interpolate
                            if idx_s != 0 or idx_e !=-1:
                                vals = np.full(self.NPTS, None)
                                vals[idx_s:idx_e] = remap2grid(sec[var]['grid'], sec[var]['values'], self.s[idx_s:idx_e])
                                st['sections'][idx_sec][var]['values'] = vals.tolist()
                            else:
                                st['sections'][idx_sec][var]['values'] = remap2grid(sec[var]['grid'], sec[var]['values'], self.s).tolist()
                            st['sections'][idx_sec][var]['grid'] = self.s

            # if vars not provided as inputs
            input_vars = st['sections'][idx_sec].keys()
            if 'fiber_orientation' not in input_vars:
                st['sections'][idx_sec]['fiber_orientation'] = {}
                st['sections'][idx_sec]['fiber_orientation']['grid'] = self.s
                st['sections'][idx_sec]['fiber_orientation']['values'] = [0. if thki != None else None for thki in st['sections'][idx_sec]['thickness']['values']]
            if 'web_flag' not in input_vars:
                st['sections'][idx_sec]['web_flag'] = False
            if 'full_circumference' not in input_vars:
                if all(['midpoint' not in input_vars, 'width' not in input_vars, 's0' not in input_vars, 's1' not in input_vars]):
                    st['sections'][idx_sec]['full_circumference'] = True
                else:
                    st['sections'][idx_sec]['full_circumference'] = False

        blade['st'] = st

        return blade


    def calc_composite_bounds(self, blade):

        # correct arc position for sections that wrap around the trailing edge
        def ArcEndPt_Correction_TE(s_in, L):
            if s_in > L:
                s_in -= L
            elif s_in < 0.:
                s_in += L
            return s_in

        # Format profile for interpolation
        profile   = copy.copy(blade['profile'])
        idx_le    = np.argmin(profile[:,0,0])
        profile[:idx_le,0,:] *= -1
        profile_d = profile*blade['pf']['chord'][np.newaxis, np.newaxis, :]

        # Loop spanwise
        for i in range(self.NPTS):
            s_all = []

            t2 = time.time()
            profile_curve = arc_length(profile_d[:,0,i], profile_d[:,1,i])
            
            # loop through composite layups
            for idx_sec, sec in enumerate(blade['st']['sections']):

                # initialize chord wise start end points
                if i == 0:
                    if sec['full_circumference']:
                        blade['st']['sections'][idx_sec]['s0'] = {}
                        blade['st']['sections'][idx_sec]['s0']['grid'] = self.s
                        blade['st']['sections'][idx_sec]['s0']['values'] = np.full(self.NPTS, -1.).tolist()
                        blade['st']['sections'][idx_sec]['s1'] = {}
                        blade['st']['sections'][idx_sec]['s1']['grid'] = self.s
                        blade['st']['sections'][idx_sec]['s1']['values'] = np.full(self.NPTS, 1.).tolist()
                    if 's0' not in blade['st']['sections'][idx_sec].keys():
                        blade['st']['sections'][idx_sec]['s0'] = {}
                        blade['st']['sections'][idx_sec]['s0']['grid'] = self.s
                        blade['st']['sections'][idx_sec]['s0']['values'] = np.full(self.NPTS, None).tolist()
                    if 's1' not in blade['st']['sections'][idx_sec].keys():
                        blade['st']['sections'][idx_sec]['s1'] = {}
                        blade['st']['sections'][idx_sec]['s1']['grid'] = self.s
                        blade['st']['sections'][idx_sec]['s1']['values'] = np.full(self.NPTS, None).tolist()

                # loop through composite layers, find end points if given a width and midpoint
                if 'midpoint' in blade['st']['sections'][idx_sec].keys() and 'width' in blade['st']['sections'][idx_sec].keys():
                    mid   = sec['midpoint']['values'][i]
                    width = sec['width']['values'][i]

                    if mid != None:
                    
                        # solve for arc-wise position of my section ends
                        mid_arc   = remap2grid(profile[:,0,i], profile_curve, mid, spline=interp1d)
                        s0_arc   = ArcEndPt_Correction_TE(mid_arc - width/2., profile_curve[-1])
                        s1_arc   = ArcEndPt_Correction_TE(mid_arc + width/2., profile_curve[-1])

                        # Convert arcwise 
                        s0, s1 = remap2grid(profile_curve, profile[:,0,i], [s0_arc, s1_arc], spline=interp1d)


                        # remove approximate duplicates
                        s0_close = np.isclose(s0,s_all,rtol=5e-3)
                        if len(s0_close)>0 and any(s0_close):
                            s0 = s_all[np.argmax(s0_close)]
                        else:
                            s_all.append(s0)
                        s1_close = np.isclose(s1,s_all,rtol=5e-3)
                        if any(s1_close):
                            s1 = s_all[np.argmax(s1_close)]
                        else:
                            s_all.append(s1)

                        # store final value
                        blade['st']['sections'][idx_sec]['s0']['values'][i] = float(s0)
                        blade['st']['sections'][idx_sec]['s1']['values'][i] = float(s1)

        # Set any end points that are fixed to other sections, loop through composites again
        for idx_sec, sec in enumerate(blade['st']['sections']):
            if 'fixed' in blade['st']['sections'][idx_sec].keys():
                if 's0' in blade['st']['sections'][idx_sec]['fixed'].keys():
                    target_name  = blade['st']['sections'][idx_sec]['fixed']['s0'][0]
                    target_point = blade['st']['sections'][idx_sec]['fixed']['s0'][1]
                    target_idx   = [i for i, sec in enumerate(blade['st']['sections']) if sec['name']==target_name][0]
                    blade['st']['sections'][idx_sec]['s0']['grid']   = blade['st']['sections'][target_idx][target_point]['grid']
                    blade['st']['sections'][idx_sec]['s0']['values'] = blade['st']['sections'][target_idx][target_point]['values']

                if 's1' in blade['st']['sections'][idx_sec]['fixed'].keys():
                    target_name = blade['st']['sections'][idx_sec]['fixed']['s1'][0]
                    target_point = blade['st']['sections'][idx_sec]['fixed']['s1'][1]
                    target_idx  = [i for i, sec in enumerate(blade['st']['sections']) if sec['name']==target_name][0]
                    blade['st']['sections'][idx_sec]['s1']['grid']   = blade['st']['sections'][target_idx][target_point]['grid']
                    blade['st']['sections'][idx_sec]['s1']['values'] = blade['st']['sections'][target_idx][target_point]['values']


        return blade

    def calc_control_points(self, blade, r_max_chord=0., r_in=[]):

        if 'ctrl_pts' not in blade.keys():
            blade['ctrl_pts'] = {}

        # Build Control Point Grid, if not provided with key word arg
        if len(r_in)==0:
            # solve for end of cylinder radius by interpolating relative thickness
            cyl_thk_min = 0.9
            idx_s       = np.argmax(blade['pf']['rthick']<1)
            idx_e       = np.argmax(np.isclose(blade['pf']['rthick'], min(blade['pf']['rthick'])))
            r_cylinder  = remap2grid(blade['pf']['rthick'][idx_e:idx_s-2:-1], blade['pf']['s'][idx_e:idx_s-2:-1], cyl_thk_min)

            # solve for max chord radius
            if r_max_chord == 0.:
                r_max_chord = blade['pf']['s'][np.argmax(blade['pf']['chord'])]

            # control point grid
            r_in = np.r_[[0.], [r_cylinder], np.linspace(r_max_chord, 1., self.NINPUT-2)]

        # Fit control points to planform variables
        blade['ctrl_pts']['theta_in']     = remap2grid(blade['pf']['s'], blade['pf']['theta'], r_in)
        blade['ctrl_pts']['chord_in']     = remap2grid(blade['pf']['s'], blade['pf']['chord'], r_in)
        blade['ctrl_pts']['precurve_in']  = remap2grid(blade['pf']['s'], blade['pf']['precurve'], r_in)
        blade['ctrl_pts']['presweep_in']  = remap2grid(blade['pf']['s'], blade['pf']['presweep'], r_in)
        blade['ctrl_pts']['thickness_in'] = remap2grid(blade['pf']['s'], blade['pf']['rthick'], r_in)

        # Fit control points to composite thickness variables variables 
        #   Note: entering 0 thickness for areas where composite section does not extend to, however the precomp region selection vars 
        #   sector_idx_strain_spar, sector_idx_strain_te) will still be None over these ranges
        idx_spar  = [i for i, sec in enumerate(blade['st']['sections']) if sec['name']==self.spar_var][0]
        idx_te    = [i for i, sec in enumerate(blade['st']['sections']) if sec['name']==self.te_var][0]
        grid_spar = blade['st']['sections'][idx_spar]['thickness']['grid']
        grid_te   = blade['st']['sections'][idx_te]['thickness']['grid']
        vals_spar = [0. if val==None else val for val in blade['st']['sections'][idx_spar]['thickness']['values']]
        vals_te   = [0. if val==None else val for val in blade['st']['sections'][idx_te]['thickness']['values']]
        blade['ctrl_pts']['sparT_in']     = remap2grid(grid_spar, vals_spar, r_in)
        blade['ctrl_pts']['teT_in']       = remap2grid(grid_te, vals_te, r_in)

        # Store additional rotorse variables
        blade['ctrl_pts']['r_in']         = r_in
        blade['ctrl_pts']['r_cylinder']   = r_cylinder
        blade['ctrl_pts']['r_max_chord']  = r_max_chord

        return blade

    def update_planform(self, blade):

        self.s                  = blade['pf']['s'] # TODO: assumes the start and end points of composite sections does not change
        blade['pf']['chord']    = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['chord_in'], self.s)
        blade['pf']['theta']    = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['theta_in'], self.s)
        blade['pf']['r']        = blade['ctrl_pts']['bladeLength']*np.array(self.s)
        blade['pf']['precurve'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['precurve_in'], self.s)
        blade['pf']['presweep'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['presweep_in'], self.s)

        idx_spar  = [i for i, sec in enumerate(blade['st']['sections']) if sec['name']==self.spar_var][0]
        idx_te    = [i for i, sec in enumerate(blade['st']['sections']) if sec['name']==self.te_var][0]

        blade['st']['sections'][idx_spar]['thickness']['grid']   = self.s
        blade['st']['sections'][idx_spar]['thickness']['values'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['sparT_in'], self.s)
        blade['st']['sections'][idx_te]['thickness']['grid']   = self.s
        blade['st']['sections'][idx_te]['thickness']['values'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['teT_in'], self.s)

        # print blade['ctrl_pts']['r_in']
        # print blade['ctrl_pts']['theta_in']
        # import matplotlib.pyplot as plt
        # plt.plot(self.s, thk_te, label='input')
        # plt.plot(self.s, thk_te2, label='fit')
        # plt.legend()
        # plt.show()

        #### TODO: Thickness not currently a design variable
        # thk_ref = [af_ref[af]['relative_thickness'] for af in blade_ref['bem_aero']['airfoil_position']['labels']]
        # blade['pf']['rthick']   = remap2grid(blade_ref['bem_aero']['airfoil_position']['grid'], thk_ref, self.s)
        # blade['pf']['p_le']     = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['r_in'], self.s)

        return blade

        
    def convert_precomp(self, blade, materials_in=[]):

        def region_stacking(i, idx, S0, S1, blade, material_dict, materials, region_loc):
            # Recieve start and end of composite sections chordwise, find which composites layers are in each
            # chordwise regions, generate the precomp composite class instance
            dp = sorted(list(set(S0+S1)))
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            # print i, dp

            for i_reg, (dp0, dp1) in enumerate(zip(dp[0:-1], dp[1:])):
                n_pliesi = []
                thki     = []
                thetai   = []
                mati     = []
                for i_sec, s0i, s1i in zip(idx, S0, S1):
                    name = blade['st']['sections'][i_sec]['name']
                    if s0i <= dp0 and s1i >= dp1:
                        
                        if name in region_loc.keys():
                            if region_loc[name][i] == None:
                                region_loc[name][i] = [i_reg]
                            else:
                                region_loc[name][i].append(i_reg)

                        n_pliesi.append(1.)
                        thki.append(blade['st']['sections'][i_sec]['thickness']['values'][i])
                        thetai.append(blade['st']['sections'][i_sec]['fiber_orientation']['values'][i])
                        mati.append(material_dict[blade['st']['sections'][i_sec]['material']])

                n_plies.append(np.array(n_pliesi))
                thk.append(np.array(thki))
                theta.append(np.array(thetai))
                mat_idx.append(np.array(mati))

            sec = CompositeSection(dp, n_plies, thk, theta, mat_idx, materials)
            return sec, region_loc

        def web_stacking(i, web_idx, web_S0, web_S1, blade, material_dict, materials, flatback, upperCSi):
            dp = []
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            if len(web_idx)>0:
                dp = np.mean((np.abs(web_S0), np.abs(web_S0)), axis=0).tolist()

                dp_all = [[-1.*s0i, -1.*s1i] for s0i, s1i in zip(web_S0, web_S1)]
                web_dp, web_ids = np.unique(dp_all, axis=0, return_inverse=True)
                for webi in np.unique(web_ids):
                    # store variable values (thickness, orientation, material) for layers that make up each web, based on the mapping array web_ids
                    n_pliesi = [1. for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thki     = [blade['st']['sections'][i_reg]['thickness']['values'][i] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thetai   = [blade['st']['sections'][i_reg]['fiber_orientation']['values'][i] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    mati     = [material_dict[blade['st']['sections'][i_reg]['material']] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]

                    n_plies.append(np.array(n_pliesi))
                    thk.append(np.array(thki))
                    theta.append(np.array(thetai))
                    mat_idx.append(np.array(mati))

            if flatback:
                dp.append(1.)
                n_plies.append(upperCSi.n_plies[-1])
                thk.append(upperCSi.t[-1])
                theta.append(upperCSi.theta[-1])
                mat_idx.append(upperCSi.mat_idx[-1])

            dp_out = sorted(list(set(dp)))
            sec = CompositeSection(dp_out, n_plies, thk, theta, mat_idx, materials)
            return sec

        ## Initialization
        if 'precomp' not in blade.keys():
            blade['precomp'] = {}

        region_loc_vars = [self.te_var, self.spar_var]
        region_loc_ss = {} # track precomp regions for user selected composite layers
        region_loc_ps = {}
        for var in region_loc_vars:
            region_loc_ss[var] = [None]*self.NPTS
            region_loc_ps[var] = [None]*self.NPTS

        ## Materials
        if 'materials' not in blade['precomp']:
            material_dict = {}
            materials     = []
            for i, mati in enumerate(materials_in):
                material_id = i
                material_dict[mati['name']] = material_id
                materials.append(Orthotropic2DMaterial(mati['E'][0]*1e3, mati['E'][1]*1e3, mati['G'][0]*1e3, mati['nu'][0]*1e3, mati['rho'], mati['name']))
            blade['precomp']['materials']     = materials
            blade['precomp']['material_dict'] = material_dict

        ## Profiles
        profile = [None]*self.NPTS
        for i in range(self.NPTS):
            profile[i] = Profile.initWithTEtoTEdata(blade['profile'][:,0,i], blade['profile'][:,1,i])

        ## Composites
        upperCS = [None]*self.NPTS
        lowerCS = [None]*self.NPTS
        websCS  = [None]*self.NPTS
        for i in range(self.NPTS):
            ss_idx  = []
            ss_S0   = []
            ss_S1   = []
            ps_idx  = []
            ps_S0   = []
            ps_S1   = []
            web_S0  = []
            web_S1  = []
            web_idx = []

            # Determine spanwise composite layer elements that are non-zero at this spanwise location,
            # determine their chord-wise start and end location on the pressure and suctions side
            for idx_sec, sec in enumerate(blade['st']['sections']):
                if not sec['web_flag']:
                    if sec['s0']['values'][i] != None and sec['thickness']['values'][i] != None:
                        if sec['s0']['values'][i] < 0. or sec['s1']['values'][i] < 0.:
                            ps_idx.append(idx_sec)
                            if sec['s0']['values'][i] > 0.:
                                ps_S1.append(min(sec['s0']['values'][i], -1))
                            else:
                                ps_S1.append(min(sec['s0']['values'][i], 0.))
                            ps_S0.append(min(sec['s1']['values'][i], 0.))
                            
                        if sec['s0']['values'][i] > 0. or sec['s1']['values'][i] > 0.:
                            ss_idx.append(idx_sec)
                            ss_S0.append(max(sec['s0']['values'][i], 0.))
                            if sec['s0']['values'][i] > sec['s1']['values'][i] and sec['s1']['values'][i]<0.:
                                ss_S1.append(max(sec['s1']['values'][i], 1.))
                            else:
                                ss_S1.append(max(sec['s1']['values'][i], 0.))
                else:
                    if sec['s0']['values'][i] != None and sec['thickness']['values'][i] != None:
                        web_idx.append(idx_sec)
                        web_S0.append(sec['s0']['values'][i])
                        web_S1.append(sec['s1']['values'][i])

            # pressure side, absolute value of S division points
            ps_S0 = np.abs(ps_S0).tolist()
            ps_S1 = np.abs(ps_S1).tolist()

            # 'web' at trailing edge needed for flatback airfoils
            if blade['profile'][0,1,i] != blade['profile'][-1,1,i]:
                flatback = True
            else:
                flatback = False

            # generate the Precomp composite stacks for chordwise regions
            upperCS[i], region_loc_ss = region_stacking(i, ss_idx, ss_S0, ss_S1, blade, blade['precomp']['material_dict'], blade['precomp']['materials'], region_loc_ss)
            lowerCS[i], region_loc_ps = region_stacking(i, ps_idx, ps_S0, ps_S1, blade, blade['precomp']['material_dict'], blade['precomp']['materials'], region_loc_ps)
            if len(web_idx)>0 or flatback:
                websCS[i] = web_stacking(i, web_idx, web_S0, web_S1, blade, blade['precomp']['material_dict'], blade['precomp']['materials'], flatback, upperCS[i])

            
            blade['precomp']['upperCS']       = upperCS
            blade['precomp']['lowerCS']       = lowerCS
            blade['precomp']['websCS']        = websCS
            blade['precomp']['profile']       = profile

            # Assumptions:
            # - pressure and suction side regions are the same (i.e. spar cap is the Nth region on both side)
            # - if the composite layer is divided into multiple regions (i.e. if the spar cap is split into 3 regions due to the web locations),
            #   the middle region is selected with int(n_reg/2), note for an even number of regions, this rounds up
            blade['precomp']['sector_idx_strain_spar'] = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.spar_var]]
            blade['precomp']['sector_idx_strain_te']   = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.te_var]]
            blade['precomp']['spar_var'] = self.spar_var
            blade['precomp']['te_var']   = self.te_var
        
        return blade

        

if __name__ == "__main__":

    ## File managment
    fname_input        = "turbine_inputs/nrel5mw_mod2.yaml"
    fname_output       = "turbine_inputs/nrel5mw_mod_out.yaml"
    flag_write_out     = True
    flag_write_precomp = True
    dir_precomp_out    = "turbine_inputs/precomp"

    ## Load and Format Blade
    tt = time.time()
    refBlade = ReferenceBlade()
    refBlade.verbose = True
    refBlade.spar_var = 'Spar_Cap_SS'
    refBlade.te_var   = 'TE_reinforcement'

    blade = refBlade.initialize(fname_input)

    ## Parmeterization




    ## save output yaml
    if flag_write_out:
        t3 = time.time()
        refBlade.write_ontology(fname_output, blade, refBlade.wt_ref)
        if refBlade.verbose:
            print 'Complete: Write Output: \t%f s'%(time.time()-t3)

    ## save precomp out
    if flag_write_precomp:
        t4 = time.time()
        materials = blade['precomp']['materials']
        upper     = blade['precomp']['upperCS']
        lower     = blade['precomp']['lowerCS']
        webs      = blade['precomp']['websCS']
        profile   = blade['precomp']['profile']
        chord     = blade['pf']['chord']
        twist     = blade['pf']['theta']
        p_le      = blade['pf']['p_le']
        precomp_write = PreCompWriter(dir_precomp_out, materials, upper, lower, webs, profile, chord, twist, p_le)
        precomp_write.execute()
        if refBlade.verbose:
            print 'Complete: Write PreComp: \t%f s'%(time.time()-t4)