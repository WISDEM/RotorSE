from __future__ import print_function
import os, sys, copy, time, warnings
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

from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp, PreCompWriter
from rotorse.geometry_tools.geometry import AirfoilShape, Curve


TURBULENCE_CLASS = commonse.enum.Enum('A B C')
TURBINE_CLASS = commonse.enum.Enum('I II III')
DRIVETRAIN_TYPE = commonse.enum.Enum('geared single_stage multi_drive pm_direct_drive')

def remap2grid(x_ref, y_ref, x, spline=PchipInterpolator):


    try:
        spline_y = spline(x_ref, y_ref)
    except:
        x_ref = np.flip(x_ref, axis=0)
        y_ref = np.flip(y_ref, axis=0)
        spline_y = spline(x_ref, y_ref)

    # error handling for x[-1] - x_ref[-1] > 0 and x[-1]~x_ref[-1]
    try:
        _ = iter(x)
        if x[-1]>max(x_ref) and np.isclose(x[-1], x_ref[-1]):
            x[-1]=x_ref[-1]
    except:
        if np.isclose(x, 0.):
            x = 0.
        if x>max(x_ref) and np.isclose(x, x_ref[-1]):
            x=x_ref[-1]

    y_out = spline_y(x)

    np.place(y_out, y_out < min(y_ref), min(y_ref))
    np.place(y_out, y_out > max(y_ref), max(y_ref))

    return y_out

# def remapWbreak(x_ref, y_ref, x0, idx_break, spline=PchipInterpolator):
#     # for interpolating on airfoil surface, split x into two sections and determine which to use
#     if x0 >= x_ref[idx_break]:
#         x = x_ref[idx_break:]
#         y = y_ref[idx_break:]
#     else:
#         x = x_ref[:idx_break+1]
#         y = y_ref[:idx_break+1]
#     return remap2grid(x, y, x0, spline=spline)

def remapAirfoil(x_ref, y_ref, x0):
    # for interpolating airfoil surface
    x = copy.copy(x_ref)
    y = copy.copy(y_ref)
    x_in = copy.copy(x0)

    idx_le = np.argmin(x)
    x[:idx_le] *= -1.

    idx = [ix0 for ix0, dx0 in enumerate(np.diff(x_in)) if dx0 >0][0]
    x_in[:idx] *= -1.

    return remap2grid(x, y, x_in)

def arc_length(x, y, z=[]):
    npts = len(x)
    arc = np.array([0.]*npts)
    # if high_fidelity:
    #     # iteratively fit a spline between points
    #     if all(np.diff(x)<0):
    #         # correct for decending data
    #         x *= -1.
    #     if not all(np.diff(x)>0):
    #         # if data has more than one zero crossing, break the data up
    #         zero_crossings = [0]+np.where(np.diff(np.sign(x)))[0].tolist()+[len(x)]
    #         # print(zero_crossings)
    #         arc = [arc_length(x[zero_crossings[idx-1]:zero_crossings[idx]+1], y[zero_crossings[idx-1]:zero_crossings[idx]+1]) for idx in range(1,len(zero_crossings))]
            
    #         for idx, arci in enumerate(arc):
    #             if idx == 0:
    #                 arc_out = list(arci)
    #             else:
    #                 arc_out = arc_out + list(arci[1:]+arc_out[-1])

    #         return arc_out
            

    #     spline = PchipInterpolator(x, y)
    #     for k in range(1, npts):
    #         a = x[k-1]
    #         b = x[k]
    #         tz = np.linspace(a, b, num=10)
    #         f = spline(t2)
    #         arc[k] = arc[k-1] + arc_length(tz, f)[-1]


    # else:
    if len(z) == len(x):
        for k in range(1, npts):
            arc[k] = arc[k-1] + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2 + (z[k] - z[k-1])**2)
    else:
        for k in range(1, npts):
            arc[k] = arc[k-1] + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)

    return arc


def rotate(xo, yo, xp, yp, angle):
    ## Rotate a point clockwise by a given angle around a given origin.
    # angle *= -1.
    qx = xo + np.cos(angle) * (xp - xo) - np.sin(angle) * (yp - yo)
    qy = yo + np.sin(angle) * (xp - xo) + np.cos(angle) * (yp - yo)
    return qx, qy

class ReferenceBlade(object):
    def __init__(self):

        # Validate input file against JSON schema
        self.validate        = True       # (bool) run IEA turbine ontology JSON validation
        self.fname_schema    = ''          # IEA turbine ontology JSON schema file

        # Grid sizes
        self.NINPUT          = 5
        self.NPTS            = 50
        self.NPTS_AfProfile  = 200
        self.NPTS_AfPolar    = 100
        self.r_in            = []          # User definied input grid (must be from 0-1)

        # 
        self.analysis_level  = 0           # 0: Precomp, 1: Precomp + write FAST model, 2: FAST/Elastodyn, 3: FAST/Beamdyn)
        self.verbose         = False

        # Precomp analyis
        self.spar_var        = ['']          # name of composite layer for RotorSE spar cap buckling analysis
        self.te_var          = ''          # name of composite layer for RotorSE trailing edge buckling analysis


        

    def initialize(self, fname_input):
        if self.verbose:
            print('Running initialization: %s' % fname_input)

        # Load input
        self.fname_input = fname_input
        self.wt_ref = self.load_ontology(self.fname_input, validate=self.validate, fname_schema=self.fname_schema)

        t1 = time.time()
        # Renaming and converting lists to dicts for simplicity
        # blade_ref = copy.deepcopy(self.wt_ref['components']['blade'])
        af_ref    = {}
        for afi in self.wt_ref['airfoils']:
            af_ref[afi['name']] = afi

        # build blade
        # blade = {}
        blade = copy.deepcopy(self.wt_ref['components']['blade'])
        blade = self.set_configuration(blade, self.wt_ref)
        blade = self.remap_composites(blade)
        blade = self.remap_planform(blade, af_ref)
        blade = self.remap_profiles(blade, af_ref)
        blade = self.remap_polars(blade, af_ref)
        blade = self.calc_composite_bounds(blade)
        blade = self.calc_control_points(blade, self.r_in)
        
        blade['analysis_level'] = self.analysis_level

        if self.verbose:
            print('Complete: Geometry Analysis: \t%f s'%(time.time()-t1))
            
        # Conversion
        if self.analysis_level < 3:
            t2 = time.time()
            blade = self.convert_precomp(blade, self.wt_ref['materials'])
            if self.verbose:
                print('Complete: Precomp Conversion: \t%f s'%(time.time()-t2))
        elif self.analysis_level == 3:
            # sonata/ anba

            # meshing with sonata

            # 
            pass

        return blade

    def update(self, blade):
        # 
        t1 = time.time()
        blade['st'] = self.calc_spanwise_grid(blade['st'])

        blade = self.update_planform(blade)
        blade = self.calc_composite_bounds(blade)

        if self.verbose:
            prin('Complete: Geometry Update: \t%f s'%(time.time()-t1))

        # Conversion
        if self.analysis_level < 3:
            t2 = time.time()
            blade = self.convert_precomp(blade)
            if self.verbose:
                print('Complete: Precomp Conversion: \t%f s'%(time.time()-t2))


        return blade

    def load_ontology(self, fname_input, validate=False, fname_schema=''):
        """ Load inputs IEA turbine ontology yaml inputs, optional validation """
        # Read IEA turbine ontology yaml input file
        with open(fname_input, 'r') as myfile:
            t_load = time.time()
            inputs = myfile.read()

        # Validate the turbine input with the IEA turbine ontology schema
        yaml = YAML()
        if validate:
            t_validate = time.time()

            with open(fname_schema, 'r') as myfile:
                schema = myfile.read()
            json.validate(yaml.load(inputs), yaml.load(schema))

            t_validate = time.time()-t_validate
            if self.verbose:
                print('Complete: Schema "%s" validation: \t%f s'%(fname_schema, t_validate))
        else:
            t_validate = 0.

        # return yaml.load(inputs)
        with open(fname_input, 'r') as myfile:
            inputs = myfile.read()

        if self.verbose:
            t_load = time.time() - t_load - t_validate
            print('Complete: Load Input File: \t%f s'%(t_load))
        
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
        blade_out = copy.deepcopy(blade)

        # Planform
        wt_out['components']['blade']['outer_shape_bem']['chord']['values']             = blade_out['pf']['chord'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['chord']['grid']               = blade_out['pf']['s'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['twist']['values']             = np.radians(blade_out['pf']['theta']).tolist()
        wt_out['components']['blade']['outer_shape_bem']['twist']['grid']               = blade_out['pf']['s'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['pitch_axis']['values']        = blade_out['pf']['p_le'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['pitch_axis']['grid']          = blade_out['pf']['s'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['reference_axis']['x']['values']  = (-1*blade_out['pf']['precurve']).tolist()
        wt_out['components']['blade']['outer_shape_bem']['reference_axis']['x']['grid']    = blade_out['pf']['s'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['reference_axis']['y']['values']  = blade_out['pf']['presweep'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['reference_axis']['y']['grid']    = blade_out['pf']['s'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['reference_axis']['z']['values']  = blade_out['pf']['r'].tolist()
        wt_out['components']['blade']['outer_shape_bem']['reference_axis']['z']['grid']    = blade_out['pf']['s'].tolist()

        # Composite layups
        st = blade_out['st']
        # for var in st['reference_axis'].keys():
        #     try:
        #         _ = st['reference_axis'][var].keys()

        #         st['reference_axis'][var]['grid'] = [float(r) for val, r in zip(st['reference_axis'][var]['values'], st['reference_axis'][var]['grid']) if val != None]
        #         st['reference_axis'][var]['values'] = [float(val) for val in st['reference_axis'][var]['values'] if val != None]
        #         reference_axis
        #         if st['reference_axis'][idx_sec][var]['values'] == []:
        #             del st['reference_axis'][var]
        #             continue
        #     except:
        #         pass

        for idx_sec, sec in enumerate(st['layers']):
            for var in st['layers'][idx_sec].keys():
                try:
                    _ = st['layers'][idx_sec][var].keys()

                    st['layers'][idx_sec][var]['grid'] = [float(r) for val, r in zip(st['layers'][idx_sec][var]['values'], st['layers'][idx_sec][var]['grid']) if val != None]
                    st['layers'][idx_sec][var]['values'] = [float(val) for val in st['layers'][idx_sec][var]['values'] if val != None]
                    
                    if st['layers'][idx_sec][var]['values'] == []:
                        del st['layers'][idx_sec][var]
                        continue
                except:
                    pass
        for idx_sec, sec in enumerate(st['webs']):
            for var in st['webs'][idx_sec].keys():
                try:
                    _ = st['webs'][idx_sec][var].keys()
                    st['webs'][idx_sec][var]['grid'] = [float(r) for val, r in zip(st['webs'][idx_sec][var]['values'], st['webs'][idx_sec][var]['grid']) if val != None]
                    st['webs'][idx_sec][var]['values'] = [float(val) for val in st['webs'][idx_sec][var]['values'] if val != None]

                    if st['layers'][idx_sec][var]['values'] == []:
                        del st['layers'][idx_sec][var]
                        continue
                except:
                    pass
        wt_out['components']['blade']['internal_structure_2d_fem'] = st

        f = open(fname, "w")
        yaml=YAML()
        yaml.default_flow_style = None
        yaml.width = float("inf")
        yaml.indent(mapping=4, sequence=6, offset=3)
        yaml.dump(wt_out, f)

    def calc_spanwise_grid(self, st):
        ### Spanwise grid
        # Finds the start and end points of all composite layers, which are required points in the new grid
        # Attempts to roughly evenly space points between the required start/end points to output the user specified grid size

        n = self.NPTS
        # Find unique composite start and end points
        r_points = copy.copy(self.r_in)
        for type_sec, idx_sec, sec in zip(['webs']*len(st['webs'])+['layers']*len(st['layers']), range(len(st['webs']))+range(len(st['layers'])), st['webs']+st['layers']):
            for var in sec.keys():
                if type(sec[var]) not in [str, bool]:
                    if 'grid' in sec[var].keys():
                        if len(sec[var]['grid']) > 0.:
                            # remove approximate duplicates
                            r0 = sec[var]['grid'][0]
                            r1 = sec[var]['grid'][-1]

                            r0_close = np.isclose(r0,r_points)
                            if len(r0_close)>0 and any(r0_close):
                                st[type_sec][idx_sec][var]['grid'][0] = r_points[np.argmax(r0_close)]
                            else:
                                r_points.append(r0)

                            r1_close = np.isclose(r1,r_points)
                            if any(r1_close):
                                st[type_sec][idx_sec][var]['grid'][-1] = r_points[np.argmax(r1_close)]
                            else:
                                r_points.append(r1)

        # Check for large enough grid size
        r_points = sorted(r_points)
        n_pts = len(r_points)
        if n_pts > n:
            grid_size_warning = "A grid size of %d was specified, but %d unique composite layer start/end points were found.  It is highly recommended to increase the grid size to >= %d to avoid errors or unrealistic results "%(n, n_pts, n_pts)
            warnings.warn(grid_size_warning)

        # estimate length of the linspaces between required start/end points
        r_points_idx = [int(n*i) for i in r_points]
        lengths = []
        for i in range(1,len(r_points_idx)):
            len_i = max([r_points_idx[i] - r_points_idx[i-1] + 2, 2])
            lengths.append(len_i)

        # Correct lengths down to the total number of requested points
        n_estimate = sum(lengths)-len(r_points_idx)+2
        n_diff = n_estimate - n
        if n_diff > 0:
            lengths[np.argmax(lengths)] -= n_diff

        # Build grid as concatenation of linspaces between required points
        grid_out = []
        for i in range(1,n_pts):
            if i == n_pts-1:
                grid_out.append(np.linspace(r_points[i-1], r_points[i], lengths[i-1]))
            else:
                grid_out.append(np.linspace(r_points[i-1], r_points[i], lengths[i-1])[:-1])
        self.s = np.concatenate(grid_out)

        return st

    
    def set_configuration(self, blade, wt_ref):

        blade['config'] = {}

        blade['config']['name']  = wt_ref['name']
        for var in wt_ref['assembly']['global']:
            blade['config'][var] = wt_ref['assembly']['global'][var]
        for var in wt_ref['assembly']['control']:
            blade['config'][var] = wt_ref['assembly']['control'][var]

        return blade

    def remap_planform(self, blade, af_ref):

        blade['pf'] = {}

        blade['pf']['s']        = self.s
        blade['pf']['chord']    = remap2grid(blade['outer_shape_bem']['chord']['grid'], blade['outer_shape_bem']['chord']['values'], self.s)
        blade['pf']['theta']    = np.degrees(remap2grid(blade['outer_shape_bem']['twist']['grid'], blade['outer_shape_bem']['twist']['values'], self.s))
        blade['pf']['p_le']     = remap2grid(blade['outer_shape_bem']['pitch_axis']['grid'], blade['outer_shape_bem']['pitch_axis']['values'], self.s)
        blade['pf']['r']        = remap2grid(blade['outer_shape_bem']['reference_axis']['z']['grid'], blade['outer_shape_bem']['reference_axis']['z']['values'], self.s)
        blade['pf']['precurve'] = -1.*remap2grid(blade['outer_shape_bem']['reference_axis']['x']['grid'], blade['outer_shape_bem']['reference_axis']['x']['values'], self.s)
        blade['pf']['presweep'] = remap2grid(blade['outer_shape_bem']['reference_axis']['y']['grid'], blade['outer_shape_bem']['reference_axis']['y']['values'], self.s)

        thk_ref = [af_ref[af]['relative_thickness'] for af in blade['outer_shape_bem']['airfoil_position']['labels']]
        blade['pf']['rthick']   = remap2grid(blade['outer_shape_bem']['airfoil_position']['grid'], thk_ref, self.s)

        return blade

    def remap_profiles(self, blade, AFref, spline=PchipInterpolator):

        # Option to correct trailing edge for closed to flatback transition
        trailing_edge_correction = True

        # Get airfoil thicknesses in decending order and cooresponding airfoil names
        AFref_thk = [AFref[af]['relative_thickness'] for af in blade['outer_shape_bem']['airfoil_position']['labels']]

        af_thk_dict = {}
        for afi in blade['outer_shape_bem']['airfoil_position']['labels']:
            afi_thk = AFref[afi]['relative_thickness']
            if afi_thk not in af_thk_dict.keys():
                af_thk_dict[afi_thk] = afi

        af_thk = sorted(af_thk_dict.keys())
        af_labels = [af_thk_dict[afi] for afi in af_thk]
        
        # Build array of reference airfoil coordinates, remapped
        AFref_n  = len(af_labels)
        AFref_xy = np.zeros((self.NPTS_AfProfile, 2, AFref_n))
        AF_fb = {}

        for afi, af_label in enumerate(af_labels[::-1]):
            points = np.column_stack((AFref[af_label]['coordinates']['x'], AFref[af_label]['coordinates']['y']))
 
            # check that airfoil points are declared from the TE suction side to TE pressure side
            idx_le = np.argmin(AFref[af_label]['coordinates']['x'])
            if np.mean(AFref[af_label]['coordinates']['y'][:idx_le]) > 0.:
                points = np.flip(points, axis=0)

            if afi == 0:
                af = AirfoilShape(points=points)
                af.redistribute(self.NPTS_AfProfile, even=False, dLE=True)
                s = af.s
                af_points = af.points
            else:
                af_points = np.column_stack((AFref_xy[:,0,0], remapAirfoil(points[:,0], points[:,1], AFref_xy[:,0,0])))

            if [1,0] not in af_points.tolist():
                af_points[:,0] -= af_points[np.argmin(af_points[:,0]), 0]
            c = max(af_points[:,0])-min(af_points[:,0])
            af_points[:,:] /= c
            AFref_xy[:,:,afi] = af_points

            # if correcting, check for flatbacks
            if trailing_edge_correction:
                if af_points[0,1] == af_points[-1,1]:
                    AF_fb[af_label] = False
                else:
                    AF_fb[af_label] = True

        
        AFref_xy = np.flip(AFref_xy, axis=2)

        if trailing_edge_correction:
            # closed to flat transition, find spanwise indexes where cylinder/sharp -> flatback
            transition = False
            for i in range(1,len(blade['outer_shape_bem']['airfoil_position']['labels'])):
                afi1 = blade['outer_shape_bem']['airfoil_position']['labels'][i]
                afi0 = blade['outer_shape_bem']['airfoil_position']['labels'][i-1]
                if AF_fb[afi1] and not AF_fb[afi0]:
                    transition = True
                    trans_thk = [AFref[afi0]['relative_thickness'], AFref[afi1]['relative_thickness']]
            if transition:
                trans_correct_idx = [i_thk for i_thk, thk in enumerate(blade['pf']['rthick']) if thk<trans_thk[0] and thk>trans_thk[1]]
            else:
                trans_correct_idx = []


        # Spanwise thickness interpolation
        spline = PchipInterpolator
        profile_spline = spline(af_thk, AFref_xy, axis=2)
        blade['profile'] = profile_spline(blade['pf']['rthick'])
        blade['profile_spline'] = profile_spline


        for i in range(self.NPTS):
            af_le = blade['profile'][np.argmin(blade['profile'][:,0,i]),:,i]
            blade['profile'][:,0,i] -= af_le[0]
            blade['profile'][:,1,i] -= af_le[1]
            c = max(blade['profile'][:,0,i]) - min(blade['profile'][:,0,i])
            blade['profile'][:,:,i] /= c


            if trailing_edge_correction:
                if i in trans_correct_idx:

                    # Find indices on Suction and Pressure side for last 85-95% and 95-100% chordwise
                    idx_85_95  = [i_x for i_x, xi in enumerate(blade['profile'][:,0,i]) if xi>0.85 and xi < 0.95]
                    idx_95_100 = [i_x for i_x, xi in enumerate(blade['profile'][:,0,i]) if xi>0.95 and xi < 1.]

                    idx_85_95_break = [i_idx for i_idx, d_idx in enumerate(np.diff(idx_85_95)) if d_idx > 1][0]+1
                    idx_85_95_SS    = idx_85_95[:idx_85_95_break]
                    idx_85_95_PS    = idx_85_95[idx_85_95_break:]

                    idx_95_100_break = [i_idx for i_idx, d_idx in enumerate(np.diff(idx_95_100)) if d_idx > 1][0]+1
                    idx_95_100_SS    = idx_95_100[:idx_95_100_break]
                    idx_95_100_PS    = idx_95_100[idx_95_100_break:]

                    # Interpolate the last 5% to the trailing edge
                    idx_in_PS = idx_85_95_PS+[-1]
                    x_corrected_PS = blade['profile'][idx_95_100_PS,0,i]
                    y_corrected_PS = remap2grid(blade['profile'][idx_in_PS,0,i], blade['profile'][idx_in_PS,1,i], x_corrected_PS)

                    idx_in_SS = [0]+idx_85_95_SS
                    x_corrected_SS = blade['profile'][idx_95_100_SS,0,i]
                    y_corrected_SS = remap2grid(blade['profile'][idx_in_SS,0,i], blade['profile'][idx_in_SS,1,i], x_corrected_SS)

                    # Overwrite profile with corrected TE
                    blade['profile'][idx_95_100_SS,1,i] = y_corrected_SS
                    blade['profile'][idx_95_100_PS,1,i] = y_corrected_PS

        return blade

    def remap_polars(self, blade, AFref, spline=PchipInterpolator):
        # TODO: does not support multiple polars at different Re, takes the first polar from list

        ## Set angle of attack grid for airfoil resampling
        # assume grid for last airfoil is sufficient
        alpha = np.array(AFref[blade['outer_shape_bem']['airfoil_position']['labels'][-1]]['polars'][0]['c_l']['grid'])
        Re    = [AFref[blade['outer_shape_bem']['airfoil_position']['labels'][-1]]['polars'][0]['re']]

        # get reference airfoil polars
        af_ref = []
        for afi in blade['outer_shape_bem']['airfoil_position']['labels']:
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
            airfoils[i].eval_unsteady(np.degrees(alpha), cl[:,i], cd[:,i], cm[:,i])

        blade['airfoils'] = airfoils

        return blade


    def remap_composites(self, blade):
        # Remap composite sections to a common grid
        t = time.time()
        
        # st = copy.deepcopy(blade_ref['internal_structure_2d_fem'])
        # print('remap_composites copy %f'%(time.time()-t))
        st = blade['internal_structure_2d_fem']
        st = self.calc_spanwise_grid(st)

        for var in st['reference_axis']:
            st['reference_axis'][var]['values'] = remap2grid(st['reference_axis'][var]['grid'], st['reference_axis'][var]['values'], self.s).tolist()
            st['reference_axis'][var]['grid'] = self.s.tolist()

        # remap
        for type_sec, idx_sec, sec in zip(['webs']*len(st['webs'])+['layers']*len(st['layers']), range(len(st['webs']))+range(len(st['layers'])), st['webs']+st['layers']):
            for var in sec.keys():
                # print(sec['name'], var)
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
                                st[type_sec][idx_sec][var]['values'] = vals.tolist()
                            else:
                                st[type_sec][idx_sec][var]['values'] = remap2grid(sec[var]['grid'], sec[var]['values'], self.s).tolist()
                            st[type_sec][idx_sec][var]['grid'] = self.s

        blade['st'] = st

        return blade


    def calc_composite_bounds(self, blade):

        #######
        def calc_axis_intersection(rotation, offset, p_le_d, side, thk=0.):
            # dimentional analysis that takes a rotation and offset from the pitch axis and calculates the airfoil intersection
            # rotation
            
            offset_x   = offset*np.cos(rotation) + p_le_d[0]
            offset_y   = offset*np.sin(rotation) + p_le_d[1]

            m_rot      = np.sin(rotation)/np.cos(rotation)       # slope of rotated axis
            plane_rot  = [m_rot, -1*m_rot*p_le_d[0]+ p_le_d[1]]  # coefficients for rotated axis line: a1*x + a0

            m_intersection     = np.sin(rotation+np.pi/2.)/np.cos(rotation+np.pi/2.)   # slope perpendicular to rotated axis
            plane_intersection = [m_intersection, -1*m_intersection*offset_x+offset_y] # coefficients for line perpendicular to rotated axis line at the offset: a1*x + a0

            # intersection between airfoil surface and the line perpendicular to the rotated/offset axis
            y_intersection = np.polyval(plane_intersection, profile_i[:,0])
            idx_inter      = np.argwhere(np.diff(np.sign(profile_i[:,1] - y_intersection))).flatten() # find closest airfoil surface points to intersection 

            # if len(idx_inter) == 0:
            #     print(blade['pf']['s'][i], blade['pf']['r'][i], blade['pf']['chord'][i], thk)
            #     import matplotlib.pyplot as plt
            #     plt.plot(profile_i[:,0], profile_i[:,1])
            #     plt.axis('equal')
            #     ymin, ymax = plt.gca().get_ylim()
            #     xmin, xmax = plt.gca().get_xlim()
            #     plt.plot(profile_i[:,0], y_intersection)
            #     plt.plot(p_le_d[0], p_le_d[1], '.')
            #     plt.axis([xmin, xmax, ymin, ymax])
            #     plt.show()

            midpoint_arc = []
            for sidei in side:
                if sidei.lower() == 'suction':
                    tangent_line = np.polyfit(profile_i[idx_inter[0]:idx_inter[0]+2, 0], profile_i[idx_inter[0]:idx_inter[0]+2, 1], 1)
                elif sidei.lower() == 'pressure':
                    tangent_line = np.polyfit(profile_i[idx_inter[1]:idx_inter[1]+2, 0], profile_i[idx_inter[1]:idx_inter[1]+2, 1], 1)

                midpoint_x = (tangent_line[1]-plane_intersection[1])/(plane_intersection[0]-tangent_line[0])
                midpoint_y = plane_intersection[0]*(tangent_line[1]-plane_intersection[1])/(plane_intersection[0]-tangent_line[0]) + plane_intersection[1]

                # convert to arc position
                if sidei.lower() == 'suction':
                    x_half = profile_i[:idx_le+1,0]
                    arc_half = profile_i_arc[:idx_le+1]
                elif sidei.lower() == 'pressure':
                    x_half = profile_i[idx_le:,0]
                    arc_half = profile_i_arc[idx_le:]

                midpoint_arc.append(remap2grid(x_half, arc_half, midpoint_x))#, spline=interp1d))

            return midpoint_arc
        ########

        # Format profile for interpolation

        profile_d = copy.copy(blade['profile'])
        profile_d[:,0,:] = profile_d[:,0,:] - blade['pf']['p_le'][np.newaxis, :]
        profile_d = np.flip(profile_d*blade['pf']['chord'][np.newaxis, np.newaxis, :], axis=0)

        for i in range(self.NPTS):
            s_all = []

            t9 = time.time()
            profile_i = copy.copy(profile_d[:,:,i])
            if list(profile_i[-1,:]) != list(profile_i[-1,:]):
                TE = np.mean((profile_i[-1,:], profile_i[-1,:]), axis=0)
                profile_i = np.row_stack((TE, profile_i, TE))

            idx_le = np.argmin(profile_i[:,0])

            profile_i_arc = arc_length(profile_i[:,0], profile_i[:,1])
            arc_L = profile_i_arc[-1]
            profile_i_arc /= arc_L

            
            # loop through composite layups
            for type_sec, idx_sec, sec in zip(['webs']*len(blade['st']['webs'])+['layers']*len(blade['st']['layers']), range(len(blade['st']['webs']))+range(len(blade['st']['layers'])), blade['st']['webs']+blade['st']['layers']):
                # for idx_sec, sec in enumerate(blade['st'][type_sec]):

                # initialize chord wise start end points
                if i == 0:
                    # print(sec['name'], blade['st'][type_sec][idx_sec].keys())
                    if all([field not in blade['st'][type_sec][idx_sec].keys() for field in ['midpoint_nd_arc','start_nd_arc','end_nd_arc','rotation','web']]):
                        blade['st'][type_sec][idx_sec]['start_nd_arc'] = {}
                        blade['st'][type_sec][idx_sec]['start_nd_arc']['grid'] = self.s
                        blade['st'][type_sec][idx_sec]['start_nd_arc']['values'] = np.full(self.NPTS, 0.).tolist()
                        blade['st'][type_sec][idx_sec]['end_nd_arc'] = {}
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['grid'] = self.s
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['values'] = np.full(self.NPTS, 1.).tolist()
                    if 'width' in blade['st'][type_sec][idx_sec].keys():
                        blade['st'][type_sec][idx_sec]['start_nd_arc'] = {}
                        blade['st'][type_sec][idx_sec]['start_nd_arc']['grid'] = self.s
                        blade['st'][type_sec][idx_sec]['start_nd_arc']['values'] = np.full(self.NPTS, None).tolist()
                        blade['st'][type_sec][idx_sec]['end_nd_arc'] = {}
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['grid'] = self.s
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['values'] = np.full(self.NPTS, None).tolist()
                    if 'start_nd_arc' not in blade['st'][type_sec][idx_sec].keys():
                        blade['st'][type_sec][idx_sec]['start_nd_arc'] = {}
                        blade['st'][type_sec][idx_sec]['start_nd_arc']['grid'] = self.s
                        blade['st'][type_sec][idx_sec]['start_nd_arc']['values'] = np.full(self.NPTS, None).tolist()
                    if 'end_nd_arc' not in blade['st'][type_sec][idx_sec].keys():
                        blade['st'][type_sec][idx_sec]['end_nd_arc'] = {}
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['grid'] = self.s
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['values'] = np.full(self.NPTS, None).tolist()
                    if 'fiber_orientation' not in blade['st'][type_sec][idx_sec].keys() and type_sec != 'webs':
                        blade['st'][type_sec][idx_sec]['fiber_orientation'] = {}
                        blade['st'][type_sec][idx_sec]['fiber_orientation']['grid'] = self.s
                        blade['st'][type_sec][idx_sec]['fiber_orientation']['values'] = np.zeros(self.NPTS).tolist() 
                    if 'rotation' in blade['st'][type_sec][idx_sec].keys():
                        if 'fixed' in blade['st'][type_sec][idx_sec]['rotation'].keys():
                            if blade['st'][type_sec][idx_sec]['rotation']['fixed'] == 'twist':
                                blade['st'][type_sec][idx_sec]['rotation']['grid'] = blade['pf']['s']
                                blade['st'][type_sec][idx_sec]['rotation']['values'] = np.radians(blade['pf']['theta'])
                            else:
                                warning_invalid_fixed_rotation_reference = 'Invalid fixed reference given for layer = "%s" rotation. Currently supported options: "twist".'%(sec['name'])
                                warnings.warn(warning_invalid_fixed_rotation_reference)


                # If non-dimensional coordinates are given, ignore other methods
                calc_bounds = True
                # if 'values' in blade['st'][type_sec][idx_sec]['start_nd_arc'].keys() and 'values' in blade['st'][type_sec][idx_sec]['end_nd_arc'].keys():
                #     if blade['st'][type_sec][idx_sec]['start_nd_arc']['values'][i] != None and blade['st'][type_sec][idx_sec]['end_nd_arc']['values'][i] != None:
                #         calc_bounds = False

                if calc_bounds:
                    if 'rotation' in blade['st'][type_sec][idx_sec].keys() and 'width' in blade['st'][type_sec][idx_sec].keys() and 'side' in blade['st'][type_sec][idx_sec].keys() and blade['st'][type_sec][idx_sec]['thickness']['values'][i] not in [None, 0., 0]:

                        # layer midpoint definied with a rotation and offset about the pitch axis
                        rotation   = sec['rotation']['values'][i] # radians
                        width      = sec['width']['values'][i]    # meters
                        p_le_d     = [0., 0.]                     # pitch axis for dimentional profile
                        side       = sec['side']
                        if 'offset_x_pa' in blade['st'][type_sec][idx_sec].keys():
                            offset = sec['offset_x_pa']['values'][i]
                        else:
                            offset = 0.

                        if rotation == None:
                            rotation = 0.
                        if width == None:
                            width = 0.
                        if side == None:
                            side = 0.
                        if offset == None:
                            offset = 0.

                        if side.lower() != 'suction' and side.lower() != 'pressure':
                            warning_invalid_side_value = 'Invalid airfoil value give: side = "%s" for layer = "%s" at r[%d] = %f. Must be set to "suction" or "pressure".'%(side, sec['name'], i, blade['pf']['r'][i])
                            warnings.warn(warning_invalid_side_value)

                        midpoint = calc_axis_intersection(rotation, offset, p_le_d, [side], thk=sec['thickness']['values'][i])[0]
                        
                        blade['st'][type_sec][idx_sec]['start_nd_arc']['values'][i] = midpoint-width/arc_L/2.
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['values'][i]   = midpoint+width/arc_L/2.

                    elif 'rotation' in blade['st'][type_sec][idx_sec].keys():
                        # web defined with a rotatio and offset about the pitch axis
                        # if 'fixed' in sec['rotation'].keys():
                        #     sec['rotation']['values']

                        rotation   = sec['rotation']['values'][i] # radians
                        p_le_d     = [0., 0.]                     # pitch axis for dimentional profile
                        if 'offset_x_pa' in blade['st'][type_sec][idx_sec].keys():
                            offset = sec['offset_x_pa']['values'][i]
                        else:
                            offset = 0.

                        if rotation == None:
                            rotation = 0
                        if offset == None:
                            offset = 0
                        
                        [blade['st'][type_sec][idx_sec]['start_nd_arc']['values'][i], blade['st'][type_sec][idx_sec]['end_nd_arc']['values'][i]] = sorted(calc_axis_intersection(rotation, offset, p_le_d, ['suction', 'pressure']))

                    elif 'midpoint_nd_arc' in blade['st'][type_sec][idx_sec].keys():
                        # fixed to LE or TE
                        width      = sec['width']['values'][i]    # meters
                        if blade['st'][type_sec][idx_sec]['midpoint_nd_arc']['fixed'].lower() == 'te':
                            midpoint = 1.
                        elif blade['st'][type_sec][idx_sec]['midpoint_nd_arc']['fixed'].lower() == 'le':
                            midpoint = profile_i_arc[idx_le]
                        else:
                            warning_invalid_side_value = 'Invalid fixed midpoint give: midpoint_nd_arc[fixed] = "%s" for layer = "%s" at r[%d] = %f. Must be set to "LE" or "TE".'%(blade['st'][type_sec][idx_sec]['midpoint_nd_arc']['fixed'], sec['name'], i, blade['pf']['r'][i])
                            warnings.warn(warning_invalid_side_value)

                        if width == None:
                            width = 0

                        blade['st'][type_sec][idx_sec]['start_nd_arc']['values'][i] = midpoint-width/arc_L/2.
                        blade['st'][type_sec][idx_sec]['end_nd_arc']['values'][i]   = midpoint+width/arc_L/2.
                        if blade['st'][type_sec][idx_sec]['end_nd_arc']['values'][i] > 1.:
                            blade['st'][type_sec][idx_sec]['end_nd_arc']['values'][i] -= 1.
                    

        # Set any end points that are fixed to other sections, loop through composites again
        for idx_sec, sec in enumerate(blade['st']['layers']):
            if 'fixed' in blade['st']['layers'][idx_sec]['start_nd_arc'].keys() and 'fixed' in blade['st']['layers'][idx_sec]['end_nd_arc'].keys():
                target_name  = blade['st']['layers'][idx_sec]['start_nd_arc']['fixed']
                target_idx   = [i for i, sec in enumerate(blade['st']['layers']) if sec['name']==target_name][0]
                blade['st']['layers'][idx_sec]['start_nd_arc']['grid']   = blade['st']['layers'][target_idx]['end_nd_arc']['grid'].tolist()
                blade['st']['layers'][idx_sec]['start_nd_arc']['values'] = blade['st']['layers'][target_idx]['end_nd_arc']['values']

                target_name  = blade['st']['layers'][idx_sec]['end_nd_arc']['fixed']
                target_idx   = [i for i, sec in enumerate(blade['st']['layers']) if sec['name']==target_name][0]
                blade['st']['layers'][idx_sec]['end_nd_arc']['grid']   = blade['st']['layers'][target_idx]['start_nd_arc']['grid'].tolist()
                blade['st']['layers'][idx_sec]['end_nd_arc']['values'] = blade['st']['layers'][target_idx]['start_nd_arc']['values']


        return blade

    def calc_control_points(self, blade, r_in=[], r_max_chord=0.):

        if 'ctrl_pts' not in blade.keys():
            blade['ctrl_pts'] = {}

        # solve for max chord radius
        if r_max_chord == 0.:
            r_max_chord = blade['pf']['s'][np.argmax(blade['pf']['chord'])]

        # solve for end of cylinder radius by interpolating relative thickness
        cyl_thk_min = 0.999
        idx_s       = np.argmax(blade['pf']['rthick']<1)
        idx_e       = np.argmax(np.isclose(blade['pf']['rthick'], min(blade['pf']['rthick'])))
        r_cylinder  = remap2grid(blade['pf']['rthick'][idx_e:idx_s-1:-1], blade['pf']['s'][idx_e:idx_s-1:-1], cyl_thk_min)

        # Build Control Point Grid, if not provided with key word arg
        if len(r_in)==0:
            # control point grid
            r_in = np.array(sorted(np.r_[[0.], [r_cylinder], np.linspace(r_max_chord, 1., self.NINPUT-2)]))

        # Fit control points to planform variables
        blade['ctrl_pts']['theta_in']     = remap2grid(blade['pf']['s'], blade['pf']['theta'], r_in)
        blade['ctrl_pts']['chord_in']     = remap2grid(blade['pf']['s'], blade['pf']['chord'], r_in)
        blade['ctrl_pts']['precurve_in']  = remap2grid(blade['pf']['s'], blade['pf']['precurve'], r_in)
        blade['ctrl_pts']['presweep_in']  = remap2grid(blade['pf']['s'], blade['pf']['presweep'], r_in)
        blade['ctrl_pts']['thickness_in'] = remap2grid(blade['pf']['s'], blade['pf']['rthick'], r_in)

        # Fit control points to composite thickness variables variables 
        #   Note: entering 0 thickness for areas where composite section does not extend to, however the precomp region selection vars 
        #   sector_idx_strain_spar, sector_idx_strain_te) will still be None over these ranges
        idx_spar  = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==self.spar_var[0].lower()][0]
        idx_te    = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==self.te_var.lower()][0]
        grid_spar = blade['st']['layers'][idx_spar]['thickness']['grid']
        grid_te   = blade['st']['layers'][idx_te]['thickness']['grid']
        vals_spar = [0. if val==None else val for val in blade['st']['layers'][idx_spar]['thickness']['values']]
        vals_te   = [0. if val==None else val for val in blade['st']['layers'][idx_te]['thickness']['values']]
        blade['ctrl_pts']['sparT_in']     = remap2grid(grid_spar, vals_spar, r_in)
        blade['ctrl_pts']['teT_in']       = remap2grid(grid_te, vals_te, r_in)

        # Store additional rotorse variables
        blade['ctrl_pts']['r_in']         = r_in
        blade['ctrl_pts']['r_cylinder']   = r_cylinder
        blade['ctrl_pts']['r_max_chord']  = r_max_chord
        blade['ctrl_pts']['bladeLength']  = arc_length(blade['pf']['precurve'], blade['pf']['presweep'], blade['pf']['r'])[-1]

        return blade

    def update_planform(self, blade):

        if blade['ctrl_pts']['r_in'][2] != blade['ctrl_pts']['r_max_chord']:
            blade['ctrl_pts']['r_in'] = np.r_[[0.], [blade['ctrl_pts']['r_cylinder']], np.linspace(blade['ctrl_pts']['r_max_chord'], 1., self.NINPUT-2)]

        self.s                  = blade['pf']['s'] # TODO: assumes the start and end points of composite sections does not change
        blade['pf']['chord']    = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['chord_in'], self.s)
        blade['pf']['theta']    = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['theta_in'], self.s)
        blade['pf']['r']        = blade['ctrl_pts']['bladeLength']*np.array(self.s)
        blade['pf']['precurve'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['precurve_in'], self.s)
        blade['pf']['presweep'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['presweep_in'], self.s)

        blade['ctrl_pts']['bladeLength']  = arc_length(blade['pf']['precurve'], blade['pf']['presweep'], blade['pf']['r'])[-1]

        for var in self.spar_var:
            idx_spar  = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==var.lower()][0]
            blade['st']['layers'][idx_spar]['thickness']['grid']   = self.s.tolist()
            blade['st']['layers'][idx_spar]['thickness']['values'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['sparT_in'], self.s).tolist()

        idx_te    = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==self.te_var.lower()][0]
        blade['st']['layers'][idx_te]['thickness']['grid']   = self.s.tolist()
        blade['st']['layers'][idx_te]['thickness']['values'] = remap2grid(blade['ctrl_pts']['r_in'], blade['ctrl_pts']['teT_in'], self.s).tolist()

        return blade

        
    def convert_precomp(self, blade, materials_in=[]):

        ##############################
        def region_stacking(i, idx, start_nd_arc, end_nd_arc, blade, material_dict, materials, region_loc):
            # Recieve start and end of composite sections chordwise, find which composites layers are in each
            # chordwise regions, generate the precomp composite class instance

            # error handling to makes sure there were no numeric errors causing values very close too, but not exactly, 0 or 1
            start_nd_arc = [0. if start_nd_arci!=0. and np.isclose(start_nd_arci,0.) else start_nd_arci for start_nd_arci in start_nd_arc]
            end_nd_arc = [0. if end_nd_arci!=0. and np.isclose(end_nd_arci,0.) else end_nd_arci for end_nd_arci in end_nd_arc]
            start_nd_arc = [1. if start_nd_arci!=1. and np.isclose(start_nd_arci,1.) else start_nd_arci for start_nd_arci in start_nd_arc]
            end_nd_arc = [1. if end_nd_arci!=1. and np.isclose(end_nd_arci,1.) else end_nd_arci for end_nd_arci in end_nd_arc]

            # region end points
            dp = sorted(list(set(start_nd_arc+end_nd_arc)))

            #initialize
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            # loop through division points, find what layers make up the stack between those bounds
            for i_reg, (dp0, dp1) in enumerate(zip(dp[0:-1], dp[1:])):
                n_pliesi = []
                thki     = []
                thetai   = []
                mati     = []
                for i_sec, start_nd_arci, end_nd_arci in zip(idx, start_nd_arc, end_nd_arc):
                    name = blade['st']['layers'][i_sec]['name']
                    if start_nd_arci <= dp0 and end_nd_arci >= dp1:
                        
                        if name in region_loc.keys():
                            if region_loc[name][i] == None:
                                region_loc[name][i] = [i_reg]
                            else:
                                region_loc[name][i].append(i_reg)

                        n_pliesi.append(1.)
                        thki.append(blade['st']['layers'][i_sec]['thickness']['values'][i])
                        if blade['st']['layers'][i_sec]['fiber_orientation']['values'][i] == None:
                            thetai.append(0.)
                        else:
                            thetai.append(blade['st']['layers'][i_sec]['fiber_orientation']['values'][i])
                        mati.append(material_dict[blade['st']['layers'][i_sec]['material']])

                n_plies.append(np.array(n_pliesi))
                thk.append(np.array(thki))
                theta.append(np.array(thetai))
                mat_idx.append(np.array(mati))

            # print('----------------------')
            # print('dp', dp)
            # print('n_plies', n_plies)
            # print('thk', thk)
            # print('theta', theta)
            # print('mat_idx', mat_idx)
            # print('materials', materials)

            sec = CompositeSection(dp, n_plies, thk, theta, mat_idx, materials)
            return sec, region_loc
            ##############################

        def web_stacking(i, web_idx, web_start_nd_arc, web_end_nd_arc, blade, material_dict, materials, flatback, upperCSi):
            dp = []
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            if len(web_idx)>0:
                dp = np.mean((np.abs(web_start_nd_arc), np.abs(web_start_nd_arc)), axis=0).tolist()

                dp_all = [[-1.*start_nd_arci, -1.*end_nd_arci] for start_nd_arci, end_nd_arci in zip(web_start_nd_arc, web_end_nd_arc)]
                web_dp, web_ids = np.unique(dp_all, axis=0, return_inverse=True)
                for webi in np.unique(web_ids):
                    # store variable values (thickness, orientation, material) for layers that make up each web, based on the mapping array web_ids
                    n_pliesi = [1. for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thki     = [blade['st']['layers'][i_reg]['thickness']['values'][i] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thetai   = [blade['st']['layers'][i_reg]['fiber_orientation']['values'][i] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thetai   = [0. if theta_ij==None else theta_ij for theta_ij in thetai]
                    mati     = [material_dict[blade['st']['layers'][i_reg]['material']] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]

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
            ##############################

        ## Initialization
        if 'precomp' not in blade.keys():
            blade['precomp'] = {}

        region_loc_vars = [self.te_var] + self.spar_var
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
                if mati['orth'] == 1 or mati['orth'] == True:
                    try:
                        iter(mati['E'])
                    except:
                        warnings.warn('Ontology input warning: Material "%s" entered as Orthogonal, must supply E, G, and nu as a list representing the 3 principle axes.'%mati['name'])
                if 'G' not in mati.keys():
                    
                    if mati['orth'] == 1 or mati['orth'] == True:
                        warning_shear_modulus_orthogonal = 'Ontology input warning: No shear modulus, G, provided for material "%s".'%mati['name']
                        warnings.warn(warning_shear_modulus_orthogonal)
                    else:
                        warning_shear_modulus_isotropic = 'Ontology input warning: No shear modulus, G, provided for material "%s".  Assuming 2G*(1 + nu) = E, which is only valid for isotropic materials.'%mati['name']
                        warnings.warn(warning_shear_modulus_isotropic)
                        mati['G'] = mati['E']/(2*(1+mati['nu']))

                material_id = i
                material_dict[mati['name']] = material_id
                if mati['orth'] == 1 or mati['orth'] == True:
                    materials.append(Orthotropic2DMaterial(mati['E'][0], mati['E'][1], mati['G'][0], mati['nu'][0], mati['rho'], mati['name']))
                else:
                    materials.append(Orthotropic2DMaterial(mati['E'], mati['E'], mati['G'], mati['nu'], mati['rho'], mati['name']))
            blade['precomp']['materials']     = materials
            blade['precomp']['material_dict'] = material_dict

        
        upperCS = [None]*self.NPTS
        lowerCS = [None]*self.NPTS
        websCS  = [None]*self.NPTS
        profile = [None]*self.NPTS

        ## Spanwise

        for i in range(self.NPTS):
            # time0 = time.time()
        
            ## Profiles
            # rotate
            
            profile_i = np.flip(copy.copy(blade['profile'][:,:,i]), axis=0)
            profile_i_rot = np.column_stack(rotate(blade['pf']['p_le'][i], 0., profile_i[:,0], profile_i[:,1], -1.*np.radians(blade['pf']['theta'][i])))
            # normalize
            profile_i_rot[:,0] -= min(profile_i_rot[:,0])
            profile_i_rot = profile_i_rot/ max(profile_i_rot[:,0])

            profile_i_rot_precomp = copy.copy(profile_i_rot)
            idx_le_precomp = np.argmax(profile_i_rot_precomp[:,0])
            if idx_le_precomp != 0:
                if profile_i_rot_precomp[0,0] == profile_i_rot_precomp[-1,0]:
                     idx_s = 1
                profile_i_rot_precomp = np.row_stack((profile_i_rot_precomp[idx_le_precomp:], profile_i_rot_precomp[idx_s:idx_le_precomp,:]))
            profile_i_rot_precomp[:,1] -= profile_i_rot_precomp[np.argmin(profile_i_rot_precomp[:,0]),1]

            if profile_i_rot_precomp[-1,0] != 1.:
                profile_i_rot_precomp = np.row_stack((profile_i_rot_precomp, profile_i_rot_precomp[0,:]))

            # 'web' at trailing edge needed for flatback airfoils
            if profile_i_rot_precomp[0,1] != profile_i_rot_precomp[-1,1] and profile_i_rot_precomp[0,0] == profile_i_rot_precomp[-1,0]:
                flatback = True
            else:
                flatback = False

            profile[i] = Profile.initWithTEtoTEdata(profile_i_rot_precomp[:,0], profile_i_rot_precomp[:,1])

            # import matplotlib.pyplot as plt
            # plt.plot(profile_i_rot_precomp[:,0], profile_i_rot_precomp[:,1])
            # plt.axis('equal')
            # plt.show()

            idx_le = np.argmin(profile_i_rot[:,0])

            profile_i_arc = arc_length(profile_i_rot[:,0], profile_i_rot[:,1])
            arc_L = profile_i_arc[-1]
            profile_i_arc /= arc_L

            loc_LE = profile_i_arc[idx_le]
            len_PS = 1.-loc_LE

            ## Composites
            ss_idx           = []
            ss_start_nd_arc  = []
            ss_end_nd_arc    = []
            ps_idx           = []
            ps_start_nd_arc  = []
            ps_end_nd_arc    = []
            web_start_nd_arc = []
            web_end_nd_arc   = []
            web_idx          = []

            # Determine spanwise composite layer elements that are non-zero at this spanwise location,
            # determine their chord-wise start and end location on the pressure and suctions side

            spline_arc2xnd = PchipInterpolator(profile_i_arc, profile_i_rot[:,0])

            time1 = time.time()
            for idx_sec, sec in enumerate(blade['st']['layers']):

                if 'web' not in sec.keys():
                    if sec['start_nd_arc']['values'][i] != None and sec['thickness']['values'][i] != None:
                        if sec['start_nd_arc']['values'][i] < loc_LE or sec['end_nd_arc']['values'][i] < loc_LE:
                            ss_idx.append(idx_sec)
                            if sec['start_nd_arc']['values'][i] < loc_LE:
                                # ss_start_nd_arc.append(sec['start_nd_arc']['values'][i])
                                ss_end_nd_arc_temp = float(spline_arc2xnd(sec['start_nd_arc']['values'][i]))
                                if ss_end_nd_arc_temp == profile_i_rot[0,0] and profile_i_rot[0,0] != 1.:
                                    ss_end_nd_arc_temp = 1.
                                ss_end_nd_arc.append(ss_end_nd_arc_temp)
                            else:
                                ss_end_nd_arc.append(1.)
                            # ss_end_nd_arc.append(min(sec['end_nd_arc']['values'][i], loc_LE)/loc_LE)
                            if sec['end_nd_arc']['values'][i] < loc_LE:
                                ss_start_nd_arc.append(float(spline_arc2xnd(sec['end_nd_arc']['values'][i])))
                            else:
                                ss_start_nd_arc.append(0.)
                            
                        if sec['start_nd_arc']['values'][i] > loc_LE or sec['end_nd_arc']['values'][i] > loc_LE:
                            ps_idx.append(idx_sec)
                            # ps_start_nd_arc.append((max(sec['start_nd_arc']['values'][i], loc_LE)-loc_LE)/len_PS)
                            # ps_end_nd_arc.append((min(sec['end_nd_arc']['values'][i], 1.)-loc_LE)/len_PS)

                            if sec['start_nd_arc']['values'][i] > loc_LE and sec['end_nd_arc']['values'][i] < loc_LE:
                                # ps_start_nd_arc.append(float(remap2grid(profile_i_arc, profile_i_rot[:,0], sec['start_nd_arc']['values'][i])))
                                ps_end_nd_arc.append(1.)
                            else:
                                ps_end_nd_arc_temp = float(spline_arc2xnd(sec['end_nd_arc']['values'][i]))
                                if ps_end_nd_arc_temp == profile_i_rot[-1,0] and profile_i_rot[-1,0] != 1.:
                                    ps_end_nd_arc_temp = 1.
                                ps_end_nd_arc.append(ps_end_nd_arc_temp)
                            if sec['start_nd_arc']['values'][i] < loc_LE:
                                ps_start_nd_arc.append(0.)
                            else:
                                ps_start_nd_arc.append(float(spline_arc2xnd(sec['start_nd_arc']['values'][i])))


                else:
                    target_name  = blade['st']['layers'][idx_sec]['web']
                    target_idx   = [k for k, webi in enumerate(blade['st']['webs']) if webi['name']==target_name][0]

                    if blade['st']['webs'][target_idx]['start_nd_arc']['values'][i] != None and blade['st']['layers'][idx_sec]['thickness']['values'][i] != None:
                        web_idx.append(idx_sec)

                        start_nd_arc = float(spline_arc2xnd(blade['st']['webs'][target_idx]['start_nd_arc']['values'][i]))
                        end_nd_arc   = float(spline_arc2xnd(blade['st']['webs'][target_idx]['end_nd_arc']['values'][i]))

                        web_start_nd_arc.append(start_nd_arc)
                        web_end_nd_arc.append(end_nd_arc)


            time1 = time.time() - time1
            # print(time1)

            # generate the Precomp composite stacks for chordwise regions
            upperCS[i], region_loc_ss = region_stacking(i, ss_idx, ss_start_nd_arc, ss_end_nd_arc, blade, blade['precomp']['material_dict'], blade['precomp']['materials'], region_loc_ss)
            lowerCS[i], region_loc_ps = region_stacking(i, ps_idx, ps_start_nd_arc, ps_end_nd_arc, blade, blade['precomp']['material_dict'], blade['precomp']['materials'], region_loc_ps)
            if len(web_idx)>0 or flatback:
                websCS[i] = web_stacking(i, web_idx, web_start_nd_arc, web_end_nd_arc, blade, blade['precomp']['material_dict'], blade['precomp']['materials'], flatback, upperCS[i])
            else:
                websCS[i] = CompositeSection([], [], [], [], [], [])


        blade['precomp']['upperCS']       = upperCS
        blade['precomp']['lowerCS']       = lowerCS
        blade['precomp']['websCS']        = websCS
        blade['precomp']['profile']       = profile

        # Assumptions:
        # - pressure and suction side regions are the same (i.e. spar cap is the Nth region on both side)
        # - if the composite layer is divided into multiple regions (i.e. if the spar cap is split into 3 regions due to the web locations),
        #   the middle region is selected with int(n_reg/2), note for an even number of regions, this rounds up
        blade['precomp']['sector_idx_strain_spar'] = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.spar_var[0]]]
        blade['precomp']['sector_idx_strain_te']   = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.te_var]]
        blade['precomp']['spar_var'] = self.spar_var
        blade['precomp']['te_var']   = self.te_var
        
        return blade

    def plot_design(self, blade, path, show_plots = True):
        
        import matplotlib.pyplot as plt
        
        # Chord
        fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
        axc.plot(blade['pf']['s'], blade['pf']['chord'])
        axc.set(xlabel='r/R' , ylabel='Chord (m)')
        fig_name = 'init_chord.png'
        fc.savefig(path + fig_name)
        
        # Theta
        ft, axt  = plt.subplots(1,1,figsize=(5.3, 4))
        axt.plot(blade['pf']['s'], blade['pf']['theta'])
        axt.set(xlabel='r/R' , ylabel='Twist (deg)')
        fig_name = 'init_theta.png'
        ft.savefig(path + fig_name)
        
        # Pitch axis
        fp, axp  = plt.subplots(1,1,figsize=(5.3, 4))
        axp.plot(blade['pf']['s'], blade['pf']['p_le']*100.)
        axp.set(xlabel='r/R' , ylabel='Pitch Axis (%)')
        fig_name = 'init_p_le.png'
        fp.savefig(path + fig_name)
        
        
        # Planform
        le = blade['pf']['p_le']*blade['pf']['chord']
        te = (1. - blade['pf']['p_le'])*blade['pf']['chord']

        fpl, axpl  = plt.subplots(1,1,figsize=(5.3, 4))
        axpl.plot(blade['pf']['s'], -le)
        axpl.plot(blade['pf']['s'], te)
        axpl.set(xlabel='r/R' , ylabel='Planform (m)')
        axpl.legend()
        fig_name = 'init_planform.png'
        fpl.savefig(path + fig_name)
        
        
        
        # Relative thickness
        frt, axrt  = plt.subplots(1,1,figsize=(5.3, 4))
        axrt.plot(blade['pf']['s'], blade['pf']['rthick']*100.)
        axrt.set(xlabel='r/R' , ylabel='Relative Thickness (%)')
        fig_name = 'init_rthick.png'
        frt.savefig(path + fig_name)
        
        # Absolute thickness
        fat, axat  = plt.subplots(1,1,figsize=(5.3, 4))
        axat.plot(blade['pf']['s'], blade['pf']['rthick']*blade['pf']['chord'])
        axat.set(xlabel='r/R' , ylabel='Absolute Thickness (m)')
        fig_name = 'init_absthick.png'
        fat.savefig(path + fig_name)
        
        # Prebend
        fpb, axpb  = plt.subplots(1,1,figsize=(5.3, 4))
        axpb.plot(blade['pf']['s'], blade['pf']['precurve'])
        axpb.set(xlabel='r/R' , ylabel='Prebend (m)')
        fig_name = 'init_prebend.png'
        fpb.savefig(path + fig_name)
        
        # Sweep
        fsw, axsw  = plt.subplots(1,1,figsize=(5.3, 4))
        axsw.plot(blade['pf']['s'], blade['pf']['presweep'])
        axsw.set(xlabel='r/R' , ylabel='Presweep (m)')
        fig_name = 'init_presweep.png'
        plt.subplots_adjust(left = 0.14)
        fsw.savefig(path + fig_name)
        
        idx_spar  = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==self.spar_var[0].lower()][0]
        idx_te    = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==self.te_var.lower()][0]
        idx_skin  = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()=='shell_skin'][0]
        
        # Spar caps thickness
        fsc, axsc  = plt.subplots(1,1,figsize=(5.3, 4))
        axsc.plot(blade['st']['layers'][idx_spar]['thickness']['grid'], blade['st']['layers'][idx_spar]['thickness']['values'])
        axsc.set(xlabel='r/R' , ylabel='Spar Caps Thickness (m)')
        fig_name = 'init_sc.png'
        plt.subplots_adjust(left = 0.14)
        fsc.savefig(path + fig_name)
        
        # TE reinf thickness
        fte, axte  = plt.subplots(1,1,figsize=(5.3, 4))
        axte.plot(blade['st']['layers'][idx_te]['thickness']['grid'], blade['st']['layers'][idx_te]['thickness']['values'])
        axte.set(xlabel='r/R' , ylabel='TE Reinf. Thickness (m)')
        fig_name = 'init_te.png'
        plt.subplots_adjust(left = 0.14)
        fte.savefig(path + fig_name)
        
        # Skin
        fsk, axsk  = plt.subplots(1,1,figsize=(5.3, 4))
        axsk.plot(blade['st']['layers'][idx_skin]['thickness']['grid'], blade['st']['layers'][idx_skin]['thickness']['values'])
        axsk.set(xlabel='r/R' , ylabel='Shell Skin Thickness (m)')
        fig_name = 'init_skin.png'
        fsk.savefig(path + fig_name)
        
        
        if show_plots:
            plt.show()
        
        
        return None        

        
    def smooth_outer_shape(self, blade, path, show_plots = True):
        
        s               = blade['pf']['s']        
        
        # Absolute Thickness
        abs_thick_init  = blade['pf']['rthick']*blade['pf']['chord']
        s_interp_at     = np.array([0.0, 0.15, 0.4, 0.6, 0.8, 1.0 ])
        f_interp1       = interp1d(s,abs_thick_init)
        abs_thick_int1  = f_interp1(s_interp_at)
        f_interp2       = PchipInterpolator(s_interp_at,abs_thick_int1)
        abs_thick_int2  = f_interp2(s)
        
        import matplotlib.pyplot as plt
        
        
        
        # Chord
        chord_init      = blade['pf']['chord']
        s_interp_c      = np.array([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0 ])
        f_interp1       = interp1d(s,chord_init)
        chord_int1      = f_interp1(s_interp_c)
        f_interp2       = PchipInterpolator(s_interp_c,chord_int1)
        chord_int2      = f_interp2(s)
        
        fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
        axc.plot(s, chord_init, c='k', label='Initial')
        axc.plot(s_interp_c, chord_int1, 'ko', label='Interp Points')
        axc.plot(s, chord_int2, c='b', label='PCHIP')
        axc.set(xlabel='r/R' , ylabel='Chord (m)')
        fig_name = 'interp_chord.png'
        axc.legend()
        fc.savefig(path + fig_name)
        

        # Relative thickness
        r_thick_interp = abs_thick_int2 / chord_int2
        r_thick_airfoils = np.array([0.18, 0.211, 0.241, 0.301, 0.36 , 0.50, 1.00])
        f_interp1        = interp1d(r_thick_interp,s)
        s_interp_rt      = f_interp1(r_thick_airfoils)
        f_interp2        = PchipInterpolator(np.flip(s_interp_rt),np.flip(r_thick_airfoils))
        r_thick_int2     = f_interp2(s)
        
        
        frt, axrt  = plt.subplots(1,1,figsize=(5.3, 4))
        axrt.plot(blade['pf']['s'], blade['pf']['rthick']*100., c='k', label='Initial')
        axrt.plot(blade['pf']['s'], r_thick_interp * 100., c='b', label='Interp')
        axrt.plot(s_interp_rt, r_thick_airfoils * 100., 'og', label='Airfoils')
        axrt.plot(blade['pf']['s'], r_thick_int2 * 100., c='g', label='Reconstructed')
        axrt.set(xlabel='r/R' , ylabel='Relative Thickness (%)')
        fig_name = 'interp_rthick.png'
        axrt.legend()
        frt.savefig(path + fig_name)

        
        fat, axat  = plt.subplots(1,1,figsize=(5.3, 4))
        axat.plot(s, abs_thick_init, c='k', label='Initial')
        axat.plot(s_interp_at, abs_thick_int1, 'ko', label='Interp Points')
        axat.plot(s, abs_thick_int2, c='b', label='PCHIP')
        axat.plot(s, r_thick_int2 * chord_int2, c='g', label='Reconstructed')
        axat.set(xlabel='r/R' , ylabel='Absolute Thickness (m)')
        fig_name = 'interp_abs_thick.png'
        axat.legend()
        fat.savefig(path + fig_name)
        
        
        
        # Planform
        le_init = blade['pf']['p_le']*blade['pf']['chord']
        te_init = (1. - blade['pf']['p_le'])*blade['pf']['chord']
        
        s_interp_le     = np.array([0.0, 0.5, 0.8, 0.9, 1.0])
        f_interp1       = interp1d(s,le_init)
        le_int1         = f_interp1(s_interp_le)
        f_interp2       = PchipInterpolator(s_interp_le,le_int1)
        le_int2         = f_interp2(s)
        
        fpl, axpl  = plt.subplots(1,1,figsize=(5.3, 4))
        axpl.plot(blade['pf']['s'], -le_init, c='k', label='LE init')
        axpl.plot(blade['pf']['s'], -le_int2, c='b', label='LE smooth')
        axpl.plot(blade['pf']['s'], te_init, c='g', label='TE init')
        axpl.plot(blade['pf']['s'], blade['pf']['chord'] - le_int2, c='r', label='TE smooth')
        axpl.set(xlabel='r/R' , ylabel='Planform (m)')
        axpl.legend()
        fig_name = 'interp_planform.png'
        fpl.savefig(path + fig_name)
        
        
        print(le_int2/blade['pf']['chord'])
        
        
        
        if show_plots:
            plt.show()
        
        
        # print(chord_int2)
        # print(s_interp_rt)
        # exit()
        
        
        return None
        
        
        
if __name__ == "__main__":

    ## File managment
    # fname_input        = "turbine_inputs/nrel5mw_mod_update.yaml"
    fname_input        = "turbine_inputs/BAR15_clean.yaml"
    # fname_input        = "turbine_inputs/IEAonshoreWT.yaml"
    # fname_input        = "turbine_inputs/test_out.yaml"
    fname_output       = "turbine_inputs/test_out_BAR.yaml"
    flag_write_out     = True
    flag_write_precomp = False
    dir_precomp_out    = "turbine_inputs/precomp"

    ## Load and Format Blade
    tt = time.time()
    refBlade = ReferenceBlade()
    refBlade.verbose  = True
    refBlade.spar_var = ['Spar_cap_ss', 'Spar_cap_ps']
    refBlade.te_var   = 'TE_reinforcement'
    refBlade.NPTS     = 50
    refBlade.validate = False
    refBlade.fname_schema = "turbine_inputs/IEAontology_schema.yaml"

    blade = refBlade.initialize(fname_input)

    ## save output yaml
    if flag_write_out:
        t3 = time.time()
        refBlade.write_ontology(fname_output, blade, refBlade.wt_ref)
        if refBlade.verbose:
            print('Complete: Write Output: \t%f s'%(time.time()-t3))

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
            print('Complete: Write PreComp: \t%f s'%(time.time()-t4))