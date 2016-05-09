# from __future__ import print_function

import numpy as np
import math
from openmdao.api import IndepVarComp, Component, ExecComp, Group, ScipyGMRES
from rotoraero import SetupRunVarSpeed, RegulatedPowerCurve, AEP, \
    RPM2RS, RS2RPM, RegulatedPowerCurveGroup, COE

from rotoraerodefaults import CCBladeGeometry, CSMDrivetrain, RayleighCDF, WeibullWithMeanCDF, RayleighCDF, CCBlade, CCBladeAirfoils, AirfoilSpline

from scipy.interpolate import RectBivariateSpline
from akima import Akima, akima_interp_with_derivs
from csystem import DirectionVector
from utilities import hstack, vstack, trapz_deriv, interp_with_deriv
from environment import PowerWind
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
import _pBEAM
import _curvefem
import _bem  # TODO: move to rotoraero
from enum import Enum
# from ccblade2 import CCBlade_to_RotorSE_connection as CCBlade
from airfoil_parameterization import AirfoilAnalysis



#######################
##  BASE COMPONENTS  ##
#######################


class ResizeCompositeSection(Component):
    def __init__(self, nstr, num_airfoils, airfoils_dof):
        super(ResizeCompositeSection, self).__init__()

        self.add_param('upperCSIn', shape=nstr, desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_param('lowerCSIn', shape=nstr, desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_param('websCSIn', shape=nstr, desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)

        self.add_param('chord_str_ref', shape=nstr, units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)')
        self.add_param('thick_str_ref', shape=nstr, units='m', desc='airfoil thickness distribution for reference section, thickness of structural layup scaled with reference thickness')
        self.add_param('afp_str', val=np.zeros((nstr, airfoils_dof)))
        self.add_param('idx_cylinder_str', val=1, pass_by_obj=True)

        self.add_param('sector_idx_strain_spar', val=np.zeros(nstr, dtype=np.int), desc='index of sector for spar (PreComp definition of sector)', pass_by_obj=True)
        self.add_param('sector_idx_strain_te', val=np.zeros(nstr, dtype=np.int), desc='index of sector for trailing-edge (PreComp definition of sector)', pass_by_obj=True)

        self.add_param('chord_str', shape=nstr, units='m', desc='structural chord distribution')
        self.add_param('sparT_str', shape=nstr, units='m', desc='structural spar cap thickness distribution')
        self.add_param('teT_str', shape=nstr, units='m', desc='structural trailing-edge panel thickness distribution')

        # out
        self.add_output('upperCSOut', shape=nstr, desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_output('lowerCSOut', shape=nstr, desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_output('websCSOut', shape=nstr, desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)
        self.add_output('dummy', shape=1)
        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        # print "ResizeCompositeSection"
        chord_str_ref = params['chord_str_ref']
        thick_str_ref = params['thick_str_ref']
        upperCSIn = params['upperCSIn']
        lowerCSIn = params['lowerCSIn']
        websCSIn = params['websCSIn']
        chord_str = params['chord_str']
        sector_idx_strain_spar = params['sector_idx_strain_spar']
        sector_idx_strain_te = params['sector_idx_strain_te']
        sparT_str = params['sparT_str']
        teT_str = params['teT_str']

        nstr = len(chord_str_ref)

        # copy data acrosss
        upperCSOut = []
        lowerCSOut = []
        websCSOut = []

        for i in range(nstr):
            upperCSOut.append(upperCSIn[i].mycopy())
            lowerCSOut.append(lowerCSIn[i].mycopy())
            websCSOut.append(websCSIn[i].mycopy())

        # scale all thicknesses with airfoil thickness
        for i in range(nstr):

            upper = upperCSOut[i]
            lower = lowerCSOut[i]
            webs = websCSOut[i]

            # factor = t_str[i]/tref[i]
            # if i < params['idx_cylinder_str']:
            factor = chord_str[i]/chord_str_ref[i]  # same as thickness ratio for constant t/c
            # else:
            #     from airfoilprep_free import getCoordinates
            #     xl, xu, yl, yu = getCoordinates(params['afp_str'][i])
            #     t_str = max(abs(yl)) + max(abs(yu))
            #     factor = t_str/thick_str_ref[i]

            for j in range(len(upper.t)):
                upper.t[j] *= factor

            for j in range(len(lower.t)):
                lower.t[j] *= factor

            for j in range(len(webs.t)):
                webs.t[j] *= factor


        # change spar and trailing edge thickness to specified values
        for i in range(nstr):

            idx_spar = sector_idx_strain_spar[i]
            idx_te = sector_idx_strain_te[i]
            upper = upperCSOut[i]
            lower = lowerCSOut[i]

            # upper and lower have same thickness for this design
            tspar = np.sum(upper.t[idx_spar])
            tte = np.sum(upper.t[idx_te])

            upper.t[idx_spar] *= sparT_str[i]/tspar
            lower.t[idx_spar] *= sparT_str[i]/tspar

            upper.t[idx_te] *= teT_str[i]/tte
            lower.t[idx_te] *= teT_str[i]/tte

        unknowns['upperCSOut'] = upperCSOut
        unknowns['lowerCSOut'] = lowerCSOut
        unknowns['websCSOut'] = websCSOut
        unknowns['dummy'] = 0.0

    def linearize(self, params, unknowns, resids):
        J = {}
        J['dummy', 'chord_str_ref'] = np.zeros((1, len(params['chord_str_ref'])))
        return J

class PreCompSections(Component):
    def __init__(self, nstr, num_airfoils, airfoils_dof):
        super(PreCompSections, self).__init__()
        self.add_param('r', shape=nstr, units='m', desc='radial positions. r[0] should be the hub location \
            while r[-1] should be the blade tip. Any number \
            of locations can be specified between these in ascending order.')
        self.add_param('chord', shape=nstr, units='m', desc='array of chord lengths at corresponding radial positions')
        self.add_param('theta', shape=nstr, units='deg', desc='array of twist angles at corresponding radial positions. \
            (positive twist decreases angle of attack)')
        self.add_param('leLoc', shape=nstr, desc='array of leading-edge positions from a reference blade axis \
            (usually blade pitch axis). locations are normalized by the local chord length.  \
            e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
            positive in -x direction for airfoil-aligned coordinate system')
        self.add_param('profile', shape=nstr, desc='airfoil shape at each radial position', pass_by_obj=True)
        self.add_param('materials', desc='list of all Orthotropic2DMaterial objects used in defining the geometry', pass_by_obj=True)
        self.add_param('upperCS', val=np.zeros(nstr), desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_param('lowerCS', shape=nstr, desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_param('websCS', shape=nstr, desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)
        self.add_param('sector_idx_strain_spar', val=np.zeros(nstr, dtype=np.int), desc='index of sector for spar (PreComp definition of sector)', pass_by_obj=True)
        self.add_param('sector_idx_strain_te', val=np.zeros(nstr, dtype=np.int), desc='index of sector for trailing-edge (PreComp definition of sector)', pass_by_obj=True)

        self.add_param('airfoil_parameterization', val=np.zeros((num_airfoils, airfoils_dof)))
        self.add_param('af_str_idx', val=np.zeros(nstr), pass_by_obj=True)
        self.add_param('airfoil_analysis_options', val={}, pass_by_obj=True)


        self.add_output('eps_crit_spar', shape=nstr, desc='critical strain in spar from panel buckling calculation')
        self.add_output('eps_crit_te', shape=nstr, desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_output('xu_strain_spar', shape=nstr, desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('xl_strain_spar', shape=nstr, desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('yu_strain_spar', shape=nstr, desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('yl_strain_spar', shape=nstr, desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('xu_strain_te', shape=nstr, desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('xl_strain_te', shape=nstr, desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_output('yu_strain_te', shape=nstr, desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('yl_strain_te', shape=nstr, desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')

        self.add_output('beam:z', shape=nstr, units='m', desc='locations of properties along beam')
        self.add_output('beam:EA', shape=nstr, units='N', desc='axial stiffness')
        self.add_output('beam:EIxx', shape=nstr, units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_output('beam:EIyy', shape=nstr, units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_output('beam:EIxy', shape=nstr, units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_output('beam:GJ', shape=nstr, units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_output('beam:rhoA', shape=nstr, units='kg/m', desc='mass per unit length')
        self.add_output('beam:rhoJ', shape=nstr, units='kg*m', desc='polar mass moment of inertia per unit length')
        self.add_output('beam:x_ec_str', shape=nstr, units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_output('beam:y_ec_str', shape=nstr, units='m', desc='y-distance to elastic center from point about which above structural properties are computed')

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.nstr = nstr
        self.num_airfoils = num_airfoils
        self.airfoils_dof = airfoils_dof


    def criticalStrainLocations(self, sector_idx_strain, x_ec_nose, y_ec_nose):

        n = len(sector_idx_strain)

        # find corresponding locations on airfoil at midpoint of sector
        xun = np.zeros(n)
        xln = np.zeros(n)
        yun = np.zeros(n)
        yln = np.zeros(n)

        for i in range(n):
            csU = self.upperCS[i]
            csL = self.lowerCS[i]
            pf = self.profile[i]
            idx = sector_idx_strain[i]

            xun[i] = 0.5*(csU.loc[idx] + csU.loc[idx+1])
            xln[i] = 0.5*(csL.loc[idx] + csL.loc[idx+1])
            yun[i] = np.interp(xun[i], pf.x, pf.yu)
            yln[i] = np.interp(xln[i], pf.x, pf.yl)

        # make dimensional and define relative to elastic center
        xu = xun*self.chord - x_ec_nose
        xl = xln*self.chord - x_ec_nose
        yu = yun*self.chord - y_ec_nose
        yl = yln*self.chord - y_ec_nose


        # switch to airfoil coordinate system
        xu, yu = yu, xu
        xl, yl = yl, xl

        return xu, xl, yu, yl


    def panelBucklingStrain(self, sector_idx_strain):
        """
        see chapter on Structural Component Design Techniques from Alastair Johnson
        section 6.2: Design of composite panels

        assumes: large aspect ratio, simply supported, uniaxial compression, flat rectangular plate

        """

        # rename
        chord = self.chord
        CS_list = self.upperCS  # TODO: assumes the upper surface is the compression one

        # initialize
        nsec = len(chord)
        eps_crit = np.zeros(nsec)

        for i in range(nsec):

            cs = CS_list[i]
            sector_idx = sector_idx_strain[i]

            # chord-wise length of sector
            sector_length = chord[i] * (cs.loc[sector_idx+1] - cs.loc[sector_idx])

            # get matrices
            A, B, D, totalHeight = cs.compositeMatrices(sector_idx)
            E = cs.effectiveEAxial(sector_idx)
            D1 = D[0, 0]
            D2 = D[1, 1]
            D3 = D[0, 1] + 2*D[2, 2]

            # use empirical formula
            Nxx = 2 * (math.pi/sector_length)**2 * (math.sqrt(D1*D2) + D3)

            eps_crit[i] = - Nxx / totalHeight / E

        return eps_crit

    def solve_nonlinear(self, params, unknowns, resids):
        # print "PreCompSections"
        self.chord = params['chord']
        self.materials = params['materials']
        self.r = params['r']
        self.profile = params['profile']
        self.upperCS = params['upperCS']
        self.lowerCS = params['lowerCS']
        self.websCS = params['websCS']
        self.theta = params['theta']
        self.leLoc = params['leLoc']
        self.sector_idx_strain_spar = params['sector_idx_strain_spar']
        self.sector_idx_strain_te = params['sector_idx_strain_te']

        # radial discretization
        nsec = len(self.r)

        # initialize variables
        self.beam_z = self.r
        self.beam_EA = np.zeros(nsec)
        self.beam_EIxx = np.zeros(nsec)
        self.beam_EIyy = np.zeros(nsec)
        self.beam_EIxy = np.zeros(nsec)
        self.beam_GJ = np.zeros(nsec)
        self.beam_rhoA = np.zeros(nsec)
        self.beam_rhoJ = np.zeros(nsec)

        # distance to elastic center from point about which structural properties are computed
        # using airfoil coordinate system
        self.beam_x_ec_str = np.zeros(nsec)
        self.beam_y_ec_str = np.zeros(nsec)

        # distance to elastic center from airfoil nose
        # using profile coordinate system
        x_ec_nose = np.zeros(nsec)
        y_ec_nose = np.zeros(nsec)

        profile = self.profile
        nstr = self.nstr

        if params['airfoil_analysis_options']['AnalysisMethod'] != 'Files':
            airfoil_parameterization = params['airfoil_parameterization']
            af_str_idx = params['af_str_idx']
            airfoil_types_str = np.zeros((8, self.airfoils_dof))
            for z in range(6):
                airfoil_types_str[z+2, :] = airfoil_parameterization[z]
            import os
            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PreCompFiles')

            pro_str = [0]*nstr
            for i in range(nstr):
                pro_str[i] = airfoil_types_str[af_str_idx[i]]
            profile = [0]*nstr
            for j in range(nstr):
                if pro_str[j][0] == 0.0:
                    profile[j] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(j+1) + '.inp'))
                else:
                    if params['airfoil_analysis_options']['AirfoilParameterization'] != 'Precomputational:T/C':
                        afanalysis = AirfoilAnalysis(pro_str[j], params['airfoil_analysis_options'])
                        xl, xu, yl, yu = afanalysis.getCoordinates(type='split')
                    else:
                        afanalysis = AirfoilAnalysis(params['airfoil_analysis_options']['BaseAirfoil'], params['airfoil_analysis_options'], computeModel=False)
                        xl, xu, yl, yu = afanalysis.getPreCompCoordinates(pro_str[j])
                    # xl, xu, yl, yu = getCoordinates([pro_str[j]])
                    xu1 = np.zeros(len(xu))
                    xl1 = np.zeros(len(xl))
                    yu1 = np.zeros(len(xu))
                    yl1 = np.zeros(len(xl))
                    for k in range(len(xu)):
                        xu1[k] = float(xu[k])
                        yu1[k] = float(yu[k])
                    for k in range(len(xl)):
                        xl1[k] = float(xl[k])
                        yl1[k] = float(yl[k])
                    x = np.append(xu1, xl1)
                    y = np.append(yu1, yl1)

                    profile[j] = Profile.initFromCoordinates(x, y)
        mat = self.materials
        csU = self.upperCS
        csL = self.lowerCS
        csW = self.websCS

        # twist rate
        th_prime = _precomp.tw_rate(self.r, self.theta)

        # arrange materials into list
        n = len(mat)
        E1 = [0]*n
        E2 = [0]*n
        G12 = [0]*n
        nu12 = [0]*n
        rho = [0]*n

        for i in range(n):
            E1[i] = mat[i].E1
            E2[i] = mat[i].E2
            G12[i] = mat[i].G12
            nu12[i] = mat[i].nu12
            rho[i] = mat[i].rho

        for i in range(nsec):

            xnode, ynode = profile[i]._preCompFormat()
            locU, n_laminaU, n_pliesU, tU, thetaU, mat_idxU = csU[i]._preCompFormat()
            locL, n_laminaL, n_pliesL, tL, thetaL, mat_idxL = csL[i]._preCompFormat()
            locW, n_laminaW, n_pliesW, tW, thetaW, mat_idxW = csW[i]._preCompFormat()

            nwebs = len(locW)

            # address a bug in f2py (need to pass in length 1 arrays even though they are not used)
            if nwebs == 0:
                locW = [0]
                n_laminaW = [0]
                n_pliesW = [0]
                tW = [0]
                thetaW = [0]
                mat_idxW = [0]


            results = _precomp.properties(self.chord[i], self.theta[i],
                th_prime[i], self.leLoc[i],
                xnode, ynode, E1, E2, G12, nu12, rho,
                locU, n_laminaU, n_pliesU, tU, thetaU, mat_idxU,
                locL, n_laminaL, n_pliesL, tL, thetaL, mat_idxL,
                nwebs, locW, n_laminaW, n_pliesW, tW, thetaW, mat_idxW)


            self.beam_EIxx[i] = results[1]  # EIedge
            self.beam_EIyy[i] = results[0]  # EIflat
            self.beam_GJ[i] = results[2]
            self.beam_EA[i] = results[3]
            self.beam_EIxy[i] = results[4]  # EIflapedge
            self.beam_x_ec_str[i] = results[12] - results[10]
            self.beam_y_ec_str[i] = results[13] - results[11]
            self.beam_rhoA[i] = results[14]
            self.beam_rhoJ[i] = results[15] + results[16]  # perpindicular axis theorem

            x_ec_nose[i] = results[13] + self.leLoc[i]*self.chord[i]
            y_ec_nose[i] = results[12]  # switch b.c of coordinate system used

        unknowns['beam:z'] = self.beam_z
        unknowns['beam:EIxx'] = self.beam_EIxx
        unknowns['beam:EIyy'] = self.beam_EIyy
        unknowns['beam:GJ'] = self.beam_GJ
        unknowns['beam:EA'] = self.beam_EA
        unknowns['beam:EIxy'] = self.beam_EIxy
        unknowns['beam:x_ec_str'] = self.beam_x_ec_str
        unknowns['beam:y_ec_str'] = self.beam_y_ec_str
        unknowns['beam:rhoA'] = self.beam_rhoA
        unknowns['beam:rhoJ'] = self.beam_rhoJ
        unknowns['eps_crit_spar'] = self.panelBucklingStrain(self.sector_idx_strain_spar)
        unknowns['eps_crit_te'] = self.panelBucklingStrain(self.sector_idx_strain_te)

        xu_strain_spar, xl_strain_spar, yu_strain_spar, \
            yl_strain_spar = self.criticalStrainLocations(self.sector_idx_strain_spar, x_ec_nose, y_ec_nose)
        xu_strain_te, xl_strain_te, yu_strain_te, \
            yl_strain_te = self.criticalStrainLocations(self.sector_idx_strain_te, x_ec_nose, y_ec_nose)

        unknowns['xu_strain_spar'] = xu_strain_spar
        unknowns['xl_strain_spar'] = xl_strain_spar
        unknowns['yu_strain_spar'] = yu_strain_spar
        unknowns['yl_strain_spar'] = yl_strain_spar
        unknowns['xu_strain_te'] = xu_strain_te
        unknowns['xl_strain_te'] = xl_strain_te
        unknowns['yu_strain_te'] = yu_strain_te
        unknowns['yl_strain_te'] = yl_strain_te

class RotorWithpBEAM(Component):

    def __init__(self, nstr):
        super(RotorWithpBEAM, self).__init__()

        self.add_param('nF', val=5, desc='number of natural frequencies to return', pass_by_obj=True)

        self.add_param('Px_defl', shape=nstr, desc='distributed load (force per unit length) in airfoil x-direction at max deflection condition')
        self.add_param('Py_defl', shape=nstr, desc='distributed load (force per unit length) in airfoil y-direction at max deflection condition')
        self.add_param('Pz_defl', shape=nstr, desc='distributed load (force per unit length) in airfoil z-direction at max deflection condition')

        self.add_param('Px_strain', shape=nstr, desc='distributed load (force per unit length) in airfoil x-direction at max strain condition')
        self.add_param('Py_strain', shape=nstr, desc='distributed load (force per unit length) in airfoil y-direction at max strain condition')
        self.add_param('Pz_strain', shape=nstr, desc='distributed load (force per unit length) in airfoil z-direction at max strain condition')

        self.add_param('Px_pc_defl', shape=nstr, desc='distributed load (force per unit length) in airfoil x-direction for deflection used in generated power curve')
        self.add_param('Py_pc_defl', shape=nstr, desc='distributed load (force per unit length) in airfoil y-direction for deflection used in generated power curve')
        self.add_param('Pz_pc_defl', shape=nstr, desc='distributed load (force per unit length) in airfoil z-direction for deflection used in generated power curve')

        self.add_param('xu_strain_spar', shape=nstr, desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_param('xl_strain_spar', shape=nstr, desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_param('yu_strain_spar', shape=nstr, desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_param('yl_strain_spar', shape=nstr, desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_param('xu_strain_te', shape=nstr, desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_param('xl_strain_te', shape=nstr, desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_param('yu_strain_te', shape=nstr, desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_param('yl_strain_te', shape=nstr, desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')

        self.add_param('Mx_damage', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil x-direction')
        self.add_param('My_damage', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil y-direction')
        self.add_param('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap')
        self.add_param('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels')
        self.add_param('eta_damage', val=1.755, desc='safety factor for fatigue')
        self.add_param('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis')
        self.add_param('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis')

        self.add_param('beam:z', shape=nstr, units='m', desc='locations of properties along beam')
        self.add_param('beam:EA', shape=nstr, units='N', desc='axial stiffness')
        self.add_param('beam:EIxx', shape=nstr, units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_param('beam:EIyy', shape=nstr, units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_param('beam:EIxy', shape=nstr, units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_param('beam:GJ', shape=nstr, units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_param('beam:rhoA', shape=nstr, units='kg/m', desc='mass per unit length')
        self.add_param('beam:rhoJ', shape=nstr, units='kg*m', desc='polar mass moment of inertia per unit length')
        self.add_param('beam:x_ec_str', shape=nstr, units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_param('beam:y_ec_str', shape=nstr, units='m', desc='y-distance to elastic center from point about which above structural properties are computed')

        # outputs
        self.add_output('blade_mass', shape=1, units='kg', desc='mass of one blades')
        self.add_output('blade_moment_of_inertia', shape=1, units='kg*m**2', desc='out of plane moment of inertia of a blade')
        self.add_output('freq', shape=5, units='Hz', desc='first nF natural frequencies of blade')
        self.add_output('dx_defl', shape=nstr, desc='deflection of blade section in airfoil x-direction under max deflection loading')
        self.add_output('dy_defl', shape=nstr, desc='deflection of blade section in airfoil y-direction under max deflection loading')
        self.add_output('dz_defl', shape=nstr, desc='deflection of blade section in airfoil z-direction under max deflection loading')
        self.add_output('dx_pc_defl', shape=nstr, desc='deflection of blade section in airfoil x-direction under power curve loading')
        self.add_output('dy_pc_defl', shape=nstr, desc='deflection of blade section in airfoil y-direction under power curve loading')
        self.add_output('dz_pc_defl', shape=nstr, desc='deflection of blade section in airfoil z-direction under power curve loading')
        self.add_output('strainU_spar', shape=nstr, desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_output('strainL_spar', shape=nstr, desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')
        self.add_output('strainU_te', shape=nstr, desc='strain in trailing-edge panels on upper surface at location xu,yu_te with loads P_te')
        self.add_output('strainL_te', shape=nstr, desc='strain in trailing-edge panels on lower surface at location xl,yl_te with loads P_te')
        self.add_output('damageU_spar', shape=nstr, desc='fatigue damage on upper surface in spar cap')
        self.add_output('damageL_spar', shape=nstr, desc='fatigue damage on lower surface in spar cap')
        self.add_output('damageU_te', shape=nstr, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_output('damageL_te', shape=nstr, desc='fatigue damage on lower surface in trailing-edge panels')

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

    def principalCS(self, EIyy, EIxx, y_ec_str, x_ec_str, EA, EIxy):

        # rename (with swap of x, y for profile c.s.)
        EIxx = np.copy(EIyy)
        EIyy = np.copy(EIxx)
        x_ec_str = np.copy(y_ec_str)
        y_ec_str = np.copy(x_ec_str)
        EA = np.copy(EA)
        EIxy = np.copy(EIxy)

        # translate to elastic center
        EIxx -= y_ec_str**2*EA
        EIyy -= x_ec_str**2*EA
        EIxy -= x_ec_str*y_ec_str*EA

        # get rotation angle
        alpha = 0.5*np.arctan2(2*EIxy, EIyy-EIxx)

        EI11 = EIxx - EIxy*np.tan(alpha)
        EI22 = EIyy + EIxy*np.tan(alpha)

        # get moments and positions in principal axes
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return EI11, EI22, EA, ca, sa

    def strain(self, blade, xu, yu, xl, yl, EI11, EI22, EA, ca, sa):

        Vx, Vy, Fz, Mx, My, Tz = blade.shearAndBending()

        # use profile c.s. to use Hansen's notation
        Vx, Vy = Vy, Vx
        Mx, My = My, Mx
        xu, yu = yu, xu
        xl, yl = yl, xl

        # convert to principal xes
        M1 = Mx*ca + My*sa
        M2 = -Mx*sa + My*ca

        x = xu*ca + yu*sa
        y = -xu*sa + yu*ca

        # compute strain
        strainU = -(M1/EI11*y - M2/EI22*x + Fz/EA)  # negative sign because 3 is opposite of z

        x = xl*ca + yl*sa
        y = -xl*sa + yl*ca

        strainL = -(M1/EI11*y - M2/EI22*x + Fz/EA)

        return strainU, strainL

    def damage(self, Mx, My, xu, yu, xl, yl, EI11, EI22, EA, ca, sa, emax=0.01, eta=1.755, m=10.0, N=365*24*3600*24):

        # use profile c.s. to use Hansen's notation
        Mx, My = My, Mx
        Fz = 0.0
        xu, yu = yu, xu
        xl, yl = yl, xl

        # convert to principal xes
        M1 = Mx*ca + My*sa
        M2 = -Mx*sa + My*ca

        x = xu*ca + yu*sa
        y = -xu*sa + yu*ca

        # compute strain
        strainU = -(M1/EI11*y - M2/EI22*x + Fz/EA)  # negative sign because 3 is opposite of z

        x = xl*ca + yl*sa
        y = -xl*sa + yl*ca

        strainL = -(M1/EI11*y - M2/EI22*x + Fz/EA)

        # number of cycles to failure
        NfU = (emax/(eta*strainU))**m
        NfL = (emax/(eta*strainL))**m

        # damage
        damageU = N/NfU
        damageL = N/NfL

        damageU = math.log(N) - m*(math.log(emax) - math.log(eta) - np.log(np.abs(strainU)))
        damageL = math.log(N) - m*(math.log(emax) - math.log(eta) - np.log(np.abs(strainL)))

        return damageU, damageL

    def solve_nonlinear(self, params, unknowns, resids):
        Px_defl = params['Px_defl']
        Py_defl = params['Py_defl']
        Pz_defl = params['Pz_defl']

        self.nF = params['nF']
        self.Px_defl = params['Px_defl']
        self.Py_defl = params['Py_defl']
        self.Pz_defl = params['Pz_defl']
        self.Px_strain = params['Px_strain']
        self.Py_strain = params['Py_strain']
        self.Pz_strain = params['Pz_strain']
        self.Px_pc_defl = params['Px_pc_defl']
        self.Py_pc_defl = params['Py_pc_defl']
        self.Pz_pc_defl = params['Pz_pc_defl']

        self.xu_strain_spar = params['xu_strain_spar']
        self.xl_strain_spar = params['xl_strain_spar']
        self.yu_strain_spar = params['yu_strain_spar']
        self.yl_strain_spar = params['yl_strain_spar']
        self.xu_strain_te = params['xu_strain_te']
        self.xu_strain_te = params['xu_strain_te']
        self.xl_strain_te = params['xl_strain_te']
        self.yu_strain_te = params['yu_strain_te']
        self.yl_strain_te = params['yl_strain_te']

        self.Mx_damage = params['Mx_damage']
        self.My_damage = params['My_damage']
        self.strain_ult_spar = params['strain_ult_spar']
        self.strain_ult_te = params['strain_ult_te']
        self.eta_damage = params['eta_damage']
        self.m_damage = params['m_damage']
        self.N_damage = params['N_damage']

        # outputs
        nsec = len(params['beam:z'])

        # create finite element objects
        p_section = _pBEAM.SectionData(nsec, params['beam:z'], params['beam:EA'], params['beam:EIxx'],
            params['beam:EIyy'], params['beam:GJ'], params['beam:rhoA'], params['beam:rhoJ'])
        # p_loads = _pBEAM.Loads(nsec)  # no loads
        p_tip = _pBEAM.TipData()  # no tip mass
        k = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')])
        p_base = _pBEAM.BaseData(k, float('inf'))  # rigid base


        # ----- tip deflection -----

        # evaluate displacements
        p_loads = _pBEAM.Loads(nsec, Px_defl, Py_defl, Pz_defl)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        self.dx_defl, self.dy_defl, self.dz_defl, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

        p_loads = _pBEAM.Loads(nsec, self.Px_pc_defl, self.Py_pc_defl, self.Pz_pc_defl)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        self.dx_pc_defl, self.dy_pc_defl, self.dz_pc_defl, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()


        # --- mass ---
        self.blade_mass = blade.mass()
        # dmass = rhot*w
        # blade_mass_beam:rhoA
        # blade_mass_beam:z

        # --- moments of inertia
        self.blade_moment_of_inertia = blade.outOfPlaneMomentOfInertia()

        # ----- natural frequencies ----
        self.freq = blade.naturalFrequencies(self.nF)


        # ----- strain -----
        EI11, EI22, EA, ca, sa = self.principalCS(params['beam:EIyy'], params['beam:EIxx'], params['beam:y_ec_str'], params['beam:x_ec_str'], params['beam:EA'], params['beam:EIxy'])

        p_loads = _pBEAM.Loads(nsec, self.Px_strain, self.Py_strain, self.Pz_strain)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        self.strainU_spar, self.strainL_spar = self.strain(blade, self.xu_strain_spar, self.yu_strain_spar,
            self.xl_strain_spar, self.yl_strain_spar, EI11, EI22, EA, ca, sa)
        self.strainU_te, self.strainL_te = self.strain(blade, self.xu_strain_te, self.yu_strain_te,
            self.xl_strain_te, self.yl_strain_te, EI11, EI22, EA, ca, sa)
        self.damageU_spar, self.damageL_spar = self.damage(self.Mx_damage, self.My_damage, self.xu_strain_spar, self.yu_strain_spar,
            self.xl_strain_spar, self.yl_strain_spar, EI11, EI22, EA, ca, sa,
            emax=self.strain_ult_spar, eta=self.eta_damage, m=self.m_damage, N=self.N_damage)
        self.damageU_te, self.damageL_te = self.damage(self.Mx_damage, self.My_damage, self.xu_strain_te, self.yu_strain_te,
            self.xl_strain_te, self.yl_strain_te, EI11, EI22, EA, ca, sa,
            emax=self.strain_ult_te, eta=self.eta_damage, m=self.m_damage, N=self.N_damage)

        unknowns['blade_mass'] = self.blade_mass
        unknowns['blade_moment_of_inertia'] = self.blade_moment_of_inertia
        unknowns['freq'] = self.freq
        unknowns['dx_defl'] = self.dx_defl
        unknowns['dy_defl'] = self.dy_defl
        unknowns['dz_defl'] = self.dz_defl
        unknowns['dx_pc_defl'] = self.dx_pc_defl
        unknowns['dy_pc_defl'] = self.dy_pc_defl
        unknowns['dz_pc_defl'] = self.dz_pc_defl
        unknowns['strainU_spar'] = self.strainU_spar
        unknowns['strainL_spar'] = self.strainL_spar
        unknowns['strainU_te'] = self.strainU_te
        unknowns['strainL_te'] = self.strainL_te
        unknowns['damageU_spar'] = self.damageU_spar
        unknowns['damageL_spar'] = self.damageL_spar
        unknowns['damageU_te'] = self.damageU_te
        unknowns['damageL_te'] = self.damageL_te

class CurveFEM(Component):
    def __init__(self, nstr):
        super(CurveFEM, self).__init__()

        """natural frequencies for curved blades"""

        self.add_param('Omega', shape=1, units='rpm', desc='rotor rotation frequency')
        self.add_param('beam:z', shape=nstr, units='m', desc='locations of properties along beam')
        self.add_param('beam:EA', shape=nstr, units='N', desc='axial stiffness')
        self.add_param('beam:EIxx', shape=nstr, units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_param('beam:EIyy', shape=nstr, units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_param('beam:EIxy', shape=nstr, units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_param('beam:GJ', shape=nstr, units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_param('beam:rhoA', shape=nstr, units='kg/m', desc='mass per unit length')
        self.add_param('beam:rhoJ', shape=nstr, units='kg*m', desc='polar mass moment of inertia per unit length')
        self.add_param('beam:x_ec_str', shape=nstr, units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_param('beam:y_ec_str', shape=nstr, units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_param('theta_str', shape=nstr, units='deg', desc='structural twist distribution')
        self.add_param('precurve_str', shape=nstr, units='m', desc='structural precuve (see FAST definition)')
        self.add_param('presweep_str', shape=nstr, units='m', desc='structural presweep (see FAST definition)')
        self.add_param('nF', val=5, desc='number of frequencies to return', pass_by_obj=True)

        self.add_output('freq', shape=5, units='Hz', desc='first nF natural frequencies')

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        # print "CurveFEM"
        r = params['beam:z']

        rhub = r[0]
        bladeLength = r[-1] - r[0]
        bladeFrac = (r - rhub) / bladeLength

        freq = _curvefem.frequencies(params['Omega'], bladeLength, rhub, bladeFrac, params['theta_str'],
                                     params['beam:rhoA'], params['beam:EIxx'], params['beam:EIyy'], params['beam:GJ'], params['beam:EA'], params['beam:rhoJ'],
                                     params['precurve_str'], params['presweep_str'])

        unknowns['freq'] = freq[:params['nF']]

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

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.naero = naero
        self.nstr = nstr

    def solve_nonlinear(self, params, unknowns, resids):
        # print "GridSetup"
        r_aero = params['initial_aero_grid']
        r_str = params['initial_str_grid']
        r_aug = np.concatenate([[0.0], r_aero, [1.0]])

        nstr = len(r_str)
        naug = len(r_aug)

        # find idx in augmented aero array that brackets the structural index
        # then find the fraction the structural value is between the two bounding indices
        unknowns['fraction'] = np.zeros(nstr)
        unknowns['idxj'] = np.zeros(nstr, dtype=np.int)
        if r_str[-1] > 1.0: ## For bug when doing finite difference test
            r_str[-1] = 1.0
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


    def solve_nonlinear(self, params, unknowns, resids):
        # print "RGrid"
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
        self.add_param('idx_cylinder_aero', val=1, desc='first idx in r_aero_unit of non-cylindrical section')  # constant twist inboard of here
        self.add_param('idx_cylinder_str', val=1, desc='first idx in r_str_unit of non-cylindrical section')
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

        self.fd_options['form'] = 'forward'
        self.fd_options['step_type'] = 'absolute'
        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        # print 'r_max_chord', params['r_max_chord']
        # print 'chord_sub', params['chord_sub']
        # print 'theta_sub', params['theta_sub']
        # print 'sparT', params['sparT']
        # print 'teT', params['teT']
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

    # def linearize(self, params, unknowns, resids):
    #     J = {}
    #     J['Rhub', 'hubFraction'] = params['bladeLength']
    #     J['Rhub', 'bladeLength'] = params['hubFraction']
    #     return J

class BladeCurvature(Component):
    def __init__(self, nstr):
        super(BladeCurvature, self).__init__()
        self.add_param('r', shape=nstr, units='m', desc='location in blade z-coordinate')
        self.add_param('precurve', shape=nstr, units='m', desc='location in blade x-coordinate')
        self.add_param('presweep', shape=nstr, units='m', desc='location in blade y-coordinate')
        self.add_param('precone', shape=1, units='deg', desc='precone angle')

        self.add_output('totalCone', shape=nstr, units='deg', desc='total cone angle from precone and curvature')
        self.add_output('x_az', shape=nstr, units='m', desc='location of blade in azimuth x-coordinate system')
        self.add_output('y_az', shape=nstr, units='m', desc='location of blade in azimuth y-coordinate system')
        self.add_output('z_az', shape=nstr, units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_output('s', shape=nstr, units='m', desc='cumulative path length along blade')

    def solve_nonlinear(self, params, unknowns, resids):

        # print "BladeCurvature"
        # self.x_az, self.y_az, self.z_az, cone, s = \
        #     _bem.definecurvature(self.r, self.precurve, self.presweep, 0.0)
        self.r = params['r']
        self.precurve = params['precurve']
        self.presweep = params['presweep']
        self.precone = params['precone']

        n = len(self.r)
        dx_dx = np.eye(3*n)

        self.x_az, x_azd, self.y_az, y_azd, self.z_az, z_azd, \
            cone, coned, s, sd = _bem.definecurvature_dv2(self.r, dx_dx[:, :n],
                self.precurve, dx_dx[:, n:2*n], self.presweep, dx_dx[:, 2*n:], 0.0, np.zeros(3*n))

        self.totalCone = self.precone + np.degrees(cone)
        self.s = self.r[0] + s

        unknowns['totalCone'] = self.totalCone
        unknowns['x_az'] = self.x_az
        unknowns['y_az'] = self.y_az
        unknowns['z_az'] = self.z_az
        unknowns['s'] = self.s

        dxaz_dr = x_azd[:n, :].T
        dxaz_dprecurve = x_azd[n:2*n, :].T
        dxaz_dpresweep = x_azd[2*n:, :].T

        dyaz_dr = y_azd[:n, :].T
        dyaz_dprecurve = y_azd[n:2*n, :].T
        dyaz_dpresweep = y_azd[2*n:, :].T

        dzaz_dr = z_azd[:n, :].T
        dzaz_dprecurve = z_azd[n:2*n, :].T
        dzaz_dpresweep = z_azd[2*n:, :].T

        dcone_dr = np.degrees(coned[:n, :]).T
        dcone_dprecurve = np.degrees(coned[n:2*n, :]).T
        dcone_dpresweep = np.degrees(coned[2*n:, :]).T

        ds_dr = sd[:n, :].T
        ds_dr[:, 0] += 1
        ds_dprecurve = sd[n:2*n, :].T
        ds_dpresweep = sd[2*n:, :].T

        J = {}
        J['x_az', 'r'] = dxaz_dr
        J['x_az', 'precurve'] = dxaz_dprecurve
        J['x_az', 'presweep'] = dxaz_dpresweep
        J['x_az', 'precone'] = np.zeros(n)
        J['y_az', 'r'] = dyaz_dr
        J['y_az', 'precurve'] = dyaz_dprecurve
        J['y_az', 'presweep'] = dyaz_dpresweep
        J['y_az', 'precone'] = np.zeros(n)
        J['z_az', 'r'] = dzaz_dr
        J['z_az', 'precurve'] = dzaz_dprecurve
        J['z_az', 'presweep'] = dzaz_dpresweep
        J['z_az', 'precone'] = np.zeros(n)
        J['totalCone', 'r'] = dcone_dr
        J['totalCone', 'precurve'] = dcone_dprecurve
        J['totalCone', 'presweep'] = dcone_dpresweep
        J['totalCone', 'precone'] = np.ones(n)
        J['s', 'r'] = ds_dr
        J['s', 'precurve'] = ds_dprecurve
        J['s', 'presweep'] = ds_dpresweep
        J['s', 'precone'] = np.zeros(n)

        self.J = J


    def list_deriv_vars(self):

        inputs = ('r', 'precurve', 'presweep', 'precone')
        outputs = ('x_az', 'y_az', 'z_az', 'totalCone', 's')

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        return self.J



class DamageLoads(Component):
    def __init__(self, nstr):
        super(DamageLoads, self).__init__()
        self.add_param('rstar', shape=18, desc='nondimensional radial locations of damage equivalent moments')
        self.add_param('Mxb', shape=18, units='N*m', desc='damage equivalent moments about blade c.s. x-direction')
        self.add_param('Myb', shape=18, units='N*m', desc='damage equivalent moments about blade c.s. y-direction')
        self.add_param('theta', shape=nstr, units='deg', desc='structural twist')
        self.add_param('r', shape=nstr, units='m', desc='structural radial locations')

        self.add_output('Mxa', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil c.s. x-direction')
        self.add_output('Mya', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil c.s. y-direction')

    def solve_nonlinear(self, params, unknowns, resids):
        # print "DamageLoads"
        self.rstar = params['rstar']
        self.Mxb = params['Mxb']
        self.Myb = params['Myb']
        self.theta = params['theta']
        self.r = params['r']

        rstar_str = (self.r-self.r[0])/(self.r[-1]-self.r[0])

        Mxb_str, self.dMxbstr_drstarstr, self.dMxbstr_drstar, self.dMxbstr_dMxb = \
            akima_interp_with_derivs(self.rstar, self.Mxb, rstar_str)

        Myb_str, self.dMybstr_drstarstr, self.dMybstr_drstar, self.dMybstr_dMyb = \
            akima_interp_with_derivs(self.rstar, self.Myb, rstar_str)

        self.Ma = DirectionVector(Mxb_str, Myb_str, 0.0).bladeToAirfoil(self.theta)
        self.Mxa = self.Ma.x
        self.Mya = self.Ma.y

        unknowns['Mxa'] = self.Mxa
        unknowns['Mya'] = self.Mya

    def list_deriv_vars(self):

        inputs = ('rstar', 'Mxb', 'Myb', 'theta', 'r')
        outputs = ('Mxa', 'Mya')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):
        J = {}

        n = len(self.r)
        drstarstr_dr = np.zeros((n, n))
        for i in range(1, n-1):
            drstarstr_dr[i, i] = 1.0/(self.r[-1] - self.r[0])
        drstarstr_dr[1:, 0] = (self.r[1:] - self.r[-1])/(self.r[-1] - self.r[0])**2
        drstarstr_dr[:-1, -1] = -(self.r[:-1] - self.r[0])/(self.r[-1] - self.r[0])**2

        dMxbstr_drstarstr = np.diag(self.dMxbstr_drstarstr)
        dMybstr_drstarstr = np.diag(self.dMybstr_drstarstr)

        dMxbstr_dr = np.dot(dMxbstr_drstarstr, drstarstr_dr)
        dMybstr_dr = np.dot(dMybstr_drstarstr, drstarstr_dr)

        dMxa_dr = np.dot(np.diag(self.Ma.dx['dx']), dMxbstr_dr)\
            + np.dot(np.diag(self.Ma.dx['dy']), dMybstr_dr)
        dMxa_drstar = np.dot(np.diag(self.Ma.dx['dx']), self.dMxbstr_drstar)\
            + np.dot(np.diag(self.Ma.dx['dy']), self.dMybstr_drstar)
        dMxa_dMxb = np.dot(np.diag(self.Ma.dx['dx']), self.dMxbstr_dMxb)
        dMxa_dMyb = np.dot(np.diag(self.Ma.dx['dy']), self.dMybstr_dMyb)
        dMxa_dtheta = np.diag(self.Ma.dx['dtheta'])

        dMya_dr = np.dot(np.diag(self.Ma.dy['dx']), dMxbstr_dr)\
            + np.dot(np.diag(self.Ma.dy['dy']), dMybstr_dr)
        dMya_drstar = np.dot(np.diag(self.Ma.dy['dx']), self.dMxbstr_drstar)\
            + np.dot(np.diag(self.Ma.dy['dy']), self.dMybstr_drstar)
        dMya_dMxb = np.dot(np.diag(self.Ma.dy['dx']), self.dMxbstr_dMxb)
        dMya_dMyb = np.dot(np.diag(self.Ma.dy['dy']), self.dMybstr_dMyb)
        dMya_dtheta = np.diag(self.Ma.dy['dtheta'])

        J['Mxa', 'rstar'] = dMxa_drstar
        J['Mxa', 'Mxb'] = dMxa_dMxb
        J['Mxa', 'Myb'] = dMxa_dMyb
        J['Mxa', 'theta'] = dMxa_dtheta
        J['Mxa', 'r'] = dMxa_dr

        J['Mya', 'rstar'] = dMya_drstar
        J['Mya', 'Mxb'] = dMya_dMxb
        J['Mya', 'Myb'] = dMya_dMyb
        J['Mya', 'theta'] = dMya_dtheta
        J['Mya', 'r'] = dMya_dr

        return J


class TotalLoads(Component):
    def __init__(self, nstr):
        super(TotalLoads, self).__init__()

        # variables
        self.add_param('aeroLoads:r', units='m', desc='radial positions along blade going toward tip')
        self.add_param('aeroLoads:Px', units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_param('aeroLoads:Py', units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_param('aeroLoads:Pz', units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_param('aeroLoads:Omega', units='rpm', desc='rotor rotation speed')
        self.add_param('aeroLoads:pitch', units='deg', desc='pitch angle')
        self.add_param('aeroLoads:azimuth', units='deg', desc='azimuthal angle')

        self.add_param('r', shape=nstr, units='m', desc='structural radial locations')
        self.add_param('theta', shape=nstr, units='deg', desc='structural twist')
        self.add_param('tilt', shape=1, units='deg', desc='tilt angle')
        self.add_param('totalCone', shape=nstr, units='deg', desc='total cone angle from precone and curvature')
        self.add_param('z_az', shape=nstr, units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_param('rhoA', shape=nstr, units='kg/m', desc='mass per unit length')

        # parameters
        self.add_param('g', val=9.81, units='m/s**2', desc='acceleration of gravity')

        # outputs
        self.add_output('Px_af', shape=nstr, desc='total distributed loads in airfoil x-direction')
        self.add_output('Py_af', shape=nstr, desc='total distributed loads in airfoil y-direction')
        self.add_output('Pz_af', shape=nstr, desc='total distributed loads in airfoil z-direction')

    def solve_nonlinear(self, params, unknowns, resids):
        # print "TotalLoads"
        self.r = params['r']
        self.theta = params['theta']
        self.tilt = params['tilt']
        self.totalCone = params['totalCone']
        self.z_az = params['z_az']
        self.rhoA = params['rhoA']
        self.g = params['g']


        # totalCone = self.precone
        # z_az = self.r*cosd(self.precone)
        totalCone = self.totalCone
        z_az = self.z_az

        # keep all in blade c.s. then rotate all at end

        # rename
        # aero = self.aeroLoads

        # --- aero loads ---

        # interpolate aerodynamic loads onto structural grid
        P_a = DirectionVector(0, 0, 0)
        P_a.x, self.dPax_dr, self.dPax_daeror, self.dPax_daeroPx = akima_interp_with_derivs(params['aeroLoads:r'], params['aeroLoads:Px'], self.r)
        P_a.y, self.dPay_dr, self.dPay_daeror, self.dPay_daeroPy = akima_interp_with_derivs(params['aeroLoads:r'], params['aeroLoads:Py'], self.r)
        P_a.z, self.dPaz_dr, self.dPaz_daeror, self.dPaz_daeroPz = akima_interp_with_derivs(params['aeroLoads:r'], params['aeroLoads:Pz'], self.r)


        # --- weight loads ---

        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -self.rhoA*self.g)

        self.P_w = weight.yawToHub(self.tilt).hubToAzimuth(params['aeroLoads:azimuth'])\
            .azimuthToBlade(totalCone)


        # --- centrifugal loads ---

        # azimuthal c.s.
        Omega = params['aeroLoads:Omega']*RPM2RS
        load = DirectionVector(0.0, 0.0, self.rhoA*Omega**2*z_az)

        self.P_c = load.azimuthToBlade(totalCone)


        # --- total loads ---
        P = P_a + self.P_w + self.P_c

        # rotate to airfoil c.s.
        theta = np.array(self.theta) + params['aeroLoads:pitch']
        self.P = P.bladeToAirfoil(theta)

        self.Px_af = self.P.x
        self.Py_af = self.P.y
        self.Pz_af = self.P.z

        unknowns['Px_af'] = self.Px_af
        unknowns['Py_af'] = self.Py_af
        unknowns['Pz_af'] = self.Pz_af

    def list_deriv_vars(self):

        inputs = ('aeroLoads:r', 'aeroLoads:Px', 'aeroLoads:Py', 'aeroLoads:Pz', 'aeroLoads:Omega',
            'aeroLoads:pitch', 'aeroLoads:azimuth', 'r', 'theta', 'tilt', 'totalCone', 'rhoA', 'z_az')
        outputs = ('Px_af', 'Py_af', 'Pz_af')

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        dPwx, dPwy, dPwz = self.P_w.dx, self.P_w.dy, self.P_w.dz
        dPcx, dPcy, dPcz = self.P_c.dx, self.P_c.dy, self.P_c.dz
        dPx, dPy, dPz = self.P.dx, self.P.dy, self.P.dz
        Omega = params['aeroLoads:Omega']*RPM2RS
        z_az = self.z_az


        dPx_dOmega = dPcx['dz']*self.rhoA*z_az*2*Omega*RPM2RS
        dPy_dOmega = dPcy['dz']*self.rhoA*z_az*2*Omega*RPM2RS
        dPz_dOmega = dPcz['dz']*self.rhoA*z_az*2*Omega*RPM2RS

        dPx_dr = np.diag(self.dPax_dr)
        dPy_dr = np.diag(self.dPay_dr)
        dPz_dr = np.diag(self.dPaz_dr)

        dPx_dprecone = np.diag(dPwx['dprecone'] + dPcx['dprecone'])
        dPy_dprecone = np.diag(dPwy['dprecone'] + dPcy['dprecone'])
        dPz_dprecone = np.diag(dPwz['dprecone'] + dPcz['dprecone'])

        dPx_dzaz = np.diag(dPcx['dz']*self.rhoA*Omega**2)
        dPy_dzaz = np.diag(dPcy['dz']*self.rhoA*Omega**2)
        dPz_dzaz = np.diag(dPcz['dz']*self.rhoA*Omega**2)

        dPx_drhoA = np.diag(-dPwx['dz']*self.g + dPcx['dz']*Omega**2*z_az)
        dPy_drhoA = np.diag(-dPwy['dz']*self.g + dPcy['dz']*Omega**2*z_az)
        dPz_drhoA = np.diag(-dPwz['dz']*self.g + dPcz['dz']*Omega**2*z_az)

        dPxaf_daeror = (dPx['dx']*self.dPax_daeror.T + dPx['dy']*self.dPay_daeror.T + dPx['dz']*self.dPaz_daeror.T).T
        dPyaf_daeror = (dPy['dx']*self.dPax_daeror.T + dPy['dy']*self.dPay_daeror.T + dPy['dz']*self.dPaz_daeror.T).T
        dPzaf_daeror = (dPz['dx']*self.dPax_daeror.T + dPz['dy']*self.dPay_daeror.T + dPz['dz']*self.dPaz_daeror.T).T

        dPxaf_dPxaero = (dPx['dx']*self.dPax_daeroPx.T).T
        dPxaf_dPyaero = (dPx['dy']*self.dPay_daeroPy.T).T
        dPxaf_dPzaero = (dPx['dz']*self.dPaz_daeroPz.T).T

        dPyaf_dPxaero = (dPy['dx']*self.dPax_daeroPx.T).T
        dPyaf_dPyaero = (dPy['dy']*self.dPay_daeroPy.T).T
        dPyaf_dPzaero = (dPy['dz']*self.dPaz_daeroPz.T).T

        dPzaf_dPxaero = (dPz['dx']*self.dPax_daeroPx.T).T
        dPzaf_dPyaero = (dPz['dy']*self.dPay_daeroPy.T).T
        dPzaf_dPzaero = (dPz['dz']*self.dPaz_daeroPz.T).T

        dPxaf_dOmega = dPx['dx']*dPx_dOmega + dPx['dy']*dPy_dOmega + dPx['dz']*dPz_dOmega
        dPyaf_dOmega = dPy['dx']*dPx_dOmega + dPy['dy']*dPy_dOmega + dPy['dz']*dPz_dOmega
        dPzaf_dOmega = dPz['dx']*dPx_dOmega + dPz['dy']*dPy_dOmega + dPz['dz']*dPz_dOmega

        dPxaf_dpitch = dPx['dtheta']
        dPyaf_dpitch = dPy['dtheta']
        dPzaf_dpitch = dPz['dtheta']

        dPxaf_dazimuth = dPx['dx']*dPwx['dazimuth'] + dPx['dy']*dPwy['dazimuth'] + dPx['dz']*dPwz['dazimuth']
        dPyaf_dazimuth = dPy['dx']*dPwx['dazimuth'] + dPy['dy']*dPwy['dazimuth'] + dPy['dz']*dPwz['dazimuth']
        dPzaf_dazimuth = dPz['dx']*dPwx['dazimuth'] + dPz['dy']*dPwy['dazimuth'] + dPz['dz']*dPwz['dazimuth']

        dPxaf_dr = dPx['dx']*dPx_dr + dPx['dy']*dPy_dr + dPx['dz']*dPz_dr
        dPyaf_dr = dPy['dx']*dPx_dr + dPy['dy']*dPy_dr + dPy['dz']*dPz_dr
        dPzaf_dr = dPz['dx']*dPx_dr + dPz['dy']*dPy_dr + dPz['dz']*dPz_dr

        dPxaf_dtheta = np.diag(dPx['dtheta'])
        dPyaf_dtheta = np.diag(dPy['dtheta'])
        dPzaf_dtheta = np.diag(dPz['dtheta'])

        dPxaf_dtilt = dPx['dx']*dPwx['dtilt'] + dPx['dy']*dPwy['dtilt'] + dPx['dz']*dPwz['dtilt']
        dPyaf_dtilt = dPy['dx']*dPwx['dtilt'] + dPy['dy']*dPwy['dtilt'] + dPy['dz']*dPwz['dtilt']
        dPzaf_dtilt = dPz['dx']*dPwx['dtilt'] + dPz['dy']*dPwy['dtilt'] + dPz['dz']*dPwz['dtilt']

        dPxaf_dprecone = dPx['dx']*dPx_dprecone + dPx['dy']*dPy_dprecone + dPx['dz']*dPz_dprecone
        dPyaf_dprecone = dPy['dx']*dPx_dprecone + dPy['dy']*dPy_dprecone + dPy['dz']*dPz_dprecone
        dPzaf_dprecone = dPz['dx']*dPx_dprecone + dPz['dy']*dPy_dprecone + dPz['dz']*dPz_dprecone

        dPxaf_drhoA = dPx['dx']*dPx_drhoA + dPx['dy']*dPy_drhoA + dPx['dz']*dPz_drhoA
        dPyaf_drhoA = dPy['dx']*dPx_drhoA + dPy['dy']*dPy_drhoA + dPy['dz']*dPz_drhoA
        dPzaf_drhoA = dPz['dx']*dPx_drhoA + dPz['dy']*dPy_drhoA + dPz['dz']*dPz_drhoA

        dPxaf_dzaz = dPx['dx']*dPx_dzaz + dPx['dy']*dPy_dzaz + dPx['dz']*dPz_dzaz
        dPyaf_dzaz = dPy['dx']*dPx_dzaz + dPy['dy']*dPy_dzaz + dPy['dz']*dPz_dzaz
        dPzaf_dzaz = dPz['dx']*dPx_dzaz + dPz['dy']*dPy_dzaz + dPz['dz']*dPz_dzaz

        J = {}
        J['Px_af', 'aeroLoads:r'] = dPxaf_daeror
        J['Px_af', 'aeroLoads:Px'] = dPxaf_dPxaero
        J['Px_af', 'aeroLoads:Py'] = dPxaf_dPyaero
        J['Px_af', 'aeroLoads:Pz'] = dPxaf_dPzaero
        J['Px_af', 'aeroLoads:Omega'] = dPxaf_dOmega
        J['Px_af', 'aeroLoads:pitch'] = dPxaf_dpitch
        J['Px_af', 'aeroLoads:azimuth'] = dPxaf_dazimuth
        J['Px_af', 'r'] = dPxaf_dr
        J['Px_af', 'theta'] = dPxaf_dtheta
        J['Px_af', 'tilt'] = dPxaf_dtilt
        J['Px_af', 'totalCone'] = dPxaf_dprecone
        J['Px_af', 'rhoA'] = dPxaf_drhoA
        J['Px_af', 'z_az'] = dPxaf_dzaz

        J['Py_af', 'aeroLoads:r'] = dPyaf_daeror
        J['Py_af', 'aeroLoads:Px'] = dPyaf_dPxaero
        J['Py_af', 'aeroLoads:Py'] = dPyaf_dPyaero
        J['Py_af', 'aeroLoads:Pz'] = dPyaf_dPzaero
        J['Py_af', 'aeroLoads:Omega'] = dPyaf_dOmega
        J['Py_af', 'aeroLoads:pitch'] = dPyaf_dpitch
        J['Py_af', 'aeroLoads:azimuth'] = dPyaf_dazimuth
        J['Py_af', 'r'] = dPyaf_dr
        J['Py_af', 'theta'] = dPyaf_dtheta
        J['Py_af', 'tilt'] = dPyaf_dtilt
        J['Py_af', 'totalCone'] = dPyaf_dprecone
        J['Py_af', 'rhoA'] = dPyaf_drhoA
        J['Py_af', 'z_az'] = dPyaf_dzaz

        J['Pz_af', 'aeroLoads:r'] = dPzaf_daeror
        J['Pz_af', 'aeroLoads:Px'] = dPzaf_dPxaero
        J['Pz_af', 'aeroLoads:Py'] = dPzaf_dPyaero
        J['Pz_af', 'aeroLoads:Pz'] = dPzaf_dPzaero
        J['Pz_af', 'aeroLoads:Omega'] = dPzaf_dOmega
        J['Pz_af', 'aeroLoads:pitch'] = dPzaf_dpitch
        J['Pz_af', 'aeroLoads:azimuth'] = dPzaf_dazimuth
        J['Pz_af', 'r'] = dPzaf_dr
        J['Pz_af', 'theta'] = dPzaf_dtheta
        J['Pz_af', 'tilt'] = dPzaf_dtilt
        J['Pz_af', 'totalCone'] = dPzaf_dprecone
        J['Pz_af', 'rhoA'] = dPzaf_drhoA
        J['Pz_af', 'z_az'] = dPzaf_dzaz

        return J



class TipDeflection(Component):
    def __init__(self):
        super(TipDeflection, self).__init__()
        # variables
        self.add_param('dx', shape=1, desc='deflection at tip in airfoil x-direction')
        self.add_param('dy', shape=1, desc='deflection at tip in airfoil y-direction')
        self.add_param('dz', shape=1, desc='deflection at tip in airfoil z-direction')
        self.add_param('theta', shape=1, units='deg', desc='twist at tip section')
        self.add_param('pitch', shape=1, units='deg', desc='blade pitch angle')
        self.add_param('azimuth', shape=1, units='deg', desc='azimuth angle')
        self.add_param('tilt', shape=1, units='deg', desc='tilt angle')
        self.add_param('totalConeTip', shape=1, units='deg', desc='total coning angle including precone and curvature')

        # parameters
        self.add_param('dynamicFactor', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation') #, pass_by_obj=True)

        # outputs
        self.add_output('tip_deflection', shape=1, units='m', desc='deflection at tip in yaw x-direction')

    def solve_nonlinear(self, params, unknowns, resids):
        # print "TipDeflection"
        self.dx = params['dx']
        self.dy = params['dy']
        self.dz = params['dz']
        self.theta = params['theta']
        self.pitch = params['pitch']
        self.azimuth = params['azimuth']
        self.tilt = params['tilt']
        self.totalConeTip = params['totalConeTip']
        self.dynamicFactor = params['dynamicFactor']

        theta = self.theta + self.pitch

        dr = DirectionVector(self.dx, self.dy, self.dz)
        self.delta = dr.airfoilToBlade(theta).bladeToAzimuth(self.totalConeTip) \
            .azimuthToHub(self.azimuth).hubToYaw(self.tilt)

        self.tip_deflection = self.dynamicFactor * self.delta.x

        unknowns['tip_deflection'] = self.tip_deflection


    def list_deriv_vars(self):

        inputs = ('dx', 'dy', 'dz', 'theta', 'pitch', 'azimuth', 'tilt', 'totalConeTip')
        outputs = ('tip_deflection',)

        return inputs, outputs


    def linearize(self, params, unknowns, resids):
        J = {}
        dx = self.dynamicFactor * self.delta.dx['dx']
        dy = self.dynamicFactor * self.delta.dx['dy']
        dz = self.dynamicFactor * self.delta.dx['dz']
        dtheta = self.dynamicFactor * self.delta.dx['dtheta']
        dpitch = self.dynamicFactor * self.delta.dx['dtheta']
        dazimuth = self.dynamicFactor * self.delta.dx['dazimuth']
        dtilt = self.dynamicFactor * self.delta.dx['dtilt']
        dtotalConeTip = self.dynamicFactor * self.delta.dx['dprecone']

        J['tip_deflection', 'dx'] = dx
        J['tip_deflection', 'dy'] = dy
        J['tip_deflection', 'dz'] = dz
        J['tip_deflection', 'theta'] = dtheta
        J['tip_deflection', 'pitch'] = dpitch
        J['tip_deflection', 'azimuth'] = dazimuth
        J['tip_deflection', 'tilt'] = dtilt
        J['tip_deflection', 'totalConeTip'] = dtotalConeTip
        J['tip_deflection', 'dynamicFactor'] = self.delta.x.tolist()

        return J

class BladeDeflection(Component):
    def __init__(self, nstr):
        super(BladeDeflection, self).__init__()
        self.add_param('dx', shape=nstr, desc='deflections in airfoil x-direction')
        self.add_param('dy', shape=nstr, desc='deflections in airfoil y-direction')
        self.add_param('dz', shape=nstr, desc='deflections in airfoil z-direction')
        self.add_param('pitch', shape=1, units='deg', desc='blade pitch angle')
        self.add_param('theta_str', shape=nstr, units='deg', desc='structural twist')

        self.add_param('r_sub_precurve0', shape=3, desc='undeflected precurve locations (internal)')
        self.add_param('Rhub0', shape=1, units='m', desc='hub radius')
        self.add_param('r_str0', shape=nstr, units='m', desc='undeflected radial locations')
        self.add_param('precurve_str0', shape=nstr, units='m', desc='undeflected precurve locations')
        self.add_param('bladeLength0', shape=1, units='m', desc='original blade length (only an actual length if no curvature)')

        self.add_output('delta_bladeLength', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
        self.add_output('delta_precurve_sub', shape=3,  units='m', desc='adjustment to precurve to account for curvature from loading')

    def solve_nonlinear(self, params, unknowns, resids):
        # print "BladeDeflection"
        self.dx = params['dx']
        self.dy = params['dy']
        self.dz = params['dz']
        self.pitch = params['pitch']
        self.theta_str = params['theta_str']
        self.r_sub_precurve0 = params['r_sub_precurve0']
        self.Rhub0 = params['Rhub0']
        self.r_str0 = params['r_str0']
        self.precurve_str0 = params['precurve_str0']
        self.bladeLength0 = params['bladeLength0']


        theta = self.theta_str + self.pitch

        dr = DirectionVector(self.dx, self.dy, self.dz)
        self.delta = dr.airfoilToBlade(theta)

        precurve_str_out = self.precurve_str0 + self.delta.x

        self.length0 = self.Rhub0 + np.sum(np.sqrt((self.precurve_str0[1:] - self.precurve_str0[:-1])**2 +
                                            (self.r_str0[1:] - self.r_str0[:-1])**2))
        self.length = self.Rhub0 + np.sum(np.sqrt((precurve_str_out[1:] - precurve_str_out[:-1])**2 +
                                           (self.r_str0[1:] - self.r_str0[:-1])**2))

        self.shortening = self.length0/self.length

        self.delta_bladeLength = self.bladeLength0 * (self.shortening - 1)
        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        self.delta_precurve_sub, self.dpcs_drsubpc0, self.dpcs_drstr0, self.dpcs_ddeltax = \
            interp_with_deriv(self.r_sub_precurve0, self.r_str0, self.delta.x)

        unknowns['delta_bladeLength'] = self.delta_bladeLength
        unknowns['delta_precurve_sub'] = self.delta_precurve_sub

    def list_deriv_vars(self):

        inputs = ('dx', 'dy', 'dz', 'pitch', 'theta_str', 'r_sub_precurve0', 'Rhub0',
            'r_str0', 'precurve_str0', 'bladeLength0')
        outputs = ('delta_bladeLength', 'delta_precurve_sub')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        n = len(self.theta_str)

        ddeltax_ddx = self.delta.dx['dx']
        ddeltax_ddy = self.delta.dx['dy']
        ddeltax_ddz = self.delta.dx['dz']
        ddeltax_dtheta = self.delta.dx['dtheta']
        ddeltax_dthetastr = ddeltax_dtheta
        ddeltax_dpitch = ddeltax_dtheta

        dl0_drhub0 = 1.0
        dl_drhub0 = 1.0
        dl0_dprecurvestr0 = np.zeros(n)
        dl_dprecurvestr0 = np.zeros(n)
        dl0_drstr0 = np.zeros(n)
        dl_drstr0 = np.zeros(n)

        precurve_str_out = self.precurve_str0 + self.delta.x


        for i in range(1, n-1):
            sm0 = math.sqrt((self.precurve_str0[i] - self.precurve_str0[i-1])**2 + (self.r_str0[i] - self.r_str0[i-1])**2)
            sm = math.sqrt((precurve_str_out[i] - precurve_str_out[i-1])**2 + (self.r_str0[i] - self.r_str0[i-1])**2)
            sp0 = math.sqrt((self.precurve_str0[i+1] - self.precurve_str0[i])**2 + (self.r_str0[i+1] - self.r_str0[i])**2)
            sp = math.sqrt((precurve_str_out[i+1] - precurve_str_out[i])**2 + (self.r_str0[i+1] - self.r_str0[i])**2)
            dl0_dprecurvestr0[i] = (self.precurve_str0[i] - self.precurve_str0[i-1]) / sm0 \
                - (self.precurve_str0[i+1] - self.precurve_str0[i]) / sp0
            dl_dprecurvestr0[i] = (precurve_str_out[i] - precurve_str_out[i-1]) / sm \
                - (precurve_str_out[i+1] - precurve_str_out[i]) / sp
            dl0_drstr0[i] = (self.r_str0[i] - self.r_str0[i-1]) / sm0 \
                - (self.r_str0[i+1] - self.r_str0[i]) / sp0
            dl_drstr0[i] = (self.r_str0[i] - self.r_str0[i-1]) / sm \
                - (self.r_str0[i+1] - self.r_str0[i]) / sp

        sfirst0 = math.sqrt((self.precurve_str0[1] - self.precurve_str0[0])**2 + (self.r_str0[1] - self.r_str0[0])**2)
        sfirst = math.sqrt((precurve_str_out[1] - precurve_str_out[0])**2 + (self.r_str0[1] - self.r_str0[0])**2)
        slast0 = math.sqrt((self.precurve_str0[n-1] - self.precurve_str0[n-2])**2 + (self.r_str0[n-1] - self.r_str0[n-2])**2)
        slast = math.sqrt((precurve_str_out[n-1] - precurve_str_out[n-2])**2 + (self.r_str0[n-1] - self.r_str0[n-2])**2)
        dl0_dprecurvestr0[0] = -(self.precurve_str0[1] - self.precurve_str0[0]) / sfirst0
        dl0_dprecurvestr0[n-1] = (self.precurve_str0[n-1] - self.precurve_str0[n-2]) / slast0
        dl_dprecurvestr0[0] = -(precurve_str_out[1] - precurve_str_out[0]) / sfirst
        dl_dprecurvestr0[n-1] = (precurve_str_out[n-1] - precurve_str_out[n-2]) / slast
        dl0_drstr0[0] = -(self.r_str0[1] - self.r_str0[0]) / sfirst0
        dl0_drstr0[n-1] = (self.r_str0[n-1] - self.r_str0[n-2]) / slast0
        dl_drstr0[0] = -(self.r_str0[1] - self.r_str0[0]) / sfirst
        dl_drstr0[n-1] = (self.r_str0[n-1] - self.r_str0[n-2]) / slast

        dl_ddeltax = dl_dprecurvestr0
        dl_ddx = dl_ddeltax * ddeltax_ddx
        dl_ddy = dl_ddeltax * ddeltax_ddy
        dl_ddz = dl_ddeltax * ddeltax_ddz
        dl_dthetastr = dl_ddeltax * ddeltax_dthetastr
        dl_dpitch = np.dot(dl_ddeltax, ddeltax_dpitch)

        dshort_dl = -self.length0/self.length**2
        dshort_dl0 = 1.0/self.length
        dshort_drhub0 = dshort_dl0*dl0_drhub0 + dshort_dl*dl_drhub0
        dshort_dprecurvestr0 = dshort_dl0*dl0_dprecurvestr0 + dshort_dl*dl_dprecurvestr0
        dshort_drstr0 = dshort_dl0*dl0_drstr0 + dshort_dl*dl_drstr0
        dshort_ddx = dshort_dl*dl_ddx
        dshort_ddy = dshort_dl*dl_ddy
        dshort_ddz = dshort_dl*dl_ddz
        dshort_dthetastr = dshort_dl*dl_dthetastr
        dshort_dpitch = dshort_dl*dl_dpitch

        dbl_dbl0 = (self.shortening - 1)
        dbl_drhub0 = self.bladeLength0 * dshort_drhub0
        dbl_dprecurvestr0 = self.bladeLength0 * dshort_dprecurvestr0
        dbl_drstr0 = self.bladeLength0 * dshort_drstr0
        dbl_ddx = self.bladeLength0 * dshort_ddx
        dbl_ddy = self.bladeLength0 * dshort_ddy
        dbl_ddz = self.bladeLength0 * dshort_ddz
        dbl_dthetastr = self.bladeLength0 * dshort_dthetastr
        dbl_dpitch = self.bladeLength0 * dshort_dpitch

        m = len(self.r_sub_precurve0)
        dpcs_ddx = self.dpcs_ddeltax*ddeltax_ddx
        dpcs_ddy = self.dpcs_ddeltax*ddeltax_ddy
        dpcs_ddz = self.dpcs_ddeltax*ddeltax_ddz
        dpcs_dpitch = np.dot(self.dpcs_ddeltax, ddeltax_dpitch)
        dpcs_dthetastr = self.dpcs_ddeltax*ddeltax_dthetastr

        J = {}

        J['delta_bladeLength', 'dx'] = np.reshape(dbl_ddx, (1, len(dbl_ddx)))
        J['delta_bladeLength', 'dy'] = np.reshape(dbl_ddy, (1, len(dbl_ddy)))
        J['delta_bladeLength', 'dz'] = np.reshape(dbl_ddz, (1, len(dbl_ddz)))
        J['delta_bladeLength', 'pitch'] = dbl_dpitch
        J['delta_bladeLength', 'theta_str'] = np.reshape(dbl_dthetastr, (1, len(dbl_dthetastr)))
        J['delta_bladeLength', 'r_sub_precurve0'] = np.zeros((1, m))
        J['delta_bladeLength', 'Rhub0'] = dbl_drhub0
        J['delta_bladeLength', 'r_str0'] = np.reshape(dbl_drstr0, (1, len(dbl_drstr0)))
        J['delta_bladeLength', 'precurve_str0'] = np.reshape(dbl_dprecurvestr0, (1, len(dbl_dprecurvestr0)))
        J['delta_bladeLength', 'bladeLength0'] = dbl_dbl0

        J['delta_precurve_sub', 'dx'] = dpcs_ddx
        J['delta_precurve_sub', 'dy'] = dpcs_ddy
        J['delta_precurve_sub', 'dz'] = dpcs_ddz
        J['delta_precurve_sub', 'pitch'] = dpcs_dpitch
        J['delta_precurve_sub', 'theta_str'] = dpcs_dthetastr
        J['delta_precurve_sub', 'r_sub_precurve0'] = self.dpcs_drsubpc0
        J['delta_precurve_sub', 'Rhub0'] = np.zeros(m)
        J['delta_precurve_sub', 'r_str0'] = self.dpcs_drstr0
        J['delta_precurve_sub', 'precurve_str0'] = np.zeros((m, n))
        J['delta_precurve_sub', 'bladeLength0'] = np.zeros(m)

        return J


class RootMoment(Component):
    """blade root bending moment"""
    def __init__(self, nstr):
        super(RootMoment, self).__init__()
        self.add_param('aeroLoads:r', units='m', desc='radial positions along blade going toward tip')
        self.add_param('aeroLoads:Px', units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_param('aeroLoads:Py', units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_param('aeroLoads:Pz', units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_param('r_str', shape=nstr, units='m')
        self.add_param('totalCone', shape=nstr, units='deg', desc='total cone angle from precone and curvature')
        self.add_param('x_az', shape=nstr, units='m', desc='location of blade in azimuth x-coordinate system')
        self.add_param('y_az', shape=nstr, units='m', desc='location of blade in azimuth y-coordinate system')
        self.add_param('z_az', shape=nstr, units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_param('s', shape=nstr, units='m', desc='cumulative path length along blade')

        self.add_output('root_bending_moment', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')

    def solve_nonlinear(self, params, unknowns, resids):
        # print "RootMoment"
        self.r_str = params['r_str']
        self.totalCone = params['totalCone']
        self.x_az = params['x_az']
        self.y_az = params['y_az']
        self.z_az = params['z_az']
        self.s = params['s']

        r = self.r_str
        x_az = self.x_az
        y_az = self.y_az
        z_az = self.z_az


        # aL = self.aeroLoads
        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        Px, self.dPx_dr, self.dPx_dalr, self.dPx_dalPx = interp_with_deriv(r, params['aeroLoads:r'], params['aeroLoads:Px'])
        Py, self.dPy_dr, self.dPy_dalr, self.dPy_dalPy = interp_with_deriv(r, params['aeroLoads:r'], params['aeroLoads:Py'])
        Pz, self.dPz_dr, self.dPz_dalr, self.dPz_dalPz = interp_with_deriv(r, params['aeroLoads:r'], params['aeroLoads:Pz'])

        # loads in azimuthal c.s.
        P = DirectionVector(Px, Py, Pz).bladeToAzimuth(self.totalCone)

        # distributed bending load in azimuth coordinate ysstem
        az = DirectionVector(x_az, y_az, z_az)
        Mp = az.cross(P)

        # integrate
        Mx = np.trapz(Mp.x, self.s)
        My = np.trapz(Mp.y, self.s)
        Mz = np.trapz(Mp.z, self.s)

        # get total magnitude
        self.root_bending_moment = math.sqrt(Mx**2 + My**2 + Mz**2)

        self.P = P
        self.az = az
        self.Mp = Mp
        self.r = r
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

        unknowns['root_bending_moment'] = self.root_bending_moment


    def list_deriv_vars(self):

        inputs = ('r_str', 'aeroLoads:r', 'aeroLoads:Px', 'aeroLoads:Py', 'aeroLoads:Pz', 'totalCone',
                  'x_az', 'y_az', 'z_az', 's')
        outputs = ('root_bending_moment',)

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        # dx_dr = -sind(self.precone)
        # dz_dr = cosd(self.precone)

        # dx_dprecone = -self.r*cosd(self.precone)*math.pi/180.0
        # dz_dprecone = -self.r*sind(self.precone)*math.pi/180.0

        dPx_dr = (self.P.dx['dx']*self.dPx_dr.T + self.P.dx['dy']*self.dPy_dr.T + self.P.dx['dz']*self.dPz_dr.T).T
        dPy_dr = (self.P.dy['dx']*self.dPx_dr.T + self.P.dy['dy']*self.dPy_dr.T + self.P.dy['dz']*self.dPz_dr.T).T
        dPz_dr = (self.P.dz['dx']*self.dPx_dr.T + self.P.dz['dy']*self.dPy_dr.T + self.P.dz['dz']*self.dPz_dr.T).T

        dPx_dalr = (self.P.dx['dx']*self.dPx_dalr.T + self.P.dx['dy']*self.dPy_dalr.T + self.P.dx['dz']*self.dPz_dalr.T).T
        dPy_dalr = (self.P.dy['dx']*self.dPx_dalr.T + self.P.dy['dy']*self.dPy_dalr.T + self.P.dy['dz']*self.dPz_dalr.T).T
        dPz_dalr = (self.P.dz['dx']*self.dPx_dalr.T + self.P.dz['dy']*self.dPy_dalr.T + self.P.dz['dz']*self.dPz_dalr.T).T

        dPx_dalPx = (self.P.dx['dx']*self.dPx_dalPx.T).T
        dPx_dalPy = (self.P.dx['dy']*self.dPy_dalPy.T).T
        dPx_dalPz = (self.P.dx['dz']*self.dPz_dalPz.T).T

        dPy_dalPx = (self.P.dy['dx']*self.dPx_dalPx.T).T
        dPy_dalPy = (self.P.dy['dy']*self.dPy_dalPy.T).T
        dPy_dalPz = (self.P.dy['dz']*self.dPz_dalPz.T).T

        dPz_dalPx = (self.P.dz['dx']*self.dPx_dalPx.T).T
        dPz_dalPy = (self.P.dz['dy']*self.dPy_dalPy.T).T
        dPz_dalPz = (self.P.dz['dz']*self.dPz_dalPz.T).T


        # dazx_dr = np.diag(self.az.dx['dx']*dx_dr + self.az.dx['dz']*dz_dr)
        # dazy_dr = np.diag(self.az.dy['dx']*dx_dr + self.az.dy['dz']*dz_dr)
        # dazz_dr = np.diag(self.az.dz['dx']*dx_dr + self.az.dz['dz']*dz_dr)

        # dazx_dprecone = (self.az.dx['dx']*dx_dprecone.T + self.az.dx['dz']*dz_dprecone.T).T
        # dazy_dprecone = (self.az.dy['dx']*dx_dprecone.T + self.az.dy['dz']*dz_dprecone.T).T
        # dazz_dprecone = (self.az.dz['dx']*dx_dprecone.T + self.az.dz['dz']*dz_dprecone.T).T

        dMpx, dMpy, dMpz = self.az.cross_deriv_array(self.P, namea='az', nameb='P')

        dMpx_dr = dMpx['dPx']*dPx_dr.T + dMpx['dPy']*dPy_dr.T + dMpx['dPz']*dPz_dr.T
        dMpy_dr = dMpy['dPx']*dPx_dr.T + dMpy['dPy']*dPy_dr.T + dMpy['dPz']*dPz_dr.T
        dMpz_dr = dMpz['dPx']*dPx_dr.T + dMpz['dPy']*dPy_dr.T + dMpz['dPz']*dPz_dr.T

        dMpx_dtotalcone = dMpx['dPx']*self.P.dx['dprecone'].T + dMpx['dPy']*self.P.dy['dprecone'].T + dMpx['dPz']*self.P.dz['dprecone'].T
        dMpy_dtotalcone = dMpy['dPx']*self.P.dx['dprecone'].T + dMpy['dPy']*self.P.dy['dprecone'].T + dMpy['dPz']*self.P.dz['dprecone'].T
        dMpz_dtotalcone = dMpz['dPx']*self.P.dx['dprecone'].T + dMpz['dPy']*self.P.dy['dprecone'].T + dMpz['dPz']*self.P.dz['dprecone'].T

        dMpx_dalr = (dMpx['dPx']*dPx_dalr.T + dMpx['dPy']*dPy_dalr.T + dMpx['dPz']*dPz_dalr.T).T
        dMpy_dalr = (dMpy['dPx']*dPx_dalr.T + dMpy['dPy']*dPy_dalr.T + dMpy['dPz']*dPz_dalr.T).T
        dMpz_dalr = (dMpz['dPx']*dPx_dalr.T + dMpz['dPy']*dPy_dalr.T + dMpz['dPz']*dPz_dalr.T).T

        dMpx_dalPx = (dMpx['dPx']*dPx_dalPx.T + dMpx['dPy']*dPy_dalPx.T + dMpx['dPz']*dPz_dalPx.T).T
        dMpy_dalPx = (dMpy['dPx']*dPx_dalPx.T + dMpy['dPy']*dPy_dalPx.T + dMpy['dPz']*dPz_dalPx.T).T
        dMpz_dalPx = (dMpz['dPx']*dPx_dalPx.T + dMpz['dPy']*dPy_dalPx.T + dMpz['dPz']*dPz_dalPx.T).T

        dMpx_dalPy = (dMpx['dPx']*dPx_dalPy.T + dMpx['dPy']*dPy_dalPy.T + dMpx['dPz']*dPz_dalPy.T).T
        dMpy_dalPy = (dMpy['dPx']*dPx_dalPy.T + dMpy['dPy']*dPy_dalPy.T + dMpy['dPz']*dPz_dalPy.T).T
        dMpz_dalPy = (dMpz['dPx']*dPx_dalPy.T + dMpz['dPy']*dPy_dalPy.T + dMpz['dPz']*dPz_dalPy.T).T

        dMpx_dalPz = (dMpx['dPx']*dPx_dalPz.T + dMpx['dPy']*dPy_dalPz.T + dMpx['dPz']*dPz_dalPz.T).T
        dMpy_dalPz = (dMpy['dPx']*dPx_dalPz.T + dMpy['dPy']*dPy_dalPz.T + dMpy['dPz']*dPz_dalPz.T).T
        dMpz_dalPz = (dMpz['dPx']*dPx_dalPz.T + dMpz['dPy']*dPy_dalPz.T + dMpz['dPz']*dPz_dalPz.T).T

        dMx_dMpx, dMx_ds = trapz_deriv(self.Mp.x, self.s)
        dMy_dMpy, dMy_ds = trapz_deriv(self.Mp.y, self.s)
        dMz_dMpz, dMz_ds = trapz_deriv(self.Mp.z, self.s)

        dMx_dr = np.dot(dMx_dMpx, dMpx_dr)
        dMy_dr = np.dot(dMy_dMpy, dMpy_dr)
        dMz_dr = np.dot(dMz_dMpz, dMpz_dr)

        dMx_dalr = np.dot(dMx_dMpx, dMpx_dalr)
        dMy_dalr = np.dot(dMy_dMpy, dMpy_dalr)
        dMz_dalr = np.dot(dMz_dMpz, dMpz_dalr)

        dMx_dalPx = np.dot(dMx_dMpx, dMpx_dalPx)
        dMy_dalPx = np.dot(dMy_dMpy, dMpy_dalPx)
        dMz_dalPx = np.dot(dMz_dMpz, dMpz_dalPx)

        dMx_dalPy = np.dot(dMx_dMpx, dMpx_dalPy)
        dMy_dalPy = np.dot(dMy_dMpy, dMpy_dalPy)
        dMz_dalPy = np.dot(dMz_dMpz, dMpz_dalPy)

        dMx_dalPz = np.dot(dMx_dMpx, dMpx_dalPz)
        dMy_dalPz = np.dot(dMy_dMpy, dMpy_dalPz)
        dMz_dalPz = np.dot(dMz_dMpz, dMpz_dalPz)

        dMx_dtotalcone = dMx_dMpx * dMpx_dtotalcone
        dMy_dtotalcone = dMy_dMpy * dMpy_dtotalcone
        dMz_dtotalcone = dMz_dMpz * dMpz_dtotalcone

        dMx_dazx = dMx_dMpx * dMpx['dazx']
        dMx_dazy = dMx_dMpx * dMpx['dazy']
        dMx_dazz = dMx_dMpx * dMpx['dazz']

        dMy_dazx = dMy_dMpy * dMpy['dazx']
        dMy_dazy = dMy_dMpy * dMpy['dazy']
        dMy_dazz = dMy_dMpy * dMpy['dazz']

        dMz_dazx = dMz_dMpz * dMpz['dazx']
        dMz_dazy = dMz_dMpz * dMpz['dazy']
        dMz_dazz = dMz_dMpz * dMpz['dazz']

        drbm_dr = (self.Mx*dMx_dr + self.My*dMy_dr + self.Mz*dMz_dr)/self.root_bending_moment
        drbm_dalr = (self.Mx*dMx_dalr + self.My*dMy_dalr + self.Mz*dMz_dalr)/self.root_bending_moment
        drbm_dalPx = (self.Mx*dMx_dalPx + self.My*dMy_dalPx + self.Mz*dMz_dalPx)/self.root_bending_moment
        drbm_dalPy = (self.Mx*dMx_dalPy + self.My*dMy_dalPy + self.Mz*dMz_dalPy)/self.root_bending_moment
        drbm_dalPz = (self.Mx*dMx_dalPz + self.My*dMy_dalPz + self.Mz*dMz_dalPz)/self.root_bending_moment
        drbm_dtotalcone = (self.Mx*dMx_dtotalcone + self.My*dMy_dtotalcone + self.Mz*dMz_dtotalcone)/self.root_bending_moment
        drbm_dazx = (self.Mx*dMx_dazx + self.My*dMy_dazx + self.Mz*dMz_dazx)/self.root_bending_moment
        drbm_dazy = (self.Mx*dMx_dazy + self.My*dMy_dazy + self.Mz*dMz_dazy)/self.root_bending_moment
        drbm_dazz = (self.Mx*dMx_dazz + self.My*dMy_dazz + self.Mz*dMz_dazz)/self.root_bending_moment
        drbm_ds = (self.Mx*dMx_ds + self.My*dMy_ds + self.Mz*dMz_ds)/self.root_bending_moment


        J = {}
        J['root_bending_moment', 'r_str'] = np.reshape(drbm_dr, (1, len(drbm_dr)))
        J['root_bending_moment', 'aeroLoads:r'] = np.reshape(drbm_dalr, (1, len(drbm_dalr)))
        J['root_bending_moment', 'aeroLoads:Px'] = np.reshape(drbm_dalPx, (1, len(drbm_dalPx)))
        J['root_bending_moment', 'aeroLoads:Py'] = np.reshape(drbm_dalPy, (1, len(drbm_dalPy)))
        J['root_bending_moment', 'aeroLoads:Pz'] = np.reshape(drbm_dalPz, (1, len(drbm_dalPz)))
        J['root_bending_moment', 'totalCone'] = np.reshape(drbm_dtotalcone, (1, len(drbm_dtotalcone)))
        J['root_bending_moment', 'x_az'] = np.reshape(drbm_dazx, (1, len(drbm_dazx)))
        J['root_bending_moment', 'y_az'] = np.reshape(drbm_dazy, (1, len(drbm_dazy)))
        J['root_bending_moment', 'z_az'] = np.reshape(drbm_dazz, (1, len(drbm_dazz)))
        J['root_bending_moment', 's'] = np.reshape(drbm_ds, (1, len(drbm_ds)))

        return J



class MassProperties(Component):
    def __init__(self):
        super(MassProperties, self).__init__()
        # variables
        self.add_param('blade_mass', shape=1, units='kg', desc='mass of one blade')
        self.add_param('blade_moment_of_inertia', shape=1, units='kg*m**2', desc='mass moment of inertia of blade about hub')
        self.add_param('tilt', shape=1, units='deg', desc='rotor tilt angle (used to translate moments of inertia from hub to yaw c.s.')

        # parameters
        self.add_param('nBlades', val=3, desc='number of blades', pass_by_obj=True)

        # outputs
        self.add_output('mass_all_blades', shape=1, units='kg', desc='mass of all blades')
        self.add_output('I_all_blades', shape=6, desc='mass moments of inertia of all blades in yaw c.s. order:Ixx, Iyy, Izz, Ixy, Ixz, Iyz')

    def solve_nonlinear(self, params, unknowns, resids):
        # print "MassProperties"
        self.blade_mass = params['blade_mass']
        self.blade_moment_of_inertia = params['blade_moment_of_inertia']
        self.tilt = params['tilt']
        self.nBlades = params['nBlades']

        self.mass_all_blades = self.nBlades * self.blade_mass

        Ibeam = self.nBlades * self.blade_moment_of_inertia

        Ixx = Ibeam
        Iyy = Ibeam/2.0  # azimuthal average for 2 blades, exact for 3+
        Izz = Ibeam/2.0
        Ixy = 0
        Ixz = 0
        Iyz = 0  # azimuthal average for 2 blades, exact for 3+

        # rotate to yaw c.s.
        I = DirectionVector(Ixx, Iyy, Izz).hubToYaw(self.tilt)  # because off-diagonal components are all zero

        self.I_all_blades = np.array([I.x, I.y, I.z, Ixy, Ixz, Iyz])
        self.Ivec = I

        unknowns['mass_all_blades'] = self.mass_all_blades
        unknowns['I_all_blades'] = self.I_all_blades

    def list_deriv_vars(self):

        inputs = ('blade_mass', 'blade_moment_of_inertia', 'tilt')
        outputs = ('mass_all_blades', 'I_all_blades')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):
        I = self.Ivec

        dIx_dmoi = self.nBlades*(I.dx['dx'] + I.dx['dy']/2.0 + I.dx['dz']/2.0)
        dIy_dmoi = self.nBlades*(I.dy['dx'] + I.dy['dy']/2.0 + I.dy['dz']/2.0)
        dIz_dmoi = self.nBlades*(I.dz['dx'] + I.dz['dy']/2.0 + I.dz['dz']/2.0)

        J = {}
        J['mass_all_blades', 'blade_mass'] = self.nBlades
        J['mass_all_blades', 'blade_moment_of_inertia'] = 0.0
        J['mass_all_blades', 'tilt'] = 0.0
        J['I_all_blades', 'blade_mass'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        J['I_all_blades', 'blade_moment_of_inertia'] = np.array([dIx_dmoi, dIy_dmoi, dIz_dmoi, 0.0, 0.0, 0.0])
        J['I_all_blades', 'tilt'] = np.array([ I.dx['dtilt'],  I.dy['dtilt'],  I.dz['dtilt'], 0.0, 0.0, 0.0])

        return J


class TurbineClass(Component):
    def __init__(self):
        super(TurbineClass, self).__init__()
        # parameters
        self.add_param('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True)

        # outputs should be constant
        self.add_output('V_mean', shape=1, units='m/s', desc='IEC mean wind speed for Rayleigh distribution')
        self.add_output('V_extreme', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_output('V_extreme_full', shape=2, units='m/s', desc='IEC extreme wind speed at hub height')
        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        self.turbine_class = params['turbine_class']

        if self.turbine_class == 'I':
            Vref = 50.0
        elif self.turbine_class == 'II':
            Vref = 42.5
        elif self.turbine_class == 'III':
            Vref = 37.5

        unknowns['V_mean'] = 0.2*Vref
        unknowns['V_extreme'] = 1.4*Vref
        unknowns['V_extreme_full'][0] = 1.4*Vref
        unknowns['V_extreme_full'][1] = 1.4*Vref

class ExtremeLoads(Component):
    def __init__(self):
        super(ExtremeLoads, self).__init__()
        # variables
        self.add_param('T', units='N', shape=((2,)), desc='rotor thrust, index 0 is at worst-case, index 1 feathered')
        self.add_param('Q', units='N*m', shape=((2,)), desc='rotor torque, index 0 is at worst-case, index 1 feathered')

        # parameters
        self.add_param('nBlades', val=3, desc='number of blades', pass_by_obj=True)

        # outputs
        self.add_output('T_extreme', shape=1, units='N', desc='rotor thrust at survival wind condition')
        self.add_output('Q_extreme', shape=1, units='N*m', desc='rotor torque at survival wind condition')

    def solve_nonlinear(self, params, unknowns, resids):
        # print "ExtremeLoads"
        n = float(params['nBlades'])
        self.T = params['T']
        self.Q = params['Q']
        self.T_extreme = (self.T[0] + self.T[1]*(n-1)) / n
        self.Q_extreme = (self.Q[0] + self.Q[1]*(n-1)) / n
        unknowns['T_extreme'] = self.T_extreme
        unknowns['Q_extreme'] = 0.0


    def list_deriv_vars(self):

        inputs = ('T', 'Q')
        outputs = ('T_extreme', 'Q_extreme')

        return inputs, outputs


    def linearize(self, params, unknowns, resids):
        n = float(params['nBlades'])
        J = {}
        J['T_extreme', 'T'] = np.reshape(np.array([[1.0/n], [(n-1)/n]]), (1, 2))
        # J['Q_extreme', 'Q'] = np.reshape(np.array([1.0/n, (n-1)/n]), (1, 2))

        return J




class GustETM(Component):
    def __init__(self):
        super(GustETM, self).__init__()
        # variables
        self.add_param('V_mean', shape=1, units='m/s', desc='IEC average wind speed for turbine class')
        self.add_param('V_hub', shape=1, units='m/s', desc='hub height wind speed')

        # parameters
        self.add_param('turbulence_class', val=Enum('A', 'B', 'C'), desc='IEC turbulence class', pass_by_obj=True)
        self.add_param('std', val=3, desc='number of standard deviations for strength of gust', pass_by_obj=True)

        # out
        self.add_output('V_gust', shape=1, units='m/s', desc='gust wind speed')


    def solve_nonlinear(self, params, unknowns, resids):
        # print "GustETM"
        self.V_mean = params['V_mean']
        self.V_hub = params['V_hub']
        self.turbulence_class = params['turbulence_class']
        self.std = params['std']


        if self.turbulence_class == 'A':
            Iref = 0.16
        elif self.turbulence_class == 'B':
            Iref = 0.14
        elif self.turbulence_class == 'C':
            Iref = 0.12

        c = 2.0

        self.sigma = c * Iref * (0.072*(self.V_mean/c + 3)*(self.V_hub/c - 4) + 10)
        self.V_gust = self.V_hub + self.std*self.sigma
        self.Iref = Iref
        self.c = c

        unknowns['V_gust'] = self.V_gust


    def list_deriv_vars(self):

        inputs = ('V_mean', 'V_hub')
        outputs = ('V_gust', )

        return inputs, outputs


    def linearize(self, params, unknowns, resids):
        Iref = self.Iref
        c = self.c
        J = {}
        J['V_gust', 'V_mean'] = self.std*(c*Iref*0.072/c*(self.V_hub/c - 4))
        J['V_gust', 'V_hub'] = 1.0 + self.std*(c*Iref*0.072*(self.V_mean/c + 3)/c)

        return J




class SetupPCModVarSpeed(Component):
    def __init__(self):
        super(SetupPCModVarSpeed, self).__init__()
        self.add_param('control:tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control:pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_param('Vrated', shape=1, units='m/s', desc='rated wind speed')
        self.add_param('R', shape=1, units='m', desc='rotor radius')
        self.add_param('Vfactor', shape=1, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation')

        self.add_output('Uhub', shape=1, units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', shape=1, units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', shape=1, units='deg', desc='pitch angles to run')
        self.add_output('azimuth', shape=1, units='deg')

    def solve_nonlinear(self, params, unknowns, resids):
        print 'control_tsr', params['control:tsr']
        self.Vrated = params['Vrated']
        self.R = params['R']
        self.Vfactor = params['Vfactor']

        self.Uhub = self.Vfactor * self.Vrated
        self.Omega = params['control:tsr']*self.Uhub/self.R*RS2RPM
        self.pitch = params['control:pitch']

        unknowns['Uhub'] = self.Uhub
        unknowns['Omega'] = self.Omega
        unknowns['pitch'] = self.pitch
        unknowns['azimuth'] = 0.0


    def list_deriv_vars(self):

        inputs = ('control:tsr', 'Vrated', 'R')
        outputs = ('Uhub', 'Omega', 'pitch')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        J = {}
        J['Uhub', 'control:tsr'] = 0.0
        J['Uhub', 'Vrated'] = self.Vfactor
        J['Uhub', 'R'] = 0.0
        J['Omega', 'control:tsr'] = self.Uhub/self.R*RS2RPM
        J['Omega', 'Vrated'] = params['control:tsr']*self.Vfactor/self.R*RS2RPM
        J['Omega', 'R'] = -params['control:tsr']*self.Uhub/self.R**2*RS2RPM
        J['pitch', 'control:tsr'] = 0.0
        J['pitch', 'Vrated'] = 0.0
        J['pitch', 'R'] = 0.0

        return J


class OutputsAero(Component):
    def __init__(self):
        super(OutputsAero, self).__init__()
        pbo = False
        # --- outputs ---
        self.add_param('AEP_in', shape=1, units='kW*h', desc='annual energy production')
        self.add_param('V_in', shape=200, units='m/s', desc='wind speeds (power curve)')
        self.add_param('P_in', shape=200, units='W', desc='power (power curve)')
        self.add_param('Omega_in', shape=1, units='rpm', desc='speed (power curve)')

        self.add_param('ratedConditions:V_in', shape=1, units='m/s', desc='rated wind speed')
        self.add_param('ratedConditions:Omega_in', shape=1, units='rpm', desc='rotor rotation speed at rated')
        self.add_param('ratedConditions:pitch_in', shape=1, units='deg', desc='pitch setting at rated')
        self.add_param('ratedConditions:T_in', shape=1, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_param('ratedConditions:Q_in', shape=1, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_param('hub_diameter_in', shape=1, units='m', desc='hub diameter')
        self.add_param('diameter_in', shape=1, units='m', desc='rotor diameter')
        self.add_param('V_extreme_in', shape=1, units='m/s', desc='survival wind speed')
        self.add_param('T_extreme_in', shape=1, units='N', desc='thrust at survival wind condition')
        self.add_param('Q_extreme_in', shape=1, units='N*m', desc='thrust at survival wind condition')

        # internal use outputs
        self.add_param('Rtip_in', shape=1, units='m', desc='tip location in z_b')
        self.add_param('precurveTip_in', shape=1, units='m', desc='tip location in x_b')
        self.add_param('presweepTip_in', val=0.0, units='m', desc='tip location in y_b')  # TODO: connect later

        # --- outputs ---
        self.add_output('AEP', shape=1, units='kW*h', desc='annual energy production')
        self.add_output('V', shape=200, units='m/s', desc='wind speeds (power curve)', pass_by_obj=pbo)
        self.add_output('P', shape=200, units='W', desc='power (power curve)', pass_by_obj=pbo)
        self.add_output('Omega', shape=1, units='rpm', desc='speed (power curve)', pass_by_obj=pbo)

        self.add_output('ratedConditions:V', shape=1, units='m/s', desc='rated wind speed', pass_by_obj=pbo)
        self.add_output('ratedConditions:Omega', shape=1, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('ratedConditions:pitch', shape=1, units='deg', desc='pitch setting at rated', pass_by_obj=pbo)
        self.add_output('ratedConditions:T', shape=1, units='N', desc='rotor aerodynamic thrust at rated', pass_by_obj=pbo)
        self.add_output('ratedConditions:Q', shape=1, units='N*m', desc='rotor aerodynamic torque at rated', pass_by_obj=pbo)

        self.add_output('hub_diameter', shape=1, units='m', desc='hub diameter', pass_by_obj=pbo)
        self.add_output('diameter', shape=1, units='m', desc='rotor diameter', pass_by_obj=pbo)
        self.add_output('V_extreme', shape=1, units='m/s', desc='survival wind speed', pass_by_obj=pbo)
        self.add_output('T_extreme', shape=1, units='N', desc='thrust at survival wind condition', pass_by_obj=pbo)
        self.add_output('Q_extreme', shape=1, units='N*m', desc='thrust at survival wind condition', pass_by_obj=pbo)

        # internal use outputs
        self.add_output('Rtip', shape=1, units='m', desc='tip location in z_b', pass_by_obj=pbo)
        self.add_output('precurveTip', shape=1, units='m', desc='tip location in x_b', pass_by_obj=pbo)
        self.add_output('presweepTip', val=0.0, units='m', desc='tip location in y_b', pass_by_obj=pbo)  # TODO: connect later

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['AEP'] = params['AEP_in']
        unknowns['V'] = params['V_in']
        unknowns['P'] = params['P_in']
        unknowns['Omega'] = params['Omega_in']
        unknowns['ratedConditions:V'] = params['ratedConditions:V_in']
        unknowns['ratedConditions:Omega'] = params['ratedConditions:Omega_in']
        unknowns['ratedConditions:pitch'] = params['ratedConditions:pitch_in']
        unknowns['ratedConditions:T'] = params['ratedConditions:T_in']
        unknowns['ratedConditions:Q'] = params['ratedConditions:Q_in']
        unknowns['hub_diameter'] = params['hub_diameter_in']
        unknowns['diameter'] = params['diameter_in']
        unknowns['V_extreme'] = params['V_extreme_in']
        unknowns['T_extreme'] = params['T_extreme_in']
        unknowns['Q_extreme'] = params['Q_extreme_in']
        unknowns['Rtip'] = params['Rtip_in']
        unknowns['precurveTip'] = params['precurveTip_in']
        unknowns['presweepTip'] = params['presweepTip_in']

    def linearize(self, params, unknowns,resids):
        J = {}
        J['AEP', 'AEP_in'] = 1
        J['V', 'V_in'] = np.diag(np.ones(len(params['V_in'])))
        J['P', 'P_in'] = np.diag(np.ones(len(params['P_in'])))
        J['Omega', 'Omega_in'] = 1
        J['ratedConditions:V', 'ratedConditions:V_in'] = 1
        J['ratedConditions:Omega', 'ratedConditions:Omega_in'] = 1
        J['ratedConditions:pitch', 'ratedConditions:pitch_in'] = 1
        J['ratedConditions:T', 'ratedConditions:T_in'] = 1
        J['ratedConditions:Q', 'ratedConditions:Q_in'] = 1
        J['hub_diameter', 'hub_diameter_in'] = 1
        J['diameter', 'diameter_in'] = 1
        J['V_extreme', 'V_extreme_in'] = 1
        J['T_extreme', 'T_extreme_in'] = 1
        J['Q_extreme', 'Q_extreme_in'] = 1
        J['Rtip', 'Rtip_in'] = 1
        J['precurveTip', 'precurveTip_in'] = 1
        J['presweepTip', 'T_presweepTip_in'] = 1

        return J

class OutputsStructures(Component):
    def __init__(self, nstr):
        super(OutputsStructures, self).__init__()

        # structural outputs
        self.add_param('mass_one_blade_in', shape=1, units='kg', desc='mass of one blade')
        self.add_param('mass_all_blades_in', shape=1,  units='kg', desc='mass of all blade')
        self.add_param('I_all_blades_in', shape=6, desc='out of plane moments of inertia in yaw-aligned c.s.')
        self.add_param('freq_in', shape=5, units='Hz', desc='1st nF natural frequencies')
        self.add_param('freq_curvefem_in', shape=5, units='Hz', desc='1st nF natural frequencies')
        self.add_param('tip_deflection_in', shape=1, units='m', desc='blade tip deflection in +x_y direction')
        self.add_param('strainU_spar_in', shape=nstr, desc='axial strain and specified locations')
        self.add_param('strainL_spar_in', shape=nstr, desc='axial strain and specified locations')
        self.add_param('strainU_te_in', shape=nstr, desc='axial strain and specified locations')
        self.add_param('strainL_te_in', shape=nstr, desc='axial strain and specified locations')
        self.add_param('eps_crit_spar_in', shape=nstr, desc='critical strain in spar from panel buckling calculation')
        self.add_param('eps_crit_te_in', shape=nstr,  desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_param('root_bending_moment_in', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')
        self.add_param('damageU_spar_in', shape=nstr, desc='fatigue damage on upper surface in spar cap')
        self.add_param('damageL_spar_in', shape=nstr, desc='fatigue damage on lower surface in spar cap')
        self.add_param('damageU_te_in', shape=nstr, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_param('damageL_te_in', shape=nstr, desc='fatigue damage on lower surface in trailing-edge panels')
        self.add_param('delta_bladeLength_out_in', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
        self.add_param('delta_precurve_sub_out_in', shape=3, units='m', desc='adjustment to precurve to account for curvature from loading')

        # structural outputs
        pbo = True
        self.add_output('mass_one_blade', shape=1, units='kg', desc='mass of one blade')
        self.add_output('mass_all_blades', shape=1,  units='kg', desc='mass of all blade')
        self.add_output('I_all_blades', shape=6, desc='out of plane moments of inertia in yaw-aligned c.s.', pass_by_obj=pbo)
        self.add_output('freq', shape=5, units='Hz', desc='1st nF natural frequencies', pass_by_obj=pbo)
        self.add_output('freq_curvefem', shape=5, units='Hz', desc='1st nF natural frequencies')
        self.add_output('tip_deflection', shape=1, units='m', desc='blade tip deflection in +x_y direction', pass_by_obj=pbo)
        self.add_output('strainU_spar', shape=nstr, desc='axial strain and specified locations')
        self.add_output('strainL_spar', shape=nstr, desc='axial strain and specified locations')
        self.add_output('strainU_te', shape=nstr, desc='axial strain and specified locations')
        self.add_output('strainL_te', shape=nstr, desc='axial strain and specified locations')
        self.add_output('eps_crit_spar', shape=nstr, desc='critical strain in spar from panel buckling calculation')
        self.add_output('eps_crit_te', shape=nstr,  desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_output('root_bending_moment', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade', pass_by_obj=pbo)
        self.add_output('damageU_spar', shape=nstr, desc='fatigue damage on upper surface in spar cap', pass_by_obj=pbo)
        self.add_output('damageL_spar', shape=nstr, desc='fatigue damage on lower surface in spar cap', pass_by_obj=pbo)
        self.add_output('damageU_te', shape=nstr, desc='fatigue damage on upper surface in trailing-edge panels', pass_by_obj=pbo)
        self.add_output('damageL_te', shape=nstr, desc='fatigue damage on lower surface in trailing-edge panels', pass_by_obj=pbo)
        self.add_output('delta_bladeLength_out', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading', pass_by_obj=pbo)
        self.add_output('delta_precurve_sub_out', shape=3, units='m', desc='adjustment to precurve to account for curvature from loading', pass_by_obj=pbo)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['mass_one_blade'] = params['mass_one_blade_in']
        print "mass_all_blades", params['mass_all_blades_in']
        unknowns['mass_all_blades'] = params['mass_all_blades_in']
        unknowns['I_all_blades'] = params['I_all_blades_in']
        unknowns['freq'] = params['freq_in']
        unknowns['freq_curvefem'] = params['freq_curvefem_in']
        unknowns['tip_deflection'] = params['tip_deflection_in']
        unknowns['strainU_spar'] = params['strainU_spar_in']
        unknowns['strainL_spar'] = params['strainL_spar_in']
        unknowns['strainU_te'] = params['strainU_te_in']
        unknowns['strainL_te'] = params['strainL_te_in']
        unknowns['eps_crit_spar'] = params['eps_crit_spar_in']
        unknowns['eps_crit_te'] = params['eps_crit_te_in']
        unknowns['root_bending_moment'] = params['root_bending_moment_in']
        unknowns['damageU_spar'] = params['damageU_spar_in']
        unknowns['damageL_spar'] = params['damageL_spar_in']
        unknowns['damageU_te'] = params['damageU_te_in']
        unknowns['damageL_te'] = params['damageL_te_in']
        unknowns['delta_bladeLength_out'] = params['delta_bladeLength_out_in']
        unknowns['delta_precurve_sub_out'] = params['delta_precurve_sub_out_in']

    def linearize(self, params, unknowns,resids):
        J = {}
        J['mass_one_blade', 'mass_one_blade_in'] = 1
        J['mass_all_blades', 'mass_all_blades_in'] = 1
        J['I_all_blades', 'I_all_blades_in'] = np.diag(np.ones(len(params['I_all_blades_in'])))
        J['freq', 'freq_in'] = np.diag(np.ones(len(params['freq_in'])))
        J['freq_curvefem', 'freq_curvefem_in'] = np.diag(np.ones(len(params['freq_curvefem_in'])))
        J['tip_deflection', 'tip_deflection_in'] = 1
        J['strainU_spar', 'strainU_spar_in'] = np.diag(np.ones(len(params['strainU_spar_in'])))
        J['strainL_spar', 'strainL_spar_in'] = np.diag(np.ones(len(params['strainL_spar_in'])))
        J['strainU_te', 'strainU_te_in'] = np.diag(np.ones(len(params['strainU_te_in'])))
        J['strainL_te', 'strainL_te_in'] = np.diag(np.ones(len(params['strainL_te_in'])))
        J['eps_crit_spar', 'eps_crit_spar_in'] = np.diag(np.ones(len(params['eps_crit_spar_in'])))
        J['eps_crit_te', 'eps_crit_te_in'] = np.diag(np.ones(len(params['eps_crit_te_in'])))
        J['root_bending_moment', 'root_bending_moment_in'] = 1
        J['damageU_spar', 'damageU_spar_in'] = np.diag(np.ones(len(params['damageU_spar_in'])))
        J['damageL_spar', 'damageL_spar_in'] = np.diag(np.ones(len(params['damageL_spar_in'])))
        J['damageU_te', 'damageU_te_in'] = np.diag(np.ones(len(params['damageU_te_in'])))
        J['damageL_te', 'damageL_te_in'] = np.diag(np.ones(len(params['damageL_te_in'])))
        J['delta_bladeLength_out', 'delta_bladeLength_out_in'] = 1
        J['delta_precurve_sub_out', 'delta_precurve_sub_out_in'] = np.diag(np.ones(len(params['delta_precurve_sub_out_in'])))

        return J

class ObjandCons(Component):
    def __init__(self, nstr, npower, num_airfoils, airfoils_dof):
        super(ObjandCons, self).__init__()
        self.add_param('COE', val=0.1)
        self.add_param('strainU_spar', val=np.zeros(nstr))
        self.add_param('strain_ult_spar', val=0.0)
        self.add_param('strainU_te', val=np.zeros(nstr))
        self.add_param('strain_ult_te', val=0.0)
        self.add_param('strainL_te', val=np.zeros(nstr))
        self.add_param('eps_crit_spar', val=np.zeros(nstr))
        self.add_param('eps_crit_te', val=np.zeros(nstr))
        self.add_param('freq_curvefem', val=np.zeros(5))
        self.add_param('ratedConditions:Omega', val=0.0)
        self.add_param('nBlades', val=3, pass_by_obj=True)
        self.add_param('airfoil_parameterization', val=np.zeros((num_airfoils,airfoils_dof)))
        self.add_param('power', val=np.zeros(npower))
        self.add_param('control:ratedPower', val=0.0)

        self.add_output('obj', val=1.0)
        self.add_output('con1', val=np.zeros(7))
        self.add_output('con2', val=np.zeros(8))
        self.add_output('con3', val=np.zeros(8))
        self.add_output('con4', val=np.zeros(8))
        self.add_output('con5', val=np.zeros(7))
        self.add_output('con6', val=np.zeros(2))
        if airfoils_dof == 8:
            self.add_output('con_freeform', val=np.zeros((num_airfoils,airfoils_dof/2)))
        self.add_output('con_power', val=0.0)
        self.airfoils_dof = airfoils_dof

    def solve_nonlinear(self, params, unknowns, resids):
        self.eta_strain = 1.35*1.3*1.0
        self.con1_indices = [0, 12, 14, 18, 22, 28, 34]
        self.con2_indices = [0, 8, 12, 14, 18, 22, 28, 34]
        self.con3_indices = self.con2_indices
        self.con4_indices = [10, 12, 14, 20, 23, 27, 31, 33]
        self.con5_indices = [10, 12, 13, 14, 21, 28, 33]
        unknowns['obj'] = params['COE']*100.0
        unknowns['con1'] = params['strainU_spar'][self.con1_indices]*self.eta_strain/params['strain_ult_spar']
        unknowns['con2'] = params['strainU_te'][self.con2_indices]*self.eta_strain/params['strain_ult_te']
        unknowns['con3'] = params['strainL_te'][self.con3_indices]*self.eta_strain/params['strain_ult_te']
        unknowns['con4'] = (params['eps_crit_spar'][self.con4_indices] - params['strainU_spar'][self.con4_indices]) / params['strain_ult_spar']
        unknowns['con5'] = (params['eps_crit_te'][self.con5_indices] - params['strainU_te'][self.con5_indices]) / params['strain_ult_te']
        unknowns['con6'] = params['freq_curvefem'][0:2] - params['nBlades']*params['ratedConditions:Omega']/60.0*1.1
        if self.airfoils_dof == 8:
            unknowns['con_freeform'] = params['airfoil_parameterization'][:, [4, 5, 6, 7]] - params['airfoil_parameterization'][:, [0, 1, 2, 3]]
        unknowns['con_power'] = (params['power'][-1] - params['control:ratedPower']) # / 1.e6

    def linearize(self, params, unknowns, resids):
        J = {}
        dcon1_dstrainU_spar = np.zeros((7, 38))
        dcon1_dstrain_ult_spar = -params['strainU_spar'][self.con1_indices]*self.eta_strain/params['strain_ult_spar']**2
        dcon2_dstrainU_te = np.zeros((8, 38))
        dcon2_dstrain_ult_te = -params['strainU_te'][self.con2_indices]*self.eta_strain/params['strain_ult_te']**2
        dcon3_dstrainL_te = np.zeros((8, 38))
        dcon3_dstrain_ult_te = -params['strainL_te'][self.con3_indices]*self.eta_strain/params['strain_ult_te']**2
        dcon4_deps_crit_spar = np.zeros((8, 38))
        dcon4_dstrainU_spar = np.zeros((8, 38))
        dcon4_dstrain_ult_spar = -(params['eps_crit_spar'][self.con4_indices] - params['strainU_spar'][self.con4_indices]) / params['strain_ult_spar']**2
        dcon5_deps_crit_te = np.zeros((7, 38))
        dcon5_dstrainU_te = np.zeros((7, 38))
        dcon5_dstrain_ult_te = -(params['eps_crit_te'][self.con5_indices] - params['strainU_te'][self.con5_indices]) / params['strain_ult_te']**2
        dcon6_dfreq = np.zeros((2, 5))
        dcon6_dfreq[0][0], dcon6_dfreq[1][1] = 1.0, 1.0
        dcon6_dOmega = -params['nBlades']*np.ones(2)/60.0*1.1
        dcon_freeform_dafp = np.zeros((24,48))
        for i in range(6):
            dcon_freeform_dafp[np.ix_(range(i*4,i*4+4), range(i*8,i*8+8))] += np.hstack((np.diag(-np.ones(4)), np.diag(np.ones(4))))

        for i in range(7):
            dcon1_dstrainU_spar[i][self.con1_indices[i]] = self.eta_strain / params['strain_ult_spar']
            dcon5_deps_crit_te[i][self.con5_indices[i]] = 1.0 / params['strain_ult_te']
            dcon5_dstrainU_te[i][self.con5_indices[i]] = -1.0 / params['strain_ult_te']
        for i in range(8):
            dcon2_dstrainU_te[i][self.con2_indices[i]] = self.eta_strain / params['strain_ult_te']
            dcon3_dstrainL_te[i][self.con3_indices[i]] = self.eta_strain / params['strain_ult_te']
            dcon4_deps_crit_spar[i][self.con4_indices[i]] = 1.0 / params['strain_ult_spar']
            dcon4_dstrainU_spar[i][self.con4_indices[i]] = -1.0 / params['strain_ult_spar']

        J['obj', 'COE'] = 100.0
        J['con1', 'strainU_spar'] = dcon1_dstrainU_spar
        J['con1', 'strain_ult_spar'] = dcon1_dstrain_ult_spar
        J['con2', 'strainU_te'] = dcon2_dstrainU_te
        J['con2', 'strain_ult_te'] = dcon2_dstrain_ult_te
        J['con3', 'strainL_te'] = dcon3_dstrainL_te
        J['con3', 'strain_ult_te'] = dcon3_dstrain_ult_te
        J['con4', 'eps_crit_spar'] = dcon4_deps_crit_spar
        J['con4', 'strainU_spar'] = dcon4_dstrainU_spar
        J['con4', 'strain_ult_spar'] = dcon4_dstrain_ult_spar
        J['con5', 'eps_crit_te'] = dcon5_deps_crit_te
        J['con5', 'strainU_te'] = dcon5_dstrainU_te
        J['con5', 'strain_ult_te'] = dcon5_dstrain_ult_te
        J['con6', 'freq_curvefem'] = dcon6_dfreq
        J['con6', 'ratedConditions:Omega'] = dcon6_dOmega
        if self.airfoils_dof == 8:
            J['con_freeform', 'airfoil_parameterization'] = dcon_freeform_dafp
        J['con_power', 'power'] = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]).reshape(1,5)
        J['con_power', 'control:ratedPower'] = -1.0
        return J


class StructureGroup(Group):
    def __init__(self, naero, nstr, num_airfoils, airfoils_dof):
        super(StructureGroup, self).__init__()
        # self.add('curvature', BladeCurvature(nstr))
        self.add('resize', ResizeCompositeSection(nstr, num_airfoils, airfoils_dof))
        # self.add('gust', GustETM())
        # self.add('setuppc',  SetupPCModVarSpeed())

        # self.add('aero_rated', CCBlade('loads', naero, 1)) # 'loads', naero, 1))
        # self.add('aero_extrm', CCBlade('loads', naero,  1))
        # self.add('aero_extrm_forces', CCBlade('power', naero, 2))
        # self.add('aero_defl_powercurve', CCBlade('loads', naero,  1))
        #
        self.add('beam', PreCompSections(nstr, num_airfoils, airfoils_dof))
        # self.add('loads_defl', TotalLoads(nstr))
        # self.add('loads_pc_defl', TotalLoads(nstr))
        # self.add('loads_strain', TotalLoads(nstr))
        # self.add('damage', DamageLoads(nstr))
        # self.add('struc', RotorWithpBEAM(nstr))
        self.add('curvefem', CurveFEM(nstr))
        # self.add('tip', TipDeflection())
        # self.add('root_moment', RootMoment(nstr))
        # self.add('mass', MassProperties())
        # self.add('extreme', ExtremeLoads())
        # self.add('blade_defl', BladeDeflection(nstr))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'

class OptimizeRotorSE(Group):
    def __init__(self, naero, nstr):
        super(OptimizeRotorSE, self).__init__()
        self.add('rotor', RotorSE(naero, nstr, False), promotes=['r_max_chord', 'chord_sub', 'theta_sub', 'control:tsr', 'mass_all_blades', 'AEP', 'obj', 'powercurve.control:Vin', 'powercurve.control:Vout', 'con1', 'con2', 'con3', 'con4', 'con5', 'con6', 'airfoil_parameterization', 'con_freeform', 'concon'])

class RotorSE(Group):
    def __init__(self, naero, nstr, npower, num_airfoils, airfoils_dof, vars=True):
        super(RotorSE, self).__init__()
        """rotor model"""
        n3 = 4
        n5 = 5
        if vars:
            self.add('initial_aero_grid', IndepVarComp('initial_aero_grid', np.zeros(naero)), promotes=['*'])
            self.add('initial_str_grid', IndepVarComp('initial_str_grid', np.zeros(nstr)), promotes=['*'])
            self.add('idx_cylinder_aero', IndepVarComp('idx_cylinder_aero', 1), promotes=['*'])
            self.add('idx_cylinder_str', IndepVarComp('idx_cylinder_str', 1), promotes=['*'])
            self.add('hubFraction', IndepVarComp('hubFraction', 0.0), promotes=['*'])
            self.add('r_aero', IndepVarComp('r_aero', np.zeros(naero)), promotes=['*'])
            self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
            self.add('chord_sub', IndepVarComp('chord_sub', np.zeros(n3),units='m'), promotes=['*'])
            self.add('theta_sub', IndepVarComp('theta_sub', np.zeros(n3), units='deg'), promotes=['*'])
            self.add('precurve_sub', IndepVarComp('precurve_sub', np.zeros(3), units='m'), promotes=['*'])
            self.add('delta_precurve_sub', IndepVarComp('delta_precurve_sub', np.zeros(3)), promotes=['*'])
            self.add('bladeLength', IndepVarComp('bladeLength', 0.0, units='m'), promotes=['*'])
            self.add('precone', IndepVarComp('precone', 0.0, units='deg'), promotes=['*'])
            self.add('tilt', IndepVarComp('tilt', 0.0, units='deg'), promotes=['*'])
            self.add('yaw', IndepVarComp('yaw', 0.0, units='deg'), promotes=['*'])
            self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])
            self.add('airfoil_files', IndepVarComp('airfoil_files', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])
            self.add('airfoil_parameterization', IndepVarComp('airfoil_parameterization', val=np.zeros((num_airfoils, airfoils_dof))), promotes=['*'])
            self.add('airfoil_analysis_options', IndepVarComp('airfoil_analysis_options', {}, pass_by_obj=True), promotes=['*'])
            self.add('rho', IndepVarComp('rho', val=1.225, units='kg/m**3', desc='density of air', pass_by_obj=True), promotes=['*'])
            self.add('mu', IndepVarComp('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air', pass_by_obj=True), promotes=['*'])
            self.add('shearExp', IndepVarComp('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True), promotes=['*'])
            self.add('hubHt', IndepVarComp('hubHt', val=np.zeros(1), units='m', desc='hub height'), promotes=['*'])
            self.add('turbine_class', IndepVarComp('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
            self.add('turbulence_class', IndepVarComp('turbulence_class', val=Enum('B', 'A', 'C'), desc='IEC turbulence class class', pass_by_obj=True), promotes=['*'])
            self.add('g', IndepVarComp('g', val=9.81, units='m/s**2', desc='acceleration of gravity', pass_by_obj=True), promotes=['*'])
            self.add('cdf_reference_height_wind_speed', IndepVarComp('cdf_reference_height_wind_speed', val=np.zeros(1), units='m', desc='reference hub height for IEC wind speed (used in CDF calculation)'), promotes=['*'])
            self.add('VfactorPC', IndepVarComp('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation'), promotes=['*'])

            # --- composite sections ---
            self.add('sparT', IndepVarComp('sparT', val=np.zeros(n5), units='m', desc='spar cap thickness parameters'), promotes=['*'])
            self.add('teT', IndepVarComp('teT', val=np.zeros(n5), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])
            self.add('chord_str_ref', IndepVarComp('chord_str_ref', val=np.zeros(nstr), units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)'), promotes=['*'])
            self.add('leLoc', IndepVarComp('leLoc', val=np.zeros(nstr), desc='array of leading-edge positions from a reference blade axis \
                (usually blade pitch axis). locations are normalized by the local chord length.  \
                e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
                positive in -x direction for airfoil-aligned coordinate system'), promotes=['*'])

            self.add('profile', IndepVarComp('profile', val=np.zeros(nstr), desc='airfoil shape at each radial position', pass_by_obj=True), promotes=['*'])
            self.add('materials', IndepVarComp('materials', val=np.zeros(6),
                desc='list of all Orthotropic2DMaterial objects used in defining the geometry', pass_by_obj=True), promotes=['*'])
            self.add('upperCS', IndepVarComp('upperCS', val=np.zeros(nstr),
                desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True), promotes=['*'])
            self.add('lowerCS', IndepVarComp('lowerCS', val=np.zeros(nstr),
                desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True), promotes=['*'])
            self.add('websCS', IndepVarComp('websCS', val=np.zeros(nstr),
                desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True), promotes=['*'])
            self.add('sector_idx_strain_spar', IndepVarComp('sector_idx_strain_spar', val=np.zeros(38,  dtype=np.int), desc='index of sector for spar (PreComp definition of sector)', pass_by_obj=True), promotes=['*'])
            self.add('sector_idx_strain_te', IndepVarComp('sector_idx_strain_te', val=np.zeros(38,  dtype=np.int), desc='index of sector for trailing-edge (PreComp definition of sector)', pass_by_obj=True), promotes=['*'])

            # --- control ---
            self.add('c_Vin', IndepVarComp('control:Vin', val=0.0, units='m/s', desc='cut-in wind speed'), promotes=['*'])
            self.add('c_Vout', IndepVarComp('control:Vout', val=0.0, units='m/s', desc='cut-out wind speed'), promotes=['*'])
            self.add('c_ratedPower', IndepVarComp('control:ratedPower', val=0.0,  units='W', desc='rated power'), promotes=['*'])
            self.add('c_minOmega', IndepVarComp('control:minOmega', val=0.0, units='rpm', desc='minimum allowed rotor rotation speed'), promotes=['*'])
            self.add('c_maxOmega', IndepVarComp('control:maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed'), promotes=['*'])
            self.add('c_tsr', IndepVarComp('control:tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)'), promotes=['*'])
            self.add('c_pitch', IndepVarComp('control:pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)'), promotes=['*'])
            self.add('pitch_extreme', IndepVarComp('pitch_extreme', val=0.0, units='deg', desc='worst-case pitch at survival wind condition'), promotes=['*'])
            self.add('pitch_extreme_full', IndepVarComp('pitch_extreme_full', val=np.array([0.0, 90.0]), units='deg', desc='worst-case pitch at survival wind condition'), promotes=['*'])
            self.add('azimuth_extreme', IndepVarComp('azimuth_extreme', val=0.0, units='deg', desc='worst-case azimuth at survival wind condition'), promotes=['*'])
            self.add('Omega_load', IndepVarComp('Omega_load', val=0.0, units='rpm', desc='worst-case azimuth at survival wind condition'), promotes=['*'])

            # --- drivetrain efficiency ---
            self.add('drivetrainType', IndepVarComp('drivetrainType', val=Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True), promotes=['*'])

            # --- fatigue ---
            self.add('rstar_damage', IndepVarComp('rstar_damage', val=np.zeros(18), desc='nondimensional radial locations of damage equivalent moments'), promotes=['*'])
            self.add('Mxb_damage', IndepVarComp('Mxb_damage', val=np.zeros(18), units='N*m', desc='damage equivalent moments about blade c.s. x-direction'), promotes=['*'])
            self.add('Myb_damage', IndepVarComp('Myb_damage', val=np.zeros(18), units='N*m', desc='damage equivalent moments about blade c.s. y-direction'), promotes=['*'])
            self.add('strain_ult_spar', IndepVarComp('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap'), promotes=['*'])
            self.add('strain_ult_te', IndepVarComp('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels'), promotes=['*'])
            self.add('eta_damage', IndepVarComp('eta_damage', val=1.755, desc='safety factor for fatigue'), promotes=['*'])
            self.add('m_damage', IndepVarComp('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis'), promotes=['*'])
            self.add('N_damage', IndepVarComp('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis'), promotes=['*'])

            # --- options ---
            self.add('nSector', IndepVarComp('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True), promotes=['*'])
            self.add('npts_coarse_power_curve', IndepVarComp('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at', pass_by_obj=True), promotes=['*'])
            self.add('npts_spline_power_curve', IndepVarComp('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve', pass_by_obj=True), promotes=['*'])
            self.add('AEP_loss_factor', IndepVarComp('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)'), promotes=['*'])
            self.add('dynamic_amplication_tip_deflection', IndepVarComp('dynamic_amplication_tip_deflection', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation'), promotes=['*'])
            self.add('nF', IndepVarComp('nF', val=5, desc='number of natural frequencies to compute', pass_by_obj=True), promotes=['*'])

            self.add('weibull_shape', IndepVarComp('weibull_shape', val=0.0), promotes=['*'])
        else:
            self.add('initial_aero_grid', IndepVarComp('initial_aero_grid', np.zeros(naero), pass_by_obj=True), promotes=['*'])
            self.add('initial_str_grid', IndepVarComp('initial_str_grid', np.zeros(nstr), pass_by_obj=True), promotes=['*'])
            self.add('idx_cylinder_aero', IndepVarComp('idx_cylinder_aero', 1, pass_by_obj=True), promotes=['*'])
            self.add('idx_cylinder_str', IndepVarComp('idx_cylinder_str', 1, pass_by_obj=True), promotes=['*'])
            self.add('hubFraction', IndepVarComp('hubFraction', 0.0, pass_by_obj=True), promotes=['*'])
            self.add('r_aero', IndepVarComp('r_aero', np.zeros(naero), pass_by_obj=True), promotes=['*'])
            self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
            self.add('chord_sub', IndepVarComp('chord_sub', np.zeros(n3),units='m'), promotes=['*'])
            self.add('theta_sub', IndepVarComp('theta_sub', np.zeros(n3), units='deg'), promotes=['*'])
            self.add('precurve_sub', IndepVarComp('precurve_sub', np.zeros(3), units='m', pass_by_obj=True), promotes=['*'])
            self.add('delta_precurve_sub', IndepVarComp('delta_precurve_sub', np.zeros(3), pass_by_obj=True), promotes=['*'])
            self.add('bladeLength', IndepVarComp('bladeLength', 0.0, units='m', pass_by_obj=True), promotes=['*'])
            self.add('precone', IndepVarComp('precone', 0.0, units='deg', pass_by_obj=True), promotes=['*'])
            self.add('tilt', IndepVarComp('tilt', 0.0, units='deg', pass_by_obj=True), promotes=['*'])
            self.add('yaw', IndepVarComp('yaw', 0.0, units='deg', pass_by_obj=True), promotes=['*'])
            self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])
            self.add('airfoil_files', IndepVarComp('airfoil_files', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])
            self.add('airfoil_parameterization', IndepVarComp('airfoil_parameterization', val=np.zeros((num_airfoils, airfoils_dof))), promotes=['*'])
            self.add('airfoil_analysis_options', IndepVarComp('airfoil_analysis_options', {}, pass_by_obj=True), promotes=['*'])
            self.add('rho', IndepVarComp('rho', val=1.225, units='kg/m**3', desc='density of air', pass_by_obj=True), promotes=['*'])
            self.add('mu', IndepVarComp('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air', pass_by_obj=True), promotes=['*'])
            self.add('shearExp', IndepVarComp('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True), promotes=['*'])
            self.add('hubHt', IndepVarComp('hubHt', val=np.zeros(1), units='m', desc='hub height'), promotes=['*'])
            self.add('turbine_class', IndepVarComp('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
            self.add('turbulence_class', IndepVarComp('turbulence_class', val=Enum('B', 'A', 'C'), desc='IEC turbulence class class', pass_by_obj=True), promotes=['*'])
            self.add('g', IndepVarComp('g', val=9.81, units='m/s**2', desc='acceleration of gravity', pass_by_obj=True), promotes=['*'])
            self.add('cdf_reference_height_wind_speed', IndepVarComp('cdf_reference_height_wind_speed', val=np.zeros(1), units='m', desc='reference hub height for IEC wind speed (used in CDF calculation)', pass_by_obj=True), promotes=['*'])
            self.add('VfactorPC', IndepVarComp('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation', pass_by_obj=True), promotes=['*'])

            # --- composite sections ---
            self.add('sparT', IndepVarComp('sparT', val=np.zeros(n5), units='m', desc='spar cap thickness parameters'), promotes=['*'])
            self.add('teT', IndepVarComp('teT', val=np.zeros(n5), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])
            self.add('chord_str_ref', IndepVarComp('chord_str_ref', val=np.zeros(nstr), units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)', pass_by_obj=True), promotes=['*'])
            self.add('leLoc', IndepVarComp('leLoc', val=np.zeros(nstr), desc='array of leading-edge positions from a reference blade axis \
                (usually blade pitch axis). locations are normalized by the local chord length.  \
                e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
                positive in -x direction for airfoil-aligned coordinate system', pass_by_obj=True), promotes=['*'])

            self.add('profile', IndepVarComp('profile', val=np.zeros(nstr), desc='airfoil shape at each radial position', pass_by_obj=True), promotes=['*'])
            self.add('materials', IndepVarComp('materials', val=np.zeros(6),
                desc='list of all Orthotropic2DMaterial objects used in defining the geometry', pass_by_obj=True), promotes=['*'])
            self.add('upperCS', IndepVarComp('upperCS', val=np.zeros(nstr),
                desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True), promotes=['*'])
            self.add('lowerCS', IndepVarComp('lowerCS', val=np.zeros(nstr),
                desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True), promotes=['*'])
            self.add('websCS', IndepVarComp('websCS', val=np.zeros(nstr),
                desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True), promotes=['*'])
            self.add('sector_idx_strain_spar', IndepVarComp('sector_idx_strain_spar', val=np.zeros(38,  dtype=np.int), desc='index of sector for spar (PreComp definition of sector)', pass_by_obj=True), promotes=['*'])
            self.add('sector_idx_strain_te', IndepVarComp('sector_idx_strain_te', val=np.zeros(38,  dtype=np.int), desc='index of sector for trailing-edge (PreComp definition of sector)', pass_by_obj=True), promotes=['*'])

            # --- control ---
            self.add('c_Vin', IndepVarComp('control:Vin', val=0.0, units='m/s', desc='cut-in wind speed', pass_by_obj=True), promotes=['*'])
            self.add('c_Vout', IndepVarComp('control:Vout', val=0.0, units='m/s', desc='cut-out wind speed', pass_by_obj=True), promotes=['*'])
            self.add('c_ratedPower', IndepVarComp('control:ratedPower', val=0.0,  units='W', desc='rated power', pass_by_obj=True), promotes=['*'])
            self.add('c_minOmega', IndepVarComp('control:minOmega', val=0.0, units='rpm', desc='minimum allowed rotor rotation speed', pass_by_obj=True), promotes=['*'])
            self.add('c_maxOmega', IndepVarComp('control:maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed', pass_by_obj=True), promotes=['*'])
            self.add('c_tsr', IndepVarComp('control:tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)'), promotes=['*'])
            self.add('c_pitch', IndepVarComp('control:pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)'), promotes=['*'])
            self.add('pitch_extreme', IndepVarComp('pitch_extreme', val=0.0, units='deg', desc='worst-case pitch at survival wind condition', pass_by_obj=True), promotes=['*'])
            self.add('pitch_extreme_full', IndepVarComp('pitch_extreme_full', val=np.array([0.0, 90.0]), units='deg', desc='worst-case pitch at survival wind condition', pass_by_obj=True), promotes=['*'])
            self.add('azimuth_extreme', IndepVarComp('azimuth_extreme', val=0.0, units='deg', desc='worst-case azimuth at survival wind condition', pass_by_obj=True), promotes=['*'])
            self.add('Omega_load', IndepVarComp('Omega_load', val=0.0, units='rpm', desc='worst-case azimuth at survival wind condition', pass_by_obj=True), promotes=['*'])

            # --- drivetrain efficiency ---
            self.add('drivetrainType', IndepVarComp('drivetrainType', val=Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True), promotes=['*'])

            # --- fatigue ---
            self.add('rstar_damage', IndepVarComp('rstar_damage', val=np.zeros(18), desc='nondimensional radial locations of damage equivalent moments', pass_by_obj=True), promotes=['*'])
            self.add('Mxb_damage', IndepVarComp('Mxb_damage', val=np.zeros(18), units='N*m', desc='damage equivalent moments about blade c.s. x-direction', pass_by_obj=True), promotes=['*'])
            self.add('Myb_damage', IndepVarComp('Myb_damage', val=np.zeros(18), units='N*m', desc='damage equivalent moments about blade c.s. y-direction', pass_by_obj=True), promotes=['*'])
            self.add('strain_ult_spar', IndepVarComp('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap'), promotes=['*'])
            self.add('strain_ult_te', IndepVarComp('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels'), promotes=['*'])
            self.add('eta_damage', IndepVarComp('eta_damage', val=1.755, desc='safety factor for fatigue'), promotes=['*'])
            self.add('m_damage', IndepVarComp('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis', pass_by_obj=True), promotes=['*'])
            self.add('N_damage', IndepVarComp('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis', pass_by_obj=True), promotes=['*'])

            # --- options ---
            self.add('nSector', IndepVarComp('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True), promotes=['*'])
            self.add('npts_coarse_power_curve', IndepVarComp('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at', pass_by_obj=True), promotes=['*'])
            self.add('npts_spline_power_curve', IndepVarComp('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve', pass_by_obj=True), promotes=['*'])
            self.add('AEP_loss_factor', IndepVarComp('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)', pass_by_obj=True), promotes=['*'])
            self.add('dynamic_amplication_tip_deflection', IndepVarComp('dynamic_amplication_tip_deflection', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation', pass_by_obj=True), promotes=['*'])
            self.add('nF', IndepVarComp('nF', val=5, desc='number of natural frequencies to compute', pass_by_obj=True), promotes=['*'])

            self.add('weibull_shape', IndepVarComp('weibull_shape', val=0.0, pass_by_obj=True), promotes=['*'])
        self.add('thick_str_ref', IndepVarComp('thick_str_ref', val=np.zeros(nstr), units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)', pass_by_obj=True), promotes=['*'])
        self.add('af_idx', IndepVarComp('af_idx', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])
        self.add('af_str_idx', IndepVarComp('af_str_idx', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])
        self.add('turbineclass', TurbineClass())
        self.add('gridsetup', GridSetup(naero, nstr))
        self.add('grid', RGrid(naero, nstr))
        self.add('spline0', GeometrySpline(naero, nstr))
        self.add('spline', GeometrySpline(naero, nstr))
        self.add('geom', CCBladeGeometry())
        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRunVarSpeed(npower))
        self.add('airfoil_analysis', CCBladeAirfoils(naero, num_airfoils, airfoils_dof))
        self.add('airfoil_spline', AirfoilSpline(naero, nstr, num_airfoils, airfoils_dof))
        self.add('analysis', CCBlade('power', naero, npower, num_airfoils, airfoils_dof))

        self.add('dt', CSMDrivetrain(npower))
        self.add('powercurve', RegulatedPowerCurveGroup(npower))
        self.add('wind', PowerWind())
        self.add('cdf', WeibullWithMeanCDF(200))
        # self.add('cdf', RayleighCDF())
        self.add('aep', AEP())

        self.add('outputs_aero', OutputsAero(), promotes=['*'])



        # connections to turbineclass
        self.connect('turbine_class', 'turbineclass.turbine_class')

        # connections to gridsetup
        self.connect('initial_aero_grid', 'gridsetup.initial_aero_grid')
        self.connect('initial_str_grid', 'gridsetup.initial_str_grid')

        # connections to grid
        self.connect('r_aero', 'grid.r_aero')
        self.connect('gridsetup.fraction', 'grid.fraction')
        self.connect('gridsetup.idxj', 'grid.idxj')

        # connections to spline0
        self.connect('r_aero', 'spline0.r_aero_unit')
        self.connect('grid.r_str', 'spline0.r_str_unit')
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
        # self.connect('precurve_sub', 'spline0.precurve_sub')

        # connections to spline
        self.connect('r_aero', 'spline.r_aero_unit')
        self.connect('grid.r_str', 'spline.r_str_unit')
        self.connect('r_max_chord', 'spline.r_max_chord')
        self.connect('chord_sub', 'spline.chord_sub')
        self.connect('theta_sub', 'spline.theta_sub')
        self.connect('precurve_sub', 'spline.precurve_sub')
        self.connect('bladeLength', 'spline.bladeLength')
        # self.connect('precurve_sub + delta_precurve_sub', 'spline.precurve_sub')
        # self.connect('bladeLength + delta_bladeLength', 'spline.bladeLength')
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

        # # connectiosn to tipspeed
        # self.connect('geom.R', 'tipspeed.R')
        # self.connect('max_tip_speed', 'tipspeed.Vtip_max')
        # self.connect('tipspeed.Omega_max', 'control:maxOmega')

        # connections to setup
        # self.connect('control', 'setup.control')
        self.connect('control:Vin', 'setup.control:Vin')
        self.connect('control:Vout', 'setup.control:Vout')
        self.connect('control:maxOmega', 'setup.control:maxOmega')
        self.connect('control:minOmega', 'setup.control:minOmega')
        self.connect('control:pitch', 'setup.control:pitch')
        self.connect('control:ratedPower', 'setup.control:ratedPower')
        self.connect('control:tsr', 'setup.control:tsr')
        self.connect('geom.R', 'setup.R')
        self.connect('npts_coarse_power_curve', 'setup.npts')

        # connections to analysis
        self.connect('spline.r_aero', 'analysis.r')
        self.connect('spline.chord_aero', 'analysis.chord')
        self.connect('spline.theta_aero', 'analysis.theta')
        self.connect('spline.precurve_aero', 'analysis.precurve')
        self.connect('spline.precurve_str', 'analysis.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'analysis.Rhub')
        self.connect('spline.Rtip', 'analysis.Rtip')
        self.connect('hubHt', 'analysis.hubHt')
        self.connect('precone', 'analysis.precone')
        self.connect('tilt', 'analysis.tilt')
        self.connect('yaw', 'analysis.yaw')

        self.connect('af_idx', 'airfoil_spline.af_idx')
        self.connect('af_str_idx', 'airfoil_spline.af_str_idx')
        self.connect('af_idx', 'airfoil_analysis.af_idx')
        self.connect('af_str_idx', 'beam.af_str_idx')
        self.connect('thick_str_ref', 'resize.thick_str_ref')

        self.connect('airfoil_parameterization', 'airfoil_spline.airfoil_parameterization')
        self.connect('airfoil_parameterization', 'airfoil_analysis.airfoil_parameterization')
        self.connect('airfoil_analysis_options', 'airfoil_analysis.airfoil_analysis_options')
        self.connect('airfoil_files', 'airfoil_analysis.airfoil_files')
        self.connect('idx_cylinder_str', 'airfoil_spline.idx_cylinder_str')
        self.connect('idx_cylinder_aero', 'airfoil_spline.idx_cylinder_aero')
        self.connect('airfoil_spline.airfoil_parameterization_full', 'analysis.airfoil_parameterization')
        self.connect('airfoil_analysis_options', 'analysis.airfoil_analysis_options')

        self.connect('airfoil_analysis.af', 'analysis.af')
        # self.connect('airfoil_parameterization', 'analysis.airfoil_parameterization')
        # self.connect('airfoil_analysis_options', 'analysis.airfoil_analysis_options')

        self.connect('nBlades', 'analysis.B')
        self.connect('rho', 'analysis.rho')
        self.connect('mu', 'analysis.mu')
        self.connect('shearExp', 'analysis.shearExp')
        self.connect('nSector', 'analysis.nSector')

        self.connect('setup.Uhub', 'analysis.Uhub')
        self.connect('setup.Omega', 'analysis.Omega')
        self.connect('setup.pitch', 'analysis.pitch')


        # connections to drivetrain
        self.connect('analysis.P', 'dt.aeroPower')
        self.connect('analysis.Q', 'dt.aeroTorque')
        self.connect('analysis.T', 'dt.aeroThrust')
        self.connect('control:ratedPower', 'dt.ratedPower')
        self.connect('drivetrainType', 'dt.drivetrainType')

        # connections to powercurve
        self.connect('control:Vin', 'powercurve.control:Vin')
        self.connect('control:Vout', 'powercurve.control:Vout')
        self.connect('control:maxOmega', 'powercurve.control:maxOmega')
        self.connect('control:minOmega', 'powercurve.control:minOmega')
        self.connect('control:pitch', 'powercurve.control:pitch')
        self.connect('control:ratedPower', 'powercurve.control:ratedPower')
        self.connect('control:tsr', 'powercurve.control:tsr')
        self.connect('setup.Uhub', 'powercurve.Vcoarse')
        self.connect('dt.power', 'powercurve.Pcoarse')
        self.connect('analysis.T', 'powercurve.Tcoarse')
        self.connect('geom.R', 'powercurve.R')
        self.connect('npts_spline_power_curve', 'powercurve.npts')

        # # setup Brent method to find rated speed
        # self.connect('control:Vin', 'brent.lower_bound')
        # self.connect('control:Vout', 'brent.upper_bound')
        # self.brent.add_param('powercurve.Vrated', low=-1e-15, high=1e15)
        # self.brent.add_constraint('powercurve.residual = 0')
        # self.brent.invalid_bracket_return = 1.0

        # connections to wind
        # self.wind.z = np.zeros(1)
        # self.wind.U = np.zeros(1)
        # self.connect('cdf_reference_mean_wind_speed', 'wind.Uref')
        self.connect('turbineclass.V_mean', 'wind.Uref')
        self.connect('cdf_reference_height_wind_speed', 'wind.zref')
        self.connect('hubHt', 'wind.z') # , src_indices=[0]) # TODO
        self.connect('shearExp', 'wind.shearExp')

        # connections to cdf
        self.connect('powercurve.V', 'cdf.x')
        self.connect('wind.U', 'cdf.xbar', src_indices=[0])
        self.connect('weibull_shape', 'cdf.k') #TODO

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to outputs
        self.connect('powercurve.V', 'V_in')
        self.connect('powercurve.P', 'P_in')
        self.connect('aep.AEP', 'AEP_in')
        self.connect('powercurve.ratedConditions:V', 'ratedConditions:V_in')
        self.connect('powercurve.ratedConditions:Omega', 'ratedConditions:Omega_in')
        self.connect('powercurve.ratedConditions:pitch', 'ratedConditions:pitch_in')
        self.connect('powercurve.ratedConditions:T', 'ratedConditions:T_in')
        self.connect('powercurve.ratedConditions:Q', 'ratedConditions:Q_in')


        self.connect('spline.diameter', 'hub_diameter_in')
        self.connect('geom.diameter', 'diameter_in')


        # --- add structures ---
        self.add('structure_group', StructureGroup(naero, nstr, num_airfoils, airfoils_dof), promotes=['*'])
        self.add('curvature', BladeCurvature(nstr))
        # self.add('resize', ResizeCompositeSection(nstr))
        self.add('gust', GustETM())
        self.add('setuppc',  SetupPCModVarSpeed())
        #
        self.add('aero_rated', CCBlade('loads', naero, 1, num_airfoils, airfoils_dof)) # 'loads', naero, 1))
        self.add('aero_extrm', CCBlade('loads', naero,  1, num_airfoils, airfoils_dof))
        #self.add('aero_extrm_forces', CCBlade('power', naero, 2))
        self.add('aero_defl_powercurve', CCBlade('loads', naero,  1, num_airfoils, airfoils_dof))
        #
        # self.add('beam', PreCompSections(nstr))
        self.add('loads_defl', TotalLoads(nstr))
        self.add('loads_pc_defl', TotalLoads(nstr))
        self.add('loads_strain', TotalLoads(nstr))
        self.add('damage', DamageLoads(nstr))
        self.add('struc', RotorWithpBEAM(nstr))
        # self.add('curvefem', CurveFEM(nstr))
        self.add('tip', TipDeflection())
        self.add('root_moment', RootMoment(nstr))
        self.add('mass', MassProperties())
        self.add('extreme', ExtremeLoads())
        self.add('blade_defl', BladeDeflection(nstr))


        self.add('output_struc', OutputsStructures(nstr), promotes=['*'])



        # connections to curvature
        self.connect('spline.r_str', 'curvature.r')
        self.connect('spline.precurve_str', 'curvature.precurve')
        self.connect('spline.presweep_str', 'curvature.presweep')
        self.connect('precone', 'curvature.precone')

        # connections to resize
        self.connect('upperCS', 'resize.upperCSIn')
        self.connect('lowerCS', 'resize.lowerCSIn')
        self.connect('websCS', 'resize.websCSIn')
        self.connect('chord_str_ref', 'resize.chord_str_ref')
        self.connect('sector_idx_strain_spar', 'resize.sector_idx_strain_spar')
        self.connect('sector_idx_strain_te', 'resize.sector_idx_strain_te')
        self.connect('spline.chord_str', 'resize.chord_str')
        self.connect('spline.sparT_str', 'resize.sparT_str')
        self.connect('spline.teT_str', 'resize.teT_str')
        self.connect('airfoil_spline.airfoil_str_parameterization_full', 'resize.afp_str')
        self.connect('idx_cylinder_str', 'resize.idx_cylinder_str')

        # connections to gust
        self.connect('turbulence_class', 'gust.turbulence_class')
        self.connect('turbineclass.V_mean', 'gust.V_mean')
        self.connect('powercurve.ratedConditions:V', 'gust.V_hub')

        # connections to setuppc
        self.connect('control:pitch', 'setuppc.control:pitch')
        self.connect('control:tsr', 'setuppc.control:tsr')
        self.connect('powercurve.ratedConditions:V', 'setuppc.Vrated')
        self.connect('geom.R', 'setuppc.R')
        self.connect('VfactorPC', 'setuppc.Vfactor')

        # connections to aero_rated (for max deflection)
        self.connect('spline.r_aero', 'aero_rated.r')
        self.connect('spline.chord_aero', 'aero_rated.chord')
        self.connect('spline.theta_aero', 'aero_rated.theta')
        self.connect('spline.precurve_aero', 'aero_rated.precurve')
        self.connect('spline.precurve_str', 'aero_rated.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_rated.Rhub')
        self.connect('spline.Rtip', 'aero_rated.Rtip')
        self.connect('hubHt', 'aero_rated.hubHt')
        self.connect('precone', 'aero_rated.precone')
        self.connect('tilt', 'aero_rated.tilt')
        self.connect('yaw', 'aero_rated.yaw')

        # self.connect('airfoil_files', 'aero_rated.airfoil_files')
        self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_rated.airfoil_parameterization')
        self.connect('airfoil_analysis_options', 'aero_rated.airfoil_analysis_options')

        self.connect('airfoil_analysis.af', 'aero_rated.af')
        # self.connect('airfoil_parameterization', 'aero_rated.airfoil_parameterization')
        # self.connect('airfoil_analysis_options', 'aero_rated.airfoil_analysis_options')

        self.connect('nBlades', 'aero_rated.B')
        self.connect('rho', 'aero_rated.rho')
        self.connect('mu', 'aero_rated.mu')
        self.connect('shearExp', 'aero_rated.shearExp')
        self.connect('nSector', 'aero_rated.nSector') # TODO: Check effect
        # self.connect('powercurve.ratedConditions:V + 3*gust.sigma', 'aero_rated.V_load')  # OpenMDAO bug
        self.connect('gust.V_gust', 'aero_rated.V_load')
        self.connect('powercurve.ratedConditions:Omega', 'aero_rated.Omega_load')
        self.connect('powercurve.ratedConditions:pitch', 'aero_rated.pitch_load')
        self.connect('powercurve.azimuth', 'aero_rated.azimuth_load')
        # self.aero_rated.azimuth_load = 180.0  # closest to tower

        # connections to aero_extrm (for max strain)
        self.connect('spline.r_aero', 'aero_extrm.r')
        self.connect('spline.chord_aero', 'aero_extrm.chord')
        self.connect('spline.theta_aero', 'aero_extrm.theta')
        self.connect('spline.precurve_aero', 'aero_extrm.precurve')
        self.connect('spline.precurve_str', 'aero_extrm.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_extrm.Rhub')
        self.connect('spline.Rtip', 'aero_extrm.Rtip')
        self.connect('hubHt', 'aero_extrm.hubHt')
        self.connect('precone', 'aero_extrm.precone')
        self.connect('tilt', 'aero_extrm.tilt')
        self.connect('yaw', 'aero_extrm.yaw')
        # self.connect('airfoil_files', 'aero_extrm.airfoil_files')
        self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_extrm.airfoil_parameterization')
        self.connect('airfoil_analysis_options', 'aero_extrm.airfoil_analysis_options')
        self.connect('airfoil_analysis.af', 'aero_extrm.af')
        # self.connect('airfoil_parameterization', 'aero_extrm.airfoil_parameterization')
        # self.connect('airfoil_analysis_options', 'aero_extrm.airfoil_analysis_options')
        self.connect('nBlades', 'aero_extrm.B')
        self.connect('rho', 'aero_extrm.rho')
        self.connect('mu', 'aero_extrm.mu')
        self.connect('shearExp', 'aero_extrm.shearExp')
        self.connect('nSector', 'aero_extrm.nSector') ## CHECK EFFECT
        self.connect('turbineclass.V_extreme', 'aero_extrm.V_load')
        self.connect('pitch_extreme', 'aero_extrm.pitch_load')
        self.connect('azimuth_extreme', 'aero_extrm.azimuth_load')
        self.connect('Omega_load', 'aero_extrm.Omega_load')
        # self.aero_extrm.Omega_load = 0.0  # parked case

        # connections to aero_extrm_forces (for tower thrust)
        # self.connect('spline.r_aero', 'aero_extrm_forces.r')
        # self.connect('spline.chord_aero', 'aero_extrm_forces.chord')
        # self.connect('spline.theta_aero', 'aero_extrm_forces.theta')
        # self.connect('spline.precurve_aero', 'aero_extrm_forces.precurve')
        # self.connect('spline.precurve_str', 'aero_extrm_forces.precurveTip', src_indices=[nstr-1])
        # self.connect('spline.Rhub', 'aero_extrm_forces.Rhub')
        # self.connect('spline.Rtip', 'aero_extrm_forces.Rtip')
        # self.connect('hubHt', 'aero_extrm_forces.hubHt')
        # self.connect('precone', 'aero_extrm_forces.precone')
        # self.connect('tilt', 'aero_extrm_forces.tilt')
        # self.connect('yaw', 'aero_extrm_forces.yaw')
        # # self.connect('airfoil_files', 'aero_extrm_forces.airfoil_files')
        # self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_extrm_forces.airfoil_parameterization')
        # self.connect('airfoil_analysis_options', 'aero_extrm_forces.airfoil_analysis_options')
        # self.connect('airfoil_analysis.af', 'aero_extrm_forces.af')
        # # self.connect('airfoil_parameterization', 'aero_extrm_forces.airfoil_parameterization')
        # # self.connect('airfoil_analysis_options', 'aero_extrm_forces.airfoil_analysis_options')
        # self.connect('nBlades', 'aero_extrm_forces.B')
        # self.connect('rho', 'aero_extrm_forces.rho')
        # self.connect('mu', 'aero_extrm_forces.mu')
        # self.connect('shearExp', 'aero_extrm_forces.shearExp')
        # self.connect('nSector', 'aero_extrm_forces.nSector')
        # # self.aero_extrm_forces.Uhub = np.zeros(2)
        # # self.aero_extrm_forces.Omega = np.zeros(2)  # parked case
        # # self.aero_extrm_forces.pitch = np.zeros(2)
        # self.connect('turbineclass.V_extreme_full', 'aero_extrm_forces.Uhub')
        # self.connect('pitch_extreme_full', 'aero_extrm_forces.pitch')
        # # self.aero_extrm_forces.pitch[1] = 90  # feathered
        # # self.aero_extrm_forces.T = np.zeros(2)
        # # self.aero_extrm_forces.Q = np.zeros(2)

        # connections to aero_defl_powercurve (for gust reversal)
        self.connect('spline.r_aero', 'aero_defl_powercurve.r')
        self.connect('spline.chord_aero', 'aero_defl_powercurve.chord')
        self.connect('spline.theta_aero', 'aero_defl_powercurve.theta')
        self.connect('spline.precurve_aero', 'aero_defl_powercurve.precurve')
        self.connect('spline.precurve_str', 'aero_defl_powercurve.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_defl_powercurve.Rhub')
        self.connect('spline.Rtip', 'aero_defl_powercurve.Rtip')
        self.connect('hubHt', 'aero_defl_powercurve.hubHt')
        self.connect('precone', 'aero_defl_powercurve.precone')
        self.connect('tilt', 'aero_defl_powercurve.tilt')
        self.connect('yaw', 'aero_defl_powercurve.yaw')
        # self.connect('airfoil_files', 'aero_defl_powercurve.airfoil_files')
        self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_defl_powercurve.airfoil_parameterization')
        self.connect('airfoil_analysis_options', 'aero_defl_powercurve.airfoil_analysis_options')
        self.connect('airfoil_analysis.af', 'aero_defl_powercurve.af')
        # self.connect('airfoil_parameterization', 'aero_defl_powercurve.airfoil_parameterization')
        # self.connect('airfoil_analysis_options', 'aero_defl_powercurve.airfoil_analysis_options')
        self.connect('nBlades', 'aero_defl_powercurve.B')
        self.connect('rho', 'aero_defl_powercurve.rho')
        self.connect('mu', 'aero_defl_powercurve.mu')
        self.connect('shearExp', 'aero_defl_powercurve.shearExp')
        self.connect('nSector', 'aero_defl_powercurve.nSector') # CHECK EFFECT
        self.connect('setuppc.Uhub', 'aero_defl_powercurve.V_load')
        self.connect('setuppc.Omega', 'aero_defl_powercurve.Omega_load')
        self.connect('setuppc.pitch', 'aero_defl_powercurve.pitch_load')
        self.connect('setuppc.azimuth', 'aero_defl_powercurve.azimuth_load')
        # self.aero_defl_powercurve.azimuth_load = 0.0

        # connections to beam
        self.connect('spline.r_str', 'beam.r')
        self.connect('spline.chord_str', 'beam.chord')
        self.connect('spline.theta_str', 'beam.theta')
        self.connect('leLoc', 'beam.leLoc')
        self.connect('profile', 'beam.profile')
        self.connect('materials', 'beam.materials')
        self.connect('resize.upperCSOut', 'beam.upperCS')
        self.connect('resize.lowerCSOut', 'beam.lowerCS')
        self.connect('resize.websCSOut', 'beam.websCS')
        self.connect('sector_idx_strain_spar', 'beam.sector_idx_strain_spar')
        self.connect('sector_idx_strain_te', 'beam.sector_idx_strain_te')
        self.connect('airfoil_parameterization', 'beam.airfoil_parameterization')
        self.connect('airfoil_analysis_options', 'beam.airfoil_analysis_options')


        # connections to loads_defl
        self.connect('aero_rated.loads:Omega', 'loads_defl.aeroLoads:Omega')
        self.connect('aero_rated.loads:Px', 'loads_defl.aeroLoads:Px')
        self.connect('aero_rated.loads:Py', 'loads_defl.aeroLoads:Py')
        self.connect('aero_rated.loads:Pz', 'loads_defl.aeroLoads:Pz')
        self.connect('aero_rated.loads:azimuth', 'loads_defl.aeroLoads:azimuth')
        self.connect('aero_rated.loads:pitch', 'loads_defl.aeroLoads:pitch')
        self.connect('aero_rated.loads:r', 'loads_defl.aeroLoads:r')

        # self.connect('aero_rated.loads:Omega', 'loads_defl.aeroLoads:Omega')
        # self.connect('aero_rated.loads:Px', 'loads_defl.aeroLoads:Px')
        # self.connect('aero_rated.loads:Py', 'loads_defl.aeroLoads:Py')
        # self.connect('aero_rated.loads:Pz', 'loads_defl.aeroLoads:Pz')
        # self.connect('aero_rated.loads:azimuth', 'loads_defl.aeroLoads:azimuth')
        # self.connect('aero_rated.loads:pitch', 'loads_defl.aeroLoads:pitch')
        # self.connect('aero_rated.loads:r', 'loads_defl.aeroLoads:r')

        self.connect('beam.beam:z', 'loads_defl.r')
        self.connect('spline.theta_str', 'loads_defl.theta')
        self.connect('tilt', 'loads_defl.tilt')
        self.connect('curvature.totalCone', 'loads_defl.totalCone')
        self.connect('curvature.z_az', 'loads_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_defl.rhoA')
        self.connect('g', 'loads_defl.g')

        # connections to loads_pc_defl
        self.connect('aero_defl_powercurve.loads:Omega', 'loads_pc_defl.aeroLoads:Omega')
        self.connect('aero_defl_powercurve.loads:Px', 'loads_pc_defl.aeroLoads:Px')
        self.connect('aero_defl_powercurve.loads:Py', 'loads_pc_defl.aeroLoads:Py')
        self.connect('aero_defl_powercurve.loads:Pz', 'loads_pc_defl.aeroLoads:Pz')
        self.connect('aero_defl_powercurve.loads:azimuth', 'loads_pc_defl.aeroLoads:azimuth')
        self.connect('aero_defl_powercurve.loads:pitch', 'loads_pc_defl.aeroLoads:pitch')
        self.connect('aero_defl_powercurve.loads:r', 'loads_pc_defl.aeroLoads:r')
        self.connect('beam.beam:z', 'loads_pc_defl.r')
        self.connect('spline.theta_str', 'loads_pc_defl.theta')
        self.connect('tilt', 'loads_pc_defl.tilt')
        self.connect('curvature.totalCone', 'loads_pc_defl.totalCone')
        self.connect('curvature.z_az', 'loads_pc_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_pc_defl.rhoA')
        self.connect('g', 'loads_pc_defl.g')


        # connections to loads_strain
        self.connect('aero_extrm.loads:Omega', 'loads_strain.aeroLoads:Omega')
        self.connect('aero_extrm.loads:Px', 'loads_strain.aeroLoads:Px')
        self.connect('aero_extrm.loads:Py', 'loads_strain.aeroLoads:Py')
        self.connect('aero_extrm.loads:Pz', 'loads_strain.aeroLoads:Pz')
        self.connect('aero_extrm.loads:azimuth', 'loads_strain.aeroLoads:azimuth')
        self.connect('aero_extrm.loads:pitch', 'loads_strain.aeroLoads:pitch')
        self.connect('aero_extrm.loads:r', 'loads_strain.aeroLoads:r')
        self.connect('beam.beam:z', 'loads_strain.r')
        self.connect('spline.theta_str', 'loads_strain.theta')
        self.connect('tilt', 'loads_strain.tilt')
        self.connect('curvature.totalCone', 'loads_strain.totalCone')
        self.connect('curvature.z_az', 'loads_strain.z_az')
        self.connect('beam.beam:rhoA', 'loads_strain.rhoA')
        self.connect('g', 'loads_strain.g')


        # connections to damage
        self.connect('rstar_damage', 'damage.rstar')
        self.connect('Mxb_damage', 'damage.Mxb')
        self.connect('Myb_damage', 'damage.Myb')
        self.connect('spline.theta_str', 'damage.theta')
        self.connect('beam.beam:z', 'damage.r')


        # connections to struc
        self.connect('beam.beam:z', 'struc.beam:z')
        self.connect('beam.beam:EA', 'struc.beam:EA')
        self.connect('beam.beam:EIxx', 'struc.beam:EIxx')
        self.connect('beam.beam:EIyy', 'struc.beam:EIyy')
        self.connect('beam.beam:EIxy', 'struc.beam:EIxy')
        self.connect('beam.beam:GJ', 'struc.beam:GJ')
        self.connect('beam.beam:rhoA', 'struc.beam:rhoA')
        self.connect('beam.beam:rhoJ', 'struc.beam:rhoJ')
        self.connect('beam.beam:x_ec_str', 'struc.beam:x_ec_str')
        self.connect('beam.beam:y_ec_str', 'struc.beam:y_ec_str')
        self.connect('nF', 'struc.nF')
        self.connect('loads_defl.Px_af', 'struc.Px_defl')
        self.connect('loads_defl.Py_af', 'struc.Py_defl')
        self.connect('loads_defl.Pz_af', 'struc.Pz_defl')
        self.connect('loads_pc_defl.Px_af', 'struc.Px_pc_defl')
        self.connect('loads_pc_defl.Py_af', 'struc.Py_pc_defl')
        self.connect('loads_pc_defl.Pz_af', 'struc.Pz_pc_defl')
        self.connect('loads_strain.Px_af', 'struc.Px_strain')
        self.connect('loads_strain.Py_af', 'struc.Py_strain')
        self.connect('loads_strain.Pz_af', 'struc.Pz_strain')
        self.connect('beam.xu_strain_spar', 'struc.xu_strain_spar')
        self.connect('beam.xl_strain_spar', 'struc.xl_strain_spar')
        self.connect('beam.yu_strain_spar', 'struc.yu_strain_spar')
        self.connect('beam.yl_strain_spar', 'struc.yl_strain_spar')
        self.connect('beam.xu_strain_te', 'struc.xu_strain_te')
        self.connect('beam.xl_strain_te', 'struc.xl_strain_te')
        self.connect('beam.yu_strain_te', 'struc.yu_strain_te')
        self.connect('beam.yl_strain_te', 'struc.yl_strain_te')
        self.connect('damage.Mxa', 'struc.Mx_damage')
        self.connect('damage.Mya', 'struc.My_damage')
        self.connect('strain_ult_spar', 'struc.strain_ult_spar')
        self.connect('strain_ult_te', 'struc.strain_ult_te')
        self.connect('eta_damage', 'struc.eta_damage')
        self.connect('m_damage', 'struc.m_damage')
        self.connect('N_damage', 'struc.N_damage')

        # connections to curvefem
        self.connect('powercurve.ratedConditions:Omega', 'curvefem.Omega')
        self.connect('beam.beam:z', 'curvefem.beam:z')
        self.connect('beam.beam:EA', 'curvefem.beam:EA')
        self.connect('beam.beam:EIxx', 'curvefem.beam:EIxx')
        self.connect('beam.beam:EIyy', 'curvefem.beam:EIyy')
        self.connect('beam.beam:EIxy', 'curvefem.beam:EIxy')
        self.connect('beam.beam:GJ', 'curvefem.beam:GJ')
        self.connect('beam.beam:rhoA', 'curvefem.beam:rhoA')
        self.connect('beam.beam:rhoJ', 'curvefem.beam:rhoJ')
        self.connect('beam.beam:x_ec_str', 'curvefem.beam:x_ec_str')
        self.connect('beam.beam:y_ec_str', 'curvefem.beam:y_ec_str')
        self.connect('spline.theta_str', 'curvefem.theta_str')
        self.connect('spline.precurve_str', 'curvefem.precurve_str')
        self.connect('spline.presweep_str', 'curvefem.presweep_str')
        self.connect('nF', 'curvefem.nF')

        # connections to tip
        self.connect('struc.dx_defl', 'tip.dx', src_indices=[nstr-1])
        self.connect('struc.dy_defl', 'tip.dy', src_indices=[nstr-1])
        self.connect('struc.dz_defl', 'tip.dz', src_indices=[nstr-1])
        self.connect('spline.theta_str', 'tip.theta', src_indices=[nstr-1])
        self.connect('aero_rated.loads:pitch', 'tip.pitch')
        self.connect('aero_rated.loads:azimuth', 'tip.azimuth')
        self.connect('tilt', 'tip.tilt')
        self.connect('curvature.totalCone', 'tip.totalConeTip', src_indices=[nstr-1])
        self.connect('dynamic_amplication_tip_deflection', 'tip.dynamicFactor')


        # connections to root moment
        self.connect('spline.r_str', 'root_moment.r_str')
        self.connect('aero_rated.loads:Px', 'root_moment.aeroLoads:Px')
        self.connect('aero_rated.loads:Py', 'root_moment.aeroLoads:Py')
        self.connect('aero_rated.loads:Pz', 'root_moment.aeroLoads:Pz')
        self.connect('aero_rated.loads:r', 'root_moment.aeroLoads:r')
        self.connect('curvature.totalCone', 'root_moment.totalCone')
        self.connect('curvature.x_az', 'root_moment.x_az')
        self.connect('curvature.y_az', 'root_moment.y_az')
        self.connect('curvature.z_az', 'root_moment.z_az')
        self.connect('curvature.s', 'root_moment.s')


        # connections to mass
        self.connect('struc.blade_mass', 'mass.blade_mass')
        self.connect('struc.blade_moment_of_inertia', 'mass.blade_moment_of_inertia')
        self.connect('nBlades', 'mass.nBlades')
        self.connect('tilt', 'mass.tilt')

        # connectsion to extreme
        # self.connect('aero_extrm_forces.T', 'extreme.T')
        # self.connect('aero_extrm_forces.Q', 'extreme.Q')
        # self.connect('nBlades', 'extreme.nBlades')

        # connections to blade_defl
        self.connect('struc.dx_pc_defl', 'blade_defl.dx')
        self.connect('struc.dy_pc_defl', 'blade_defl.dy')
        self.connect('struc.dz_pc_defl', 'blade_defl.dz')
        self.connect('aero_defl_powercurve.loads:pitch', 'blade_defl.pitch')
        self.connect('spline0.theta_str', 'blade_defl.theta_str')
        self.connect('spline0.r_sub_precurve', 'blade_defl.r_sub_precurve0')
        self.connect('spline0.Rhub', 'blade_defl.Rhub0')
        self.connect('spline0.r_str', 'blade_defl.r_str0')
        self.connect('spline0.precurve_str', 'blade_defl.precurve_str0')
        self.connect('bladeLength', 'blade_defl.bladeLength0')
        # self.connect('precurve_sub', 'blade_defl.precurve_sub0')

        # connect to outputs
        self.connect('turbineclass.V_extreme', 'V_extreme_in')
        self.connect('extreme.T_extreme', 'T_extreme_in')
        self.connect('extreme.Q_extreme', 'Q_extreme_in')
        self.connect('struc.blade_mass', 'mass_one_blade_in')
        self.connect('mass.mass_all_blades', 'mass_all_blades_in')
        self.connect('mass.I_all_blades', 'I_all_blades_in')
        self.connect('struc.freq', 'freq_in')
        self.connect('curvefem.freq', 'freq_curvefem_in')
        self.connect('tip.tip_deflection', 'tip_deflection_in')
        self.connect('struc.strainU_spar', 'strainU_spar_in')
        self.connect('struc.strainL_spar', 'strainL_spar_in')
        self.connect('struc.strainU_te', 'strainU_te_in')
        self.connect('struc.strainL_te', 'strainL_te_in')
        self.connect('root_moment.root_bending_moment', 'root_bending_moment_in')
        self.connect('beam.eps_crit_spar', 'eps_crit_spar_in')
        self.connect('beam.eps_crit_te', 'eps_crit_te_in')
        self.connect('struc.damageU_spar', 'damageU_spar_in')
        self.connect('struc.damageL_spar', 'damageL_spar_in')
        self.connect('struc.damageU_te', 'damageU_te_in')
        self.connect('struc.damageL_te', 'damageL_te_in')
        self.connect('blade_defl.delta_bladeLength', 'delta_bladeLength_out_in')
        self.connect('blade_defl.delta_precurve_sub', 'delta_precurve_sub_out_in')

        self.connect('spline.Rtip', 'Rtip_in')
        self.connect('spline.precurve_str', 'precurveTip_in', src_indices=[nstr-1])

        #self.connect('Omega', 'Omega_in')

        #COE Objective
        self.add('coe', COE(), promotes=['*'])
        self.add('obj_cons', ObjandCons(nstr, npower, num_airfoils, airfoils_dof), promotes=['*'])
        self.connect('analysis.P', 'power')
        #self.add('obj_cmp', ExecComp('obj = COE*100.0', COE=0.1), promotes=['*'])
        # self.add('obj_cmp', ExecComp('obj = (mass_all_blades + 589154)*100.0 / AEP', mass_all_blades=50000.0, AEP=1000000.0), promotes=['*'])
        #eta_strain = 1.35*1.3*1.0
        #self.add('obj_con1', ExecComp('con1 = strainU_spar[[0, 12, 14, 18, 22, 28, 34]]*eta_strain/strain_ult_spar', strainU_spar=np.zeros(nstr), eta_strain=eta_strain, strain_ult_spar=0.0, con1=np.zeros(7)), promotes=['*'])
        #self.add('obj_con2', ExecComp('con2 = strainU_te[[0, 8, 12, 14, 18, 22, 28, 34]]*eta_strain/strain_ult_te', strainU_te=np.zeros(nstr), eta_strain=eta_strain, strain_ult_te=0.0, con2=np.zeros(8)), promotes=['*'])
        #self.add('obj_con3', ExecComp('con3 = strainL_te[[0, 8, 12, 14, 18, 22, 28, 34]]*eta_strain/strain_ult_te', strainL_te=np.zeros(nstr), eta_strain=eta_strain, strain_ult_te=0.0, con3=np.zeros(8)), promotes=['*'])
        #self.add('obj_con4', ExecComp('con4 = (eps_crit_spar[[10, 12, 14, 20, 23, 27, 31, 33]] - strainU_spar[[10, 12, 14, 20, 23, 27, 31, 33]]) / strain_ult_spar', eps_crit_spar=np.zeros(nstr), strainU_spar=np.zeros(nstr), strain_ult_spar=0.0, con4=np.zeros(8)), promotes=['*'])
        #self.add('obj_con5', ExecComp('con5 = (eps_crit_te[[10, 12, 13, 14, 21, 28, 33]] - strainU_te[[10, 12, 13, 14, 21, 28, 33]]) / strain_ult_te', eps_crit_te=np.zeros(nstr), strainU_te=np.zeros(nstr), strain_ult_te=0.0, con5=np.zeros(7)), promotes=['*'])
        #self.add('obj_con6', ExecComp('con6 = freq_curvefem[0:2] - nBlades*ratedConditions_Omega/60.0*1.1', freq_curvefem=np.zeros(5), nBlades=3, ratedConditions_Omega=0.0, con6=np.zeros(2)), promotes=['*'])
        #self.add('obj_con_freeform', ExecComp('con_freeform = airfoil_parameterization[:, [4, 5, 6, 7]] - airfoil_parameterization[:, [0, 1, 2, 3]]', airfoil_parameterization=np.zeros((6,8)), con_freeform=np.zeros((6,4))), promotes=['*'])
        #self.add('obj_concon', ExecComp('concon = (mass_all_blades + 589154)*100.0 / AEP', mass_all_blades=50000.0, AEP=1000000.0), promotes=['*'])
        ## self.add('obj_con7', ExecComp('con7 = ratedConditions_T / 1e6 - 700000./1e6', ratedConditions_T=1.0, promotes=['*']))
        # self.connect('ratedConditions.T', 'ratedConditions_T')
        #self.connect('ratedConditions:Omega', 'ratedConditions_Omega')


