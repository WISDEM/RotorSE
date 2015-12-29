from __future__ import print_function

import numpy as np
import math
from openmdao.api import Component
from openmdao.api import ExecComp, IndepVarComp, Group, NLGaussSeidel
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
from openmdao.api import IndepVarComp, Component, Problem, Group, SqliteRecorder, BaseRecorder
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from rotoraero import SetupRunVarSpeed, RegulatedPowerCurve, AEP, \
    RPM2RS, RS2RPM
from rotoraerodefaults import CCBladeGeometry, CCBlade, CSMDrivetrain, RayleighCDF, Brent, WeibullWithMeanCDF
# from openmdao.lib.drivers.api import Brent
from scipy.interpolate import RectBivariateSpline
from akima import Akima, akima_interp_with_derivs
from commonse.csystem import DirectionVector
# from commonse.utilities import hstack, vstack, trapz_deriv, interp_with_deriv
# from commonse.environment import PowerWind
from utilities import hstack, vstack, trapz_deriv, interp_with_deriv
from environment import PowerWind
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
import _pBEAM
import _curvefem
import _bem  # TODO: move to rotoraero
from enum import Enum


#######################
##  BASE COMPONENTS  ##
#######################

class BeamPropertiesBase(Component):

    def __init__(self):
        super(BeamPropertiesBase).__init__()



class StrucBase(Component):

    def __init__(self):
        super(StrucBase, self).__init__()
        # all inputs/outputs in airfoil coordinate system
    
        # inputs
        # beam = VarTree(BeamProperties(), iotype='in', desc='beam properties') # TODO: Import Beam
    
        self.add_param('nF', val=5, desc='number of natural frequencies to return')
    
        self.add_param('Px_defl', shape=1, desc='distributed load (force per unit length) in airfoil x-direction at max deflection condition')
        self.add_param('Py_defl', shape=1, desc='distributed load (force per unit length) in airfoil y-direction at max deflection condition')
        self.add_param('Pz_defl', shape=1, desc='distributed load (force per unit length) in airfoil z-direction at max deflection condition')
    
        self.add_param('Px_strain', shape=1, desc='distributed load (force per unit length) in airfoil x-direction at max strain condition')
        self.add_param('Py_strain', shape=1, desc='distributed load (force per unit length) in airfoil y-direction at max strain condition')
        self.add_param('Pz_strain', shape=1, desc='distributed load (force per unit length) in airfoil z-direction at max strain condition')
    
        self.add_param('Px_pc_defl', shape=1, desc='distributed load (force per unit length) in airfoil x-direction for deflection used in generated power curve')
        self.add_param('Py_pc_defl', shape=1, desc='distributed load (force per unit length) in airfoil y-direction for deflection used in generated power curve')
        self.add_param('Pz_pc_defl', shape=1, desc='distributed load (force per unit length) in airfoil z-direction for deflection used in generated power curve')
    
        self.add_param('xu_strain_spar', shape=1, desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_param('xl_strain_spar', shape=1, desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_param('yu_strain_spar', shape=1, desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_param('yl_strain_spar', shape=1, desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_param('xu_strain_te', shape=1, desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_param('xl_strain_te', shape=1, desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_param('yu_strain_te', shape=1, desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_param('yl_strain_te', shape=1, desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')
    
        self.add_param('Mx_damage', shape=1, units='N*m', desc='damage equivalent moments about airfoil x-direction')
        self.add_param('My_damage', shape=1, units='N*m', desc='damage equivalent moments about airfoil y-direction')
        self.add_param('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap')
        self.add_param('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels')
        self.add_param('eta_damage', val=1.755, desc='safety factor for fatigue')
        self.add_param('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis')
        self.add_param('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis')
    
        # outputs
        self.add_output('blade_mass', shape=1, desc='mass of one blades')
        self.add_output('blade_moment_of_inertia', shape=1, desc='out of plane moment of inertia of a blade')
        self.add_output('freq', shape=1, units='Hz', desc='first nF natural frequencies of blade')
        self.add_output('dx_defl', shape=1, desc='deflection of blade section in airfoil x-direction under max deflection loading')
        self.add_output('dy_defl', shape=1, desc='deflection of blade section in airfoil y-direction under max deflection loading')
        self.add_output('dz_defl', shape=1, desc='deflection of blade section in airfoil z-direction under max deflection loading')
        self.add_output('dx_pc_defl', shape=1, desc='deflection of blade section in airfoil x-direction under power curve loading')
        self.add_output('dy_pc_defl', shape=1, desc='deflection of blade section in airfoil y-direction under power curve loading')
        self.add_output('dz_pc_defl', shape=1, desc='deflection of blade section in airfoil z-direction under power curve loading')
        self.add_output('strainU_spar', shape=1, desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_output('strainL_spar', shape=1, desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')
        self.add_output('strainU_te', shape=1, desc='strain in trailing-edge panels on upper surface at location xu,yu_te with loads P_te')
        self.add_output('strainL_te', shape=1, desc='strain in trailing-edge panels on lower surface at location xl,yl_te with loads P_te')
        self.add_output('damageU_spar', shape=1, desc='fatigue damage on upper surface in spar cap')
        self.add_output('damageL_spar', shape=1, desc='fatigue damage on lower surface in spar cap')
        self.add_output('damageU_te', shape=1, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_output('damageL_te', shape=1, desc='fatigue damage on lower surface in trailing-edge panels')

class ResizeCompositeSection(Component):
    def __init__(self):
        super(ResizeCompositeSection, self).__init__()

        self.add_param('upperCSIn', shape=1, desc='list of CompositeSection objections defining the properties for upper surface')
        self.add_param('lowerCSIn', shape=1, desc='list of CompositeSection objections defining the properties for lower surface')
        self.add_param('websCSIn', shape=1, desc='list of CompositeSection objections defining the properties for shear webs')

        # TODO: remove fixed t/c assumption
        self.add_param('chord_str_ref', shape=17, units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)')

        self.add_param('sector_idx_strain_spar', val=0, dtype=np.int, desc='index of sector for spar (PreComp definition of sector)')
        self.add_param('sector_idx_strain_te', val=0, dtype=np.int, desc='index of sector for trailing-edge (PreComp definition of sector)')

        self.add_param('chord_str', shape=17, units='m', desc='structural chord distribution')
        self.add_param('sparT_str', shape=4, units='m', desc='structural spar cap thickness distribution')
        self.add_param('teT_str', shape=4, units='m', desc='structural trailing-edge panel thickness distribution')

        # out
        self.add_output('upperCSOut', shape=4, desc='list of CompositeSection objections defining the properties for upper surface')
        self.add_output('lowerCSOut', shape=4, desc='list of CompositeSection objections defining the properties for lower surface')
        self.add_output('websCSOut', shape=4, desc='list of CompositeSection objections defining the properties for shear webs')

    def solve_nonlinear(self, params, unknowns, resids):

        chord_str_ref = params['chord_str_ref']
        upperCSIn = params['upperCSIn']
        lowerCSIn = params['lowerCSIn']
        websCSIn = params['websCSIn']
        chord_str = params['chord']
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
            factor = chord_str[i]/chord_str_ref[i]  # same as thickness ratio for constant t/c

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

class PreCompSections(Component):
    def __init__(self):
        super(PreCompSections, self).__init__()
        self.add_param('r', shape=17, units='m', desc='radial positions. r[0] should be the hub location \
            while r[-1] should be the blade tip. Any number \
            of locations can be specified between these in ascending order.')
        self.add_param('chord', shape=17, units='m', desc='array of chord lengths at corresponding radial positions')
        self.add_param('theta', shape=17, units='deg', desc='array of twist angles at corresponding radial positions. \
            (positive twist decreases angle of attack)')
        self.add_param('leLoc', shape=17, desc='array of leading-edge positions from a reference blade axis \
            (usually blade pitch axis). locations are normalized by the local chord length.  \
            e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
            positive in -x direction for airfoil-aligned coordinate system')
        self.add_param('profile', shape=38, desc='airfoil shape at each radial position')
        self.add_param('materials', desc='list of all Orthotropic2DMaterial objects used in defining the geometry')
        self.add_param('upperCS', shape=4, desc='list of CompositeSection objections defining the properties for upper surface')
        self.add_param('lowerCS', shape=4, desc='list of CompositeSection objections defining the properties for lower surface')
        self.add_param('websCS', shape=4, desc='list of CompositeSection objections defining the properties for shear webs')

        self.add_param('sector_idx_strain_spar', val=0, dtype=np.int, desc='index of sector for spar (PreComp definition of sector)')
        self.add_param('sector_idx_strain_te', val=0, dtype=np.int, desc='index of sector for trailing-edge (PreComp definition of sector)')


        self.add_output('eps_crit_spar', shape=1, desc='critical strain in spar from panel buckling calculation')
        self.add_output('eps_crit_te', shape=1, desc='critical strain in trailing-edge panels from panel buckling calculation')

        self.add_output('xu_strain_spar', shape=1, desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('xl_strain_spar', shape=1, desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('yu_strain_spar', shape=1, desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('yl_strain_spar', shape=1, desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('xu_strain_te', shape=1, desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('xl_strain_te', shape=1, desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_output('yu_strain_te', shape=1, desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('yl_strain_te', shape=1, desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')


        self.add_param('beam:z', shape=1, units='m', desc='locations of properties along beam')
        self.add_param('beam:EA', shape=1, units='N', desc='axial stiffness')
        self.add_param('beam:EIxx', shape=1, units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_param('beam:EIyy', shape=1, units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_param('beam:EIxy', shape=1, units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_param('beam:GJ', shape=1, units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_param('beam:rhoA', shape=1, units='kg/m', desc='mass per unit length')
        self.add_param('beam:rhoJ', shape=1, units='kg*m', desc='polar mass moment of inertia per unit length')
        self.add_param('beam:x_ec_str', shape=1, units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_param('beam:y_ec_str', shape=1, units='m', desc='y-distance to elastic center from point about which above structural properties are computed')

        self.add_output('beam:properties', shape=1, desc='beam properties')

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

        self.chord = params['chord']
        self.materials = params['materials']
        self.properties = params['properties']
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
        self.properties.z = self.r
        self.properties.EA = np.zeros(nsec)
        self.properties.EIxx = np.zeros(nsec)
        self.properties.EIyy = np.zeros(nsec)
        self.properties.EIxy = np.zeros(nsec)
        self.properties.GJ = np.zeros(nsec)
        self.properties.rhoA = np.zeros(nsec)
        self.properties.rhoJ = np.zeros(nsec)

        # distance to elastic center from point about which structural properties are computed
        # using airfoil coordinate system
        self.properties.x_ec_str = np.zeros(nsec)
        self.properties.y_ec_str = np.zeros(nsec)

        # distance to elastic center from airfoil nose
        # using profile coordinate system
        x_ec_nose = np.zeros(nsec)
        y_ec_nose = np.zeros(nsec)

        profile = self.profile
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


            self.properties.EIxx[i] = results[1]  # EIedge
            self.properties.EIyy[i] = results[0]  # EIflat
            self.properties.GJ[i] = results[2]
            self.properties.EA[i] = results[3]
            self.properties.EIxy[i] = results[4]  # EIflapedge
            self.properties.x_ec_str[i] = results[12] - results[10]
            self.properties.y_ec_str[i] = results[13] - results[11]
            self.properties.rhoA[i] = results[14]
            self.properties.rhoJ[i] = results[15] + results[16]  # perpindicular axis theorem

            x_ec_nose[i] = results[13] + self.leLoc[i]*self.chord[i]
            y_ec_nose[i] = results[12]  # switch b.c of coordinate system used

        eps_crit_spar = self.panelBucklingStrain(self.sector_idx_strain_spar)
        eps_crit_te = self.panelBucklingStrain(self.sector_idx_strain_te)

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

    def __init__(self):
        super(RotorWithpBEAM, self).__init__()

        self.add_param('nF', val=5, desc='number of natural frequencies to return')

        self.add_param('Px_defl', shape=1, desc='distributed load (force per unit length) in airfoil x-direction at max deflection condition')
        self.add_param('Py_defl', shape=1, desc='distributed load (force per unit length) in airfoil y-direction at max deflection condition')
        self.add_param('Pz_defl', shape=1, desc='distributed load (force per unit length) in airfoil z-direction at max deflection condition')

        self.add_param('Px_strain', shape=1, desc='distributed load (force per unit length) in airfoil x-direction at max strain condition')
        self.add_param('Py_strain', shape=1, desc='distributed load (force per unit length) in airfoil y-direction at max strain condition')
        self.add_param('Pz_strain', shape=1, desc='distributed load (force per unit length) in airfoil z-direction at max strain condition')

        self.add_param('Px_pc_defl', shape=1, desc='distributed load (force per unit length) in airfoil x-direction for deflection used in generated power curve')
        self.add_param('Py_pc_defl', shape=1, desc='distributed load (force per unit length) in airfoil y-direction for deflection used in generated power curve')
        self.add_param('Pz_pc_defl', shape=1, desc='distributed load (force per unit length) in airfoil z-direction for deflection used in generated power curve')

        self.add_param('xu_strain_spar', shape=1, desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_param('xl_strain_spar', shape=1, desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_param('yu_strain_spar', shape=1, desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_param('yl_strain_spar', shape=1, desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_param('xu_strain_te', shape=1, desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_param('xl_strain_te', shape=1, desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_param('yu_strain_te', shape=1, desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_param('yl_strain_te', shape=1, desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')

        self.add_param('Mx_damage', shape=1, units='N*m', desc='damage equivalent moments about airfoil x-direction')
        self.add_param('My_damage', shape=1, units='N*m', desc='damage equivalent moments about airfoil y-direction')
        self.add_param('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap')
        self.add_param('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels')
        self.add_param('eta_damage', val=1.755, desc='safety factor for fatigue')
        self.add_param('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis')
        self.add_param('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis')

        # outputs
        self.add_output('blade_mass', shape=1, desc='mass of one blades')
        self.add_output('blade_moment_of_inertia', shape=1, desc='out of plane moment of inertia of a blade')
        self.add_output('freq', shape=1, units='Hz', desc='first nF natural frequencies of blade')
        self.add_output('dx_defl', shape=1, desc='deflection of blade section in airfoil x-direction under max deflection loading')
        self.add_output('dy_defl', shape=1, desc='deflection of blade section in airfoil y-direction under max deflection loading')
        self.add_output('dz_defl', shape=1, desc='deflection of blade section in airfoil z-direction under max deflection loading')
        self.add_output('dx_pc_defl', shape=1, desc='deflection of blade section in airfoil x-direction under power curve loading')
        self.add_output('dy_pc_defl', shape=1, desc='deflection of blade section in airfoil y-direction under power curve loading')
        self.add_output('dz_pc_defl', shape=1, desc='deflection of blade section in airfoil z-direction under power curve loading')
        self.add_output('strainU_spar', shape=1, desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_output('strainL_spar', shape=1, desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')
        self.add_output('strainU_te', shape=1, desc='strain in trailing-edge panels on upper surface at location xu,yu_te with loads P_te')
        self.add_output('strainL_te', shape=1, desc='strain in trailing-edge panels on lower surface at location xl,yl_te with loads P_te')
        self.add_output('damageU_spar', shape=1, desc='fatigue damage on upper surface in spar cap')
        self.add_output('damageL_spar', shape=1, desc='fatigue damage on lower surface in spar cap')
        self.add_output('damageU_te', shape=1, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_output('damageL_te', shape=1, desc='fatigue damage on lower surface in trailing-edge panels')

    def principalCS(self, beam):

        # rename (with swap of x, y for profile c.s.)
        EIxx = np.copy(beam.EIyy)
        EIyy = np.copy(beam.EIxx)
        x_ec_str = np.copy(beam.y_ec_str)
        y_ec_str = np.copy(beam.x_ec_str)
        EA = np.copy(beam.EA)
        EIxy = np.copy(beam.EIxy)

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

        beam = params['beam']
        Px_defl = params['Px_defl']
        Py_defl = params['Py_defl']
        Pz_defl = params['Pz_defl']

        nsec = len(beam.z)


        # create finite element objects
        p_section = _pBEAM.SectionData(nsec, beam.z, beam.EA, beam.EIxx,
            beam.EIyy, beam.GJ, beam.rhoA, beam.rhoJ)
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

        # --- moments of inertia
        self.blade_moment_of_inertia = blade.outOfPlaneMomentOfInertia()

        # ----- natural frequencies ----
        self.freq = blade.naturalFrequencies(self.nF)


        # ----- strain -----
        EI11, EI22, EA, ca, sa = self.principalCS(beam)

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


class CurveFEM(Component):
    def __init__(self):
        super(CurveFEM, self).__init__()

        """natural frequencies for curved blades"""

        self.add_param('Omega', shape=1, units='rpm', desc='rotor rotation frequency')
        self.add_param('beam:z', shape=17, units='m', desc='locations of properties along beam')
        self.add_param('beam:EA', shape=1, units='N', desc='axial stiffness')
        self.add_param('beam:EIxx', shape=1, units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_param('beam:EIyy', shape=1, units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_param('beam:EIxy', shape=1, units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_param('beam:GJ', shape=1, units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_param('beam:rhoA', shape=1, units='kg/m', desc='mass per unit length')
        self.add_param('beam:rhoJ', shape=1, units='kg*m', desc='polar mass moment of inertia per unit length')
        self.add_param('beam:x_ec_str', shape=1, units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_param('beam:y_ec_str', shape=1, units='m', desc='y-distance to elastic center from point about which above structural properties are computed')

        # self.add_output('beam:properties', shape=1, desc='beam properties')
        self.add_param('theta_str', shape=17, units='deg', desc='structural twist distribution')
        self.add_param('precurve_str', shape=17, units='m', desc='structural precuve (see FAST definition)')
        self.add_param('presweep_str', shape=17, units='m', desc='structural presweep (see FAST definition)')
        self.add_param('nF', val=5, desc='number of frequencies to return')

        self.add_output('freq', shape=1, units='Hz', desc='first nF natural frequencies')



    def solve_nonlinear(self, params, unknowns, resids):

        beam = self.beam
        r = beam.z

        rhub = r[0]
        bladeLength = r[-1] - r[0]
        bladeFrac = (r - rhub) / bladeLength

        freq = _curvefem.frequencies(self.Omega, bladeLength, rhub, bladeFrac, self.theta_str,
                                     beam.rhoA, beam.EIxx, beam.EIyy, beam.GJ, beam.EA, beam.rhoJ,
                                     self.precurve_str, self.presweep_str)

        self.freq = freq[:self.nF]


class GridSetup(Component):
    def __init__(self):
        super(GridSetup, self).__init__()

        """preprocessing step.  inputs and outputs should not change during optimization"""

        # should be constant
        self.add_param('initial_aero_grid', shape=17, desc='initial aerodynamic grid on unit radius')
        self.add_param('initial_str_grid', shape=38, desc='initial structural grid on unit radius')

        # outputs are also constant during optimization
        self.add_output('fraction', shape=1, desc='fractional location of structural grid on aero grid')
        self.add_output('idxj', shape=1, dtype=np.int, desc='index of augmented aero grid corresponding to structural index')

    def solve_nonlinear(self, params, unknowns, resids):

        r_aero = self.initial_aero_grid
        r_str = self.initial_str_grid
        r_aug = np.concatenate([[0.0], r_aero, [1.0]])

        nstr = len(r_str)
        naug = len(r_aug)

        # find idx in augmented aero array that brackets the structural index
        # then find the fraction the structural value is between the two bounding indices
        self.fraction = np.zeros(nstr)
        self.idxj = np.zeros(nstr, dtype=np.int)

        for i in range(nstr):
            for j in range(1, naug):
                if r_aug[j] >= r_str[i]:
                    j -= 1
                    break
            self.idxj[i] = j
            self.fraction[i] = (r_str[i] - r_aug[j]) / (r_aug[j+1] - r_aug[j])



class RGrid(Component):
    def __init__(self):
        super(RGrid, self).__init__()
        # variables
        self.add_param('r_aero', shape=17, desc='new aerodynamic grid on unit radius')

        # parameters
        self.add_param('fraction', shape=1, desc='fractional location of structural grid on aero grid')
        self.add_param('idxj', shape=1, dtype=np.int, desc='index of augmented aero grid corresponding to structural index')

        # outputs
        self.add_output('r_str', shape=17, desc='corresponding structural grid corresponding to new aerodynamic grid')


        missing_deriv_policy = 'assume_zero'


    def solve_nonlinear(self, params, unknowns, resids):

        r_aug = np.concatenate([[0.0], self.r_aero, [1.0]])

        nstr = len(self.fraction)
        self.r_str = np.zeros(nstr)
        for i in range(nstr):
            j = self.idxj[i]
            self.r_str[i] = r_aug[j] + self.fraction[i]*(r_aug[j+1] - r_aug[j])


    def list_deriv_vars(self):

        inputs = ('r_aero',)
        outputs = ('r_str', )

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        nstr = len(self.fraction)
        naero = len(self.r_aero)
        J = np.zeros((nstr, naero))

        for i in range(nstr):
            j = self.idxj[i]
            if j > 0 and j < naero+1:
                J[i, j-1] = 1 - self.fraction[i]
            if j > -1 and j < naero:
                J[i, j] = self.fraction[i]

        return J



class GeometrySpline(Component):
    def __init__(self):
        super(GeometrySpline, self).__init__()
        # variables
        self.add_param('r_aero_unit', desc='locations where airfoils are defined on unit radius')
        self.add_param('r_str_unit', desc='locations where airfoils are defined on unit radius')
        self.add_param('r_max_chord', desc='location of max chord on unit radius')
        self.add_param('chord_sub', units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
        self.add_param('theta_sub', units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
        self.add_param('precurve_sub', units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_param('bladeLength', shape=1, units='m', desc='blade length (if not precurved or swept) otherwise length of blade before curvature')
        self.add_param('sparT', shape=4, units='m', desc='thickness values of spar cap')
        self.add_param('teT', shape=4, units='m', desc='thickness values of trailing edge panels')

        # Kulfan parameters
        self.add_param('A1_upper_sub', shape=1, desc='thickness at control points')
        self.add_param('A2_upper_sub', shape=1, desc='thickness at control points')
        self.add_param('A3_upper_sub', shape=1, desc='thickness at control points')
        # self.add_param('A4_upper_sub', desc='thickness at control points')
        self.add_param('A1_lower_sub', shape=1, desc='thickness at control points')
        self.add_param('A2_lower_sub', shape=1, desc='thickness at control points')
        self.add_param('A3_lower_sub', shape=1, desc='thickness at control points')
        # A4_lower_sub', desc='thickness at control points')

        # parameters
        self.add_param('idx_cylinder_aero', shape=1, desc='first idx in r_aero_unit of non-cylindrical section')  # constant twist inboard of here
        self.add_param('idx_cylinder_str', shape=1, desc='first idx in r_str_unit of non-cylindrical section')
        self.add_param('hubFraction', shape=1, desc='hub location as fraction of radius')

        # out
        self.add_output('Rhub', shape=1, units='m', desc='dimensional radius of hub')
        self.add_output('Rtip', shape=1, units='m', desc='dimensional radius of tip')
        self.add_output('r_aero', shape=17, units='m', desc='dimensional aerodynamic grid')
        self.add_output('r_str', shape=17, units='m', desc='dimensional structural grid')
        self.add_output('chord_aero', shape=17, units='m', desc='chord at airfoil locations')
        self.add_output('chord_str', shape=17, units='m', desc='chord at structural locations')
        self.add_output('theta_aero', shape=17, units='deg', desc='twist at airfoil locations')
        self.add_output('theta_str', shape=17, units='deg', desc='twist at structural locations')
        self.add_output('precurve_aero', shape=17, units='m', desc='precurve at airfoil locations')
        self.add_output('precurve_str', val=np.zeros(17), units='m', desc='precurve at structural locations')
        self.add_output('presweep_str', shape=17, units='m', desc='presweep at structural locations')
        self.add_output('sparT_str', shape=4, units='m', desc='dimensional spar cap thickness distribution')
        self.add_output('teT_str', shape=4, units='m', desc='dimensional trailing-edge panel thickness distribution')
        self.add_output('r_sub_precurve', shape=17, desc='precurve locations (used internally)')

        self.add_output('diameter', shape=1) #TODO

        # self.add_output('A1_lower_aero', shape=1, units='m', desc='chord at airfoil locations')
        # self.add_output('A2_lower_aero', shape=1, units='m', desc='chord at airfoil locations')
        # self.add_output('A3_lower_aero', shape=1, units='m', desc='chord at airfoil locations')
        # self.add_output('# A4_lower_aero', shape=1, units='m', desc='chord at airfoil locations')
        # self.add_output('A1_upper_aero', shape=1, units='m', desc='chord at airfoil locations')
        # self.add_output('A2_upper_aero', shape=1, units='m', desc='chord at airfoil locations')
        # self.add_output('A3_upper_aero', shape=1, units='m', desc='chord at airfoil locations')
        # self.add_output('A4_upper_aero', shape=1, units='m', desc='chord at airfoil locations')


    def solve_nonlinear(self, params, unknowns, resids):

        Rhub = self.hubFraction * self.bladeLength
        Rtip = Rhub + self.bladeLength

        # setup chord parmeterization
        nc = len(self.chord_sub)
        r_max_chord = Rhub + (Rtip-Rhub)*self.r_max_chord
        rc = np.linspace(r_max_chord, Rtip, nc-1)
        rc = np.concatenate([[Rhub], rc])
        chord_spline = Akima(rc, self.chord_sub)

        # setup theta parmeterization
        nt = len(self.theta_sub)
        idxc_aero = self.idx_cylinder_aero
        idxc_str = self.idx_cylinder_str
        r_cylinder = Rhub + (Rtip-Rhub)*self.r_aero_unit[idxc_aero]
        rt = np.linspace(r_cylinder, Rtip, nt)
        theta_spline = Akima(rt, self.theta_sub)

        # setup precurve parmeterization
        precurve_spline = Akima(rc, np.concatenate([[0.0], self.precurve_sub]))
        self.r_sub_precurve = rc[1:]

        Rhub = self.hubFraction * self.bladeLength
        Rtip = Rhub + self.bladeLength
        # rthick = [13.8375/Rtip, 15.8874998/Rtip,  24.08750021/Rtip,  28.1874998/Rtip, 40.4874998/Rtip, Rtip/Rtip]
        rthick = [13.8375/Rtip, 24.087500/Rtip, 40.4874998/Rtip, Rtip/Rtip]
        A1_lower_spline = Akima(rthick, self.A1_lower_sub)
        A2_lower_spline = Akima(rthick, self.A2_lower_sub)
        A3_lower_spline = Akima(rthick, self.A3_lower_sub)
        # A4_lower_spline = Akima(rthick, self.A4_lower_sub)
        A1_upper_spline = Akima(rthick, self.A1_upper_sub)
        A2_upper_spline = Akima(rthick, self.A2_upper_sub)
        A3_upper_spline = Akima(rthick, self.A3_upper_sub)

        # make dimensional and evaluate splines
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.r_aero = Rhub + (Rtip-Rhub)*self.r_aero_unit
        self.r_str = Rhub + (Rtip-Rhub)*self.r_str_unit
        self.chord_aero, _, _, _ = chord_spline.interp(self.r_aero)
        self.chord_str, _, _, _ = chord_spline.interp(self.r_str)
        theta_outer_aero, _, _, _ = theta_spline.interp(self.r_aero[idxc_aero:])
        theta_outer_str, _, _, _ = theta_spline.interp(self.r_str[idxc_str:])
        self.theta_aero = np.concatenate([theta_outer_aero[0]*np.ones(idxc_aero), theta_outer_aero])
        self.theta_str = np.concatenate([theta_outer_str[0]*np.ones(idxc_str), theta_outer_str])
        self.precurve_aero, _, _, _ = precurve_spline.interp(self.r_aero)
        self.precurve_str, _, _, _ = precurve_spline.interp(self.r_str)
        self.presweep_str = np.zeros_like(self.precurve_str)  # TODO: for now
        self.A1_lower_aero, _, _, _ = A1_lower_spline.interp(self.r_aero/Rtip)
        self.A2_lower_aero, _, _, _ = A2_lower_spline.interp(self.r_aero/Rtip)
        self.A3_lower_aero, _, _, _ = A3_lower_spline.interp(self.r_aero/Rtip)
        # self.A4_lower_aero, _, _, _ = A4_lower_spline.interp(self.r_aero/Rtip)
        self.A1_upper_aero, _, _, _ = A1_upper_spline.interp(self.r_aero/Rtip)
        self.A2_upper_aero, _, _, _ = A2_upper_spline.interp(self.r_aero/Rtip)
        self.A3_upper_aero, _, _, _ = A3_upper_spline.interp(self.r_aero/Rtip)
        # self.A4_upper_aero, _, _, _ = A4_upper_spline.interp(self.r_aero/Rtip)

        N = 200
        cont = 4
        y_coor_upper_grid = np.zeros((N/2, cont))
        y_coor_lower_grid = np.zeros((N/2, cont))
        for i in range(len(self.A1_lower_sub)):
            wl = np.zeros(3)
            wu = np.zeros(3)
            dz = 0

            wl[0] = self.A1_lower_sub[i]
            wl[1] = self.A2_lower_sub[i]
            wl[2] = self.A3_lower_sub[i]
            # wl[3] = self.A4_lower_sub[i]
            wu[0] = self.A1_upper_sub[i]
            wu[1] = self.A2_upper_sub[i]
            wu[2] = self.A3_upper_sub[i]
            # wu[3] = self.A4_upper_sub[i]

            airfoil_CST = CST_shape(wl, wu, dz, N, Compute=False)
            coor = airfoil_CST.airfoil_coor(True)
            y_coor_lower_grid[:, i] = coor[1]
            y_coor_upper_grid[:, i] = coor[3]
            xl = 1.0 - coor[0]
            xu = coor[2]

        kx = min(len(xu)-1, 3)
        ky = min(len(rthick)-1, 3)
        y_coor_upper_spline = RectBivariateSpline(xu, rthick, y_coor_upper_grid, kx=kx, ky=ky) #, s=0.001)
        y_coor_lower_spline = RectBivariateSpline(xl, rthick, y_coor_lower_grid, kx=kx, ky=ky) #, s=0.001)
        # import os
        # basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PreCompFiles')
        initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
        0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
        0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
        0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
        0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
        1.0])
        xu_real = np.zeros(len(xu))
        xl_real = np.zeros(len(xu))
        yu_real = np.zeros(len(xu))
        yl_real = np.zeros(len(xu))
        ncomp = len(initial_str_grid)
        profile = [0]*ncomp
        for j in range(ncomp):
            # filename = open(os.path.join(basepath, 'shape_' + str(j+1) + '.inp'), 'w')

            y_new_upper = y_coor_upper_spline.ev(xu, initial_str_grid[j])
            y_new_lower = y_coor_lower_spline.ev(xl, initial_str_grid[j])

            for k in range(len(xu)):
                xu_real[k] = float(xu[k])
                xl_real[k] = float(xl[k])
                yu_real[k] = float(y_new_upper[k])
                yl_real[k] = float(y_new_lower[k])
            x_real = np.append(xu_real, 1-xl_real)
            y_real = np.append(yu_real, yl_real)
            profile[j] = Profile.initFromCoordinates(x_real, y_real)
        self.profile = profile

        # setup sparT parameterization
        nt = len(self.sparT)
        rt = np.linspace(0.0, Rtip, nt)
        sparT_spline = Akima(rt, self.sparT)
        teT_spline = Akima(rt, self.teT)

        self.sparT_str, _, _, _ = sparT_spline.interp(self.r_str)
        self.teT_str, _, _, _ = teT_spline.interp(self.r_str)



class BladeCurvature(Component):
    def __init__(self):
        super(BladeCurvature, self).__init__()
        self.add_param('r', shape=17, units='m', desc='location in blade z-coordinate')
        self.add_param('precurve', shape=17, units='m', desc='location in blade x-coordinate')
        self.add_param('presweep', shape=17, units='m', desc='location in blade y-coordinate')
        self.add_param('precone', shape=1, units='deg', desc='precone angle')

        self.add_output('totalCone', shape=1, units='deg', desc='total cone angle from precone and curvature')
        self.add_output('x_az', shape=1, units='m', desc='location of blade in azimuth x-coordinate system')
        self.add_output('y_az', shape=1, units='m', desc='location of blade in azimuth y-coordinate system')
        self.add_output('z_az', shape=1, units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_output('s', shape=1, units='m', desc='cumulative path length along blade')

    def solve_nonlinear(self, params, unknowns, resids):

        # self.x_az, self.y_az, self.z_az, cone, s = \
        #     _bem.definecurvature(self.r, self.precurve, self.presweep, 0.0)

        n = len(self.r)
        dx_dx = np.eye(3*n)

        self.x_az, x_azd, self.y_az, y_azd, self.z_az, z_azd, \
            cone, coned, s, sd = _bem.definecurvature_dv2(self.r, dx_dx[:, :n],
                self.precurve, dx_dx[:, n:2*n], self.presweep, dx_dx[:, 2*n:], 0.0, np.zeros(3*n))

        self.totalCone = self.precone + np.degrees(cone)
        self.s = self.r[0] + s

        dxaz_dr = x_azd[:n, :].T
        dxaz_dprecurve = x_azd[n:2*n, :].T
        dxaz_dpresweep = x_azd[2*n:, :].T
        dx = hstack([dxaz_dr, dxaz_dprecurve, dxaz_dpresweep, np.zeros(n)])

        dyaz_dr = y_azd[:n, :].T
        dyaz_dprecurve = y_azd[n:2*n, :].T
        dyaz_dpresweep = y_azd[2*n:, :].T
        dy = hstack([dyaz_dr, dyaz_dprecurve, dyaz_dpresweep, np.zeros(n)])

        dzaz_dr = z_azd[:n, :].T
        dzaz_dprecurve = z_azd[n:2*n, :].T
        dzaz_dpresweep = z_azd[2*n:, :].T
        dz = hstack([dzaz_dr, dzaz_dprecurve, dzaz_dpresweep, np.zeros(n)])

        dcone_dr = np.degrees(coned[:n, :]).T
        dcone_dprecurve = np.degrees(coned[n:2*n, :]).T
        dcone_dpresweep = np.degrees(coned[2*n:, :]).T
        dcone = hstack([dcone_dr, dcone_dprecurve, dcone_dpresweep, np.ones(n)])

        ds_dr = sd[:n, :].T
        ds_dr[:, 0] += 1
        ds_dprecurve = sd[n:2*n, :].T
        ds_dpresweep = sd[2*n:, :].T
        ds = hstack([ds_dr, ds_dprecurve, ds_dpresweep, np.zeros(n)])

        self.J = vstack([dx, dy, dz, dcone, ds])


    def list_deriv_vars(self):

        inputs = ('r', 'precurve', 'presweep', 'precone')
        outputs = ('x_az', 'y_az', 'z_az', 'totalCone', 's')

        return inputs, outputs


    def provideJ(self):

        return self.J



class DamageLoads(Component):
    def __init__(self):
        super(DamageLoads, self).__init__()
        self.add_param('rstar', shape=1, desc='nondimensional radial locations of damage equivalent moments')
        self.add_param('Mxb', shape=1, units='N*m', desc='damage equivalent moments about blade c.s. x-direction')
        self.add_param('Myb', shape=1, units='N*m', desc='damage equivalent moments about blade c.s. y-direction')
        self.add_param('theta', shape=17, units='deg', desc='structural twist')
        self.add_param('r', shape=17, units='m', desc='structural radial locations')

        self.add_output('Mxa', shape=1, units='N*m', desc='damage equivalent moments about airfoil c.s. x-direction')
        self.add_output('Mya', shape=1, units='N*m', desc='damage equivalent moments about airfoil c.s. y-direction')

    def solve_nonlinear(self, params, unknowns, resids):

        rstar_str = (self.r-self.r[0])/(self.r[-1]-self.r[0])

        Mxb_str, self.dMxbstr_drstarstr, self.dMxbstr_drstar, self.dMxbstr_dMxb = \
            akima_interp_with_derivs(self.rstar, self.Mxb, rstar_str)

        Myb_str, self.dMybstr_drstarstr, self.dMybstr_drstar, self.dMybstr_dMyb = \
            akima_interp_with_derivs(self.rstar, self.Myb, rstar_str)

        self.Ma = DirectionVector(Mxb_str, Myb_str, 0.0).bladeToAirfoil(self.theta)
        self.Mxa = self.Ma.x
        self.Mya = self.Ma.y


    def list_deriv_vars(self):

        inputs = ('rstar', 'Mxb', 'Myb', 'theta', 'r')
        outputs = ('Mxa', 'Mya')

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

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
        # (self.Ma.dx['dx'] * self.dMxbstr_dMxb.T).T
        dMxa_dMyb = np.dot(np.diag(self.Ma.dx['dy']), self.dMybstr_dMyb)
        dMxa_dtheta = np.diag(self.Ma.dx['dtheta'])

        dMya_dr = np.dot(np.diag(self.Ma.dy['dx']), dMxbstr_dr)\
            + np.dot(np.diag(self.Ma.dy['dy']), dMybstr_dr)
        dMya_drstar = np.dot(np.diag(self.Ma.dy['dx']), self.dMxbstr_drstar)\
            + np.dot(np.diag(self.Ma.dy['dy']), self.dMybstr_drstar)
        dMya_dMxb = np.dot(np.diag(self.Ma.dy['dx']), self.dMxbstr_dMxb)
        dMya_dMyb = np.dot(np.diag(self.Ma.dy['dy']), self.dMybstr_dMyb)
        dMya_dtheta = np.diag(self.Ma.dy['dtheta'])

        dMxa = hstack([dMxa_drstar, dMxa_dMxb, dMxa_dMyb, dMxa_dtheta, dMxa_dr])
        dMya = hstack([dMya_drstar, dMya_dMxb, dMya_dMyb, dMya_dtheta, dMya_dr])

        J = vstack([dMxa, dMya])

        return J


class TotalLoads(Component):
    def __init__(self):
        super(TotalLoads, self).__init__()
        # variables
        self.add_param('aeroLoads:r', units='m', desc='radial positions along blade going toward tip')
        self.add_param('aeroLoads:Px', units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_param('aeroLoads:Py', units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_param('aeroLoads:Pz', units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_param('aeroLoads:V', units='m/s', desc='hub height wind speed')
        self.add_param('aeroLoads:Omega', units='rpm', desc='rotor rotation speed')
        self.add_param('aeroLoads:pitch', units='deg', desc='pitch angle')
        self.add_param('aeroLoads:azimuth', units='deg', desc='azimuthal angle')

        # self.add_param('aeroLoads', shape=1, desc='aerodynamic loads in blade c.s.')
        self.add_param('r', shape=17, units='m', desc='structural radial locations')
        self.add_param('theta', shape=17, units='deg', desc='structural twist')
        self.add_param('tilt', shape=1, units='deg', desc='tilt angle')
        self.add_param('totalCone', shape=1, units='deg', desc='total cone angle from precone and curvature')
        self.add_param('z_az', shape=1, units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_param('rhoA', shape=1, units='kg/m', desc='mass per unit length')

        # parameters
        self.add_param('g', val=9.81, units='m/s**2', desc='acceleration of gravity')

        # outputs
        self.add_output('Px_af', shape=1, desc='total distributed loads in airfoil x-direction')
        self.add_output('Py_af', shape=1, desc='total distributed loads in airfoil y-direction')
        self.add_output('Pz_af', shape=1, desc='total distributed loads in airfoil z-direction')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

        # totalCone = self.precone
        # z_az = self.r*cosd(self.precone)
        totalCone = self.totalCone
        z_az = self.z_az

        # keep all in blade c.s. then rotate all at end

        # rename
        aero = self.aeroLoads

        # --- aero loads ---

        # interpolate aerodynamic loads onto structural grid
        P_a = DirectionVector(0, 0, 0)
        P_a.x, self.dPax_dr, self.dPax_daeror, self.dPax_daeroPx = akima_interp_with_derivs(aero.r, aero.Px, self.r)
        P_a.y, self.dPay_dr, self.dPay_daeror, self.dPay_daeroPy = akima_interp_with_derivs(aero.r, aero.Py, self.r)
        P_a.z, self.dPaz_dr, self.dPaz_daeror, self.dPaz_daeroPz = akima_interp_with_derivs(aero.r, aero.Pz, self.r)


        # --- weight loads ---

        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -self.rhoA*self.g)

        self.P_w = weight.yawToHub(self.tilt).hubToAzimuth(aero.azimuth)\
            .azimuthToBlade(totalCone)


        # --- centrifugal loads ---

        # azimuthal c.s.
        Omega = aero.Omega*RPM2RS
        load = DirectionVector(0.0, 0.0, self.rhoA*Omega**2*z_az)

        self.P_c = load.azimuthToBlade(totalCone)


        # --- total loads ---
        P = P_a + self.P_w + self.P_c

        # rotate to airfoil c.s.
        theta = np.array(self.theta) + aero.pitch
        self.P = P.bladeToAirfoil(theta)

        self.Px_af = self.P.x
        self.Py_af = self.P.y
        self.Pz_af = self.P.z



    def list_deriv_vars(self):

        inputs = ('aeroLoads.r', 'aeroLoads.Px', 'aeroLoads.Py', 'aeroLoads.Pz', 'aeroLoads.Omega',
            'aeroLoads.pitch', 'aeroLoads.azimuth', 'r', 'theta', 'tilt', 'totalCone', 'rhoA', 'z_az')
        outputs = ('Px_af', 'Py_af', 'Pz_af')

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        dPwx, dPwy, dPwz = self.P_w.dx, self.P_w.dy, self.P_w.dz
        dPcx, dPcy, dPcz = self.P_c.dx, self.P_c.dy, self.P_c.dz
        dPx, dPy, dPz = self.P.dx, self.P.dy, self.P.dz
        Omega = self.aeroLoads.Omega*RPM2RS
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

        dPx = hstack([dPxaf_daeror, dPxaf_dPxaero, dPxaf_dPyaero, dPxaf_dPzaero, dPxaf_dOmega, dPxaf_dpitch, dPxaf_dazimuth,
            dPxaf_dr, dPxaf_dtheta, dPxaf_dtilt, dPxaf_dprecone, dPxaf_drhoA, dPxaf_dzaz])
        dPy = hstack([dPyaf_daeror, dPyaf_dPxaero, dPyaf_dPyaero, dPyaf_dPzaero, dPyaf_dOmega, dPyaf_dpitch, dPyaf_dazimuth,
            dPyaf_dr, dPyaf_dtheta, dPyaf_dtilt, dPyaf_dprecone, dPyaf_drhoA, dPyaf_dzaz])
        dPz = hstack([dPzaf_daeror, dPzaf_dPxaero, dPzaf_dPyaero, dPzaf_dPzaero, dPzaf_dOmega, dPzaf_dpitch, dPzaf_dazimuth,
            dPzaf_dr, dPzaf_dtheta, dPzaf_dtilt, dPzaf_dprecone, dPzaf_drhoA, dPzaf_dzaz])

        J = vstack([dPx, dPy, dPz])

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
        self.add_param('dynamicFactor', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation')

        # outputs
        self.add_output('tip_deflection', shape=1, desc='deflection at tip in yaw x-direction')


    def solve_nonlinear(self, params, unknowns, resids):

        theta = self.theta + self.pitch

        dr = DirectionVector(self.dx, self.dy, self.dz)
        self.delta = dr.airfoilToBlade(theta).bladeToAzimuth(self.totalConeTip) \
            .azimuthToHub(self.azimuth).hubToYaw(self.tilt)

        self.tip_deflection = self.dynamicFactor * self.delta.x


    def list_deriv_vars(self):

        inputs = ('dx', 'dy', 'dz', 'theta', 'pitch', 'azimuth', 'tilt', 'totalConeTip')
        outputs = ('tip_deflection',)

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

        dx = self.dynamicFactor * self.delta.dx['dx']
        dy = self.dynamicFactor * self.delta.dx['dy']
        dz = self.dynamicFactor * self.delta.dx['dz']
        dtheta = self.dynamicFactor * self.delta.dx['dtheta']
        dpitch = self.dynamicFactor * self.delta.dx['dtheta']
        dazimuth = self.dynamicFactor * self.delta.dx['dazimuth']
        dtilt = self.dynamicFactor * self.delta.dx['dtilt']
        dtotalConeTip = self.dynamicFactor * self.delta.dx['dprecone']

        J = np.array([[dx, dy, dz, dtheta, dpitch, dazimuth, dtilt, dtotalConeTip]])

        return J

class BladeDeflection(Component):
    def __init__(self):
        super(BladeDeflection, self).__init__()
        self.add_param('dx', shape=1, desc='deflections in airfoil x-direction')
        self.add_param('dy', shape=1, desc='deflections in airfoil y-direction')
        self.add_param('dz', shape=1, desc='deflections in airfoil z-direction')
        self.add_param('pitch', shape=1, units='deg', desc='blade pitch angle')
        self.add_param('theta_str', shape=17, units='deg', desc='structural twist')

        self.add_param('r_sub_precurve0', shape=17, desc='undeflected precurve locations (internal)')
        self.add_param('Rhub0', shape=1, units='m', desc='hub radius')
        self.add_param('r_str0', shape=17, units='m', desc='undeflected radial locations')
        self.add_param('precurve_str0', units='m', desc='undeflected precurve locations')

        self.add_param('bladeLength0', shape=1, units='m', desc='original blade length (only an actual length if no curvature)')

        self.add_output('delta_bladeLength', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
        self.add_output('delta_precurve_sub', shape=1,  units='m', desc='adjustment to precurve to account for curvature from loading')

    def solve_nonlinear(self, params, unknowns, resids):

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



    def list_deriv_vars(self):

        inputs = ('dx', 'dy', 'dz', 'pitch', 'theta_str', 'r_sub_precurve0', 'Rhub0',
            'r_str0', 'precurve_str0', 'bladeLength0')
        outputs = ('delta_bladeLength', 'delta_precurve_sub')

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

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

        dbl = np.concatenate([dbl_ddx, dbl_ddy, dbl_ddz, [dbl_dpitch], dbl_dthetastr,
            np.zeros(m), [dbl_drhub0], dbl_drstr0, dbl_dprecurvestr0, [dbl_dbl0]])
        dpcs = hstack([dpcs_ddx, dpcs_ddy, dpcs_ddz, dpcs_dpitch, dpcs_dthetastr,
            self.dpcs_drsubpc0, np.zeros(m), self.dpcs_drstr0, np.zeros((m, n)), np.zeros(m)])


        J = vstack([dbl, dpcs])

        return J


class RootMoment(Component):
    """blade root bending moment"""
    def __init__(self):
        super(RootMoment, self).__init__()
        self.add_param('r_str', shape=17)
        self.add_param('aeroLoads:r', shape=1, units='m', desc='radial positions along blade going toward tip')
        self.add_param('aeroLoads:Px', shape=1, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_param('aeroLoads:Py', shape=1, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_param('aeroLoads:Pz', shape=1, units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_param('aeroLoads:V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_param('aeroLoads:Omega',shape=1,  units='rpm', desc='rotor rotation speed')
        self.add_param('aeroLoads:pitch', shape=1, units='deg', desc='pitch angle')
        self.add_param('aeroLoads:azimuth', shape=1, units='deg', desc='azimuthal angle')

        self.add_param('totalCone', shape=1, units='deg', desc='total cone angle from precone and curvature')
        self.add_param('x_az', shape=1, units='m', desc='location of blade in azimuth x-coordinate system')
        self.add_param('y_az', shape=1, units='m', desc='location of blade in azimuth y-coordinate system')
        self.add_param('z_az', shape=1, units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_param('s', shape=1, units='m', desc='cumulative path length along blade')

        self.add_output('root_bending_moment', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

        r = self.r_str
        x_az = self.x_az
        y_az = self.y_az
        z_az = self.z_az


        aL = self.aeroLoads
        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        Px, self.dPx_dr, self.dPx_dalr, self.dPx_dalPx = interp_with_deriv(r, aL.r, aL.Px)
        Py, self.dPy_dr, self.dPy_dalr, self.dPy_dalPy = interp_with_deriv(r, aL.r, aL.Py)
        Pz, self.dPz_dr, self.dPz_dalr, self.dPz_dalPz = interp_with_deriv(r, aL.r, aL.Pz)

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


    def list_deriv_vars(self):

        inputs = ('r_str', 'aeroLoads.r', 'aeroLoads.Px', 'aeroLoads.Py', 'aeroLoads.Pz', 'totalCone',
                  'x_az', 'y_az', 'z_az', 's')
        outputs = ('root_bending_moment',)

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):

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


        J = np.array([np.concatenate([drbm_dr, drbm_dalr, drbm_dalPx, drbm_dalPy, drbm_dalPz,
            drbm_dtotalcone, drbm_dazx, drbm_dazy, drbm_dazz, drbm_ds])])

        return J



class MassProperties(Component):
    def __init__(self):
        super(MassProperties, self).__init__()
        # variables
        self.add_param('blade_mass', shape=1, units='kg', desc='mass of one blade')
        self.add_param('blade_moment_of_inertia', shape=1, units='kg*m**2', desc='mass moment of inertia of blade about hub')
        self.add_param('tilt', shape=1, units='deg', desc='rotor tilt angle (used to translate moments of inertia from hub to yaw c.s.')

        # parameters
        self.add_param('nBlades', val=0, desc='number of blades')

        # outputs
        self.add_output('mass_all_blades', shape=1, desc='mass of all blades')
        self.add_output('I_all_blades', shape=1, desc='mass moments of inertia of all blades in yaw c.s. order:Ixx, Iyy, Izz, Ixy, Ixz, Iyz')

    def solve_nonlinear(self, params, unknowns, resids):

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


    def list_deriv_vars(self):

        inputs = ('blade_mass', 'blade_moment_of_inertia', 'tilt')
        outputs = ('mass_all_blades', 'I_all_blades')

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):
        I = self.Ivec

        dIx_dmoi = self.nBlades*(I.dx['dx'] + I.dx['dy']/2.0 + I.dx['dz']/2.0)
        dIy_dmoi = self.nBlades*(I.dy['dx'] + I.dy['dy']/2.0 + I.dy['dz']/2.0)
        dIz_dmoi = self.nBlades*(I.dz['dx'] + I.dz['dy']/2.0 + I.dz['dz']/2.0)

        dm = np.array([self.nBlades, 0.0, 0.0])
        dIxx = np.array([0.0, dIx_dmoi, I.dx['dtilt']])
        dIyy = np.array([0.0, dIy_dmoi, I.dy['dtilt']])
        dIzz = np.array([0.0, dIz_dmoi, I.dz['dtilt']])

        J = vstack([dm, dIxx, dIyy, dIzz, np.zeros((3, 3))])

        return J


class TurbineClass(Component):
    def __init__(self):
        super(TurbineClass, self).__init__()
        # parameters
        self.add_param('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class')

        # outputs should be constant
        self.add_output('V_mean', shape=1, units='m/s', desc='IEC mean wind speed for Rayleigh distribution')
        self.add_output('V_extreme', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')

    def solve_nonlinear(self, params, unknowns, resids):
        if self.turbine_class == 'I':
            Vref = 50.0
        elif self.turbine_class == 'II':
            Vref = 42.5
        elif self.turbine_class == 'III':
            Vref = 37.5

        self.V_mean = 0.2*Vref
        self.V_extreme = 1.4*Vref



class ExtremeLoads(Component):
    def __init__(self):
        super(ExtremeLoads, self).__init__()
        # variables
        self.add_param('T', val=np.zeros(2), units='N', shape=((2,)), desc='rotor thrust, index 0 is at worst-case, index 1 feathered')
        self.add_param('Q', val=np.zeros(2), units='N*m', shape=((2,)), desc='rotor torque, index 0 is at worst-case, index 1 feathered')

        # parameters
        self.add_param('nBlades', val=0, desc='number of blades')

        # outputs
        self.add_output('T_extreme', shape=1, units='N', desc='rotor thrust at survival wind condition')
        self.add_output('Q_extreme', shape=1, units='N*m', desc='rotor torque at survival wind condition')


    def solve_nonlinear(self, params, unknowns, resids):
        n = float(self.nBlades)
        self.T_extreme = (self.T[0] + self.T[1]*(n-1)) / n
        self.Q_extreme = (self.Q[0] + self.Q[1]*(n-1)) / n


    def list_deriv_vars(self):

        inputs = ('T', 'Q')
        outputs = ('T_extreme', 'Q_extreme')

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):
        n = float(self.nBlades)

        J = np.array([[1.0/n, (n-1)/n, 0.0, 0.0],
                      [0.0, 0.0, 1.0/n, (n-1)/n]])

        return J




class GustETM(Component):
    def __init__(self):
        super(GustETM, self).__init__()
        # variables
        self.add_param('V_mean', shape=1, units='m/s', desc='IEC average wind speed for turbine class')
        self.add_param('V_hub', shape=1, units='m/s', desc='hub height wind speed')

        # parameters
        self.add_param('turbulence_class', val=Enum('B','A', 'B', 'C'), desc='IEC turbulence class')
        self.add_param('std', val=3, desc='number of standard deviations for strength of gust')

        # out
        self.add_output('V_gust', shape=1, units='m/s', desc='gust wind speed')


    def solve_nonlinear(self, params, unknowns, resids):
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


    def list_deriv_vars(self):

        inputs = ('V_mean', 'V_hub')
        outputs = ('V_gust', )

        return inputs, outputs


    def jacobian(self, params, unknowns, resids):
        Iref = self.Iref
        c = self.c

        J = np.array([[self.std*(c*Iref*0.072/c*(self.V_hub/c - 4)),
            1.0 + self.std*(c*Iref*0.072*(self.V_mean/c + 3)/c)]])

        return J




class SetupPCModVarSpeed(Component):
    def __init__(self):
        super(SetupPCModVarSpeed, self).__init__()
        self.add_param('control:Vin', shape=1, units='m/s', desc='cut-in wind speed')
        self.add_param('control:Vout', shape=1, units='m/s', desc='cut-out wind speed')
        self.add_param('control:ratedPower', shape=1, units='W', desc='rated power')
        self.add_param('control:minOmega', shape=1, units='rpm', desc='minimum allowed rotor rotation speed')
        self.add_param('control:maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
        self.add_param('control:tsr', shape=1, desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_param('control:pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')

        self.add_param('Vrated', shape=1, units='m/s', desc='rated wind speed')
        self.add_param('R', shape=1, units='m', desc='rotor radius')
        self.add_param('Vfactor', shape=1, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation')

        self.add_output('Uhub', shape=1, units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', shape=1, units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', shape=1, units='deg', desc='pitch angles to run')

        missing_deriv_policy = 'assume_zero'

    def solve_nonlinear(self, params, unknowns, resids):

        self.Uhub = self.Vfactor * self.Vrated
        self.Omega = self.control.tsr*self.Uhub/self.R*RS2RPM
        self.pitch = self.control.pitch

    def list_deriv_vars(self):

        inputs = ('control.tsr', 'Vrated', 'R')
        outputs = ('Uhub', 'Omega', 'pitch')

        return inputs, outputs

    def jacobian(self, params, unknowns, resids):

        dU = np.array([0.0, self.Vfactor, 0.0])
        dOmega = np.array([self.Uhub/self.R*RS2RPM,
            self.control.tsr*self.Vfactor/self.R*RS2RPM,
            -self.control.tsr*self.Uhub/self.R**2*RS2RPM])
        dpitch = np.zeros(3)

        J = vstack([dU, dOmega, dpitch])

        return J

# class SellarDis2(Component):
#     """Component containing Discipline 2."""
#
#     def __init__(self):
#         super(SellarDis2, self).__init__()
#
#         # Global Design Variable
#         self.add_param('z', val=np.zeros(2))
#
#         # Coupling parameter
#         self.add_param('y1', val=1.0)
#
#         # Coupling output
#         self.add_output('y2', val=1.0)
#
#     def solve_nonlinear(self, params, unknowns, resids):
#         """Evaluates the equation
#         y2 = y1**(.5) + z1 + z2"""
#
#         z1 = params['z'][0]
#         z2 = params['z'][1]
#         y1 = params['y1']
#
#         # Note: this may cause some issues. However, y1 is constrained to be
#         # above 3.16, so lets just let it converge, and the optimizer will
#         # throw it out
#         y1 = abs(y1)
#
#         unknowns['y2'] = y1**.5 + z1 + z2
#
#     def jacobian(self, params, unknowns, resids):
#         """ Jacobian for Sellar discipline 2."""
#         J = {}
#
#         J['y2', 'y1'] = .5*params['y1']**-.5
#         J['y2', 'z'] = np.array([[1.0, 1.0]])
#
#         return J

# class init_Rotor(Component):
#     def __init__(self):
#         super(init_Rotor, self).__init__()
#         # self.add_param('eta_strain', val=1.35*1.3*1.0)
#         # self.add_param('eta_dfl', val=1.35*1.1*1.0)
#         # self.add_param('strain_ult_spar', val=1.0e-2)
#         # self.add_param('strain_ult_te', val=2500*1e-6)
#         # self.add_param('freq_margin', val=1.1)
#
#         # --- geometry inputs ---
#         self.add_param('initial_aero_grid', shape=17, desc='initial aerodynamic grid on unit radius')
#         self.add_param('initial_str_grid', shape=38, desc='initial structural grid on unit radius')
#         self.add_param('idx_cylinder_aero', shape=1, desc='first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here')
#         self.add_param('idx_cylinder_str', shape=1, desc='first idx in r_str_unit of non-cylindrical section')
#         self.add_param('hubFraction', shape=1, desc='hub location as fraction of radius')
#         self.add_param('r_aero', shape=17,desc='new aerodynamic grid on unit radius')
#         self.add_param('r_max_chord', shape=1,desc='location of max chord on unit radius')
#         self.add_param('chord_sub', shape=4, units='m', desc='chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip')
#         self.add_param('theta_sub', shape=4, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip')
#         self.add_param('precurve_sub', val=np.zeros(3), units='m', desc='precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)')
#         self.add_param('delta_precurve_sub', shape=1, units='m', desc='adjustment to precurve to account for curvature from loading')
#         self.add_param('bladeLength', shape=1, units='m', deriv_ignore=True, desc='blade length (if not precurved or swept) otherwise length of blade before curvature')
#         self.add_param('delta_bladeLength', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
#         self.add_param('precone', val=0.0, desc='precone angle', units='deg', pass_by_obj=True)
#         self.add_param('tilt', val=0.0, desc='shaft tilt', units='deg', pass_by_obj=True)
#         self.add_param('yaw', val=0.0, desc='yaw error', units='deg', pass_by_obj=True)
#         self.add_param('nBlades', val=3, desc='number of blades', pass_by_obj=True)
#         # self.add_param('airfoil_files', desc='names of airfoil file', deriv_ignore=True)
#         self.add_param('airfoil_files', shape=1, desc='names of airfoil file', pass_by_obj=True)
#
#         # --- atmosphere inputs ---
#         self.add_param('rho', val=1.225, units='kg/m**3', desc='density of air', pass_by_obj=True)
#         self.add_param('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air', pass_by_obj=True)
#         self.add_param('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True)
#         self.add_param('hubHt', shape=1, units='m', desc='hub height')
#         self.add_param('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class')
#         self.add_param('turbulence_class', val=Enum('B', 'A', 'C'), desc='IEC turbulence class class')
#         self.add_param('g', val=9.81, units='m/s**2', desc='acceleration of gravity', pass_by_obj=True)
#         self.add_param('cdf_reference_height_wind_speed', shape=1, desc='reference hub height for IEC wind speed (used in CDF calculation)')
#         # self.add_param('cdf_reference_mean_wind_speed')
#         # self.add_param('weibull_shape')
#
#         self.add_param('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation')
#
#
#         # --- composite sections ---
#         self.add_param('sparT', shape=4, units='m', desc='spar cap thickness parameters')
#         self.add_param('teT', shape=4, units='m', desc='trailing-edge thickness parameters')
#         self.add_param('chord_str_ref', shape=17, units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)')
#         self.add_param('leLoc', shape=1, desc='array of leading-edge positions from a reference blade axis \
#             (usually blade pitch axis). locations are normalized by the local chord length.  \
#             e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
#             positive in -x direction for airfoil-aligned coordinate system')
#         self.add_param('profile', shape=38, desc='airfoil shape at each radial position')
#         self.add_param('materials', shape=38,
#             desc='list of all Orthotropic2DMaterial objects used in defining the geometry')
#         self.add_param('upperCS', shape=1,
#             desc='list of CompositeSection objections defining the properties for upper surface')
#         self.add_param('lowerCS', shape=1,
#             desc='list of CompositeSection objections defining the properties for lower surface')
#         self.add_param('websCS', shape=1,
#             desc='list of CompositeSection objections defining the properties for shear webs')
#         self.add_param('sector_idx_strain_spar', shape=1,  dtype=np.int, desc='index of sector for spar (PreComp definition of sector)')
#         self.add_param('sector_idx_strain_te', shape=1, dtype=np.int, desc='index of sector for trailing-edge (PreComp definition of sector)')
#
#
#         # --- control ---
#         self.add_param('control:Vin', shape=1, units='m/s', desc='cut-in wind speed')
#         self.add_param('control:Vout', shape=1, units='m/s', desc='cut-out wind speed')
#         self.add_param('control:ratedPower',shape=1,  units='W', desc='rated power')
#         self.add_param('control:minOmega', shape=1, units='rpm', desc='minimum allowed rotor rotation speed')
#         self.add_param('control:maxOmega', shape=1, units='rpm', desc='maximum allowed rotor rotation speed')
#         self.add_param('control:tsr', shape=1, desc='tip-speed ratio in Region 2 (should be optimized externally)')
#         self.add_param('control:pitch', shape=1, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
#
#         self.add_param('pitch_extreme', shape=1, units='deg', desc='worst-case pitch at survival wind condition')
#         self.add_param('azimuth_extreme', shape=1, units='deg', desc='worst-case azimuth at survival wind condition')
#
#         # --- drivetrain efficiency ---
#         self.add_param('drivetrainType', val=Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'))
#
#         # --- fatigue ---
#         self.add_param('rstar_damage', shape=1, desc='nondimensional radial locations of damage equivalent moments')
#         self.add_param('Mxb_damage', shape=1, units='N*m', desc='damage equivalent moments about blade c.s. x-direction')
#         self.add_param('Myb_damage', shape=1, units='N*m', desc='damage equivalent moments about blade c.s. y-direction')
#         self.add_param('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap')
#         self.add_param('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels')
#         self.add_param('eta_damage', val=1.755, desc='safety factor for fatigue')
#         self.add_param('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis')
#         self.add_param('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis')
#
#         # --- options ---
#         self.add_param('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power')
#         self.add_param('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at')
#         self.add_param('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve')
#         self.add_param('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)')
#         self.add_param('dynamic_amplication_tip_deflection', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation')
#         self.add_param('nF', val=5, desc='number of natural frequencies to compute')
#
#         self.add_param('A1_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#         self.add_param('A2_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#         self.add_param('A3_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#         self.add_param('A4_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#         self.add_param('A1_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#         self.add_param('A2_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#         self.add_param('A3_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#         self.add_param('A4_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
#
#         # --- outputs ---
#         self.add_output('AEP', shape=1, units='kW*h', desc='annual energy production')
#         self.add_output('V', shape=1, units='m/s', desc='wind speeds (power curve)')
#         self.add_output('P', shape=1, units='W', desc='power (power curve)')
#         self.add_output('Omega', shape=1, units='rpm', desc='speed (power curve)')
#
#         self.add_output('ratedConditions:V', shape=1, units='m/s', desc='rated wind speed')
#         self.add_output('ratedConditions:Omega', shape=1, units='rpm', desc='rotor rotation speed at rated')
#         self.add_output('ratedConditions:pitch', shape=1, units='deg', desc='pitch setting at rated')
#         self.add_output('ratedConditions:T', shape=1, units='N', desc='rotor aerodynamic thrust at rated')
#         self.add_output('ratedConditions:Q', shape=1, units='N*m', desc='rotor aerodynamic torque at rated')
#
#         self.add_output('hub_diameter', shape=1, units='m', desc='hub diameter')
#         self.add_output('diameter', shape=1, units='m', desc='rotor diameter')
#         self.add_output('V_extreme', shape=1, units='m/s', desc='survival wind speed')
#         self.add_output('T_extreme', shape=1, units='N', desc='thrust at survival wind condition')
#         self.add_output('Q_extreme', shape=1, units='N*m', desc='thrust at survival wind condition')
#
#         # structural outputs
#         self.add_output('mass_one_blade', shape=1, units='kg', desc='mass of one blade')
#         self.add_output('mass_all_blades', shape=1,  units='kg', desc='mass of all blade')
#         self.add_output('I_all_blades', shape=1, desc='out of plane moments of inertia in yaw-aligned c.s.')
#         self.add_output('freq', shape=1, units='Hz', desc='1st nF natural frequencies')
#         self.add_output('freq_curvefem', shape=1, units='Hz', desc='1st nF natural frequencies')
#         self.add_output('tip_deflection', shape=1, units='m', desc='blade tip deflection in +x_y direction')
#         self.add_output('strainU_spar', shape=1, desc='axial strain and specified locations')
#         self.add_output('strainL_spar', shape=1, desc='axial strain and specified locations')
#         self.add_output('strainU_te', shape=1, desc='axial strain and specified locations')
#         self.add_output('strainL_te', shape=1, desc='axial strain and specified locations')
#         self.add_output('eps_crit_spar', shape=1, desc='critical strain in spar from panel buckling calculation')
#         self.add_output('eps_crit_te', shape=1,  desc='critical strain in trailing-edge panels from panel buckling calculation')
#         self.add_output('root_bending_moment', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')
#         self.add_output('damageU_spar', shape=1, desc='fatigue damage on upper surface in spar cap')
#         self.add_output('damageL_spar', shape=1, desc='fatigue damage on lower surface in spar cap')
#         self.add_output('damageU_te', shape=1, desc='fatigue damage on upper surface in trailing-edge panels')
#         self.add_output('damageL_te', shape=1, desc='fatigue damage on lower surface in trailing-edge panels')
#         self.add_output('delta_bladeLength_out', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
#         self.add_output('delta_precurve_sub_out', shape=1, units='m', desc='adjustment to precurve to account for curvature from loading')
#
#         # internal use outputs
#         self.add_output('Rtip', shape=1, units='m', desc='tip location in z_b')
#         self.add_output('precurveTip', shape=1, units='m', desc='tip location in x_b')
#         self.add_output('presweepTip', val=0.0, units='m', desc='tip location in y_b')  # TODO: connect later

class Aerodynamics(Group):
    def __init__(self):
        super(Aerodynamics, self).__init__()
        # self.add('init', init_Rotor(), promotes=['*'])
        self.add('turbineclass', TurbineClass(), promotes=['*'])
        self.add('gridsetup', GridSetup(), promotes=['*'])
        self.add('grid', RGrid(), promotes=['*'])
        self.add('spline0', GeometrySpline(), promotes=['*'])
        self.add('spline', GeometrySpline(), promotes=['*'])
        self.add('geom', CCBladeGeometry(), promotes=['*'])
        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRunVarSpeed(), promotes=['*'])
        self.add('analysis', CCBlade(), promotes=['*'])
        self.add('dt', CSMDrivetrain(), promotes=['*'])
        self.add('powercurve', RegulatedPowerCurve(), promotes=['*'])
        # self.add('brent', Brent())
        self.add('wind', PowerWind(), promotes=['*'])
        # self.add('cdf', WeibullWithMeanCDF())
        self.add('cdf', RayleighCDF())
        self.add('aep', AEP(), promotes=['*'])

# class Structures(Group):
#     def __init__(self):
#         super(Structures, self).__init__()
#         self.add('curvature', BladeCurvature(), promotes=['*'])
#         self.add('resize', ResizeCompositeSection(), promotes=['*'])
#         self.add('gust', GustETM(), promotes=['*'])
#         self.add('setuppc',  SetupPCModVarSpeed(), promotes=['*'])
#         self.add('aero_rated', CCBlade(), promotes=['*'])
#         self.add('aero_extrm', CCBlade(), promotes=['*'])
#         self.add('aero_extrm_forces', CCBlade(), promotes=['*'])
#         self.add('aero_defl_powercurve', CCBlade(), promotes=['*'])
#         self.add('beam', PreCompSections(), promotes=['*'])
#         self.add('loads_defl', TotalLoads(), promotes=['*'])
#         self.add('loads_pc_defl', TotalLoads(), promotes=['*'])
#         self.add('loads_strain', TotalLoads(), promotes=['*'])
#         self.add('damage', DamageLoads(), promotes=['*'])
#         self.add('struc', RotorWithpBEAM(), promotes=['*'])
#         self.add('curvefem', CurveFEM(), promotes=['*'])
#         self.add('tip', TipDeflection(), promotes=['*'])
#         self.add('root_moment', RootMoment(), promotes=['*'])
#         self.add('mass', MassProperties(), promotes=['*'])
#         self.add('extreme', ExtremeLoads(), promotes=['*'])
#         self.add('blade_defl', BladeDeflection(), promotes=['*'])

class Outputs(Component):
    def __init__(self):
        super(Outputs, self).__init__()

        # --- outputs ---
        self.add_param('AEP_in', shape=1, units='kW*h', desc='annual energy production')
        self.add_param('V_in', shape=1, units='m/s', desc='wind speeds (power curve)')
        self.add_param('P_in', shape=1, units='W', desc='power (power curve)')
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

        # structural outputs
        self.add_param('mass_one_blade_in', shape=1, units='kg', desc='mass of one blade')
        self.add_param('mass_all_blades_in', shape=1,  units='kg', desc='mass of all blade')
        self.add_param('I_all_blades_in', shape=1, desc='out of plane moments of inertia in yaw-aligned c.s.')
        self.add_param('freq_in', shape=1, units='Hz', desc='1st nF natural frequencies')
        self.add_param('freq_curvefem_in', shape=1, units='Hz', desc='1st nF natural frequencies')
        self.add_param('tip_deflection_in', shape=1, units='m', desc='blade tip deflection in +x_y direction')
        self.add_param('strainU_spar_in', shape=1, desc='axial strain and specified locations')
        self.add_param('strainL_spar_in', shape=1, desc='axial strain and specified locations')
        self.add_param('strainU_te_in', shape=1, desc='axial strain and specified locations')
        self.add_param('strainL_te_in', shape=1, desc='axial strain and specified locations')
        self.add_param('eps_crit_spar_in', shape=1, desc='critical strain in spar from panel buckling calculation')
        self.add_param('eps_crit_te_in', shape=1,  desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_param('root_bending_moment_in', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')
        self.add_param('damageU_spar_in', shape=1, desc='fatigue damage on upper surface in spar cap')
        self.add_param('damageL_spar_in', shape=1, desc='fatigue damage on lower surface in spar cap')
        self.add_param('damageU_te_in', shape=1, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_param('damageL_te_in', shape=1, desc='fatigue damage on lower surface in trailing-edge panels')
        self.add_param('delta_bladeLength_out_in', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
        self.add_param('delta_precurve_sub_out_in', shape=1, units='m', desc='adjustment to precurve to account for curvature from loading')

        # internal use outputs
        self.add_param('Rtip_in', shape=1, units='m', desc='tip location in z_b')
        self.add_param('precurveTip_in', shape=1, units='m', desc='tip location in x_b')
        self.add_param('presweepTip_in', val=0.0, units='m', desc='tip location in y_b')  # TODO: connect later

        # --- outputs ---
        self.add_output('AEP', shape=1, units='kW*h', desc='annual energy production')
        self.add_output('V', shape=1, units='m/s', desc='wind speeds (power curve)')
        self.add_output('P', shape=1, units='W', desc='power (power curve)')
        self.add_output('Omega', shape=1, units='rpm', desc='speed (power curve)')

        self.add_output('ratedConditions:V', shape=1, units='m/s', desc='rated wind speed')
        self.add_output('ratedConditions:Omega', shape=1, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('ratedConditions:pitch', shape=1, units='deg', desc='pitch setting at rated')
        self.add_output('ratedConditions:T', shape=1, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('ratedConditions:Q', shape=1, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_output('hub_diameter', shape=1, units='m', desc='hub diameter')
        self.add_output('diameter', shape=1, units='m', desc='rotor diameter')
        self.add_output('V_extreme', shape=1, units='m/s', desc='survival wind speed')
        self.add_output('T_extreme', shape=1, units='N', desc='thrust at survival wind condition')
        self.add_output('Q_extreme', shape=1, units='N*m', desc='thrust at survival wind condition')

        # structural outputs
        self.add_output('mass_one_blade', shape=1, units='kg', desc='mass of one blade')
        self.add_output('mass_all_blades', shape=1,  units='kg', desc='mass of all blade')
        self.add_output('I_all_blades', shape=1, desc='out of plane moments of inertia in yaw-aligned c.s.')
        self.add_output('freq', shape=1, units='Hz', desc='1st nF natural frequencies')
        self.add_output('freq_curvefem', shape=1, units='Hz', desc='1st nF natural frequencies')
        self.add_output('tip_deflection', shape=1, units='m', desc='blade tip deflection in +x_y direction')
        self.add_output('strainU_spar', shape=1, desc='axial strain and specified locations')
        self.add_output('strainL_spar', shape=1, desc='axial strain and specified locations')
        self.add_output('strainU_te', shape=1, desc='axial strain and specified locations')
        self.add_output('strainL_te', shape=1, desc='axial strain and specified locations')
        self.add_output('eps_crit_spar', shape=1, desc='critical strain in spar from panel buckling calculation')
        self.add_output('eps_crit_te', shape=1,  desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_output('root_bending_moment', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')
        self.add_output('damageU_spar', shape=1, desc='fatigue damage on upper surface in spar cap')
        self.add_output('damageL_spar', shape=1, desc='fatigue damage on lower surface in spar cap')
        self.add_output('damageU_te', shape=1, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_output('damageL_te', shape=1, desc='fatigue damage on lower surface in trailing-edge panels')
        self.add_output('delta_bladeLength_out', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
        self.add_output('delta_precurve_sub_out', shape=1, units='m', desc='adjustment to precurve to account for curvature from loading')

        # internal use outputs
        self.add_output('Rtip', shape=1, units='m', desc='tip location in z_b')
        self.add_output('precurveTip', shape=1, units='m', desc='tip location in x_b')
        self.add_output('presweepTip', val=0.0, units='m', desc='tip location in y_b')  # TODO: connect later

class RotorSE(Group):
    def __init__(self):
        super(RotorSE, self).__init__()
        """rotor model"""
        # self.add('aerodynamics', Aerodynamics(), promotes=['aep'])
        # self.add('structures', Structures(), promotes=['aep'])
        self.configure()


        self.add('obj_cmp', ExecComp('obj = -AEP', AEP=1.0), promotes=['*'])

    def configure(self):
        n = 17
        n2 = 38
        n3 = 4
        self.add('initial_aero_grid', IndepVarComp('initial_aero_grid', np.zeros(n)), promotes=['*'])
        self.add('initial_str_grid', IndepVarComp('initial_str_grid', np.zeros(n2)), promotes=['*'])
        self.add('idx_cylinder_aero', IndepVarComp('idx_cylinder_aero', 0.0), promotes=['*'])
        self.add('idx_cylinder_str', IndepVarComp('idx_cylinder_str', 0.0), promotes=['*'])
        self.add('hubFraction', IndepVarComp('hubFraction', 0.0), promotes=['*'])
        self.add('r_aero', IndepVarComp('r_aero', np.zeros(n)), promotes=['*'])
        self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
        self.add('chord_sub', IndepVarComp('chord_sub', np.zeros(n3)), promotes=['*'])
        self.add('theta_sub', IndepVarComp('theta_sub', np.zeros(n3)), promotes=['*'])
        self.add('precurve_sub', IndepVarComp('precurve_sub', np.zeros(3)), promotes=['*'])
        self.add('delta_precurve_sub', IndepVarComp('delta_precurve_sub', 0.0), promotes=['*'])
        self.add('bladeLength', IndepVarComp('bladeLength', 0.0), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])
        self.add('airfoil_files', IndepVarComp('airfoil_files', val=np.zeros(n), pass_by_obj=True), promotes=['*'])
        self.add('rho', IndepVarComp('rho', val=1.225, units='kg/m**3', desc='density of air', pass_by_obj=True), promotes=['*'])
        self.add('mu', IndepVarComp('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air', pass_by_obj=True), promotes=['*'])
        self.add('shearExp', IndepVarComp('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', val=0.0, units='m', desc='hub height'), promotes=['*'])
        self.add('turbine_class', IndepVarComp('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
        self.add('turbulence_class', IndepVarComp('turbulence_class', val=Enum('B', 'A', 'C'), desc='IEC turbulence class class', pass_by_obj=True), promotes=['*'])
        self.add('g', IndepVarComp('g', val=9.81, units='m/s**2', desc='acceleration of gravity', pass_by_obj=True), promotes=['*'])
        self.add('cdf_reference_height_wind_speed', IndepVarComp('cdf_reference_height_wind_speed', val=0.0, desc='reference hub height for IEC wind speed (used in CDF calculation)'), promotes=['*'])
        self.add('VfactorPC', IndepVarComp('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation'), promotes=['*'])

        # --- composite sections ---
        self.add('sparT', IndepVarComp('sparT', val=np.zeros(n3), units='m', desc='spar cap thickness parameters'), promotes=['*'])
        self.add('teT', IndepVarComp('teT', val=np.zeros(n3), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])
        self.add('chord_str_ref', IndepVarComp('chord_str_ref', val=np.zeros(n), units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)'), promotes=['*'])
        self.add('leLoc', IndepVarComp('leLoc', val=np.zeros(n), desc='array of leading-edge positions from a reference blade axis \
            (usually blade pitch axis). locations are normalized by the local chord length.  \
            e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
            positive in -x direction for airfoil-aligned coordinate system'), promotes=['*'])

        self.add('profile', IndepVarComp('profile', val=np.zeros(n2), desc='airfoil shape at each radial position'), promotes=['*'])
        self.add('materials', IndepVarComp('materials', val=np.zeros(n2),
            desc='list of all Orthotropic2DMaterial objects used in defining the geometry'), promotes=['*'])
        self.add('upperCS', IndepVarComp('upperCS', val=0.0,
            desc='list of CompositeSection objections defining the properties for upper surface'), promotes=['*'])
        self.add('lowerCS', IndepVarComp('lowerCS', val=0.0,
            desc='list of CompositeSection objections defining the properties for lower surface'), promotes=['*'])
        self.add('websCS', IndepVarComp('websCS', val=0.0,
            desc='list of CompositeSection objections defining the properties for shear webs'), promotes=['*'])
        self.add('sector_idx_strain_spar', IndepVarComp('sector_idx_strain_spar', val=0,  dtype=np.int, desc='index of sector for spar (PreComp definition of sector)'), promotes=['*'])
        self.add('sector_idx_strain_te', IndepVarComp('sector_idx_strain_te', val=0, dtype=np.int, desc='index of sector for trailing-edge (PreComp definition of sector)'), promotes=['*'])

        # --- control ---
        self.add('control:Vin', IndepVarComp('control:Vin', val=0.0, units='m/s', desc='cut-in wind speed'), promotes=['*'])
        self.add('control:Vout', IndepVarComp('control:Vout', val=0.0, units='m/s', desc='cut-out wind speed'), promotes=['*'])
        self.add('control:ratedPower', IndepVarComp('control:ratedPower', val=0.0,  units='W', desc='rated power'), promotes=['*'])
        self.add('control:minOmega', IndepVarComp('control:minOmega', val=0.0, units='rpm', desc='minimum allowed rotor rotation speed'), promotes=['*'])
        self.add('control:maxOmega', IndepVarComp('control:maxOmega', val=0.0, units='rpm', desc='maximum allowed rotor rotation speed'), promotes=['*'])
        self.add('control:tsr', IndepVarComp('control:tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)'), promotes=['*'])
        self.add('control:pitch', IndepVarComp('control:pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)'), promotes=['*'])
        self.add('pitch_extreme', IndepVarComp('pitch_extreme', val=0.0, units='deg', desc='worst-case pitch at survival wind condition'), promotes=['*'])
        self.add('azimuth_extreme', IndepVarComp('azimuth_extreme', val=0.0, units='deg', desc='worst-case azimuth at survival wind condition'), promotes=['*'])

        # --- drivetrain efficiency ---
        self.add('drivetrainType', IndepVarComp('drivetrainType', val=Enum('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), pass_by_obj=True), promotes=['*'])


        # --- fatigue ---
        self.add('rstar_damage', IndepVarComp('rstar_damage', val=0.0, desc='nondimensional radial locations of damage equivalent moments'), promotes=['*'])
        self.add('Mxb_damage', IndepVarComp('Mxb_damage', val=0.0, units='N*m', desc='damage equivalent moments about blade c.s. x-direction'), promotes=['*'])
        self.add('Myb_damage', IndepVarComp('Myb_damage', val=0.0, units='N*m', desc='damage equivalent moments about blade c.s. y-direction'), promotes=['*'])
        self.add('strain_ult_spar', IndepVarComp('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap'), promotes=['*'])
        self.add('strain_ult_te', IndepVarComp('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels'), promotes=['*'])
        self.add('eta_damage', IndepVarComp('eta_damage', val=1.755, desc='safety factor for fatigue'), promotes=['*'])
        self.add('m_damage', IndepVarComp('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis'), promotes=['*'])
        self.add('N_damage', IndepVarComp('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis'), promotes=['*'])


        # --- options ---
        self.add('nSector', IndepVarComp('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power'), promotes=['*'])
        self.add('npts_coarse_power_curve', IndepVarComp('npts_coarse_power_curve', val=20, desc='number of points to evaluate aero analysis at'), promotes=['*'])
        self.add('npts_spline_power_curve', IndepVarComp('npts_spline_power_curve', val=200, desc='number of points to use in fitting spline to power curve'), promotes=['*'])
        self.add('AEP_loss_factor', IndepVarComp('AEP_loss_factor', val=1.0, desc='availability and other losses (soiling, array, etc.)'), promotes=['*'])
        self.add('dynamic_amplication_tip_deflection', IndepVarComp('dynamic_amplication_tip_deflection', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation'), promotes=['*'])
        self.add('nF', IndepVarComp('nF', val=5, desc='number of natural frequencies to compute'), promotes=['*'])

        self.add('weibull_shape', IndepVarComp('weibull_shape', val=0.0), promotes=['*'])

        # self.add_param('A1_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
        # self.add_param('A2_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
        # self.add_param('A3_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
        # self.add_param('A4_upper_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
        # self.add_param('A1_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
        # self.add_param('A2_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
        # self.add_param('A3_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)
        # self.add_param('A4_lower_sub', shape=6, units='deg', desc='twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip', cs_step=1e-20) #, fd_step=0.01)

        # self.add('init', init_Rotor()) #, promotes=['*'])
        self.add('turbineclass', TurbineClass()) #, promotes=['*'])
        self.add('gridsetup', GridSetup()) #, promotes=['*'])
        self.add('grid', RGrid()) #, promotes=['*'])
        self.add('spline0', GeometrySpline()) #, promotes=['*'])
        self.add('spline', GeometrySpline()) #, promotes=['*'])
        self.add('geom', CCBladeGeometry()) #, promotes=['*'])
        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRunVarSpeed()) #, promotes=['*'])
        self.add('analysis', CCBlade()) #, promotes=['*'])
        self.add('dt', CSMDrivetrain()) #, promotes=['*'])
        self.add('powercurve', RegulatedPowerCurve()) #, promotes=['*'])
        # self.add('brent', Brent())
        self.add('wind', PowerWind()) #, promotes=['*'])
        # self.add('cdf', WeibullWithMeanCDF())
        self.add('cdf', RayleighCDF())
        self.add('aep', AEP()) #, promotes=['*'])

        # self.brent.workflow.add(['powercurve'])

        # self.driver.workflow.add(['turbineclass', 'gridsetup', 'grid', 'spline0', 'spline',
        #     'geom', 'setup', 'analysis', 'dt', 'brent', 'wind', 'cdf', 'aep'])

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
        # self.connect('A1_upper_sub', 'spline0.A1_upper_sub')
        # self.connect('A2_upper_sub', 'spline0.A2_upper_sub')
        # self.connect('A3_upper_sub', 'spline0.A3_upper_sub')
        # self.connect('A4_upper_sub', 'spline0.A4_upper_sub')
        # self.connect('A1_lower_sub', 'spline0.A1_lower_sub')
        # self.connect('A2_lower_sub', 'spline0.A2_lower_sub')
        # self.connect('A3_lower_sub', 'spline0.A3_lower_sub')
        # self.connect('A4_lower_sub', 'spline0.A4_lower_sub')
        # self.connect('precurve_sub', 'spline0.precurve_sub')

        # connections to spline
        self.connect('r_aero', 'spline.r_aero_unit')
        self.connect('grid.r_str', 'spline.r_str_unit')
        self.connect('r_max_chord', 'spline.r_max_chord')
        self.connect('chord_sub', 'spline.chord_sub')
        self.connect('theta_sub', 'spline.theta_sub')
        # self.connect('precurve_sub + delta_precurve_sub', 'spline.precurve_sub') #TODO
        # self.connect('bladeLength + delta_bladeLength', 'spline.bladeLength')
        self.connect('idx_cylinder_aero', 'spline.idx_cylinder_aero')
        self.connect('idx_cylinder_str', 'spline.idx_cylinder_str')
        self.connect('hubFraction', 'spline.hubFraction')
        self.connect('sparT', 'spline.sparT')
        self.connect('teT', 'spline.teT')
        # self.connect('A1_upper_sub', 'spline.A1_upper_sub')
        # self.connect('A2_upper_sub', 'spline.A2_upper_sub')
        # self.connect('A3_upper_sub', 'spline.A3_upper_sub')
        # self.connect('A4_upper_sub', 'spline.A4_upper_sub')
        # self.connect('A1_lower_sub', 'spline.A1_lower_sub')
        # self.connect('A2_lower_sub', 'spline.A2_lower_sub')
        # self.connect('A3_lower_sub', 'spline.A3_lower_sub')
        # self.connect('A4_lower_sub', 'spline.A4_lower_sub')

        # connections to geom
        # self.spline['precurve_str'] = np.zeros(1)
        self.connect('spline.Rtip', 'geom.Rtip')
        self.connect('precone', 'geom.precone')
        # self.connect('spline.precurve_str[-1]', 'geom.precurveTip') # TODO

        # # connectiosn to tipspeed
        # self.connect('geom.R', 'tipspeed.R')
        # self.connect('max_tip_speed', 'tipspeed.Vtip_max')
        # self.connect('tipspeed.Omega_max', 'control.maxOmega')

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
        # self.connect('spline.precurve_str[-1]', 'analysis.precurveTip')
        self.connect('spline.Rhub', 'analysis.Rhub')
        self.connect('spline.Rtip', 'analysis.Rtip')
        self.connect('hubHt', 'analysis.hubHt')
        self.connect('precone', 'analysis.precone')
        self.connect('tilt', 'analysis.tilt')
        self.connect('yaw', 'analysis.yaw')
        self.connect('airfoil_files', 'analysis.airfoil_files')
        self.connect('nBlades', 'analysis.B')
        self.connect('rho', 'analysis.rho')
        self.connect('mu', 'analysis.mu')
        self.connect('shearExp', 'analysis.shearExp')
        self.connect('nSector', 'analysis.nSector')
        self.connect('setup.Uhub', 'analysis.Uhub')
        self.connect('setup.Omega', 'analysis.Omega')
        self.connect('setup.pitch', 'analysis.pitch')
        # self.connect('spline.A1_lower_aero', 'analysis.A1_lower')
        # self.connect('spline.A2_lower_aero', 'analysis.A2_lower')
        # self.connect('spline.A3_lower_aero', 'analysis.A3_lower')
        # self.connect('spline.A4_lower_aero', 'analysis.A4_lower')
        # self.connect('spline.A1_upper_aero', 'analysis.A1_upper')
        # self.connect('spline.A2_upper_aero', 'analysis.A2_upper')
        # self.connect('spline.A3_upper_aero', 'analysis.A3_upper')
        # self.connect('spline.A4_upper_aero', 'analysis.A4_upper')
        self.analysis.run_case = 'power'

        # connections to drivetrain
        self.connect('analysis.P', 'dt.aeroPower')
        self.connect('analysis.Q', 'dt.aeroTorque')
        self.connect('analysis.T', 'dt.aeroThrust')
        self.connect('control:ratedPower', 'dt.ratedPower')
        self.connect('drivetrainType', 'dt.drivetrainType')

        # connections to powercurve
        # self.connect('control', 'powercurve.control')
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
        # self.connect('control.Vin', 'brent.lower_bound')
        # self.connect('control.Vout', 'brent.upper_bound')
        # self.brent.add_param('powercurve.Vrated', low=-1e-15, high=1e15)
        # self.brent.add_constraint('powercurve.residual = 0')
        # self.brent.invalid_bracket_return = 1.0

        # connections to wind
        self.wind.z = np.zeros(1)
        self.wind.U = np.zeros(1)
        # self.connect('cdf_reference_mean_wind_speed', 'wind.Uref')
        self.connect('turbineclass.V_mean', 'wind.Uref')
        self.connect('cdf_reference_height_wind_speed', 'wind.zref')
        # self.connect('hubHt', 'wind.z[0]') #TODO
        self.connect('shearExp', 'wind.shearExp')

        # connections to cdf
        # self.connect('powercurve.V', 'cdf.x')
        # self.connect('wind.U[0]', 'cdf.xbar')
        # self.connect('weibull_shape', 'cdf.k') #TODO

        # connections to aep
        # self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to outputs
        self.connect('powercurve.V', 'V_in')
        self.connect('powercurve.P', 'P_in')
        self.connect('aep.AEP', 'AEP_in')
        # self.connect('powercurve.ratedConditions:', 'ratedConditions')
        self.connect('powercurve.ratedConditions:V', 'ratedConditions:V_in')
        self.connect('powercurve.ratedConditions:Omega', 'ratedConditions:Omega_in')
        self.connect('powercurve.ratedConditions:pitch', 'ratedConditions:pitch_in')
        self.connect('powercurve.ratedConditions:T', 'ratedConditions:T_in')
        self.connect('powercurve.ratedConditions:Q', 'ratedConditions:Q_in')


        self.connect('spline.diameter', 'hub_diameter_in')
        self.connect('geom.diameter', 'diameter_in')


        # --- add structures ---
        self.add('curvature', BladeCurvature()) #, promotes=['*'])
        self.add('resize', ResizeCompositeSection()) #, promotes=['*'])
        self.add('gust', GustETM()) #, promotes=['*'])
        self.add('setuppc',  SetupPCModVarSpeed()) #, promotes=['*'])
        self.add('aero_rated', CCBlade()) #, promotes=['*'])
        self.add('aero_extrm', CCBlade()) #, promotes=['*'])
        self.add('aero_extrm_forces', CCBlade()) #, promotes=['*'])
        self.add('aero_defl_powercurve', CCBlade()) #, promotes=['*'])
        self.add('beam', PreCompSections()) #, promotes=['*'])
        self.add('loads_defl', TotalLoads()) #, promotes=['*'])
        self.add('loads_pc_defl', TotalLoads()) #, promotes=['*'])
        self.add('loads_strain', TotalLoads()) #, promotes=['*'])
        self.add('damage', DamageLoads()) #, promotes=['*'])
        self.add('struc', RotorWithpBEAM())
        self.add('curvefem', CurveFEM()) #, promotes=['*'])
        self.add('tip', TipDeflection()) #, promotes=['*'])
        self.add('root_moment', RootMoment())
        self.add('mass', MassProperties()) #, promotes=['*'])
        self.add('extreme', ExtremeLoads()) #, promotes=['*'])
        self.add('blade_defl', BladeDeflection()) #, promotes=['*'])


        self.add('output', Outputs(), promotes=['*'])


        # self.driver.workflow.add(['curvature', 'resize', 'gust', 'setuppc', 'aero_rated', 'aero_extrm',
        #     'aero_extrm_forces', 'aero_defl_powercurve', 'beam', 'loads_defl', 'loads_pc_defl',
        #     'loads_strain', 'damage', 'struc', 'curvefem', 'tip', 'root_moment', 'mass', 'extreme',
        #     'blade_defl'])
        #
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

        # connections to gust
        self.connect('turbulence_class', 'gust.turbulence_class')
        self.connect('turbineclass.V_mean', 'gust.V_mean')
        self.connect('powercurve.ratedConditions:V', 'gust.V_hub')

        # connections to setuppc
        # self.connect('control', 'setuppc.control')
        self.connect('control:Vin', 'setuppc.control:Vin')
        self.connect('control:Vout', 'setuppc.control:Vout')
        self.connect('control:maxOmega', 'setuppc.control:maxOmega')
        self.connect('control:minOmega', 'setuppc.control:minOmega')
        self.connect('control:pitch', 'setuppc.control:pitch')
        self.connect('control:ratedPower', 'setuppc.control:ratedPower')
        self.connect('control:tsr', 'setuppc.control:tsr')
        self.connect('powercurve.ratedConditions:V', 'setuppc.Vrated')
        self.connect('geom.R', 'setuppc.R')
        self.connect('VfactorPC', 'setuppc.Vfactor')

        # connections to aero_rated (for max deflection)
        self.connect('spline.r_aero', 'aero_rated.r')
        self.connect('spline.chord_aero', 'aero_rated.chord')
        self.connect('spline.theta_aero', 'aero_rated.theta')
        self.connect('spline.precurve_aero', 'aero_rated.precurve')
        # self.connect('spline.precurve_str[-1]', 'aero_rated.precurveTip') TODO
        self.connect('spline.Rhub', 'aero_rated.Rhub')
        self.connect('spline.Rtip', 'aero_rated.Rtip')
        self.connect('hubHt', 'aero_rated.hubHt')
        self.connect('precone', 'aero_rated.precone')
        self.connect('tilt', 'aero_rated.tilt')
        self.connect('yaw', 'aero_rated.yaw')
        self.connect('airfoil_files', 'aero_rated.airfoil_files')
        self.connect('nBlades', 'aero_rated.B')
        self.connect('rho', 'aero_rated.rho')
        self.connect('mu', 'aero_rated.mu')
        self.connect('shearExp', 'aero_rated.shearExp')
        self.connect('nSector', 'aero_rated.nSector')
        # self.connect('powercurve.ratedConditions.V + 3*gust.sigma', 'aero_rated.V_load')  # OpenMDAO bug
        self.connect('gust.V_gust', 'aero_rated.V_load')
        self.connect('powercurve.ratedConditions:Omega', 'aero_rated.Omega_load')
        self.connect('powercurve.ratedConditions:pitch', 'aero_rated.pitch_load')
        # self.connect('spline.A1_lower_aero', 'aero_rated.A1_lower')
        # self.connect('spline.A2_lower_aero', 'aero_rated.A2_lower')
        # self.connect('spline.A3_lower_aero', 'aero_rated.A3_lower')
        # self.connect('spline.A4_lower_aero', 'aero_rated.A4_lower')
        # self.connect('spline.A1_upper_aero', 'aero_rated.A1_upper')
        # self.connect('spline.A2_upper_aero', 'aero_rated.A2_upper')
        # self.connect('spline.A3_upper_aero', 'aero_rated.A3_upper')
        # self.connect('spline.A4_upper_aero', 'aero_rated.A4_upper')
        self.aero_rated.azimuth_load = 180.0  # closest to tower
        self.aero_rated.run_case = 'loads'

        # connections to aero_extrm (for max strain)
        self.connect('spline.r_aero', 'aero_extrm.r')
        self.connect('spline.chord_aero', 'aero_extrm.chord')
        self.connect('spline.theta_aero', 'aero_extrm.theta')
        self.connect('spline.precurve_aero', 'aero_extrm.precurve')
        # self.connect('spline.precurve_str[-1]', 'aero_extrm.precurveTip') # TODO
        self.connect('spline.Rhub', 'aero_extrm.Rhub')
        self.connect('spline.Rtip', 'aero_extrm.Rtip')
        self.connect('hubHt', 'aero_extrm.hubHt')
        self.connect('precone', 'aero_extrm.precone')
        self.connect('tilt', 'aero_extrm.tilt')
        self.connect('yaw', 'aero_extrm.yaw')
        self.connect('airfoil_files', 'aero_extrm.airfoil_files')
        self.connect('nBlades', 'aero_extrm.B')
        self.connect('rho', 'aero_extrm.rho')
        self.connect('mu', 'aero_extrm.mu')
        self.connect('shearExp', 'aero_extrm.shearExp')
        self.connect('nSector', 'aero_extrm.nSector')
        self.connect('turbineclass.V_extreme', 'aero_extrm.V_load')
        self.connect('pitch_extreme', 'aero_extrm.pitch_load')
        self.connect('azimuth_extreme', 'aero_extrm.azimuth_load')
        # self.connect('spline.A1_lower_aero', 'aero_extrm.A1_lower')
        # self.connect('spline.A2_lower_aero', 'aero_extrm.A2_lower')
        # self.connect('spline.A3_lower_aero', 'aero_extrm.A3_lower')
        # self.connect('spline.A4_lower_aero', 'aero_extrm.A4_lower')
        # self.connect('spline.A1_upper_aero', 'aero_extrm.A1_upper')
        # self.connect('spline.A2_upper_aero', 'aero_extrm.A2_upper')
        # self.connect('spline.A3_upper_aero', 'aero_extrm.A3_upper')
        # self.connect('spline.A4_upper_aero', 'aero_extrm.A4_upper')
        self.aero_extrm.Omega_load = 0.0  # parked case
        self.aero_extrm.run_case = 'loads'

        # connections to aero_extrm_forces (for tower thrust)
        self.connect('spline.r_aero', 'aero_extrm_forces.r')
        self.connect('spline.chord_aero', 'aero_extrm_forces.chord')
        self.connect('spline.theta_aero', 'aero_extrm_forces.theta')
        self.connect('spline.precurve_aero', 'aero_extrm_forces.precurve')
        # self.connect('spline.precurve_str[-1]', 'aero_extrm_forces.precurveTip')
        self.connect('spline.Rhub', 'aero_extrm_forces.Rhub')
        self.connect('spline.Rtip', 'aero_extrm_forces.Rtip')
        self.connect('hubHt', 'aero_extrm_forces.hubHt')
        self.connect('precone', 'aero_extrm_forces.precone')
        self.connect('tilt', 'aero_extrm_forces.tilt')
        self.connect('yaw', 'aero_extrm_forces.yaw')
        self.connect('airfoil_files', 'aero_extrm_forces.airfoil_files')
        self.connect('nBlades', 'aero_extrm_forces.B')
        self.connect('rho', 'aero_extrm_forces.rho')
        self.connect('mu', 'aero_extrm_forces.mu')
        self.connect('shearExp', 'aero_extrm_forces.shearExp')
        self.connect('nSector', 'aero_extrm_forces.nSector')
        self.aero_extrm_forces.Uhub = np.zeros(2)
        self.aero_extrm_forces.Omega = np.zeros(2)  # parked case
        self.aero_extrm_forces.pitch = np.zeros(2)
        # self.connect('turbineclass.V_extreme', 'aero_extrm_forces.Uhub[0]')
        # self.connect('turbineclass.V_extreme', 'aero_extrm_forces.Uhub[1]')
        # self.connect('pitch_extreme', 'aero_extrm_forces.pitch[0]')
        self.aero_extrm_forces.pitch[1] = 90  # feathered
        self.aero_extrm_forces.run_case = 'power'
        self.aero_extrm_forces.T = np.zeros(2)
        self.aero_extrm_forces.Q = np.zeros(2)
        # self.connect('spline.A1_lower_aero', 'aero_extrm_forces.A1_lower')
        # self.connect('spline.A2_lower_aero', 'aero_extrm_forces.A2_lower')
        # self.connect('spline.A3_lower_aero', 'aero_extrm_forces.A3_lower')
        # # self.connect('spline.A4_lower_aero', 'aero_extrm_forces.A4_lower')
        # self.connect('spline.A1_upper_aero', 'aero_extrm_forces.A1_upper')
        # self.connect('spline.A2_upper_aero', 'aero_extrm_forces.A2_upper')
        # self.connect('spline.A3_upper_aero', 'aero_extrm_forces.A3_upper')
        # self.connect('spline.A4_upper_aero', 'aero_extrm_forces.A4_upper')

        # connections to aero_defl_powercurve (for gust reversal)
        self.connect('spline.r_aero', 'aero_defl_powercurve.r')
        self.connect('spline.chord_aero', 'aero_defl_powercurve.chord')
        self.connect('spline.theta_aero', 'aero_defl_powercurve.theta')
        self.connect('spline.precurve_aero', 'aero_defl_powercurve.precurve')
        # self.connect('spline.precurve_str[-1]', 'aero_defl_powercurve.precurveTip') TODO
        self.connect('spline.Rhub', 'aero_defl_powercurve.Rhub')
        self.connect('spline.Rtip', 'aero_defl_powercurve.Rtip')
        self.connect('hubHt', 'aero_defl_powercurve.hubHt')
        self.connect('precone', 'aero_defl_powercurve.precone')
        self.connect('tilt', 'aero_defl_powercurve.tilt')
        self.connect('yaw', 'aero_defl_powercurve.yaw')
        self.connect('airfoil_files', 'aero_defl_powercurve.airfoil_files')
        self.connect('nBlades', 'aero_defl_powercurve.B')
        self.connect('rho', 'aero_defl_powercurve.rho')
        self.connect('mu', 'aero_defl_powercurve.mu')
        self.connect('shearExp', 'aero_defl_powercurve.shearExp')
        self.connect('nSector', 'aero_defl_powercurve.nSector')
        self.connect('setuppc.Uhub', 'aero_defl_powercurve.V_load')
        self.connect('setuppc.Omega', 'aero_defl_powercurve.Omega_load')
        self.connect('setuppc.pitch', 'aero_defl_powercurve.pitch_load')
        # self.connect('spline.A1_lower_aero', 'aero_defl_powercurve.A1_lower')
        # self.connect('spline.A2_lower_aero', 'aero_defl_powercurve.A2_lower')
        # self.connect('spline.A3_lower_aero', 'aero_defl_powercurve.A3_lower')
        # # self.connect('spline.A4_lower_aero', 'aero_defl_powercurve.A4_lower')
        # self.connect('spline.A1_upper_aero', 'aero_defl_powercurve.A1_upper')
        # self.connect('spline.A2_upper_aero', 'aero_defl_powercurve.A2_upper')
        # self.connect('spline.A3_upper_aero', 'aero_defl_powercurve.A3_upper')
        # self.connect('spline.A4_upper_aero', 'aero_defl_powercurve.A4_upper')
        self.aero_defl_powercurve.azimuth_load = 0.0
        self.aero_defl_powercurve.run_case = 'loads'

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


        # connections to loads_defl
        # self.connect('aero_rated.loads', 'loads_defl.aeroLoads')
        self.connect('aero_rated.loads:Omega', 'loads_defl.aeroLoads:Omega')
        self.connect('aero_rated.loads:Px', 'loads_defl.aeroLoads:Px')
        self.connect('aero_rated.loads:Py', 'loads_defl.aeroLoads:Py')
        self.connect('aero_rated.loads:Pz', 'loads_defl.aeroLoads:Pz')
        self.connect('aero_rated.loads:V', 'loads_defl.aeroLoads:V')
        self.connect('aero_rated.loads:azimuth', 'loads_defl.aeroLoads:azimuth')
        self.connect('aero_rated.loads:pitch', 'loads_defl.aeroLoads:pitch')
        self.connect('aero_rated.loads:r', 'loads_defl.aeroLoads:r')

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
        self.connect('aero_defl_powercurve.loads:V', 'loads_pc_defl.aeroLoads:V')
        self.connect('aero_defl_powercurve.loads:azimuth', 'loads_pc_defl.aeroLoads:azimuth')
        self.connect('aero_defl_powercurve.loads:pitch', 'loads_pc_defl.aeroLoads:pitch')
        self.connect('aero_defl_powercurve.loads:r', 'loads_pc_defl.aeroLoads:r')

        # self.connect('aero_defl_powercurve.loads', 'loads_pc_defl.aeroLoads') #TODO
        self.connect('beam.beam:z', 'loads_pc_defl.r')
        self.connect('spline.theta_str', 'loads_pc_defl.theta')
        self.connect('tilt', 'loads_pc_defl.tilt')
        self.connect('curvature.totalCone', 'loads_pc_defl.totalCone')
        self.connect('curvature.z_az', 'loads_pc_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_pc_defl.rhoA')
        self.connect('g', 'loads_pc_defl.g')


        # connections to loads_strain
        # self.connect('aero_extrm.loads', 'loads_strain.aeroLoads')
        self.connect('aero_extrm.loads:Omega', 'loads_strain.aeroLoads:Omega')
        self.connect('aero_extrm.loads:Px', 'loads_strain.aeroLoads:Px')
        self.connect('aero_extrm.loads:Py', 'loads_strain.aeroLoads:Py')
        self.connect('aero_extrm.loads:Pz', 'loads_strain.aeroLoads:Pz')
        self.connect('aero_extrm.loads:V', 'loads_strain.aeroLoads:V')
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
        # self.connect('beam.properties', 'struc.beam')
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA') # TODO
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA')
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA')
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA')
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA')
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA')
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA')
        # self.connect('beam.beam:EA', 'struc.beam.beam:EA')
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
        # self.connect('beam.beam:', 'curvefem.beam') #TODO
        self.connect('spline.theta_str', 'curvefem.theta_str')
        self.connect('spline.precurve_str', 'curvefem.precurve_str')
        self.connect('spline.presweep_str', 'curvefem.presweep_str')
        self.connect('nF', 'curvefem.nF')

        # connections to tip
        # self.struc.dx_defl = np.zeros(1)
        # self.struc.dy_defl = np.zeros(1)
        # self.struc.dz_defl = np.zeros(1)
        # self.spline.theta_str = np.zeros(1)
        # self.curvature.totalCone = np.zeros(1)
        # self.connect('struc.dx_defl', 'tip.dx')
        # self.connect('struc.dy_defl', 'tip.dy')
        # self.connect('struc.dz_defl', 'tip.dz')
        # self.connect('struc.dx_defl[-1]', 'tip.dx')
        # self.connect('struc.dy_defl[-1]', 'tip.dy')
        # self.connect('struc.dz_defl[-1]', 'tip.dz')
        # self.connect('spline.theta_str[-1]', 'tip.theta')
        self.connect('aero_rated.loads:pitch', 'tip.pitch')
        self.connect('aero_rated.loads:azimuth', 'tip.azimuth')
        self.connect('tilt', 'tip.tilt')
        # self.connect('curvature.totalCone[-1]', 'tip.totalConeTip')
        self.connect('dynamic_amplication_tip_deflection', 'tip.dynamicFactor')


        # connections to root moment
        self.connect('spline.r_str', 'root_moment.r_str')
        # self.connect('aero_rated.loads', 'root_moment.aeroLoads')
        self.connect('aero_rated.loads:Omega', 'root_moment.aeroLoads:Omega')
        self.connect('aero_rated.loads:Px', 'root_moment.aeroLoads:Px')
        self.connect('aero_rated.loads:Py', 'root_moment.aeroLoads:Py')
        self.connect('aero_rated.loads:Pz', 'root_moment.aeroLoads:Pz')
        self.connect('aero_rated.loads:V', 'root_moment.aeroLoads:V')
        self.connect('aero_rated.loads:azimuth', 'root_moment.aeroLoads:azimuth')
        self.connect('aero_rated.loads:pitch', 'root_moment.aeroLoads:pitch')
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
        self.connect('aero_extrm_forces.T', 'extreme.T')
        self.connect('aero_extrm_forces.Q', 'extreme.Q')
        self.connect('nBlades', 'extreme.nBlades')

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
        # self.connect('struc.freq', 'freq')
        self.connect('curvefem.freq', 'freq_curvefem_in')
        self.connect('tip.tip_deflection', 'tip_deflection_in')
        self.connect('struc.strainU_spar', 'strainU_spar_in')
        self.connect('struc.strainL_spar', 'strainL_spar_in')
        self.connect('struc.strainU_te', 'strainU_te_in')
        self.connect('struc.strainL_te', 'strainL_te_in')
        # self.connect('root_moment.root_bending_moment', 'root_bending_moment')
        self.connect('beam.eps_crit_spar', 'eps_crit_spar_in')
        self.connect('beam.eps_crit_te', 'eps_crit_te_in')
        self.connect('struc.damageU_spar', 'damageU_spar_in')
        self.connect('struc.damageL_spar', 'damageL_spar_in')
        self.connect('struc.damageU_te', 'damageU_te_in')
        self.connect('struc.damageL_te', 'damageL_te_in')
        self.connect('blade_defl.delta_bladeLength', 'delta_bladeLength_out_in')
        self.connect('blade_defl.delta_precurve_sub', 'delta_precurve_sub_out_in')

        self.connect('spline.Rtip', 'Rtip_in')
        # self.connect('spline.precurve_str[-1]', 'precurveTip') #TODO