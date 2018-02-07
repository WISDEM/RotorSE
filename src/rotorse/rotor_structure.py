# from __future__ import print_function
import numpy as np
import copy
import os
from openmdao.api import IndepVarComp, Component, Group, Problem
from ccblade.ccblade_component import CCBladePower, CCBladeLoads, CCBladeGeometry
from commonse import gravity
from commonse.csystem import DirectionVector
from commonse.utilities import trapz_deriv, interp_with_deriv
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from akima import Akima, akima_interp_with_derivs
from rotor_geometry import RotorGeometry
from rotor_aeropower import AirfoilProperties
import _pBEAM
import _curvefem
import ccblade._bem as _bem  # TODO: move to rotoraero

from rotorse import RPM2RS, RS2RPM, TURBULENCE_CLASS, TURBINE_CLASS, r_aero, r_str

naero = len(r_aero)
nstr = len(r_str)

class CompositeProperties():
    
    # === materials and composite layup  ===
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PreCompFiles')
    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
                      0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                      0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
    sector_idx_strain_spar = [2]*nstr  # (Array): index of sector for spar (PreComp definition of sector)
    sector_idx_strain_te = [3]*nstr  # (Array): index of sector for trailing-edge (PreComp definition of sector)
    chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
                              3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
                              4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
                              4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
                              3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
                              2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)

    upperCS = [0]*nstr
    lowerCS = [0]*nstr
    websCS  = [0]*nstr
    profile = [0]*nstr
    
    web1 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, np.nan])
    web2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, np.nan])
    web3 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    for i in range(nstr):
        webLoc = []
        if not np.isnan(web1[i]): webLoc.append(web1[i])
        if not np.isnan(web2[i]): webLoc.append(web2[i])
        if not np.isnan(web3[i]): webLoc.append(web3[i])

        upperCS[i], lowerCS[i], websCS[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))


# ---------------------
# Base Components
# ---------------------

class BeamPropertiesBase(Component):
    def __init__(self):
        super(BeamPropertiesBase, self).__init__()
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

class StrucBase(Component):
    def __init__(self):
        super(StrucBase, self).__init__()
        # all inputs/outputs in airfoil coordinate system
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
        self.add_param('lifetime', val=20.0, units='year', desc='number of years used in fatigue analysis')

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

'''
class AeroLoads(Component):
    def __init__(self):
        super(AeroLoads, self).__init__()
        self.add_param('r', shape=1, units='m', desc='radial positions along blade going toward tip')
        self.add_param('Px', shape=1, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_param('Py', shape=1, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_param('Pz', shape=1, units='N/m', desc='distributed loads in blade-aligned z-direction')

        self.add_param('V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_param('Omega', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_param('pitch', shape=1, units='deg', desc='pitch angle')
        self.add_param('T', shape=1, units='deg', desc='azimuthal angle')
'''
        
# ---------------------
# Components
# ---------------------

class ResizeCompositeSection(Component):
    def __init__(self):
        super(ResizeCompositeSection, self).__init__()
        
        self.add_param('chord_str', shape=nstr, units='m', desc='structural chord distribution')
        self.add_param('sparT_str', shape=nstr, units='m', desc='structural spar cap thickness distribution')
        self.add_param('teT_str', shape=nstr, units='m', desc='structural trailing-edge panel thickness distribution')

        # out
        self.add_output('upperCSOut', shape=nstr, desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_output('lowerCSOut', shape=nstr, desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_output('websCSOut', shape=nstr, desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-5

    def solve_nonlinear(self, params, unknowns, resids):

        chord_str = params['chord_str']
        sparT_str = params['sparT_str']
        teT_str = params['teT_str']

        # copy data acrosss
        upperCSOut = copy.deepcopy(CompositeProperties.upperCS)
        lowerCSOut = copy.deepcopy(CompositeProperties.lowerCS)
        websCSOut  = copy.deepcopy(CompositeProperties.websCS)

        # scale all thicknesses with airfoil thickness
        # TODO: remove fixed t/c assumption
        # factor = t_str / tref
        factor = chord_str / CompositeProperties.chord_str_ref  # same as thickness ratio for constant t/c
        for i in range(nstr):

            upperCSOut[i].t = [m*factor[i] for m in upperCSOut[i].t]
            lowerCSOut[i].t = [m*factor[i] for m in lowerCSOut[i].t]
            websCSOut[i].t  = [m*factor[i] for m in websCSOut[i].t]

            idx_spar = CompositeProperties.sector_idx_strain_spar[i]
            idx_te = CompositeProperties.sector_idx_strain_te[i]

            # upper and lower have same thickness for this design
            tspar = np.sum(upperCSOut[i].t[idx_spar])
            tte = np.sum(upperCSOut[i].t[idx_te])

            upperCSOut[i].t[idx_spar] *= sparT_str[i]/tspar
            lowerCSOut[i].t[idx_spar] *= sparT_str[i]/tspar

            upperCSOut[i].t[idx_te] *= teT_str[i]/tte
            lowerCSOut[i].t[idx_te] *= teT_str[i]/tte

        unknowns['upperCSOut'] = upperCSOut
        unknowns['lowerCSOut'] = lowerCSOut
        unknowns['websCSOut'] = websCSOut



class PreCompSections(BeamPropertiesBase):
    def __init__(self):
        super(PreCompSections, self).__init__()
        self.add_param('r', shape=nstr, units='m', desc='radial positions. r[0] should be the hub location \
            while r[-1] should be the blade tip. Any number \
            of locations can be specified between these in ascending order.')
        self.add_param('chord', shape=nstr, units='m', desc='array of chord lengths at corresponding radial positions')
        self.add_param('theta', shape=nstr, units='deg', desc='array of twist angles at corresponding radial positions. \
            (positive twist decreases angle of attack)')
        self.add_param('upperCS', val=np.zeros(nstr), desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_param('lowerCS', shape=nstr, desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_param('websCS', shape=nstr, desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)

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

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-5


    def criticalStrainLocations(self, sector_idx_strain, x_ec_nose, y_ec_nose):

        # find corresponding locations on airfoil at midpoint of sector
        xun = np.zeros(nstr)
        xln = np.zeros(nstr)
        yun = np.zeros(nstr)
        yln = np.zeros(nstr)

        for i in range(nstr):
            csU = self.upperCS[i]
            csL = self.lowerCS[i]
            pf = CompositeProperties.profile[i]
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
            Nxx = 2 * (np.pi/sector_length)**2 * (np.sqrt(D1*D2) + D3)

            eps_crit[i] = - Nxx / totalHeight / E

        return eps_crit

    def solve_nonlinear(self, params, unknowns, resids):

        self.chord = params['chord']
        self.r = params['r']
        self.upperCS = params['upperCS']
        self.lowerCS = params['lowerCS']
        self.websCS = params['websCS']
        self.theta = params['theta']

        # Load materials
	basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PreCompFiles')
	self.materials = np.array( Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp')) )

        
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

            xnode, ynode = CompositeProperties.profile[i]._preCompFormat()
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
                th_prime[i], CompositeProperties.leLoc[i],
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

            x_ec_nose[i] = results[13] + CompositeProperties.leLoc[i]*self.chord[i]
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
        unknowns['eps_crit_spar'] = self.panelBucklingStrain(CompositeProperties.sector_idx_strain_spar)
        unknowns['eps_crit_te'] = self.panelBucklingStrain(CompositeProperties.sector_idx_strain_te)

        xu_strain_spar, xl_strain_spar, yu_strain_spar, yl_strain_spar = self.criticalStrainLocations(CompositeProperties.sector_idx_strain_spar, x_ec_nose, y_ec_nose)
        xu_strain_te, xl_strain_te, yu_strain_te, yl_strain_te = self.criticalStrainLocations(CompositeProperties.sector_idx_strain_te, x_ec_nose, y_ec_nose)

        unknowns['xu_strain_spar'] = xu_strain_spar
        unknowns['xl_strain_spar'] = xl_strain_spar
        unknowns['yu_strain_spar'] = yu_strain_spar
        unknowns['yl_strain_spar'] = yl_strain_spar
        unknowns['xu_strain_te'] = xu_strain_te
        unknowns['xl_strain_te'] = xl_strain_te
        unknowns['yu_strain_te'] = yu_strain_te
        unknowns['yl_strain_te'] = yl_strain_te

        
class BladeCurvature(Component):
    def __init__(self):
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

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

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


class CurveFEM(Component):
    def __init__(self):
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

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-5

    def solve_nonlinear(self, params, unknowns, resids):

        r = params['beam:z']

        rhub = r[0]
        bladeLength = r[-1] - r[0]
        bladeFrac = (r - rhub) / bladeLength

        freq = _curvefem.frequencies(params['Omega'], bladeLength, rhub, bladeFrac, params['theta_str'],
                                     params['beam:rhoA'], params['beam:EIxx'], params['beam:EIyy'], params['beam:GJ'], params['beam:EA'], params['beam:rhoJ'],
                                     params['precurve_str'], params['presweep_str'])

        unknowns['freq'] = freq[:params['nF']]



class RotorWithpBEAM(StrucBase):

    def __init__(self):
        super(RotorWithpBEAM, self).__init__()

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-5

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

        # use profil ec.s. to use Hansen's notation
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

        damageU = np.log(N) - m*(np.log(emax) - np.log(eta) - np.log(np.abs(strainU)))
        damageL = np.log(N) - m*(np.log(emax) - np.log(eta) - np.log(np.abs(strainL)))

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
        self.N_damage = 365*24*3600*params['lifetime']

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
        

class DamageLoads(Component):
    def __init__(self):
        super(DamageLoads, self).__init__()
        self.add_param('rstar', shape=naero+1, desc='nondimensional radial locations of damage equivalent moments')
        self.add_param('Mxb', shape=naero+1, units='N*m', desc='damage equivalent moments about blade c.s. x-direction')
        self.add_param('Myb', shape=naero+1, units='N*m', desc='damage equivalent moments about blade c.s. y-direction')
        self.add_param('theta', shape=nstr, units='deg', desc='structural twist')
        self.add_param('r', shape=nstr, units='m', desc='structural radial locations')

        self.add_output('Mxa', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil c.s. x-direction')
        self.add_output('Mya', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil c.s. y-direction')

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
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
    def __init__(self):
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

        # outputs
        self.add_output('Px_af', shape=nstr, desc='total distributed loads in airfoil x-direction')
        self.add_output('Py_af', shape=nstr, desc='total distributed loads in airfoil y-direction')
        self.add_output('Pz_af', shape=nstr, desc='total distributed loads in airfoil z-direction')

        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'


    def solve_nonlinear(self, params, unknowns, resids):

        self.r = params['r']
        self.theta = params['theta']
        self.tilt = params['tilt']
        self.totalCone = params['totalCone']
        self.z_az = params['z_az']
        self.rhoA = params['rhoA']


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
        weight = DirectionVector(0.0, 0.0, -self.rhoA*gravity)

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

        dPx_drhoA = np.diag(-dPwx['dz']*gravity + dPcx['dz']*Omega**2*z_az)
        dPy_drhoA = np.diag(-dPwy['dz']*gravity + dPcy['dz']*Omega**2*z_az)
        dPz_drhoA = np.diag(-dPwz['dz']*gravity + dPcz['dz']*Omega**2*z_az)

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

	#self.deriv_options['form'] = 'central'
	#self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

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
        self.delta = dr.airfoilToBlade(theta).bladeToAzimuth(self.totalConeTip).azimuthToHub(self.azimuth).hubToYaw(self.tilt)

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
        dtheta = dpitch = self.dynamicFactor * self.delta.dx['dtheta']
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

# class ReverseTipDeflection(Component):
#     def __init__(self):
#         super(ReverseTipDeflection, self).__init__()
#         # variables
#         self.add_param('dx', shape=1) # deflection at tip in airfoil c.s.
#         self.add_param('dy', shape=1)
#         self.add_param('dz', shape=1)
#         self.add_param('theta', shape=1)
#         self.add_param('pitch', shape=1)
#         self.add_param('azimuth', shape=1)
#         self.add_param('tilt', shape=1)
#         self.add_param('precone', shape=1)
#         self.add_param('yawW', shape=1)
#         self.add_param('dynamicFactor', val=1.2)
#         self.add_output('tip_deflection', shape=1)
#
#     def solve_nonlinear(self, params, unknowns, resids):
#
#         theta = params['theta'] + params['pitch']
#
#         dr = DirectionVector(params['dx'], params['dy'], params['dz'])
#         self.delta = dr.airfoilToBlade(theta).bladeToAzimuth(params['precone']) \
#             .azimuthToHub(params['azimuth']).hubToYaw(params['tilt']).yawToWind(180.0-params['yawW'])
#
#         unknowns['tip_deflection'] = params['dynamicFactor'] * self.delta.x

class BladeDeflection(Component):
    def __init__(self):
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

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

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
            sm0 = np.sqrt((self.precurve_str0[i] - self.precurve_str0[i-1])**2 + (self.r_str0[i] - self.r_str0[i-1])**2)
            sm = np.sqrt((precurve_str_out[i] - precurve_str_out[i-1])**2 + (self.r_str0[i] - self.r_str0[i-1])**2)
            sp0 = np.sqrt((self.precurve_str0[i+1] - self.precurve_str0[i])**2 + (self.r_str0[i+1] - self.r_str0[i])**2)
            sp = np.sqrt((precurve_str_out[i+1] - precurve_str_out[i])**2 + (self.r_str0[i+1] - self.r_str0[i])**2)
            dl0_dprecurvestr0[i] = (self.precurve_str0[i] - self.precurve_str0[i-1]) / sm0 \
                - (self.precurve_str0[i+1] - self.precurve_str0[i]) / sp0
            dl_dprecurvestr0[i] = (precurve_str_out[i] - precurve_str_out[i-1]) / sm \
                - (precurve_str_out[i+1] - precurve_str_out[i]) / sp
            dl0_drstr0[i] = (self.r_str0[i] - self.r_str0[i-1]) / sm0 \
                - (self.r_str0[i+1] - self.r_str0[i]) / sp0
            dl_drstr0[i] = (self.r_str0[i] - self.r_str0[i-1]) / sm \
                - (self.r_str0[i+1] - self.r_str0[i]) / sp

        sfirst0 = np.sqrt((self.precurve_str0[1] - self.precurve_str0[0])**2 + (self.r_str0[1] - self.r_str0[0])**2)
        sfirst = np.sqrt((precurve_str_out[1] - precurve_str_out[0])**2 + (self.r_str0[1] - self.r_str0[0])**2)
        slast0 = np.sqrt((self.precurve_str0[n-1] - self.precurve_str0[n-2])**2 + (self.r_str0[n-1] - self.r_str0[n-2])**2)
        slast = np.sqrt((precurve_str_out[n-1] - precurve_str_out[n-2])**2 + (self.r_str0[n-1] - self.r_str0[n-2])**2)
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
    def __init__(self):
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
        self.add_output('Mxyz', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_output('Fxyz', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')

        #self.deriv_options['type'] = 'fd'
	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'
        #self.deriv_options['step_size'] = 1e-5

    def solve_nonlinear(self, params, unknowns, resids):

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

        # print 'al.Pz: ', aL.Pz #check=0

        Fx = np.trapz(Px, params['s'])
        Fy = np.trapz(Py, params['s'])
        Fz = np.trapz(Pz, params['s'])

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
        self.root_bending_moment = np.sqrt(Mx**2 + My**2 + Mz**2)

        self.P = P
        self.az = az
        self.Mp = Mp
        self.r = r
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

        unknowns['Mxyz'] = np.array([Mx, My, Mz])
        unknowns['Fxyz'] = np.array([Fx,Fy,Fz])
        # print 'Forces: ', unknowns['Fxyz']
        unknowns['root_bending_moment'] = self.root_bending_moment


    def list_deriv_vars(self):

        inputs = ('r_str', 'aeroLoads:r', 'aeroLoads:Px', 'aeroLoads:Py', 'aeroLoads:Pz', 'totalCone',
                  'x_az', 'y_az', 'z_az', 's')
        outputs = ('root_bending_moment',)

        return inputs, outputs


    def linearize(self, params, unknowns, resids):

        # dx_dr = -sind(self.precone)
        # dz_dr = cosd(self.precone)

        # dx_dprecone = -self.r*cosd(self.precone)*np.pi/180.0
        # dz_dprecone = -self.r*sind(self.precone)*np.pi/180.0

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

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'


    def solve_nonlinear(self, params, unknowns, resids):

        self.blade_mass = params['blade_mass']
        self.blade_moment_of_inertia = params['blade_moment_of_inertia']
        self.tilt = params['tilt']
        self.nBlades = params['nBlades']

        self.mass_all_blades = self.nBlades * self.blade_mass

        Ibeam = self.nBlades * self.blade_moment_of_inertia

        Ixx = Ibeam
        Iyy = Ibeam/2.0  # azimuthal average for 2 blades, exact for 3+
        Izz = Ibeam/2.0
        Ixy = 0.0
        Ixz = 0.0
        Iyz = 0.0  # azimuthal average for 2 blades, exact for 3+

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
        self.add_param('turbulence_class', val=TURBULENCE_CLASS['A'], desc='IEC turbulence class', pass_by_obj=True)
        self.add_param('std', val=3, desc='number of standard deviations for strength of gust', pass_by_obj=True)

        # out
        self.add_output('V_gust', shape=1, units='m/s', desc='gust wind speed')

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        self.V_mean = params['V_mean']
        self.V_hub = params['V_hub']
        self.turbulence_class = params['turbulence_class']
        self.std = params['std']


        if self.turbulence_class == TURBULENCE_CLASS['A']:
            Iref = 0.16
        elif self.turbulence_class == TURBULENCE_CLASS['B']:
            Iref = 0.14
        elif self.turbulence_class == TURBULENCE_CLASS['C']:
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

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

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

        inputs = ('control:tsr', 'control:pitch', 'Vrated', 'R')
        outputs = ('Uhub', 'Omega', 'pitch')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        J = {}
        J['Uhub', 'control:tsr'] = 0.0
        J['Uhub', 'Vrated'] = self.Vfactor
        J['Uhub', 'R'] = 0.0
        J['Uhub', 'control:pitch'] = 0.0
        J['Omega', 'control:tsr'] = self.Uhub/self.R*RS2RPM
        J['Omega', 'Vrated'] = params['control:tsr']*self.Vfactor/self.R*RS2RPM
        J['Omega', 'R'] = -params['control:tsr']*self.Uhub/self.R**2*RS2RPM
        J['Omega', 'control:pitch'] = 0.0
        J['pitch', 'control:tsr'] = 0.0
        J['pitch', 'Vrated'] = 0.0
        J['pitch', 'R'] = 0.0
        J['pitch', 'control:pitch'] = 1.0

        return J


class ConstraintsStructures(Component):
    def __init__(self):
        super(ConstraintsStructures, self).__init__()

        self.add_param('nBlades', val=3, desc='number of blades', pass_by_obj=True)
        self.add_param('freq', shape=5, units='Hz', desc='1st nF natural frequencies')
        self.add_param('freq_curvefem', shape=5, units='Hz', desc='1st nF natural frequencies')
        self.add_param('Omega', shape=1, units='rpm', desc='rotation speed')
        self.add_param('strainU_spar', shape=nstr, desc='axial strain and specified locations')
        self.add_param('strainL_spar', shape=nstr, desc='axial strain and specified locations')
        self.add_param('strainU_te', shape=nstr, desc='axial strain and specified locations')
        self.add_param('strainL_te', shape=nstr, desc='axial strain and specified locations')
        self.add_param('strain_ult_spar', val=0.0, desc='ultimate strain in spar cap')
        self.add_param('strain_ult_te', val=0.0, desc='uptimate strain in trailing-edge panels')
        self.add_param('eps_crit_spar', shape=nstr, desc='critical strain in spar from panel buckling calculation')
        self.add_param('eps_crit_te', shape=nstr, desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_param('damageU_spar', shape=nstr, desc='fatigue damage on upper surface in spar cap')
        self.add_param('damageL_spar', shape=nstr, desc='fatigue damage on lower surface in spar cap')
        self.add_param('damageU_te', shape=nstr, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_param('damageL_te', shape=nstr, desc='fatigue damage on lower surface in trailing-edge panels')
        self.add_param('eta_damage', val=0.0, desc='strain safety factor')
        self.add_param('eta_freq', val=0.0, desc='resonance frequency safety factor')

        self.add_output('Pn_margin', shape=5, desc='Blade natural frequency (pBeam) relative to blade passing frequency')
        self.add_output('P1_margin', shape=5, desc='Blade natural frequency (pBeam) relative to rotor passing frequency')
        self.add_output('Pn_margin_cfem', shape=5, desc='Blade natural frequency (curvefem) relative to blade passing frequency')
        self.add_output('P1_margin_cfem', shape=5, desc='Blade natural frequency (curvefem) relative to rotor passing frequency')
        self.add_output('rotor_strain_sparU', shape=nstr, desc='Strain in upper spar relative to ultimate allowable')
        self.add_output('rotor_strain_sparL', shape=nstr, desc='Strain in lower spar relative to ultimate allowable')
        self.add_output('rotor_strain_teU', shape=nstr, desc='Strain in upper trailing edge relative to ultimate allowable')
        self.add_output('rotor_strain_teL', shape=nstr, desc='Strain in lower trailing edge relative to ultimate allowable')
        self.add_output('rotor_buckling_sparU', shape=nstr, desc='Buckling in upper spar relative to ultimate allowable')
        self.add_output('rotor_buckling_sparL', shape=nstr, desc='Buckling in lower spar relative to ultimate allowable')
        self.add_output('rotor_buckling_teU', shape=nstr, desc='Buckling in upper trailing edge relative to ultimate allowable')
        self.add_output('rotor_buckling_teL', shape=nstr, desc='Buckling in lower trailing edge relative to ultimate allowable')
        self.add_output('rotor_damage_sparU', shape=nstr, desc='Damage in upper spar relative to ultimate allowable')
        self.add_output('rotor_damage_sparL', shape=nstr, desc='Damage in lower spar relative to ultimate allowable')
        self.add_output('rotor_damage_teU', shape=nstr, desc='Damage in upper trailing edge relative to ultimate allowable')
        self.add_output('rotor_damage_teL', shape=nstr, desc='Damage in lower trailing edge relative to ultimate allowable')

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        omega = params['Omega'] / 60.0 #Hz
        eta_f = params['eta_freq']
        eta_s = params['eta_damage']
        strain_ult_spar = params['strain_ult_spar']
        strain_ult_te   = params['strain_ult_te']
        eps_crit_spar   = params['eps_crit_spar']
        eps_crit_te     = params['eps_crit_te']
        
        unknowns['Pn_margin'] = (params['nBlades']*omega*eta_f) / params['freq']
        unknowns['P1_margin'] = (                  omega*eta_f) / params['freq']
        
        unknowns['Pn_margin_cfem'] = (params['nBlades']*omega*eta_f) / params['freq_curvefem']
        unknowns['P1_margin_cfem'] = (                  omega*eta_f) / params['freq_curvefem']

        unknowns['rotor_strain_sparU'] = params['strainU_spar'] * eta_s / strain_ult_spar
        unknowns['rotor_strain_sparL'] = params['strainL_spar'] * eta_s / strain_ult_spar
        unknowns['rotor_strain_teU']   = params['strainU_te'] * eta_s / strain_ult_te
        unknowns['rotor_strain_teL']   = params['strainL_te'] * eta_s / strain_ult_te

        # TODO: Don't understand these!
        unknowns['rotor_buckling_sparU'] = (eps_crit_spar - params['strainU_spar']) / strain_ult_spar
        unknowns['rotor_buckling_sparL'] = (eps_crit_spar - params['strainL_spar']) / strain_ult_spar
        unknowns['rotor_buckling_teU']   = (eps_crit_te   - params['strainU_te']  ) / strain_ult_te
        unknowns['rotor_buckling_teL']   = (eps_crit_te   - params['strainL_te']  ) / strain_ult_te

        unknowns['rotor_damage_sparU'] = params['damageU_spar']
        unknowns['rotor_damage_sparL'] = params['damageL_spar']
        unknowns['rotor_damage_teU']   = params['damageU_te']
        unknowns['rotor_damage_teL']   = params['damageL_te']


    def linearize(self, params, unknowns, resids):
        omega = params['Omega'] / 60.0 #Hz
        eta_f = params['eta_freq']
        eta_s = params['eta_damage']
        strain_ult_spar = params['strain_ult_spar']
        strain_ult_te   = params['strain_ult_te']
        eps_crit_spar   = params['eps_crit_spar']
        eps_crit_te     = params['eps_crit_te']

        J = {}

        myones = np.ones((nstr,))

        J['Pn_margin','Omega']    = (params['nBlades']*eta_f) / params['freq']
        J['Pn_margin','eta_freq'] = (params['nBlades']*omega) / params['freq']
        J['Pn_margin','freq']     = -np.diag(unknowns['Pn_margin'])  / params['freq']
        J['P1_margin','Omega']    = eta_f / params['freq']
        J['P1_margin','eta_freq'] = omega / params['freq']
        J['P1_margin','freq']     = -np.diag(unknowns['P1_margin'])  / params['freq']
        
        J['Pn_margin_cfem','Omega']     = (params['nBlades']*eta_f) / params['freq_curvefem']
        J['Pn_margin_cfem','eta_freq']  = (params['nBlades']*omega) / params['freq_curvefem']
        J['Pn_margin_cfem','freq_cfem'] = -np.diag(unknowns['Pn_margin_cfem'])  / params['freq_curvefem']
        J['P1_margin_cfem','Omega']     = eta_f / params['freq_curvefem']
        J['P1_margin_cfem','eta_freq']  = omega / params['freq_curvefem']
        J['P1_margin_cfem','freq_cfem'] = -np.diag(unknowns['P1_margin_cfem'])  / params['freq_curvefem']
        
        J['rotor_strain_sparU', 'eta_damage'] = params['strainU_spar'] / strain_ult_spar
        J['rotor_strain_sparL', 'eta_damage'] = params['strainL_spar'] / strain_ult_spar
        J['rotor_strain_teU'  , 'eta_damage'] = params['strainU_te']   / strain_ult_te
        J['rotor_strain_teL'  , 'eta_damage'] = params['strainL_te']   / strain_ult_te

        J['rotor_strain_sparU', 'strainU_spar'] = eta_s * np.diag(myones) / strain_ult_spar
        J['rotor_strain_sparL', 'strainL_spar'] = eta_s * np.diag(myones) / strain_ult_spar
        J['rotor_strain_teU'  , 'strainU_te']   = eta_s * np.diag(myones) / strain_ult_te
        J['rotor_strain_teL'  , 'strainL_te']   = eta_s * np.diag(myones) / strain_ult_te

        J['rotor_strain_sparU', 'strain_ult_spar'] = -unknowns['rotor_strain_sparU'] / strain_ult_spar
        J['rotor_strain_sparL', 'strain_ult_spar'] = -unknowns['rotor_strain_sparL'] / strain_ult_spar
        J['rotor_strain_teU'  , 'strain_ult_te']   = -unknowns['rotor_strain_teU']   / strain_ult_te
        J['rotor_strain_teL'  , 'strain_ult_te']   = -unknowns['rotor_strain_teL']   / strain_ult_te

        J['rotor_buckling_sparU', 'eps_crit_spar'] = J['rotor_buckling_sparL', 'eps_crit_spar'] = np.diag(np.ones(len(params['eps_crit_spar']))) / strain_ult_spar
        J['rotor_buckling_teU'  , 'eps_crit_te']   = J['rotor_buckling_teL'  , 'eps_crit_te']   = np.diag(np.ones(len(params['eps_crit_te'])))   / strain_ult_te

        J['rotor_buckling_sparU', 'strainU_spar'] = -np.diag(myones) / strain_ult_spar
        J['rotor_buckling_sparL', 'strainL_spar'] = -np.diag(myones) / strain_ult_spar
        J['rotor_buckling_teU'  , 'strainU_te']   = -np.diag(myones) / strain_ult_te
        J['rotor_buckling_teL'  , 'strainL_te']   = -np.diag(myones) / strain_ult_te

        J['rotor_buckling_sparU', 'strain_ult_spar'] = -unknowns['rotor_buckling_sparU'] / strain_ult_spar
        J['rotor_buckling_sparL', 'strain_ult_spar'] = -unknowns['rotor_buckling_sparL'] / strain_ult_spar
        J['rotor_buckling_teU'  , 'strain_ult_te']   = -unknowns['rotor_buckling_teU']   / strain_ult_te
        J['rotor_buckling_teL'  , 'strain_ult_te']   = -unknowns['rotor_buckling_teL']   / strain_ult_te
        
        J['rotor_damage_sparU', 'damageU_spar'] = np.diag(myones)
        J['rotor_damage_sparL', 'damageL_spar'] = np.diag(myones)
        J['rotor_damage_teU', 'damageU_te']     = np.diag(myones)
        J['rotor_damage_teL', 'damageL_te']     = np.diag(myones)

        return J
    
class OutputsStructures(Component):
    def __init__(self):
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
        # additional drivetrain moments output
        self.add_param('Fxyz_1_in', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_param('Fxyz_2_in', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_param('Fxyz_3_in', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_param('Fxyz_4_in', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_param('Fxyz_5_in', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_param('Fxyz_6_in', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_param('Mxyz_1_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_param('Mxyz_2_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_param('Mxyz_3_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_param('Mxyz_4_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_param('Mxyz_5_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_param('Mxyz_6_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_param('TotalCone_in', shape=1, units='rad', desc='total cone angle for blades at rated')
        self.add_param('Pitch_in', shape=1, units='rad', desc='pitch angle at rated')
        self.add_param('nBlades', val=3, desc='Number of blades on rotor', pass_by_obj=True)

        # structural outputs
        self.add_output('mass_one_blade', shape=1, units='kg', desc='mass of one blade')
        self.add_output('mass_all_blades', shape=1,  units='kg', desc='mass of all blade')
        self.add_output('I_all_blades', shape=6, desc='out of plane moments of inertia in yaw-aligned c.s.')
        self.add_output('freq', shape=5, units='Hz', desc='1st nF natural frequencies')
        self.add_output('freq_curvefem', shape=5, units='Hz', desc='1st nF natural frequencies')
        self.add_output('tip_deflection', shape=1, units='m', desc='blade tip deflection in +x_y direction')
        self.add_output('strainU_spar', shape=nstr, desc='axial strain and specified locations')
        self.add_output('strainL_spar', shape=nstr, desc='axial strain and specified locations')
        self.add_output('strainU_te', shape=nstr, desc='axial strain and specified locations')
        self.add_output('strainL_te', shape=nstr, desc='axial strain and specified locations')
        self.add_output('eps_crit_spar', shape=nstr, desc='critical strain in spar from panel buckling calculation')
        self.add_output('eps_crit_te', shape=nstr,  desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_output('root_bending_moment', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')
        self.add_output('damageU_spar', shape=nstr, desc='fatigue damage on upper surface in spar cap')
        self.add_output('damageL_spar', shape=nstr, desc='fatigue damage on lower surface in spar cap')
        self.add_output('damageU_te', shape=nstr, desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_output('damageL_te', shape=nstr, desc='fatigue damage on lower surface in trailing-edge panels')
        self.add_output('delta_bladeLength_out', shape=1, units='m', desc='adjustment to blade length to account for curvature from loading')
        self.add_output('delta_precurve_sub_out', shape=3, units='m', desc='adjustment to precurve to account for curvature from loading')
        # additional drivetrain moments output
        self.add_output('Fxyz_1', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_output('Fxyz_2', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_output('Fxyz_3', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_output('Fxyz_4', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_output('Fxyz_5', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_output('Fxyz_6', val=np.zeros((3,)), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_output('Mxyz_1', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_output('Mxyz_2', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_output('Mxyz_3', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_output('Mxyz_4', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_output('Mxyz_5', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_output('Mxyz_6', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_output('Fxyz_total', val=np.zeros((3,)), units='N', desc='Total force [x,y,z] at the blade root in *hub* c.s.')
        self.add_output('Mxyz_total', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in *hub* c.s.')
        self.add_output('TotalCone', shape=1, units='rad', desc='total cone angle for blades at rated')
        self.add_output('Pitch', shape=1, units='rad', desc='pitch angle at rated')


    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['mass_one_blade'] = params['mass_one_blade_in']
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
        unknowns['TotalCone'] = params['TotalCone_in']
        unknowns['Pitch'] = params['Pitch_in']

        for k in range(1,7):
            kstr = '_'+str(k)
            unknowns['Fxyz'+kstr] = params['Fxyz'+kstr+'_in']
            unknowns['Mxyz'+kstr] = params['Mxyz'+kstr+'_in']

        # TODO: This is meant to sum up forces and torques across all blades while taking into account coordinate systems
        # This may not be necessary as CCBlade returns total thrust (T) and torque (Q), which are the only non-zero F & M entries anyway
        # The difficulty is that the answers don't match exactly.
        F_hub   = np.array([params['Fxyz_1_in'], params['Fxyz_2_in'], params['Fxyz_3_in'], params['Fxyz_4_in'], params['Fxyz_5_in'], params['Fxyz_6_in']])
        M_hub   = np.array([params['Mxyz_1_in'], params['Mxyz_2_in'], params['Mxyz_3_in'], params['Mxyz_4_in'], params['Mxyz_5_in'], params['Mxyz_6_in']])

        nBlades = params['nBlades']
        angles  = np.linspace(0, 360, nBlades+1)
        # Initialize summation
        F_hub_tot = np.zeros((3,))
        M_hub_tot = np.zeros((3,))
        dFx_dF    = np.zeros(F_hub.shape)
        dFy_dF    = np.zeros(F_hub.shape)
        dFz_dF    = np.zeros(F_hub.shape)
        dMx_dM    = np.zeros(M_hub.shape)
        dMy_dM    = np.zeros(M_hub.shape)
        dMz_dM    = np.zeros(M_hub.shape)
        # Convert from blade to hub c.s.
        for row in xrange(nBlades):
            myF = DirectionVector.fromArray(F_hub[row,:]).azimuthToHub(angles[row])
            myM = DirectionVector.fromArray(M_hub[row,:]).azimuthToHub(angles[row])
            
            F_hub_tot += myF.toArray()
            M_hub_tot += myM.toArray()

            dFx_dF[row,:] = np.array([myF.dx['dx'], myF.dx['dy'], myF.dx['dz']])
            dFy_dF[row,:] = np.array([myF.dy['dx'], myF.dy['dy'], myF.dy['dz']])
            dFz_dF[row,:] = np.array([myF.dz['dx'], myF.dz['dy'], myF.dz['dz']])

            dMx_dM[row,:] = np.array([myM.dx['dx'], myM.dx['dy'], myM.dx['dz']])
            dMy_dM[row,:] = np.array([myM.dy['dx'], myM.dy['dy'], myM.dy['dz']])
            dMz_dM[row,:] = np.array([myM.dz['dx'], myM.dz['dy'], myM.dz['dz']])

        # Now sum over all blades
        unknowns['Fxyz_total'] = F_hub_tot
        unknowns['Mxyz_total'] = M_hub_tot
        self.J = {}
        for k in range(6):
            kstr = '_'+str(k+1)
            self.J['Fxyz_total','Fxyz'+kstr+'_in'] = np.vstack([dFx_dF[k,:], dFy_dF[k,:], dFz_dF[k,:]])
            self.J['Mxyz_total','Mxyz'+kstr+'_in'] = np.vstack([dMx_dM[k,:], dMy_dM[k,:], dMz_dM[k,:]])

    def linearize(self, params, unknowns, resids):
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

        for k in range(1,7):
            kstr = '_'+str(k)
            J['Fxyz'+kstr, 'Fxyz'+kstr+'_in'] = np.diag(np.ones(len(params['Fxyz'+kstr+'_in'])))
            J['Mxyz'+kstr, 'Mxyz'+kstr+'_in'] = np.diag(np.ones(len(params['Mxyz'+kstr+'_in'])))
        J['TotalCone', 'TotalCone_in'] = 1
        J['Pitch', 'Pitch_in'] = 1
        for key in self.J.keys(): J[key] = self.J[key]
        return J



class RotorStructure(Group):
    def __init__(self):
        super(RotorStructure, self).__init__()
        """rotor model"""

        self.add('hubHt', IndepVarComp('hubHt', val=90.0), promotes=['*'])
        #self.add('rho', IndepVarComp('rho', val=1.225), promotes=['*'])
        #self.add('mu', IndepVarComp('mu', val=1.81e-5), promotes=['*'])
        #self.add('shearExp', IndepVarComp('shearExp', val=0.2), promotes=['*'])

        self.add('VfactorPC', IndepVarComp('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation'), promotes=['*'])
        self.add('turbulence_class', IndepVarComp('turbulence_class', val=TURBULENCE_CLASS['A'], desc='IEC turbulence class class', pass_by_obj=True), promotes=['*'])
        self.add('gust_stddev', IndepVarComp('gust_stddev', val=3, pass_by_obj=True), promotes=['*'])
        self.add('airfoil_files', IndepVarComp('airfoil_files', AirfoilProperties.airfoil_files, pass_by_obj=True), promotes=['*'])

        # --- computation options ---
        self.add('tiploss', IndepVarComp('tiploss', True, pass_by_obj=True), promotes=['*'])
        self.add('hubloss', IndepVarComp('hubloss', True, pass_by_obj=True), promotes=['*'])
        self.add('wakerotation', IndepVarComp('wakerotation', True, pass_by_obj=True), promotes=['*'])
        self.add('usecd', IndepVarComp('usecd', True, pass_by_obj=True), promotes=['*'])
        self.add('nSector', IndepVarComp('nSector', val=4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power', pass_by_obj=True), promotes=['*'])
        
        # --- control ---
        self.add('c_tsr', IndepVarComp('control:tsr', val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)'), promotes=['*'])
        self.add('c_pitch', IndepVarComp('control:pitch', val=0.0, units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)'), promotes=['*'])
        self.add('pitch_extreme', IndepVarComp('pitch_extreme', val=0.0, units='deg', desc='worst-case pitch at survival wind condition'), promotes=['*'])
        self.add('azimuth_extreme', IndepVarComp('azimuth_extreme', val=0.0, units='deg', desc='worst-case azimuth at survival wind condition'), promotes=['*'])
        
        # --- composite sections ---
        #self.add('sparT', IndepVarComp('sparT', val=np.zeros(5), units='m', desc='spar cap thickness parameters'), promotes=['*'])
        #self.add('teT', IndepVarComp('teT', val=np.zeros(5), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])

        # --- fatigue ---
        self.add('rstar_damage', IndepVarComp('rstar_damage', val=np.zeros(naero+1), desc='nondimensional radial locations of damage equivalent moments'), promotes=['*'])
        self.add('Mxb_damage', IndepVarComp('Mxb_damage', val=np.zeros(naero+1), units='N*m', desc='damage equivalent moments about blade c.s. x-direction'), promotes=['*'])
        self.add('Myb_damage', IndepVarComp('Myb_damage', val=np.zeros(naero+1), units='N*m', desc='damage equivalent moments about blade c.s. y-direction'), promotes=['*'])
        self.add('strain_ult_spar', IndepVarComp('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap'), promotes=['*'])
        self.add('strain_ult_te', IndepVarComp('strain_ult_te', val=2500*1e-6, desc='uptimate strain in trailing-edge panels'), promotes=['*'])
        self.add('eta_damage', IndepVarComp('eta_damage', val=1.755, desc='safety factor for fatigue'), promotes=['*'])
        self.add('m_damage', IndepVarComp('m_damage', val=10.0, desc='slope of S-N curve for fatigue analysis'), promotes=['*'])
        self.add('lifetime', IndepVarComp('lifetime', val=20.0, units='year', desc='project lifetime for fatigue analysis'), promotes=['*'])
        self.add('eta_freq', IndepVarComp('eta_freq', val=1.1, desc='ultimate strain in spar cap'), promotes=['*'])


        # --- options ---
        self.add('dynamic_amplication_tip_deflection', IndepVarComp('dynamic_amplication_tip_deflection', val=1.2, desc='a dynamic amplification factor to adjust the static deflection calculation'), promotes=['*'])
        self.add('nF', IndepVarComp('nF', val=5, desc='number of natural frequencies to compute', pass_by_obj=True), promotes=['*'])


        # Geometry
        self.add('rotorGeometry', RotorGeometry(), promotes=['*'])

        # --- add structures ---
        self.add('curvature', BladeCurvature())
        self.add('resize', ResizeCompositeSection())
        self.add('gust', GustETM())
        self.add('setuppc',  SetupPCModVarSpeed())
        self.add('beam', PreCompSections())

        self.add('aero_rated', CCBladeLoads(naero, 1))
        self.add('aero_extrm', CCBladeLoads(naero,  1))
        self.add('aero_extrm_forces', CCBladePower(naero, 2))
        self.add('aero_defl_powercurve', CCBladeLoads(naero,  1))

        self.add('loads_defl', TotalLoads())
        self.add('loads_pc_defl', TotalLoads())
        self.add('loads_strain', TotalLoads())

        self.add('damage', DamageLoads())
        self.add('struc', RotorWithpBEAM())
        self.add('curvefem', CurveFEM())
        self.add('tip', TipDeflection())
        self.add('root_moment', RootMoment())
        self.add('mass', MassProperties())
        self.add('extreme', ExtremeLoads())
        self.add('blade_defl', BladeDeflection())

        self.add('aero_0', CCBladeLoads(naero,  1))
        self.add('aero_120', CCBladeLoads(naero,  1))
        self.add('aero_240', CCBladeLoads(naero,  1))
        self.add('root_moment_0', RootMoment())
        self.add('root_moment_120', RootMoment())
        self.add('root_moment_240', RootMoment())

        self.add('output_struc', OutputsStructures(), promotes=['*'])
        self.add('constraints', ConstraintsStructures(), promotes=['*'])

        # connections to curvature
        self.connect('spline.r_str', 'curvature.r')
        self.connect('spline.precurve_str', 'curvature.precurve')
        self.connect('spline.presweep_str', 'curvature.presweep')
        self.connect('precone', 'curvature.precone')

        # connections to resize
        self.connect('spline.chord_str', 'resize.chord_str')
        self.connect('spline.sparT_str', 'resize.sparT_str')
        self.connect('spline.teT_str', 'resize.teT_str')

        # connections to gust
        self.connect('turbulence_class', 'gust.turbulence_class')
        self.connect('turbineclass.V_mean', 'gust.V_mean')
        self.connect('gust.V_hub', 'setuppc.Vrated')
        self.connect('gust_stddev', 'gust.std')
        
        # connections to setuppc
        self.connect('control:pitch', 'setuppc.control:pitch')
        self.connect('control:tsr', 'setuppc.control:tsr')
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
        self.connect('airfoil_files', 'aero_rated.airfoil_files')
        self.connect('nBlades', 'aero_rated.B')
        self.connect('nSector', 'aero_rated.nSector')
        self.connect('gust.V_gust', 'aero_rated.V_load')
        self.aero_rated.azimuth_load = 180.0  # closest to tower

        self.connect('aero_rated.Omega_load', ['curvefem.Omega','aero_0.Omega_load','aero_120.Omega_load','aero_240.Omega_load'])
        
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
        self.connect('airfoil_files', 'aero_extrm.airfoil_files')
        self.connect('nBlades', 'aero_extrm.B')
        self.connect('nSector', 'aero_extrm.nSector')
        self.connect('turbineclass.V_extreme', 'aero_extrm.V_load')
        self.connect('pitch_extreme', 'aero_extrm.pitch_load')
        self.connect('azimuth_extreme', 'aero_extrm.azimuth_load')
        self.aero_extrm.Omega_load = 0.0  # parked case

        # connections to aero_extrm_forces (for tower thrust)
        self.connect('spline.r_aero', 'aero_extrm_forces.r')
        self.connect('spline.chord_aero', 'aero_extrm_forces.chord')
        self.connect('spline.theta_aero', 'aero_extrm_forces.theta')
        self.connect('spline.precurve_aero', 'aero_extrm_forces.precurve')
        self.connect('spline.precurve_str', 'aero_extrm_forces.precurveTip', src_indices=[nstr-1])
        self.connect('spline.Rhub', 'aero_extrm_forces.Rhub')
        self.connect('spline.Rtip', 'aero_extrm_forces.Rtip')
        self.connect('hubHt', 'aero_extrm_forces.hubHt')
        self.connect('precone', 'aero_extrm_forces.precone')
        self.connect('tilt', 'aero_extrm_forces.tilt')
        self.connect('yaw', 'aero_extrm_forces.yaw')
        self.connect('airfoil_files', 'aero_extrm_forces.airfoil_files')
        self.connect('nBlades', 'aero_extrm_forces.B')
        self.connect('nSector', 'aero_extrm_forces.nSector')
        self.aero_extrm_forces.Uhub = np.zeros(2)
        self.aero_extrm_forces.Omega = np.zeros(2)  # parked case
        self.aero_extrm_forces.pitch = np.zeros(2)
        self.connect('turbineclass.V_extreme_full', 'aero_extrm_forces.Uhub')
        self.aero_extrm_forces.pitch = np.array([0.0, 90.0])  # feathered
        self.aero_extrm_forces.T = np.zeros(2)
        self.aero_extrm_forces.Q = np.zeros(2)

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
        self.connect('airfoil_files', 'aero_defl_powercurve.airfoil_files')
        self.connect('nBlades', 'aero_defl_powercurve.B')
        self.connect('nSector', 'aero_defl_powercurve.nSector')
        self.connect('setuppc.Uhub', 'aero_defl_powercurve.V_load')
        self.connect('setuppc.Omega', 'aero_defl_powercurve.Omega_load')
        self.connect('setuppc.pitch', 'aero_defl_powercurve.pitch_load')
        self.connect('setuppc.azimuth', 'aero_defl_powercurve.azimuth_load')
        self.aero_defl_powercurve.azimuth_load = 0.0

        # connections to beam
        self.connect('spline.r_str', 'beam.r')
        self.connect('spline.chord_str', 'beam.chord')
        self.connect('spline.theta_str', 'beam.theta')
        self.connect('resize.upperCSOut', 'beam.upperCS')
        self.connect('resize.lowerCSOut', 'beam.lowerCS')
        self.connect('resize.websCSOut', 'beam.websCS')

        self.connect('aero_0.rho', ['aero_120.rho','aero_240.rho','aero_defl_powercurve.rho','aero_extrm_forces.rho','aero_extrm.rho','aero_rated.rho'])
        self.connect('aero_0.mu', ['aero_120.mu','aero_240.mu','aero_defl_powercurve.mu','aero_extrm_forces.mu','aero_extrm.mu','aero_rated.mu'])
        self.connect('aero_0.shearExp', ['aero_120.shearExp','aero_240.shearExp','aero_defl_powercurve.shearExp','aero_extrm_forces.shearExp','aero_extrm.shearExp','aero_rated.shearExp'])
        
        # connections to loads_defl
        self.connect('aero_rated.loads:Omega', 'loads_defl.aeroLoads:Omega')
        self.connect('aero_rated.loads:Px', 'loads_defl.aeroLoads:Px')
        self.connect('aero_rated.loads:Py', 'loads_defl.aeroLoads:Py')
        self.connect('aero_rated.loads:Pz', 'loads_defl.aeroLoads:Pz')
        self.connect('aero_rated.loads:azimuth', 'loads_defl.aeroLoads:azimuth')
        self.connect('aero_rated.loads:pitch', 'loads_defl.aeroLoads:pitch')
        self.connect('aero_rated.loads:r', 'loads_defl.aeroLoads:r')

        self.connect('beam.beam:z', 'loads_defl.r')
        self.connect('spline.theta_str', 'loads_defl.theta')
        self.connect('tilt', 'loads_defl.tilt')
        self.connect('curvature.totalCone', 'loads_defl.totalCone')
        self.connect('curvature.z_az', 'loads_defl.z_az')
        self.connect('beam.beam:rhoA', 'loads_defl.rhoA')

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
        self.connect('lifetime', 'struc.lifetime')

        # connections to curvefem
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


        ### adding for the drivetrain root moment calculations:
        # TODO - number and value of azimuth angles should be arbitrary user inputs
        # connections to aero_0 (for rated loads at 0 azimuth angle)
        self.connect('spline.r_aero', ['aero_0.r','aero_120.r','aero_240.r'])
        self.connect('spline.chord_aero', ['aero_0.chord', 'aero_120.chord', 'aero_240.chord'])
        self.connect('spline.theta_aero', ['aero_0.theta', 'aero_120.theta', 'aero_240.theta'])
        self.connect('spline.precurve_aero', ['aero_0.precurve', 'aero_120.precurve', 'aero_240.precurve'])
        self.connect('spline.precurve_str', ['aero_0.precurveTip', 'aero_120.precurveTip', 'aero_240.precurveTip'], src_indices=[naero-1])
        self.connect('spline.Rhub', ['aero_0.Rhub', 'aero_120.Rhub', 'aero_240.Rhub'])
        self.connect('spline.Rtip', ['aero_0.Rtip', 'aero_120.Rtip', 'aero_240.Rtip'])
        self.connect('hubHt', ['aero_0.hubHt', 'aero_120.hubHt', 'aero_240.hubHt'])
        self.connect('precone', ['aero_0.precone', 'aero_120.precone', 'aero_240.precone'])
        self.connect('tilt', ['aero_0.tilt', 'aero_120.tilt', 'aero_240.tilt'])
	self.connect('airfoil_files', ['aero_0.airfoil_files', 'aero_120.airfoil_files', 'aero_240.airfoil_files'])
        self.connect('yaw', ['aero_0.yaw', 'aero_120.yaw', 'aero_240.yaw'])
        self.connect('nBlades', ['aero_0.B','aero_120.B', 'aero_240.B'])
        self.connect('nSector', ['aero_0.nSector','aero_120.nSector','aero_240.nSector'])
        self.connect('gust.V_gust', ['aero_0.V_load','aero_120.V_load','aero_240.V_load'])

        self.add('pitch_load89', IndepVarComp('pitch_load89', val=89.0, units='deg'), promotes=['*'])
        self.add('azimuth_load0', IndepVarComp('azimuth_load0', val=0.0, units='deg'), promotes=['*'])
        self.add('azimuth_load120', IndepVarComp('azimuth_load120', val=120.0, units='deg'), promotes=['*'])
        self.add('azimuth_load240', IndepVarComp('azimuth_load240', val=240.0, units='deg'), promotes=['*'])
        self.connect('pitch_load89', ['aero_0.pitch_load','aero_120.pitch_load','aero_240.pitch_load'])
        self.connect('azimuth_load0', 'aero_0.azimuth_load')
        self.connect('azimuth_load120', 'aero_120.azimuth_load')
        self.connect('azimuth_load240', 'aero_240.azimuth_load')

        self.connect('tiploss', ['aero_0.tiploss','aero_120.tiploss','aero_240.tiploss','aero_defl_powercurve.tiploss','aero_extrm_forces.tiploss','aero_extrm.tiploss','aero_rated.tiploss'])
        self.connect('hubloss', ['aero_0.hubloss','aero_120.hubloss','aero_240.hubloss','aero_defl_powercurve.hubloss','aero_extrm_forces.hubloss','aero_extrm.hubloss','aero_rated.hubloss'])
        self.connect('wakerotation', ['aero_0.wakerotation','aero_120.wakerotation','aero_240.wakerotation','aero_defl_powercurve.wakerotation','aero_extrm_forces.wakerotation','aero_extrm.wakerotation','aero_rated.wakerotation'])
        self.connect('usecd', ['aero_0.usecd','aero_120.usecd','aero_240.usecd','aero_defl_powercurve.usecd','aero_extrm_forces.usecd','aero_extrm.usecd','aero_rated.usecd'])
        
        # connections to root moment for drivetrain
        self.connect('spline.r_str', ['root_moment_0.r_str', 'root_moment_120.r_str', 'root_moment_240.r_str'])
        self.connect('aero_rated.loads:Px', ['root_moment_0.aeroLoads:Px', 'root_moment_120.aeroLoads:Px', 'root_moment_240.aeroLoads:Px'])
        self.connect('aero_rated.loads:Py', ['root_moment_0.aeroLoads:Py', 'root_moment_120.aeroLoads:Py', 'root_moment_240.aeroLoads:Py'])
        self.connect('aero_rated.loads:Pz', ['root_moment_0.aeroLoads:Pz', 'root_moment_120.aeroLoads:Pz', 'root_moment_240.aeroLoads:Pz'])
        self.connect('aero_rated.loads:r', ['root_moment_0.aeroLoads:r', 'root_moment_120.aeroLoads:r', 'root_moment_240.aeroLoads:r'])
        self.connect('curvature.totalCone', ['root_moment_0.totalCone', 'root_moment_120.totalCone', 'root_moment_240.totalCone'])
        self.connect('curvature.x_az', ['root_moment_0.x_az','root_moment_120.x_az','root_moment_240.x_az'])
        self.connect('curvature.y_az', ['root_moment_0.y_az','root_moment_120.y_az','root_moment_240.y_az'])
        self.connect('curvature.z_az', ['root_moment_0.z_az','root_moment_120.z_az','root_moment_240.z_az'])
        self.connect('curvature.s', ['root_moment_0.s','root_moment_120.s','root_moment_240.s'])

        # connections to root Mxyz outputs
        self.connect('root_moment_0.Mxyz','Mxyz_1_in')
        self.connect('root_moment_120.Mxyz','Mxyz_2_in')
        self.connect('root_moment_240.Mxyz','Mxyz_3_in')
        self.connect('curvature.totalCone','TotalCone_in', src_indices=[nstr-1])
        self.connect('aero_0.pitch_load','Pitch_in')
        self.connect('root_moment_0.Fxyz', 'Fxyz_1_in')
        self.connect('root_moment_120.Fxyz', 'Fxyz_2_in')
        self.connect('root_moment_240.Fxyz', 'Fxyz_3_in')
        #azimuths not passed. assumed 0,120,240 in drivese function

        # Connections to constraints not accounted for by promotes=*
        self.connect('aero_rated.Omega_load', 'Omega')
        
if __name__ == '__main__':
	rotor = Problem()

	rotor.root = RotorStructure()

	#rotor.setup(check=False)
	rotor.setup()

	# === blade grid ===
	rotor['idx_cylinder_aero'] = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
	rotor['idx_cylinder_str'] = 14  # (Int): first idx in r_str_unit of non-cylindrical section
	rotor['hubFraction'] = 0.025  # (Float): hub location as fraction of radius
	# ------------------

	# === blade geometry ===
	rotor['r_max_chord'] = 0.23577  # (Float): location of max chord on unit radius
	rotor['chord_sub'] = np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
	rotor['theta_sub'] = np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
	rotor['precurve_sub'] = np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
	# rotor['delta_precurve_sub'] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
	rotor['sparT'] = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
	rotor['teT'] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
	rotor['bladeLength'] = 61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
	# rotor['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
	rotor['precone'] = 2.5  # (Float, deg): precone angle
	rotor['tilt'] = 5.0  # (Float, deg): shaft tilt
	rotor['yaw'] = 0.0  # (Float, deg): yaw error
	rotor['nBlades'] = 3  # (Int): number of blades
	# ------------------

	# === atmosphere ===
	rotor['aero_0.rho'] = 1.225  # (Float, kg/m**3): density of air
	rotor['aero_0.mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
	rotor['aero_0.shearExp'] = 0.25  # (Float): shear exponent
	rotor['hubHt'] = np.array([90.0])  # (Float, m): hub height
	rotor['turbine_class'] = TURBINE_CLASS['I']  # (Enum): IEC turbine class
	rotor['turbulence_class'] = TURBULENCE_CLASS['B']  # (Enum): IEC turbulence class class
        rotor['gust_stddev'] = 3
	# ----------------------

	# === control ===
	rotor['control:tsr'] = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
	rotor['control:pitch'] = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
	rotor['pitch_extreme'] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
	rotor['azimuth_extreme'] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
	rotor['VfactorPC'] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
	# ----------------------

	# === aero and structural analysis options ===
	rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
	rotor['nF'] = 5  # (Int): number of natural frequencies to compute
	rotor['dynamic_amplication_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
	# ----------------------


	# === fatigue ===
	rotor['rstar_damage'] = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
	    0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
	rotor['Mxb_damage'] = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
	    1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
	    1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
	rotor['Myb_damage'] = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
	    1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
	    3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
	rotor['strain_ult_spar'] = 1.0e-2  # (Float): ultimate strain in spar cap
	rotor['strain_ult_te'] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
	rotor['eta_damage'] = 1.35*1.3*1.0  # (Float): safety factor for fatigue
	rotor['m_damage'] = 10.0  # (Float): slope of S-N curve for fatigue analysis
	rotor['lifetime'] = 20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
        rotor['eta_freq'] = 1.1
	# ----------------


        # Adding in only in rotor_structure- otherwise would have been connected in larger assembly
        rotor['gust.V_hub'] = 11.7386065326
        rotor['aero_rated.Omega_load'] = 12.0
        rotor['aero_rated.pitch_load'] = rotor['control:pitch']
        
        
	# from myutilities import plt

	# === run and outputs ===
	rotor.run()

	print 'mass_one_blade =', rotor['mass_one_blade']
	print 'mass_all_blades =', rotor['mass_all_blades']
	print 'I_all_blades =', rotor['I_all_blades']
	print 'freq =', rotor['freq']
	print 'tip_deflection =', rotor['tip_deflection']
	print 'root_bending_moment =', rotor['root_bending_moment']

        #for io in rotor.root.unknowns:
        #    print(io + ' ' + str(rotor.root.unknowns[io]))



        import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(rotor['spline.r_str'], rotor['strainU_spar'], label='suction')
	plt.plot(rotor['spline.r_str'], rotor['strainL_spar'], label='pressure')
	plt.plot(rotor['spline.r_str'], rotor['eps_crit_spar'], label='critical')
	plt.ylim([-5e-3, 5e-3])
	plt.xlabel('r')
	plt.ylabel('strain')
	plt.legend()
	# plt.save('/Users/sning/Desktop/strain_spar.pdf')
	# plt.save('/Users/sning/Desktop/strain_spar.png')

	plt.figure()
	plt.plot(rotor['spline.r_str'], rotor['strainU_te'], label='suction')
	plt.plot(rotor['spline.r_str'], rotor['strainL_te'], label='pressure')
	plt.plot(rotor['spline.r_str'], rotor['eps_crit_te'], label='critical')
	plt.ylim([-5e-3, 5e-3])
	plt.xlabel('r')
	plt.ylabel('strain')
	plt.legend()
	# plt.save('/Users/sning/Desktop/strain_te.pdf')
	# plt.save('/Users/sning/Desktop/strain_te.png')

	plt.show()
	# ----------------
        f = open('deriv_structure.dat','w')
        out = rotor.check_total_derivatives(f)
        f.close()
        for k in out.keys():
            if ( (out[k]['rel error'][0] > 1e-6) and (out[k]['abs error'][0] > 1e-6) ):
                 print k
