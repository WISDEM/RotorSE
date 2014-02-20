#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

import numpy as np
import math
from openmdao.main.api import VariableTree, Component, Assembly
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Enum, Str, List

from rotoraero import SetupRun, RegulatedPowerCurve, AEP, VarSpeedMachine, RatedConditions, AeroLoads, RPM2RS
from rotoraerodefaults import CCBladeGeometry, CCBlade, CSMDrivetrain, RayleighCDF
from openmdao.lib.drivers.api import Brent
from commonse.csystem import DirectionVector
from commonse.utilities import cosd, sind, hstack, vstack, trapz_deriv, interp_with_deriv
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from akima import Akima, akima_interp_with_derivs
import _pBEAM
# from _rotor import rotor_dv as _rotor



# ---------------------
# Variable Trees
# ---------------------

class BeamProperties(VariableTree):

    z = Array(units='m', desc='locations of properties along beam')
    EA = Array(units='N', desc='axial stiffness')
    EIxx = Array(units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
    EIyy = Array(units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
    EIxy = Array(units='N*m**2', desc='coupled flap-edge stiffness')
    GJ = Array(units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
    rhoA = Array(units='kg/m', desc='mass per unit length')
    rhoJ = Array(units='kg*m', desc='polar mass moment of inertia per unit length')
    x_ec_str = Array(units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
    y_ec_str = Array(units='m', desc='y-distance to elastic center from point about which above structural properties are computed')



# class Orthotropic2DMaterial(VariableTree):

#     E1 = Float(units='N/m**2', desc='Young''s modulus in first principal direction')
#     E2 = Float(units='N/m**2', desc='Young''s modulus in second principal direction')
#     G12 = Float(units='N/m**2', desc='shear modulus')
#     nu12 = Float(desc='Poisson''s ratio (nu12*E22 = nu21*E11)')
#     rho = Float(units='kg/m**3', desc='density')


# class Sector(VariableTree):

#     num_plies = Array(dtype=np.int)
#     thickness = Array(units='m')
#     orientation = Array(units='deg')
#     materials = List(Orthotropic2DMaterial)



# ---------------------
# Base Components
# ---------------------

class BeamPropertiesBase(Component):

    properties = VarTree(BeamProperties(), iotype='out')



class StrucBase(Component):

    # all inputs/outputs in airfoil coordinate system

    # inputs
    beam = VarTree(BeamProperties(), iotype='in')

    nF = Int(iotype='in', desc='number of natural frequencies to return')

    Px_defl = Array(iotype='in')
    Py_defl = Array(iotype='in')
    Pz_defl = Array(iotype='in')

    Px_strain = Array(iotype='in')
    Py_strain = Array(iotype='in')
    Pz_strain = Array(iotype='in')

    xu_strain_spar = Array(iotype='in')
    xl_strain_spar = Array(iotype='in')
    yu_strain_spar = Array(iotype='in')
    yl_strain_spar = Array(iotype='in')
    xu_strain_te = Array(iotype='in')
    xl_strain_te = Array(iotype='in')
    yu_strain_te = Array(iotype='in')
    yl_strain_te = Array(iotype='in')

    Mx_damage = Array(iotype='in')
    My_damage = Array(iotype='in')
    strain_ult_spar = Float(0.01, iotype='in')
    strain_ult_te = Float(2500*1e-6, iotype='in')
    eta_damage = Float(1.755, iotype='in')
    m_damage = Float(10.0, iotype='in')
    N_damage = Float(365*24*3600*20.0, iotype='in')

    # outputs
    blade_mass = Float(iotype='out', desc='mass of one blades')
    blade_moment_of_inertia = Float(iotype='out', desc='out of plane moment of inertia of a blade')
    freq = Array(iotype='out')
    dx_defl = Array(iotype='out')
    dy_defl = Array(iotype='out')
    dz_defl = Array(iotype='out')
    strainU_spar = Array(iotype='out')
    strainL_spar = Array(iotype='out')
    strainU_te = Array(iotype='out')
    strainL_te = Array(iotype='out')
    damageU_spar = Array(iotype='out')
    damageL_spar = Array(iotype='out')
    damageU_te = Array(iotype='out')
    damageL_te = Array(iotype='out')





# ---------------------
# Components
# ---------------------


class ResizeCompositeSection(Component):

    upperCSIn = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for upper surface')
    lowerCSIn = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for lower surface')
    websCSIn = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for shear webs')

    chord_str_ref = Array(iotype='in', units='m')

    sector_idx_strain_spar = Array(iotype='in', dtype=np.int)
    sector_idx_strain_te = Array(iotype='in', dtype=np.int)

    chord_str = Array(iotype='in', units='m')
    sparT_str = Array(iotype='in', units='m')
    teT_str = Array(iotype='in', units='m')

    # out
    upperCSOut = List(CompositeSection, iotype='out',
        desc='list of CompositeSection objections defining the properties for upper surface')
    lowerCSOut = List(CompositeSection, iotype='out',
        desc='list of CompositeSection objections defining the properties for lower surface')
    websCSOut = List(CompositeSection, iotype='out',
        desc='list of CompositeSection objections defining the properties for shear webs')


    def execute(self):

        nstr = len(self.chord_str_ref)

        # copy data acrosss
        self.upperCSOut = []
        self.lowerCSOut = []
        self.websCSOut = []
        for i in range(nstr):
            self.upperCSOut.append(self.upperCSIn[i].mycopy())
            self.lowerCSOut.append(self.lowerCSIn[i].mycopy())
            self.websCSOut.append(self.websCSIn[i].mycopy())


        # scale all thicknesses with airfoil thickness
        for i in range(nstr):

            upper = self.upperCSOut[i]
            lower = self.lowerCSOut[i]
            webs = self.websCSOut[i]

            # factor = t_str[i]/tref[i]
            factor = self.chord_str[i]/self.chord_str_ref[i]  # same as thickness ratio for constant t/c

            for j in range(len(upper.t)):
                upper.t[j] *= factor

            for j in range(len(lower.t)):
                lower.t[j] *= factor

            for j in range(len(webs.t)):
                webs.t[j] *= factor


        # change spar and trailing edge thickness to specified values
        for i in range(nstr):

            idx_spar = self.sector_idx_strain_spar[i]
            idx_te = self.sector_idx_strain_te[i]
            upper = self.upperCSOut[i]
            lower = self.lowerCSOut[i]

            # upper and lower have same thickness for this design
            tspar = np.sum(upper.t[idx_spar])
            tte = np.sum(upper.t[idx_te])

            upper.t[idx_spar] *= self.sparT_str[i]/tspar
            lower.t[idx_spar] *= self.sparT_str[i]/tspar

            upper.t[idx_te] *= self.teT_str[i]/tte
            lower.t[idx_te] *= self.teT_str[i]/tte


class PreCompSections(BeamPropertiesBase):

    r = Array(iotype='in', units='m', desc='radial positions. r[0] should be the hub location \
        while r[-1] should be the blade tip. Any number \
        of locations can be specified between these in ascending order.')
    chord = Array(iotype='in', units='m', desc='array of chord lengths at corresponding radial positions')
    theta = Array(iotype='in', units='deg', desc='array of twist angles at corresponding radial positions. \
        (positive twist decreases angle of attack)')
    leLoc = Array(iotype='in', desc='array of leading-edge positions from a reference blade axis \
        (usually blade pitch axis). locations are normalized by the local chord length.  \
        e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
        positive in -x direction for airfoil-aligned coordinate system')
    profile = List(Profile, iotype='in', desc='airfoil shape at each radial position')
    materials = List(Orthotropic2DMaterial, iotype='in',
        desc='list of all Orthotropic2DMaterial objects used in defining the geometry')
    upperCS = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for upper surface')
    lowerCS = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for lower surface')
    websCS = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for shear webs')

    sector_idx_strain_spar = Array(iotype='in', dtype=np.int)
    sector_idx_strain_te = Array(iotype='in', dtype=np.int)


    eps_crit_spar = Array(iotype='out')
    eps_crit_te = Array(iotype='out')

    xu_strain_spar = Array(iotype='out')
    xl_strain_spar = Array(iotype='out')
    yu_strain_spar = Array(iotype='out')
    yl_strain_spar = Array(iotype='out')
    xu_strain_te = Array(iotype='out')
    xl_strain_te = Array(iotype='out')
    yu_strain_te = Array(iotype='out')
    yl_strain_te = Array(iotype='out')


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




    def execute(self):


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

        self.eps_crit_spar = self.panelBucklingStrain(self.sector_idx_strain_spar)
        self.eps_crit_te = self.panelBucklingStrain(self.sector_idx_strain_te)

        self.xu_strain_spar, self.xl_strain_spar, self.yu_strain_spar, \
            self.yl_strain_spar = self.criticalStrainLocations(self.sector_idx_strain_spar, x_ec_nose, y_ec_nose)
        self.xu_strain_te, self.xl_strain_te, self.yu_strain_te, \
            self.yl_strain_te = self.criticalStrainLocations(self.sector_idx_strain_te, x_ec_nose, y_ec_nose)




class RotorWithpBEAM(StrucBase):


    def principalCS(self, beam):

        # rename (with swap of x, y for profile c.s.)
        EIxx = beam.EIyy
        EIyy = beam.EIxx
        x_ec_str = beam.y_ec_str
        y_ec_str = beam.x_ec_str
        EA = beam.EA
        EIxy = beam.EIxy

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




    def execute(self):

        beam = self.beam
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
        p_loads = _pBEAM.Loads(nsec, self.Px_defl, self.Py_defl, self.Pz_defl)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        self.dx_defl, self.dy_defl, self.dz_defl, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()


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






class GridSetup(Component):
    """preprocessing step.  inputs and outputs should not change during optimization"""

    # should be constant
    initial_aero_grid = Array(iotype='in')
    initial_str_grid = Array(iotype='in')

    # outputs are also constant during optimization
    fraction = Array(iotype='out')
    idxj = Array(iotype='out', dtype=np.int)

    def execute(self):

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

    # variables
    r_aero = Array(iotype='in')

    # parameters
    fraction = Array(iotype='in')
    idxj = Array(iotype='in', dtype=np.int)

    # outputs
    r_str = Array(iotype='out')


    missing_deriv_policy = 'assume_zero'


    def execute(self):

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


    def provideJ(self):

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

    # variables
    r_aero_unit = Array(iotype='in', desc='locations where airfoils are defined on unit radius')
    r_str_unit = Array(np.array([0]), iotype='in', desc='locations where airfoils are defined on unit radius')
    r_max_chord = Float(iotype='in')
    chord_sub = Array(iotype='in', units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
    theta_sub = Array(iotype='in', units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
    bladeLength = Float(iotype='in', units='m')
    sparT = Array(iotype='in', units='m')  # first is multiplier, then thickness values
    teT = Array(iotype='in', units='m')  # first is multiplier, then thickness values

    # parameters
    idx_cylinder_aero = Int(iotype='in', desc='first idx in r_aero_unit of non-cylindrical section')  # constant twist inboard of here
    idx_cylinder_str = Int(iotype='in', desc='first idx in r_str_unit of non-cylindrical section')
    hubFraction = Float(iotype='in')

    # out
    Rhub = Float(iotype='out', units='m')
    Rtip = Float(iotype='out', units='m')
    r_aero = Array(iotype='out', units='m')
    r_str = Array(iotype='out', units='m')
    chord_aero = Array(iotype='out', units='m', desc='chord at airfoil locations')
    chord_str = Array(iotype='out', units='m', desc='chord at airfoil locations')
    theta_aero = Array(iotype='out', units='deg', desc='twist at airfoil locations')
    theta_str = Array(iotype='out', units='deg', desc='twist at airfoil locations')
    sparT_str = Array(iotype='out', units='m')
    teT_str = Array(iotype='out', units='m')


    def execute(self):

        # Rtip = self.bladeLength
        # Rhub = self.hubFraction * Rtip
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

        # ----- TODO: the below is not generalized at all... -----------
        # setup sparT parameterization
        nt = len(self.sparT) - 1
        rt = np.linspace(r_cylinder, Rtip, nt)
        sparT_spline = Akima(rt, self.sparT[1:])

        self.sparT_cylinder = np.array([0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05457, 0.03859, 0.03812, 0.03906, 0.04799, 0.05363, 0.05833])
        self.teT_cylinder = np.array([0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05739, 0.05457, 0.03859, 0.05765, 0.05765, 0.04731, 0.04167])

        sparT_str_in = self.sparT[0]*self.sparT_cylinder
        sparT_str_out, _, _, _ = sparT_spline.interp(self.r_str[idxc_str:])
        self.sparT_str = np.concatenate([sparT_str_in, [sparT_str_out[0]], sparT_str_out])

        # trailing edge thickness
        teT_str_in = self.teT[0]*self.teT_cylinder
        self.teT_str = np.concatenate((teT_str_in, self.teT[1]*np.ones(9), self.teT[2]*np.ones(7), self.teT[3]*np.ones(8), self.teT[4]*np.ones(2)))


    # def list_deriv_vars(self):
    #     pass
    #     # naero = len(self.r_aero_unit)
    #     # nstr = len(self.r_str_unit)
    #     # ncs = len(self.chord_sub)
    #     # nts = len(self.theta_sub)
    #     # nst = len(self.sparT)
    #     # ntt = len(self.teT)

    #     # n = naero + nstr + ncs + nts + nst + ntt + 2

    #     # dRtip = np.zeros(n)
    #     # dRhub = np.zeros(n)
    #     # dRtip[naero + nstr + 1 + ncs + nts] = 1.0
    #     # dRhub[naero + nstr + 1 + ncs + nts] = self.hubFraction

    #     # draero = np.zeros((naero, n))
    #     # draero[:, naero + nstr + 1 + ncs + nts] = (1.0 - self.r_aero_unit)*self.hubFraction + self.r_aero_unit
    #     # draero[:, :naero] = Rtip-Rhub

    #     # drstr = np.zeros((nstr, n))
    #     # drstr[:, naero + nstr + 1 + ncs + nts] = (1.0 - self.r_str_unit)*self.hubFraction + self.r_str_unit
    #     # drstr[:, naero:nstr] = Rtip-Rhub

    #     # TODO: do with Tapenade




    #     # inputs = ('r_aero_unit', 'r_str_unit', 'r_max_chord', 'chord_sub', 'theta_sub', 'bladeLength', 'sparT', 'teT')
    #     # outputs = ('Rhub', 'Rtip', 'r_aero', 'r_str', 'chord_aero', 'chord_str',
    #     #     'theta_aero', 'theta_str', 'sparT_str', 'teT_str')

    #     # return inputs, outputs

    # def provideJ(self):
    #     pass


    #     # J =

    #     # return J



# # TODO: move to rotoraero
# # also why not use bem.f90
# class BladeCurvature(Component):

#     r = Array(iotype='in')
#     precurve = Array(iotype='in')
#     presweep = Array(iotype='in')
#     precone = Float(iotype='in')

#     totalCone = Array(iotype='out')
#     x_az = Array(iotype='out')
#     y_az = Array(iotype='out')
#     z_az = Array(iotype='out')
#     s = Array(iotype='out')

#     def execute(self):

#         n = len(self.r)
#         precone = self.precone


#         # azimuthal position
#         self.x_az = -self.r*sind(precone) + self.precurve*cosd(precone)
#         self.z_az = self.r*cosd(precone) + self.precurve*sind(precone)
#         self.y_az = self.presweep


#         # total precone angle
#         x = self.precurve  # compute without precone and add in rotation after
#         z = self.r
#         y = self.presweep

#         totalCone = np.zeros(n)
#         totalCone[0] = math.atan2(-(x[1] - x[0]), z[1] - z[0])
#         totalCone[1:n-1] = 0.5*(np.arctan2(-(x[1:n-1] - x[:n-2]), z[1:n-1] - z[:n-2])
#             + np.arctan2(-(x[2:] - x[1:n-1]), z[2:] - z[1:n-1]))
#         totalCone[n-1] = math.atan2(-(x[n-1] - x[n-2]), z[n-1] - z[n-2])

#         self.totalCone = precone + np.degrees(totalCone)


#         # total path length of blade  (TODO: need to do something with this.  This should be a geometry preprocessing step just like rotoraero)
#         s = np.zeros(n)
#         s[0] = self.r[0]
#         for i in range(1, n):
#             s[i] = s[i-1] + math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2 + (z[i] - z[i-1])**2)

#         self.s = s


class DamageLoads(Component):

    rstar = Array(iotype='in')
    Mxb = Array(iotype='in')
    Myb = Array(iotype='in')
    theta = Array(iotype='in')
    r = Array(iotype='in')

    Mxa = Array(iotype='out')
    Mya = Array(iotype='out')

    def execute(self):

        rstar = (self.r-self.r[0])/(self.r[-1]-self.r[0])

        Mxb = np.interp(rstar, self.rstar, self.Mxb)
        Myb = np.interp(rstar, self.rstar, self.Myb)

        Ma = DirectionVector(Mxb, Myb, 0.0).bladeToAirfoil(self.theta)
        self.Mxa = Ma.x
        self.Mya = Ma.y



class TotalLoads(Component):

    # variables
    aeroLoads = VarTree(AeroLoads(), iotype='in')  # aerodynamic loads in blade c.s.
    r = Array(iotype='in')
    theta = Array(iotype='in')
    tilt = Float(iotype='in')
    precone = Float(iotype='in')
    # totalCone = Array(iotype='in')
    # z_az = Array(iotype='in')
    rhoA = Array(iotype='in')

    # parameters
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')

    # outputs
    Px_af = Array(iotype='out')  # total loads in af c.s.
    Py_af = Array(iotype='out')
    Pz_af = Array(iotype='out')

    missing_deriv_policy = 'assume_zero'

    def execute(self):

        totalCone = self.precone
        z_az = self.r*cosd(self.precone)

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
            'aeroLoads.pitch', 'aeroLoads.azimuth', 'r', 'theta', 'tilt', 'precone', 'rhoA')
        outputs = ('Px_af', 'Py_af', 'Pz_af')

        return inputs, outputs


    def provideJ(self):

        dPwx, dPwy, dPwz = self.P_w.dx, self.P_w.dy, self.P_w.dz
        dPcx, dPcy, dPcz = self.P_c.dx, self.P_c.dy, self.P_c.dz
        dPx, dPy, dPz = self.P.dx, self.P.dy, self.P.dz
        Omega = self.aeroLoads.Omega*RPM2RS
        z_az = self.r*cosd(self.precone)


        dPx_dOmega = dPcx['dz']*self.rhoA*z_az*2*Omega*RPM2RS
        dPy_dOmega = dPcy['dz']*self.rhoA*z_az*2*Omega*RPM2RS
        dPz_dOmega = dPcz['dz']*self.rhoA*z_az*2*Omega*RPM2RS

        dPx_dr = np.diag(self.dPax_dr + dPcx['dz']*self.rhoA*Omega**2*cosd(self.precone))
        dPy_dr = np.diag(self.dPay_dr + dPcy['dz']*self.rhoA*Omega**2*cosd(self.precone))
        dPz_dr = np.diag(self.dPaz_dr + dPcz['dz']*self.rhoA*Omega**2*cosd(self.precone))

        dPx_dprecone = dPwx['dprecone'] + dPcx['dprecone'] - dPcx['dz']*self.rhoA*Omega**2*self.r*sind(self.precone)*math.pi/180.0
        dPy_dprecone = dPwy['dprecone'] + dPcy['dprecone'] - dPcy['dz']*self.rhoA*Omega**2*self.r*sind(self.precone)*math.pi/180.0
        dPz_dprecone = dPwz['dprecone'] + dPcz['dprecone'] - dPcz['dz']*self.rhoA*Omega**2*self.r*sind(self.precone)*math.pi/180.0

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

        dPx = hstack([dPxaf_daeror, dPxaf_dPxaero, dPxaf_dPyaero, dPxaf_dPzaero, dPxaf_dOmega, dPxaf_dpitch, dPxaf_dazimuth,
            dPxaf_dr, dPxaf_dtheta, dPxaf_dtilt, dPxaf_dprecone, dPxaf_drhoA])
        dPy = hstack([dPyaf_daeror, dPyaf_dPxaero, dPyaf_dPyaero, dPyaf_dPzaero, dPyaf_dOmega, dPyaf_dpitch, dPyaf_dazimuth,
            dPyaf_dr, dPyaf_dtheta, dPyaf_dtilt, dPyaf_dprecone, dPyaf_drhoA])
        dPz = hstack([dPzaf_daeror, dPzaf_dPxaero, dPzaf_dPyaero, dPzaf_dPzaero, dPzaf_dOmega, dPzaf_dpitch, dPzaf_dazimuth,
            dPzaf_dr, dPzaf_dtheta, dPzaf_dtilt, dPzaf_dprecone, dPzaf_drhoA])

        J = vstack([dPx, dPy, dPz])

        return J





class TipDeflection(Component):

    # variables
    dx = Float(iotype='in')  # deflection at tip in airfoil c.s.
    dy = Float(iotype='in')
    dz = Float(iotype='in')
    theta = Float(iotype='in')
    pitch = Float(iotype='in')
    azimuth = Float(iotype='in')
    tilt = Float(iotype='in')
    precone = Float(iotype='in')

    # parameters
    dynamicFactor = Float(1.2, iotype='in')

    # outputs
    tip_deflection = Float(iotype='out')


    def execute(self):

        theta = self.theta + self.pitch

        dr = DirectionVector(self.dx, self.dy, self.dz)
        self.delta = dr.airfoilToBlade(theta).bladeToAzimuth(self.precone) \
            .azimuthToHub(self.azimuth).hubToYaw(self.tilt)

        self.tip_deflection = self.dynamicFactor * self.delta.x


    def list_deriv_vars(self):

        inputs = ('dx', 'dy', 'dz', 'theta', 'pitch', 'azimuth', 'tilt', 'precone')
        outputs = ('tip_deflection',)

        return inputs, outputs


    def provideJ(self):

        dx = self.dynamicFactor * self.delta.dx['dx']
        dy = self.dynamicFactor * self.delta.dx['dy']
        dz = self.dynamicFactor * self.delta.dx['dz']
        dtheta = self.dynamicFactor * self.delta.dx['dtheta']
        dpitch = self.dynamicFactor * self.delta.dx['dtheta']
        dazimuth = self.dynamicFactor * self.delta.dx['dazimuth']
        dtilt = self.dynamicFactor * self.delta.dx['dtilt']
        dprecone = self.dynamicFactor * self.delta.dx['dprecone']

        J = np.array([[dx, dy, dz, dtheta, dpitch, dazimuth, dtilt, dprecone]])

        return J




class RootMoment(Component):
    """docstring for RootMoment"""

    r_str = Array(iotype='in')
    aeroLoads = VarTree(AeroLoads(), iotype='in')  # aerodynamic loads in blade c.s.
    precone = Float(iotype='in')
    # totalCone = Array(iotype='in')
    # x_az = Array(iotype='in')
    # y_az = Array(iotype='in')
    # z_az = Array(iotype='in')
    # s = Array(iotype='in')

    root_bending_moment = Float(iotype='out')

    missing_deriv_policy = 'assume_zero'

    def execute(self):

        r = self.r_str
        x_az = -r*sind(self.precone)
        z_az = r*cosd(self.precone)
        y_az = np.zeros_like(r)


        aL = self.aeroLoads
        # Px = np.interp(r, aL.r, aL.Px)
        # Py = np.interp(r, aL.r, aL.Py)
        # Pz = np.interp(r, aL.r, aL.Pz)
        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        Px, self.dPx_dr, self.dPx_dalr, self.dPx_dalPx = interp_with_deriv(r, aL.r, aL.Px)
        Py, self.dPy_dr, self.dPy_dalr, self.dPy_dalPy = interp_with_deriv(r, aL.r, aL.Py)
        Pz, self.dPz_dr, self.dPz_dalr, self.dPz_dalPz = interp_with_deriv(r, aL.r, aL.Pz)

        # loads in azimuthal c.s.
        P = DirectionVector(Px, Py, Pz).bladeToAzimuth(self.precone)

        # distributed bending load in azimuth coordinate ysstem
        az = DirectionVector(x_az, y_az, z_az)
        Mp = az.cross(P)

        # integrate
        Mx = np.trapz(Mp.x, r)
        My = np.trapz(Mp.y, r)
        Mz = np.trapz(Mp.z, r)

        # get total magnitude
        self.root_bending_moment = math.sqrt(Mx**2 + My**2 + Mz**2)



        # self.root_bending_moment, self.J = _rotor.rootmoment_with_derivs(r, aL.r, aL.Px, aL.Py, aL.Pz, self.precone)




        self.P = P
        self.az = az
        self.Mp = Mp
        self.r = r
        self.Mx = Mx
        self.My = My
        self.Mz = Mz


    def list_deriv_vars(self):

        inputs = ('r_str', 'aeroLoads.r', 'aeroLoads.Px', 'aeroLoads.Py', 'aeroLoads.Pz', 'precone')
        outputs = ('root_bending_moment',)

        return inputs, outputs


    def provideJ(self):

        dx_dr = -sind(self.precone)
        dz_dr = cosd(self.precone)

        dx_dprecone = -self.r*cosd(self.precone)*math.pi/180.0
        dz_dprecone = -self.r*sind(self.precone)*math.pi/180.0

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


        dazx_dr = np.diag(self.az.dx['dx']*dx_dr + self.az.dx['dz']*dz_dr)
        dazy_dr = np.diag(self.az.dy['dx']*dx_dr + self.az.dy['dz']*dz_dr)
        dazz_dr = np.diag(self.az.dz['dx']*dx_dr + self.az.dz['dz']*dz_dr)

        dazx_dprecone = (self.az.dx['dx']*dx_dprecone.T + self.az.dx['dz']*dz_dprecone.T).T
        dazy_dprecone = (self.az.dy['dx']*dx_dprecone.T + self.az.dy['dz']*dz_dprecone.T).T
        dazz_dprecone = (self.az.dz['dx']*dx_dprecone.T + self.az.dz['dz']*dz_dprecone.T).T

        dMpx, dMpy, dMpz = self.az.cross_deriv_array(self.P, namea='az', nameb='P')

        dMpx_dr = (dMpx['dPx']*dPx_dr.T + dMpx['dPy']*dPy_dr.T + dMpx['dPz']*dPz_dr.T
            + dMpx['dazx']*dazx_dr.T + dMpx['dazy']*dazy_dr.T + dMpx['dazz']*dazz_dr.T).T
        dMpy_dr = (dMpy['dPx']*dPx_dr.T + dMpy['dPy']*dPy_dr.T + dMpy['dPz']*dPz_dr.T
            + dMpy['dazx']*dazx_dr.T + dMpy['dazy']*dazy_dr.T + dMpy['dazz']*dazz_dr.T).T
        dMpz_dr = (dMpz['dPx']*dPx_dr.T + dMpz['dPy']*dPy_dr.T + dMpz['dPz']*dPz_dr.T
            + dMpz['dazx']*dazx_dr.T + dMpz['dazy']*dazy_dr.T + dMpz['dazz']*dazz_dr.T).T

        dMpx_dprecone = (dMpx['dPx']*self.P.dx['dprecone'].T + dMpx['dPy']*self.P.dy['dprecone'].T + dMpx['dPz']*self.P.dz['dprecone'].T
            + dMpx['dazx']*dazx_dprecone.T + dMpx['dazy']*dazy_dprecone.T + dMpx['dazz']*dazz_dprecone.T).T
        dMpy_dprecone = (dMpy['dPx']*self.P.dx['dprecone'].T + dMpy['dPy']*self.P.dy['dprecone'].T + dMpy['dPz']*self.P.dz['dprecone'].T
            + dMpy['dazx']*dazx_dprecone.T + dMpy['dazy']*dazy_dprecone.T + dMpy['dazz']*dazz_dprecone.T).T
        dMpz_dprecone = (dMpz['dPx']*self.P.dx['dprecone'].T + dMpz['dPy']*self.P.dy['dprecone'].T + dMpz['dPz']*self.P.dz['dprecone'].T
            + dMpz['dazx']*dazx_dprecone.T + dMpz['dazy']*dazy_dprecone.T + dMpz['dazz']*dazz_dprecone.T).T

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

        dMx_dMpx, dMx_dr = trapz_deriv(self.Mp.x, self.r)
        dMy_dMpy, dMy_dr = trapz_deriv(self.Mp.y, self.r)
        dMz_dMpz, dMz_dr = trapz_deriv(self.Mp.z, self.r)

        dMx_dr = dMx_dr + np.dot(dMx_dMpx, dMpx_dr)
        dMy_dr = dMy_dr + np.dot(dMy_dMpy, dMpy_dr)
        dMz_dr = dMz_dr + np.dot(dMz_dMpz, dMpz_dr)

        dMx_dprecone = np.dot(dMx_dMpx, dMpx_dprecone)
        dMy_dprecone = np.dot(dMy_dMpy, dMpy_dprecone)
        dMz_dprecone = np.dot(dMz_dMpz, dMpz_dprecone)

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

        drbm_dr = (self.Mx*dMx_dr + self.My*dMy_dr + self.Mz*dMz_dr)/self.root_bending_moment
        drbm_dprecone = (self.Mx*dMx_dprecone + self.My*dMy_dprecone + self.Mz*dMz_dprecone)/self.root_bending_moment
        drbm_dalr = (self.Mx*dMx_dalr + self.My*dMy_dalr + self.Mz*dMz_dalr)/self.root_bending_moment
        drbm_dalPx = (self.Mx*dMx_dalPx + self.My*dMy_dalPx + self.Mz*dMz_dalPx)/self.root_bending_moment
        drbm_dalPy = (self.Mx*dMx_dalPy + self.My*dMy_dalPy + self.Mz*dMz_dalPy)/self.root_bending_moment
        drbm_dalPz = (self.Mx*dMx_dalPz + self.My*dMy_dalPz + self.Mz*dMz_dalPz)/self.root_bending_moment

        J = np.array([np.concatenate([drbm_dr, drbm_dalr, drbm_dalPx, drbm_dalPy, drbm_dalPz, [drbm_dprecone]])])

        return J


        # return self.J



class MassProperties(Component):

    # variables
    blade_mass = Float(iotype='in')
    blade_moment_of_inertia = Float(iotype='in')
    tilt = Float(iotype='in')

    # parameters
    nBlades = Int(iotype='in')

    # outputs
    mass_all_blades = Float(iotype='out')
    I_all_blades = Array(iotype='out')

    def execute(self):

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

    def provideJ(self):
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

    # parameters
    turbine_class = Enum('I', ('I', 'II', 'III'), iotype='in')

    # outputs should be constant
    V_mean = Float(iotype='out', units='m/s')
    V_extreme = Float(iotype='out', units='m/s')

    def execute(self):
        if self.turbine_class == 'I':
            Vref = 50.0
        elif self.turbine_class == 'II':
            Vref = 42.5
        elif self.turbine_class == 'III':
            Vref = 37.5

        self.V_mean = 0.2*Vref
        self.V_extreme = 1.4*Vref



class ExtremeLoads(Component):

    # variables
    T = Array(np.zeros(2), iotype='in', units='N', shape=((2,)), desc='index 0 is at worst-case, index 1 feathered')
    Q = Array(np.zeros(2), iotype='in', units='N*m', shape=((2,)), desc='index 0 is at worst-case, index 1 feathered')

    # parameters
    nBlades = Int(iotype='in')

    # outputs
    T_extreme = Float(iotype='out', units='N')
    Q_extreme = Float(iotype='out', units='N*m')


    def execute(self):
        n = float(self.nBlades)
        self.T_extreme = (self.T[0] + self.T[1]*(n-1)) / n
        self.Q_extreme = (self.Q[0] + self.Q[1]*(n-1)) / n


    def list_deriv_vars(self):

        inputs = ('T', 'Q')
        outputs = ('T_extreme', 'Q_extreme')

        return inputs, outputs


    def provideJ(self):
        n = float(self.nBlades)

        J = np.array([[1.0/n, (n-1)/n, 0.0, 0.0],
                      [0.0, 0.0, 1.0/n, (n-1)/n]])

        return J




class GustETM(Component):

    # variables
    V_mean = Float(iotype='in', units='m/s')
    V_hub = Float(iotype='in', units='m/s')

    # parameters
    turbulence_class = Enum('B', ('A', 'B', 'C'), iotype='in')
    std = Int(3, iotype='in')

    # out
    V_gust = Float(iotype='out', units='m/s')


    def execute(self):

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


    def provideJ(self):
        Iref = self.Iref
        c = self.c

        J = np.array([[self.std*(c*Iref*0.072/c*(self.V_hub/c - 4)),
            1.0 + self.std*(c*Iref*0.072*(self.V_mean/c + 3)/c)]])

        return J



# class RotorVS(RotorAeroVS):

#     # replace
#     beam = Slot(BeamPropertiesBase)
#     struc = Slot(StrucBase)

#     # inputs
#     nBlades = Int(iotype='in')
#     theta = Array(iotype='in')
#     tilt = Float(iotype='in')
#     precone = Float(iotype='in')
#     precurve = Array(iotype='in')
#     presweep = Array(iotype='in')
#     g = Float(9.81, iotype='in', units='m/s**2')
#     nF = Int(5, iotype='in')

#     V_extreme = Float(iotype='in')  # TODO: can combine with Ubar based on machine class
#     pitch_extreme = Float(iotype='in')
#     azimuth_extreme = Float(iotype='in')

#     # outputs
#     mass_one_blade = Float(iotype='out', units='kg', desc='mass of one blade')
#     mass_all_blades = Float(iotype='out', units='kg', desc='mass of all blade')
#     I_all_blades = Array(iotype='out', desc='out of plane moments of inertia in yaw-aligned c.s.')
#     freq = Array(iotype='out', units='Hz', desc='1st nF natural frequencies')
#     tip_deflection = Float(iotype='out', units='m', desc='blade tip deflection in +x_y direction')
#     strain = Array(iotype='out', desc='axial strain and specified locations')
#     strain = Array(iotype='out', desc='axial strain and specified locations')
#     root_bending_moment = Float(iotype='out', units='N*m')

#     def configure(self):
#         super(RotorVS, self).configure()

#         self.add('aero_rated', AeroBase())
#         self.add('aero_extrm', AeroBase())
#         self.add('beam', BeamPropertiesBase())
#         self.add('curve', BladeCurvature())
#         self.add('loads_defl', TotalLoads())
#         self.add('loads_strain', TotalLoads())
#         self.add('struc', StrucBase())
#         self.add('tip', TipDeflection())
#         self.add('root_moment', RootMoment())
#         self.add('mass', MassProperties())


#         self.driver.workflow.add(['aero_rated', 'aero_extrm', 'beam', 'curve',
#             'loads_defl', 'loads_strain', 'struc', 'tip', 'root_moment', 'mass'])

#         # connections to aero_rated (for max deflection)
#         self.connect('powercurve.ratedConditions.V', 'aero_rated.V_load')  # add turbulent fluctuation (3*sigma)
#         self.connect('powercurve.ratedConditions.Omega', 'aero_rated.Omega_load')
#         self.connect('powercurve.ratedConditions.pitch', 'aero_rated.pitch_load')
#         self.aero_rated.azimuth_load = 180.0  # closest to tower
#         self.aero_rated.run_case = 'loads'

#         # connections to aero_extrm (for max strain)
#         self.connect('V_extreme', 'aero_extrm.V_load')
#         self.connect('pitch_extreme', 'aero_extrm.pitch_load')
#         self.connect('azimuth_extreme', 'aero_extrm.azimuth_load')
#         self.aero_extrm.Omega_load = 0.0  # parked case
#         self.aero_extrm.run_case = 'loads'

#         # connections to curve
#         self.connect('beam.properties.z', 'curve.r')
#         self.connect('precurve', 'curve.precurve')
#         self.connect('presweep', 'curve.presweep')
#         self.connect('precone', 'curve.precone')

#         # connections to loads_defl
#         self.connect('aero_rated.loads', 'loads_defl.aeroLoads')
#         self.connect('beam.properties.z', 'loads_defl.r')
#         self.connect('theta', 'loads_defl.theta')
#         self.connect('tilt', 'loads_defl.tilt')
#         self.connect('curve.totalCone', 'loads_defl.totalCone')
#         self.connect('curve.z_az', 'loads_defl.z_az')
#         self.connect('beam.properties.rhoA', 'loads_defl.rhoA')
#         self.connect('g', 'loads_defl.g')

#         # connections to loads_strain
#         self.connect('aero_extrm.loads', 'loads_strain.aeroLoads')
#         self.connect('beam.properties.z', 'loads_strain.r')
#         self.connect('theta', 'loads_strain.theta')
#         self.connect('tilt', 'loads_strain.tilt')
#         self.connect('curve.totalCone', 'loads_strain.totalCone')
#         self.connect('curve.z_az', 'loads_strain.z_az')
#         self.connect('beam.properties.rhoA', 'loads_strain.rhoA')
#         self.connect('g', 'loads_strain.g')


#         # connections to struc
#         self.connect('beam.properties', 'struc.beam')
#         self.connect('nF', 'struc.nF')
#         self.connect('loads_defl.Px_af', 'struc.Px_defl')
#         self.connect('loads_defl.Py_af', 'struc.Py_defl')
#         self.connect('loads_defl.Pz_af', 'struc.Pz_defl')
#         self.connect('loads_strain.Px_af', 'struc.Px_strain')
#         self.connect('loads_strain.Py_af', 'struc.Py_strain')
#         self.connect('loads_strain.Pz_af', 'struc.Pz_strain')

#         # connections to tip
#         self.connect('struc.dx_defl', 'tip.dx')
#         self.connect('struc.dy_defl', 'tip.dy')
#         self.connect('struc.dz_defl', 'tip.dz')
#         self.connect('theta', 'tip.theta')
#         self.connect('aero_rated.loads.pitch', 'tip.pitch')
#         self.connect('aero_rated.loads.azimuth', 'tip.azimuth')
#         self.connect('tilt', 'tip.tilt')
#         self.connect('curve.totalCone', 'tip.totalCone')

#         # connections to root moment
#         self.connect('beam.properties.z', 'root_moment.r_str')
#         self.connect('aero_rated.loads', 'root_moment.aeroLoads')
#         self.connect('curve.totalCone', 'root_moment.totalCone')
#         self.connect('curve.x_az', 'root_moment.x_az')
#         self.connect('curve.y_az', 'root_moment.y_az')
#         self.connect('curve.z_az', 'root_moment.z_az')
#         self.connect('curve.s', 'root_moment.s')

#         # connections to mass
#         self.connect('struc.blade_mass', 'mass.blade_mass')
#         self.connect('struc.blade_moment_of_inertia', 'mass.blade_moment_of_inertia')
#         self.connect('nBlades', 'mass.nBlades')
#         self.connect('tilt', 'mass.tilt')


#         # input passthroughs
#         self.create_passthrough('struc.x_strain')
#         self.create_passthrough('struc.y_strain')
#         self.create_passthrough('struc.z_strain')

#         # connect to outputs
#         self.connect('struc.blade_mass', 'mass_one_blade')
#         self.connect('mass.mass_all_blades', 'mass_all_blades')
#         self.connect('mass.I_all_blades', 'I_all_blades')
#         self.connect('struc.freq', 'freq')
#         self.connect('tip.tip_deflection', 'tip_deflection')
#         self.connect('struc.strain', 'strain')
#         self.connect('root_moment.root_bending_moment', 'root_bending_moment')









class RotorTS(Assembly):
    """rotor for tip-speed study"""

    # --- geometry inputs ---
    initial_aero_grid = Array(iotype='in')
    initial_str_grid = Array(iotype='in')
    idx_cylinder_aero = Int(iotype='in', desc='first idx in r_aero_unit of non-cylindrical section')  # constant twist inboard of here
    idx_cylinder_str = Int(iotype='in', desc='first idx in r_str_unit of non-cylindrical section')
    hubFraction = Float(iotype='in')
    r_aero = Array(iotype='in')
    r_max_chord = Float(iotype='in')
    chord_sub = Array(iotype='in', units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
    theta_sub = Array(iotype='in', units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
    bladeLength = Float(iotype='in', units='m', deriv_ignore=True)
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg', deriv_ignore=True)
    tilt = Float(0.0, iotype='in', desc='shaft tilt', units='deg', deriv_ignore=True)
    yaw = Float(0.0, iotype='in', desc='yaw error', units='deg', deriv_ignore=True)
    nBlades = Int(3, iotype='in', desc='number of blades', deriv_ignore=True)
    airfoil_files = List(Str, iotype='in', desc='names of airfoil file', deriv_ignore=True)

    # --- atmosphere inputs ---
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air', deriv_ignore=True)
    mu = Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air', deriv_ignore=True)
    shearExp = Float(0.2, iotype='in', desc='shear exponent', deriv_ignore=True)
    hubHt = Float(iotype='in', units='m')
    turbine_class = Enum('I', ('I', 'II', 'III'), iotype='in')
    turbulence_class = Enum('B', ('A', 'B', 'C'), iotype='in')
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity', deriv_ignore=True)

    # --- composite sections ---
    sparT = Array(iotype='in', units='m')  # first is multiplier, then thickness values
    teT = Array(iotype='in', units='m')  # first is multiplier, then thickness values
    chord_str_ref = Array(iotype='in', units='m')
    leLoc = Array(iotype='in', desc='array of leading-edge positions from a reference blade axis \
        (usually blade pitch axis). locations are normalized by the local chord length.  \
        e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
        positive in -x direction for airfoil-aligned coordinate system')
    profile = List(Profile, iotype='in', desc='airfoil shape at each radial position')
    materials = List(Orthotropic2DMaterial, iotype='in',
        desc='list of all Orthotropic2DMaterial objects used in defining the geometry')
    upperCS = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for upper surface')
    lowerCS = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for lower surface')
    websCS = List(CompositeSection, iotype='in',
        desc='list of CompositeSection objections defining the properties for shear webs')
    sector_idx_strain_spar = Array(iotype='in', dtype=np.int)
    sector_idx_strain_te = Array(iotype='in', dtype=np.int)


    # --- control ---
    control = VarTree(VarSpeedMachine(), iotype='in')
    # max_tip_speed = Float(iotype='in', units='m/s', desc='maximum tip speed')
    pitch_extreme = Float(iotype='in')
    azimuth_extreme = Float(iotype='in')

    # --- drivetrain efficiency ---
    drivetrainType = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')

    # --- fatigue ---
    rstar_damage = Array(iotype='in')
    Mxb_damage = Array(iotype='in')
    Myb_damage = Array(iotype='in')
    strain_ult_spar = Float(0.01, iotype='in')
    strain_ult_te = Float(2500*1e-6, iotype='in')
    eta_damage = Float(1.755, iotype='in')
    m_damage = Float(10.0, iotype='in')
    N_damage = Float(365*24*3600*20.0, iotype='in')

    # --- options ---
    nSector = Int(4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power')
    npts_coarse_power_curve = Int(20, iotype='in', desc='number of points to evaluate aero analysis at')
    npts_spline_power_curve = Int(200, iotype='in', desc='number of points to use in fitting spline to power curve')
    AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')
    dynamic_amplication_tip_deflection = Float(1.2, iotype='in')
    nF = Int(5, iotype='in', desc='number of natural frequencies to compute')

    # --- outputs ---
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')
    V = Array(iotype='out', units='m/s', desc='wind speeds (power curve)')
    P = Array(iotype='out', units='W', desc='power (power curve)')
    ratedConditions = VarTree(RatedConditions(), iotype='out')
    hub_diameter = Float(iotype='out', units='m')
    diameter = Float(iotype='out', units='m')
    V_extreme = Float(iotype='out', units='m/s')
    T_extreme = Float(iotype='out', units='N')
    Q_extreme = Float(iotype='out', units='N*m')

    # outputs
    mass_one_blade = Float(iotype='out', units='kg', desc='mass of one blade')
    mass_all_blades = Float(iotype='out', units='kg', desc='mass of all blade')
    I_all_blades = Array(iotype='out', desc='out of plane moments of inertia in yaw-aligned c.s.')
    freq = Array(iotype='out', units='Hz', desc='1st nF natural frequencies')
    tip_deflection = Float(iotype='out', units='m', desc='blade tip deflection in +x_y direction')
    strainU_spar = Array(iotype='out', desc='axial strain and specified locations')
    strainL_spar = Array(iotype='out', desc='axial strain and specified locations')
    strainU_te = Array(iotype='out', desc='axial strain and specified locations')
    strainL_te = Array(iotype='out', desc='axial strain and specified locations')
    eps_crit_spar = Array(iotype='out')
    eps_crit_te = Array(iotype='out')
    root_bending_moment = Float(iotype='out', units='N*m')
    damageU_spar = Array(iotype='out')
    damageL_spar = Array(iotype='out')
    damageU_te = Array(iotype='out')
    damageL_te = Array(iotype='out')


    def configure(self):

        self.add('turbineclass', TurbineClass())
        self.add('gridsetup', GridSetup())
        self.add('grid', RGrid())
        self.add('spline', GeometrySpline())
        self.add('geom', CCBladeGeometry())
        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRun())
        self.add('analysis', CCBlade())
        self.add('dt', CSMDrivetrain())
        self.add('powercurve', RegulatedPowerCurve())
        self.add('brent', Brent())
        self.add('cdf', RayleighCDF())
        self.add('aep', AEP())

        self.brent.workflow.add(['powercurve'])

        self.driver.workflow.add(['turbineclass', 'gridsetup', 'grid', 'spline', 'geom',
            'setup', 'analysis', 'dt', 'brent', 'cdf', 'aep'])

        # connections to turbineclass
        self.connect('turbine_class', 'turbineclass.turbine_class')

        # connections to gridsetup
        self.connect('initial_aero_grid', 'gridsetup.initial_aero_grid')
        self.connect('initial_str_grid', 'gridsetup.initial_str_grid')

        # connections to grid
        self.connect('r_aero', 'grid.r_aero')
        self.connect('gridsetup.fraction', 'grid.fraction')
        self.connect('gridsetup.idxj', 'grid.idxj')

        # connections to spline
        self.connect('r_aero', 'spline.r_aero_unit')
        self.connect('grid.r_str', 'spline.r_str_unit')
        self.connect('r_max_chord', 'spline.r_max_chord')
        self.connect('chord_sub', 'spline.chord_sub')
        self.connect('theta_sub', 'spline.theta_sub')
        self.connect('bladeLength', 'spline.bladeLength')
        self.connect('idx_cylinder_aero', 'spline.idx_cylinder_aero')
        self.connect('idx_cylinder_str', 'spline.idx_cylinder_str')
        self.connect('hubFraction', 'spline.hubFraction')
        self.connect('sparT', 'spline.sparT')
        self.connect('teT', 'spline.teT')

        # connections to geom
        self.connect('spline.Rtip', 'geom.Rtip')
        self.connect('precone', 'geom.precone')

        # # connectiosn to tipspeed
        # self.connect('geom.R', 'tipspeed.R')
        # self.connect('max_tip_speed', 'tipspeed.Vtip_max')
        # self.connect('tipspeed.Omega_max', 'control.maxOmega')

        # connections to setup
        self.connect('control', 'setup.control')
        self.connect('geom.R', 'setup.R')
        self.connect('npts_coarse_power_curve', 'setup.npts')

        # connections to analysis
        self.connect('spline.r_aero', 'analysis.r')
        self.connect('spline.chord_aero', 'analysis.chord')
        self.connect('spline.theta_aero', 'analysis.theta')
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
        self.analysis.run_case = 'power'

        # connections to drivetrain
        self.connect('analysis.P', 'dt.aeroPower')
        self.connect('analysis.Q', 'dt.aeroTorque')
        self.connect('analysis.T', 'dt.aeroThrust')
        self.connect('control.ratedPower', 'dt.ratedPower')
        self.connect('drivetrainType', 'dt.drivetrainType')

        # connections to powercurve
        self.connect('control', 'powercurve.control')
        self.connect('setup.Uhub', 'powercurve.Vcoarse')
        self.connect('dt.power', 'powercurve.Pcoarse')
        self.connect('analysis.T', 'powercurve.Tcoarse')
        self.connect('geom.R', 'powercurve.R')
        self.connect('npts_spline_power_curve', 'powercurve.npts')

        # setup Brent method to find rated speed
        self.connect('control.Vin', 'brent.lower_bound')
        self.connect('control.Vout', 'brent.upper_bound')
        self.brent.add_parameter('powercurve.Vrated', low=-1e-15, high=1e15)
        self.brent.add_constraint('powercurve.residual = 0')

        # connections to cdf
        self.connect('powercurve.V', 'cdf.x')
        self.connect('turbineclass.V_mean', 'cdf.xbar')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('powercurve.P', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')

        # connections to outputs
        self.connect('powercurve.V', 'V')
        self.connect('powercurve.P', 'P')
        self.connect('aep.AEP', 'AEP')
        self.connect('powercurve.ratedConditions', 'ratedConditions')
        self.connect('2*spline.Rhub', 'hub_diameter')
        self.connect('2*geom.R', 'diameter')


        # --- add structures ---
        self.add('resize', ResizeCompositeSection())
        self.add('gust', GustETM())
        self.add('aero_rated', CCBlade())
        self.add('aero_extrm', CCBlade())
        self.add('aero_extrm_forces', CCBlade())
        self.add('beam', PreCompSections())
        self.add('loads_defl', TotalLoads())
        self.add('loads_strain', TotalLoads())
        self.add('damage', DamageLoads())
        self.add('struc', RotorWithpBEAM())
        self.add('tip', TipDeflection())
        self.add('root_moment', RootMoment())
        self.add('mass', MassProperties())
        self.add('extreme', ExtremeLoads())


        self.driver.workflow.add(['resize', 'gust', 'aero_rated', 'aero_extrm', 'aero_extrm_forces',
            'beam', 'loads_defl', 'loads_strain', 'damage', 'struc', 'tip', 'root_moment', 'mass', 'extreme'])


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
        self.connect('powercurve.ratedConditions.V', 'gust.V_hub')

        # connections to aero_rated (for max deflection)
        self.connect('spline.r_aero', 'aero_rated.r')
        self.connect('spline.chord_aero', 'aero_rated.chord')
        self.connect('spline.theta_aero', 'aero_rated.theta')
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
        self.connect('powercurve.ratedConditions.Omega', 'aero_rated.Omega_load')
        self.connect('powercurve.ratedConditions.pitch', 'aero_rated.pitch_load')
        self.aero_rated.azimuth_load = 180.0  # closest to tower
        self.aero_rated.run_case = 'loads'

        # connections to aero_extrm (for max strain)
        self.connect('spline.r_aero', 'aero_extrm.r')
        self.connect('spline.chord_aero', 'aero_extrm.chord')
        self.connect('spline.theta_aero', 'aero_extrm.theta')
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
        self.aero_extrm.Omega_load = 0.0  # parked case
        self.aero_extrm.run_case = 'loads'

        # connections to aero_extrm (for tower thrust)
        self.connect('spline.r_aero', 'aero_extrm_forces.r')
        self.connect('spline.chord_aero', 'aero_extrm_forces.chord')
        self.connect('spline.theta_aero', 'aero_extrm_forces.theta')
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
        self.connect('turbineclass.V_extreme', 'aero_extrm_forces.Uhub[0]')
        self.connect('turbineclass.V_extreme', 'aero_extrm_forces.Uhub[1]')
        self.connect('pitch_extreme', 'aero_extrm_forces.pitch[0]')
        self.aero_extrm_forces.pitch[1] = 90  # feathered
        self.aero_extrm_forces.run_case = 'power'
        self.aero_extrm_forces.T = np.zeros(2)
        self.aero_extrm_forces.Q = np.zeros(2)

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
        self.connect('aero_rated.loads', 'loads_defl.aeroLoads')
        self.connect('beam.properties.z', 'loads_defl.r')
        self.connect('spline.theta_str', 'loads_defl.theta')
        self.connect('tilt', 'loads_defl.tilt')
        self.connect('precone', 'loads_defl.precone')
        self.connect('beam.properties.rhoA', 'loads_defl.rhoA')
        self.connect('g', 'loads_defl.g')


        # connections to loads_strain
        self.connect('aero_extrm.loads', 'loads_strain.aeroLoads')
        self.connect('beam.properties.z', 'loads_strain.r')
        self.connect('spline.theta_str', 'loads_strain.theta')
        self.connect('tilt', 'loads_strain.tilt')
        self.connect('precone', 'loads_strain.precone')
        self.connect('beam.properties.rhoA', 'loads_strain.rhoA')
        self.connect('g', 'loads_strain.g')


        # connections to damage
        self.connect('rstar_damage', 'damage.rstar')
        self.connect('Mxb_damage', 'damage.Mxb')
        self.connect('Myb_damage', 'damage.Myb')
        self.connect('spline.theta_str', 'damage.theta')
        self.connect('beam.properties.z', 'damage.r')


        # connections to struc
        self.connect('beam.properties', 'struc.beam')
        self.connect('nF', 'struc.nF')
        self.connect('loads_defl.Px_af', 'struc.Px_defl')
        self.connect('loads_defl.Py_af', 'struc.Py_defl')
        self.connect('loads_defl.Pz_af', 'struc.Pz_defl')
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


        # rstar = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500, 0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])
        # Mxb = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003, 1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002, 1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])
        # Myb = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003, 1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002, 3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])
        # theta = np.array([13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 13.2783, 12.9342402117, 12.4835185594, 11.4807962375, 10.9555376235, 10.2141732458, 9.50474414552, 8.79808349002, 8.12523177814, 6.8138304713, 6.42067815056, 5.58414310075, 4.96394649167, 4.44089107951, 4.38490319227, 4.35830230526, 4.3314093512, 3.86273855446, 3.38640148153, 1.57771432025, 0.953398905137, 0.504982546655, 0.0995167038088, -0.0878099])

        # connections to tip
        self.struc.dx_defl = np.zeros(1)
        self.struc.dy_defl = np.zeros(1)
        self.struc.dz_defl = np.zeros(1)
        self.spline.theta_str = np.zeros(1)
        self.connect('struc.dx_defl[-1]', 'tip.dx')
        self.connect('struc.dy_defl[-1]', 'tip.dy')
        self.connect('struc.dz_defl[-1]', 'tip.dz')
        self.connect('spline.theta_str[-1]', 'tip.theta')
        self.connect('aero_rated.loads.pitch', 'tip.pitch')
        self.connect('aero_rated.loads.azimuth', 'tip.azimuth')
        self.connect('tilt', 'tip.tilt')
        self.connect('precone', 'tip.precone')
        self.connect('dynamic_amplication_tip_deflection', 'tip.dynamicFactor')


        # connections to root moment
        self.connect('spline.r_str', 'root_moment.r_str')
        self.connect('aero_rated.loads', 'root_moment.aeroLoads')
        self.connect('precone', 'root_moment.precone')


        # connections to mass
        self.connect('struc.blade_mass', 'mass.blade_mass')
        self.connect('struc.blade_moment_of_inertia', 'mass.blade_moment_of_inertia')
        self.connect('nBlades', 'mass.nBlades')
        self.connect('tilt', 'mass.tilt')

        # connectsion to extreme
        self.connect('aero_extrm_forces.T', 'extreme.T')
        self.connect('aero_extrm_forces.Q', 'extreme.Q')
        self.connect('nBlades', 'extreme.nBlades')

        # connect to outputs
        self.connect('turbineclass.V_extreme', 'V_extreme')
        self.connect('extreme.T_extreme', 'T_extreme')
        self.connect('extreme.Q_extreme', 'Q_extreme')
        self.connect('struc.blade_mass', 'mass_one_blade')
        self.connect('mass.mass_all_blades', 'mass_all_blades')
        self.connect('mass.I_all_blades', 'I_all_blades')
        self.connect('struc.freq', 'freq')
        self.connect('tip.tip_deflection', 'tip_deflection')
        self.connect('struc.strainU_spar', 'strainU_spar')
        self.connect('struc.strainL_spar', 'strainL_spar')
        self.connect('struc.strainU_te', 'strainU_te')
        self.connect('struc.strainL_te', 'strainL_te')
        self.connect('root_moment.root_bending_moment', 'root_bending_moment')
        self.connect('beam.eps_crit_spar', 'eps_crit_spar')
        self.connect('beam.eps_crit_te', 'eps_crit_te')
        self.connect('struc.damageU_spar', 'damageU_spar')
        self.connect('struc.damageL_spar', 'damageL_spar')
        self.connect('struc.damageU_te', 'damageU_te')
        self.connect('struc.damageL_te', 'damageL_te')




if __name__ == '__main__':

    import os

    rotor = RotorTS()

    # --- blade grid ---
    rotor.initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
    rotor.initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154, 0.0114340970275,
        0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455, 0.11111057,
        0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319, 0.36666667,
        0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333, 0.667358391486,
        0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724, 1.0])
    rotor.idx_cylinder_aero = 3
    rotor.idx_cylinder_str = 14
    rotor.hubFraction = 0.025

    # --- geometry -----
    rotor.hubHt = 80.0
    rotor.precone = 2.5
    rotor.tilt = -5.0
    rotor.yaw = 0.0
    rotor.nBlades = 3
    rotor.turbine_class = 'I'

    # --- atmosphere ---
    rotor.rho = 1.225
    rotor.mu = 1.81206e-5
    rotor.shearExp = 0.2

    # --- operational conditions ---
    rotor.control.Vin = 3.0
    rotor.control.Vout = 25.0
    rotor.control.ratedPower = 5e6
    rotor.control.minOmega = 0.0
    rotor.control.maxOmega = 12.0
    rotor.control.tsr = 7.55
    rotor.control.pitch = 0.0

    rotor.pitch_extreme = 0.0
    rotor.azimuth_extreme = 0.0


    # --- airfoil files ---
    basepath = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]
    rotor.airfoil_files = af

    # --- aero analysis options ---
    rotor.npts_coarse_power_curve = 20
    rotor.npts_spline_power_curve = 200
    rotor.AEP_loss_factor = 1.0
    rotor.nSector = 4
    rotor.drivetrainType = 'geared'

    # --- materials and composite layup  ---
    basepath = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(rotor.initial_str_grid)
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    rotor.leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    rotor.sector_idx_strain = [2]*ncomp
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    # # web 1
    # ib_idx = 7
    # ob_idx = 36
    # ib_webc = 0.4114
    # ob_webc = 0.1886

    # web1 = web_loc(r_str, chord_str, le_str, ib_idx, ob_idx, ib_webc, ob_webc)

    # # web 2
    # ib_idx = 7
    # ob_idx = 36
    # ib_webc = 0.5886
    # ob_webc = 0.6114

    # web2 = web_loc(r_str, chord_str, le_str, ib_idx, ob_idx, ib_webc, ob_webc)


    for i in range(ncomp):

        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))

    rotor.materials = materials
    rotor.upperCS = upper
    rotor.lowerCS = lower
    rotor.websCS = webs
    rotor.profile = profile
    # --------------------------------------


    # --- structural options ---
    rotor.g = 9.81
    rotor.nF = 5

    # --- design variables ---
    rotor.r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333, 0.5,
        0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
    rotor.r_max_chord = 0.23577
    rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]
    rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]
    rotor.bladeLength = 63.0

    # TODO: calculate these
    rotor.x_strain = np.array([1.62953277093, 1.65378760431, 1.6848244664, 1.69280026956, 1.70125331952, 1.70918845043, 1.71710285872, 1.7379700867, 1.77191347081, 1.77769022081, 1.66549686777, 1.46236108547, 1.24412860472, 0.946027322623, 0.855159937668, 0.793030786156, 0.741219389531, 0.678388044164, 0.656104367978, 0.626195492951, 0.594991331198, 0.535300387358, 0.49436378615, 0.466270531861, 0.437964716508, 0.374175959337, 0.35504758904, 0.343037547915, 0.30479534701, 0.285300267572, 0.270977493752, 0.256272902802, 0.243171569781, 0.215017502212, 0.190697407308, 0.170514014614, 0.149901126795, 0.13789004202])
    rotor.y_strain = np.array([-0.00113112893698, -0.00114796526232, -0.0172978765403, -0.0173797630875, -0.0174665494665, -0.0175480184371, -0.0176292746511, -0.0015645000705, -0.02731068479, -0.0305932407033, -0.109340525932, -0.125949563315, -0.12117411121, -0.128218415335, -0.149327182633, -0.14656631794, -0.140266481404, -0.135652296602, -0.132851488119, -0.1287750113, -0.124992337883, -0.118409965897, -0.110552832125, -0.0979009471516, -0.0937777322172, -0.084620004667, -0.0782777410193, -0.0716491412175, -0.0584114110202, -0.0591831317106, -0.0558251490105, -0.0488944452438, -0.050399265575, -0.0549454662041, -0.0623708745921, -0.0713825225821, -0.0823128686813, -0.0695992262681])
    rotor.z_strain = np.array([1.50142903976, 1.80306613137, 1.90155987557, 2.00005361977, 2.10470322299, 2.20319696719, 2.30169071139, 2.86802974054, 3.00345863882, 3.10195238302, 5.60738700113, 7.00476699698, 8.3405884027, 10.5074507751, 11.7632460137, 13.5115099732, 15.863048116, 18.5162233505, 19.9690060774, 22.0189071286, 24.0749640388, 26.12486509, 28.1747661411, 32.2807241025, 33.5303634821, 36.3866820639, 38.5350768593, 40.4864841663, 42.5425410764, 43.5397902365, 44.5924421276, 46.5438494346, 48.698400089, 52.7982021914, 56.2208598023, 58.9540612039, 61.6934184645, 63.0600191653])


    rotor.run()

    # outputs
    print 'AEP =', rotor.AEP
    print 'diameter =', rotor.diameter
    print 'ratedConditions.V =', rotor.ratedConditions.V
    print 'ratedConditions.Omega =', rotor.ratedConditions.Omega
    print 'ratedConditions.pitch =', rotor.ratedConditions.pitch
    print 'ratedConditions.T =', rotor.ratedConditions.T
    print 'ratedConditions.Q =', rotor.ratedConditions.Q
    print 'mass_one_blade =', rotor.mass_one_blade
    print 'mass_all_blades =', rotor.mass_all_blades
    print 'I_all_blades =', rotor.I_all_blades
    print 'freq =', rotor.freq
    print 'tip_deflection =', rotor.tip_deflection
    print 'root_bending_moment =', rotor.root_bending_moment

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P)
    plt.figure()
    plt.plot(rotor.spline.r_str, rotor.strainU)
    plt.plot(rotor.spline.r_str, rotor.strainL)
    plt.plot(rotor.spline.r_str, rotor.eps_crit)
    plt.ylim([-5e-3, 5e-3])
    plt.show()

    # TODO: the strain is not right
    # TODO: more variable trees?