#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

import numpy as np
import math
from openmdao.main.api import VariableTree, Component, Assembly, Driver
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Str, List, Slot, Enum, Bool

import _pBEAM
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from rotoraero import RotorAeroVS, AeroBase, AeroLoads, RPM2RS
from commonse import DirectionVector, cosd, sind, _akima


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



class Orthotropic2DMaterial(VariableTree):

    E1 = Float(units='N/m**2', desc='Young''s modulus in first principal direction')
    E2 = Float(units='N/m**2', desc='Young''s modulus in second principal direction')
    G12 = Float(units='N/m**2', desc='shear modulus')
    nu12 = Float(desc='Poisson''s ratio (nu12*E22 = nu21*E11)')
    rho = Float(units='kg/m**3', desc='density')


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



class PreComp(BeamPropertiesBase):

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

    sector_idx_buckling = Array(iotype='in', dtype=np.int)


    eps_crit = Array(iotype='out')


    def panelBucklingStrain(self):
        """
        see chapter on Structural Component Design Techniques from Alastair Johnson
        section 6.2: Design of composite panels

        assumes: large aspect ratio, simply supported, uniaxial compression, flat rectangular plate

        """

        # rename
        chord = self.chord
        CS_list = self.upperCS  # TODO: assumes the upper surface is the compression one
        sector_idx_list = self.sector_idx_buckling

        # initialize
        nsec = len(chord)
        self.eps_crit = np.zeros(nsec)

        for i in range(nsec):

            cs = CS_list[i]
            sector_idx = sector_idx_list[i]

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

            self.eps_crit[i] = - Nxx / totalHeight / E




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

        self.panelBucklingStrain()



class SectorPanelBuckling(Component):

    # from surface outward
    num_plies = Array(iotype='in', dtype=np.int)
    thickness = Array(iotype='in', units='m')
    orientation = Array(iotype='in', units='deg')
    materials = List(Orthotropic2DMaterial, iotype='in')
    sector_length = Float(iotype='in', units='m')

    eps_crit = Float(iotype='out')


    def __Qbar(self, material, theta):
        """Computes the lamina stiffness matrix

        Returns
        -------
        Qbar : numpy matrix
            the lamina stifness matrix

        Notes
        -----
        Transforms a specially orthotropic lamina from principal axis to
        an arbitrary axis defined by the ply orientation.
        [sigma_x; sigma_y; tau_xy]^T = Qbar * [epsilon_x; epsilon_y, gamma_xy]^T
        See [1]_ for further details.

        References
        ----------
        .. [1] J. C. Halpin. Primer on Composite Materials Analysis. Technomic, 2nd edition, 1992.


        """

        E11 = material.E1
        E22 = material.E2
        nu12 = material.nu12
        nu21 = nu12*E22/E11
        G12 = material.G12
        denom = (1 - nu12*nu21)

        c = cosd(theta)
        s = sind(theta)
        c2 = c*c
        s2 = s*s
        cs = c*s

        Q = np.mat([[E11/denom, nu12*E22/denom, 0],
                    [nu12*E22/denom, E22/denom, 0],
                    [0, 0, G12]])
        T12 = np.mat([[c2, s2, cs],
                      [s2, c2, -cs],
                      [-cs, cs, 0.5*(c2-s2)]])
        Tinv = np.mat([[c2, s2, -2*cs],
                       [s2, c2, 2*cs],
                       [cs, -cs, c2-s2]])

        return Tinv*Q*T12



    def execute(self):

        t = self.thickness
        n_plies = self.num_plies
        theta = self.orientation
        materials = self.materials
        n = len(theta)


        # ----- ABD matrices -----

        # heights (z - absolute, h - relative to mid-plane)
        z = np.zeros(n+1)
        for i in range(n):
            z[i+1] = z[i] + t[i]*n_plies[i]

        z_mid = (z[-1] - z[0]) / 2.0
        h = z - z_mid

        # ABD matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for i in range(n):
            Qbar = self.__Qbar(materials[i], theta[i])
            A += Qbar*(h[i+1] - h[i])
            B += 0.5*Qbar*(h[i+1]**2 - h[i]**2)
            D += 1.0/3.0*Qbar*(h[i+1]**3 - h[i]**3)

        totalHeight = z[-1] - z[0]


        # ----- effective axial modulus -----

        # Estimates the effective axial modulus of elasticity for the laminate
        S = np.vstack((np.hstack((A, B)), np.hstack((B, D))))

        # E_eff_x = N_x/h/eps_xx and eps_xx = S^{-1}(0,0)*N_x (approximately)
        detS = np.linalg.det(S)
        Eaxial = detS/np.linalg.det(S[1:, 1:])/totalHeight

        # ----- panel buckling -----
        # see chapter on Structural Component Design Techniques from Alastair Johnson
        # section 6.2: Design of composite panels
        # assumes: large aspect ratio, simply supported, uniaxial compression, flat rectangular plate

        D1 = D[0, 0]
        D2 = D[1, 1]
        D3 = D[0, 1] + 2*D[2, 2]

        # use empirical formula
        Nxx = 2 * (math.pi/self.sector_length)**2 * (math.sqrt(D1*D2) + D3)

        self.eps_crit = - Nxx / totalHeight / Eaxial





# class CompositeSection(Component):

#     break_loc = Array(iotype='in')

#     sectors = List(Sector, iotype='in')


# class PreCompCompositeSection(Component):

#     file_in = Str(iotype='in')

#     break_loc = Array(iotype='out')
#     sectors = List(Sector, iotype='out')


#     def execute(self):





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

    x_strain = Array(iotype='in')
    y_strain = Array(iotype='in')
    z_strain = Array(iotype='in')

    # outputs
    blade_mass = Float(iotype='out')
    freq = Array(iotype='out')

    dx_defl = Array(iotype='out')
    dy_defl = Array(iotype='out')
    dz_defl = Array(iotype='out')

    strain = Array(iotype='out')



class RotorWithpBEAM(StrucBase):


    def execute(self):

        beam = self.beam
        nsec = len(beam.z)


        # translate to elastic center and rotate to principal axes
        EI11 = np.zeros(nsec)
        EI22 = np.zeros(nsec)
        ca = np.zeros(nsec)
        sa = np.zeros(nsec)

        EA = beam.EA
        EIxx = beam.EIxx
        EIyy = beam.EIyy
        EIxy = beam.EIxy
        x_ec_str = beam.x_ec_str
        y_ec_str = beam.y_ec_str


        for i in range(nsec):

            # translate to elastic center
            EItemp = np.array([EIxx[i], EIyy[i], EIxy[i]]) + \
                np.array([-y_ec_str[i]**2, -x_ec_str[i]**2, -x_ec_str[i]*y_ec_str[i]])*EA[i]

            # use profile c.s. for conveneince in using Hansen's notation
            EI = DirectionVector.fromArray(EItemp).airfoilToProfile()

            # let alpha = 1/2 beta and use half-angle identity (avoid arctan issues)
            cb = (EI.y - EI.x) / math.sqrt((2*EI.z)**2 + (EI.y - EI.x)**2)  # EI.z is EIxy
            sa[i] = math.sqrt((1-cb)/2)
            ca[i] = math.sqrt((1+cb)/2)
            ta = sa[i]/ca[i]
            EI11[i] = EI.x - EI.z*ta
            EI22[i] = EI.y + EI.z*ta


        def a2p(x, y, z):  # rotate from airfoil c.s. to principal c.s.

            v = DirectionVector(x, y, 0.0).airfoilToProfile()

            r1 = v.x*ca + v.y*sa
            r2 = -v.x*sa + v.y*ca

            return r1, r2, z


        def p2a(r1, r2, r3):  # rotate from principal c.s. to airfoil c.s.

            x = r1*ca - r2*sa
            y = r1*sa + r2*ca

            v = DirectionVector(x, y, 0.0).profileToAirfoil()

            return v.x, v.y, r3


        # create finite element objects
        p_section = _pBEAM.SectionData(nsec, beam.z, EA, EI11,
            EI22, beam.GJ, beam.rhoA, beam.rhoJ)
        # p_loads = _pBEAM.Loads(nsec)  # no loads
        p_tip = _pBEAM.TipData()  # no tip mass
        k = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')])
        p_base = _pBEAM.BaseData(k, float('inf'))  # rigid base


        # ----- tip deflection -----

        # from airfoil to principal
        P1_defl, P2_defl, P3_defl = a2p(self.Px_defl, self.Py_defl, self.Pz_defl)

        # evaluate displacements
        p_loads = _pBEAM.Loads(nsec, P1_defl, P2_defl, P3_defl)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        dr1_defl, dr2_defl, dr3_defl, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

        # from principal to airfoil
        self.dx_defl, self.dy_defl, self.dz_defl = p2a(dr1_defl, dr2_defl, dr3_defl)


        # --- mass ---
        self.blade_mass = blade.mass()


        # ----- natural frequencies ----
        self.freq = blade.naturalFrequencies(self.nF)


        # ----- strain -----
        # from airfoil to principal
        P1_strain, P2_strain, P3_strain = a2p(self.Px_strain, self.Py_strain, self.Pz_strain)
        r1_strain, r2_strain, r3_strain = a2p(self.x_strain, self.y_strain, self.z_strain)

        p_loads = _pBEAM.Loads(nsec, P1_strain, P2_strain, P3_strain)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        self.strain = blade.axialStrain(len(r1_strain), r1_strain, r2_strain, r3_strain)




# ---------------------
# Components
# ---------------------

class BeamCurvature(Component):

    r = Array(iotype='in')
    precurve = Array(iotype='in')
    presweep = Array(iotype='in')
    precone = Float(iotype='in')

    totalCone = Array(iotype='out')
    z_az = Array(iotype='out')
    s = Array(iotype='out')

    def execute(self):

        n = len(self.r)
        precone = self.precone


        # azimuthal position
        # x_az = -self.r*sind(precone) + self.precurve*cosd(precone)
        z_az = self.r*cosd(precone) + self.precurve*sind(precone)
        # y_az = self.presweep


        # total precone angle
        x = self.precurve  # compute without precone and add in rotation after
        z = self.r
        y = self.presweep

        totalCone = np.zeros(n)
        totalCone[0] = math.atan2(-(x[1] - x[0]), z[1] - z[0])
        totalCone[1:n-1] = 0.5*(np.arctan2(-(x[1:n-1] - x[:n-2]), z[1:n-1] - z[:n-2])
            + np.arctan2(-(x[2:] - x[1:n-1]), z[2:] - z[1:n-1]))
        totalCone[n-1] = math.atan2(-(x[n-1] - x[n-2]), z[n-1] - z[n-2])


        # total path length of blade  (TODO: need to do something with this.  This should be a geometry preprocessing step just like rotoraero)
        s = np.zeros(n)
        s[0] = self.r[0]
        for i in range(1, n):
            s[i] = s[i-1] + math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2 + (z[i] - z[i-1])**2)


        # save variables of interest
        self.totalCone = precone + np.degrees(totalCone)  # change to degrees
        self.z_az = z_az
        self.s = s



class TotalLoads(Component):

    aeroLoads = VarTree(AeroLoads(), iotype='in')  # aerodynamic loads in blade c.s.

    r = Array(iotype='in')
    theta = Array(iotype='in')
    totalCone = Array(iotype='in')
    z_az = Array(iotype='in')
    rhoA = Array(iotype='in')
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')


    Px_af = Array(iotype='out')  # total loads in af c.s.
    Py_af = Array(iotype='out')
    Pz_af = Array(iotype='out')


    def execute(self):

        # keep all in blade c.s. then rotate all at end

        # rename
        aero = self.aeroLoads

        # --- aero loads ---

        # interpolate aerodynamic loads onto structural grid
        P_a = DirectionVector(0, 0, 0)
        P_a.x = _akima.interpolate(aero.r, aero.Px, self.r)
        P_a.y = _akima.interpolate(aero.r, aero.Py, self.r)
        P_a.z = _akima.interpolate(aero.r, aero.Pz, self.r)


        # --- weight loads ---

        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -self.rhoA*self.g)

        P_w = weight.yawToHub(aero.tilt).hubToAzimuth(aero.azimuth)\
            .azimuthToBlade(self.totalCone)


        # --- centrifugal loads ---

        # azimuthal c.s.
        Omega = aero.Omega*RPM2RS
        load = DirectionVector(0.0, 0.0, self.rhoA*Omega**2*self.z_az)

        P_c = load.azimuthToBlade(self.totalCone)


        # --- total loads ---
        P = P_a + P_w + P_c

        # rotate to airfoil c.s.
        theta = np.array(self.theta) + aero.pitch
        P = P.bladeToAirfoil(theta)

        self.Px_af = P.x
        self.Py_af = P.y
        self.Pz_af = P.z









class TipDeflection(Component):

    dx = Array(iotype='in')  # airfoil c.s.
    dy = Array(iotype='in')
    dz = Array(iotype='in')

    theta = Array(iotype='in')
    pitch = Float(iotype='in')
    azimuth = Float(iotype='in')
    tilt = Float(iotype='in')
    totalCone = Array(iotype='in')

    tip_deflection = Float(iotype='out')


    def execute(self):

        theta = np.array(self.theta) + self.pitch

        dr = DirectionVector(self.dx, self.dy, self.dz)
        delta = dr.airfoilToBlade(theta).bladeToAzimuth(self.totalCone) \
            .azimuthToHub(self.azimuth).hubToYaw(self.tilt)

        self.tip_deflection = delta.x[-1]






class RotorStruc(Assembly):

    # replace
    beam = Slot(BeamPropertiesBase)
    struc = Slot(StrucBase)

    # inputs
    theta = Array(iotype='in')
    precone = Float(iotype='in')
    precurve = Array(iotype='in')
    presweep = Array(iotype='in')
    g = Float(9.81, iotype='in', units='m/s**2')
    nF = Int(5, iotype='in')

    # connection input
    aero_loads_defl = VarTree(AeroLoads(), iotype='in')
    aero_loads_strain = VarTree(AeroLoads(), iotype='in')



    def configure(self):

        self.add('beam', BeamPropertiesBase())
        self.add('curve', BeamCurvature())
        self.add('loads_defl', TotalLoads())
        self.add('loads_strain', TotalLoads())
        self.add('struc', StrucBase())
        self.add('tip', TipDeflection())


        self.driver.workflow.add(['beam', 'curve', 'loads_defl', 'loads_strain', 'struc', 'tip'])

        # connections to curve
        self.connect('beam.properties.z', 'curve.r')
        self.connect('precurve', 'curve.precurve')
        self.connect('presweep', 'curve.presweep')
        self.connect('precone', 'curve.precone')

        # connections to loads_defl
        self.connect('aero_loads_defl', 'loads_defl.aeroLoads')
        self.connect('beam.properties.z', 'loads_defl.r')
        self.connect('theta', 'loads_defl.theta')
        self.connect('curve.totalCone', 'loads_defl.totalCone')
        self.connect('curve.z_az', 'loads_defl.z_az')
        self.connect('beam.properties.rhoA', 'loads_defl.rhoA')
        self.connect('g', 'loads_defl.g')

        # connections to loads_strain
        self.connect('aero_loads_strain', 'loads_strain.aeroLoads')
        self.connect('beam.properties.z', 'loads_strain.r')
        self.connect('theta', 'loads_strain.theta')
        self.connect('curve.totalCone', 'loads_strain.totalCone')
        self.connect('curve.z_az', 'loads_strain.z_az')
        self.connect('beam.properties.rhoA', 'loads_strain.rhoA')
        self.connect('g', 'loads_strain.g')


        # connections to struc
        self.connect('beam.properties', 'struc.beam')
        self.connect('nF', 'struc.nF')
        self.connect('loads_defl.Px_af', 'struc.Px_defl')
        self.connect('loads_defl.Py_af', 'struc.Py_defl')
        self.connect('loads_defl.Pz_af', 'struc.Pz_defl')
        self.connect('loads_strain.Px_af', 'struc.Px_strain')
        self.connect('loads_strain.Py_af', 'struc.Py_strain')
        self.connect('loads_strain.Pz_af', 'struc.Pz_strain')

        # connections to tip
        self.connect('struc.dx_defl', 'tip.dx')
        self.connect('struc.dy_defl', 'tip.dy')
        self.connect('struc.dz_defl', 'tip.dz')
        self.connect('theta', 'tip.theta')
        self.connect('aero_loads_defl.pitch', 'tip.pitch')
        self.connect('aero_loads_defl.azimuth', 'tip.azimuth')
        self.connect('aero_loads_defl.tilt', 'tip.tilt')
        self.connect('curve.totalCone', 'tip.totalCone')


        # input passthroughs
        self.create_passthrough('struc.x_strain')
        self.create_passthrough('struc.y_strain')
        self.create_passthrough('struc.z_strain')

        # output passthroughs
        self.create_passthrough('struc.blade_mass')
        self.create_passthrough('struc.freq')
        self.create_passthrough('tip.tip_deflection')
        self.create_passthrough('struc.strain')










class RotorVSVP(RotorAeroVS):

    # load conditions
    azimuth_rated = Float(iotype='in')

    V_extreme = Float(iotype='in')  # TODO: can combine with Ubar based on machine class
    pitch_extreme = Float(iotype='in')
    azimuth_extreme = Float(iotype='in')



    def configure(self):

        self.add('analysis3', AeroBase())
        self.add('beam', BeamPropertiesBase())
        self.add('pBEAM', RotorStruc())

        self.driver.workflow.add(['analysis3', 'beam', 'pBEAM'])

        # connections to analysis3
        self.connect('rated.V', 'analysis3.V_load')
        self.connect('rated.Omega', 'analysis3.Omega_load')
        self.connect('rated.pitch', 'analysis3.pitch_load')
        self.connect('azimuth_rated', 'analysis3.azimuth_load')
        self.analysis3.run_case = 'loads'

        # connections to pBEAM
        self.connect('beam.properties', 'pBEAM.properties')
        self.connect('analysis3.loads', 'pBEAM.ratedLoads')


        # passthrough
        self.create_passthrough('pBEAM.tip_deflection')

        # # connections to analysis4
        # self.connect('V_extreme', 'analysis4.V_load')
        # self.connect('pitch_extreme', 'analysis4.pitch_load')
        # self.connect('azimuth_extreme', 'analysis4.azimuth_load')
        # self.analysis4.Omega_load = 0.0  # not rotating
        # self.analysis4.run_case = 'loads'


        # connect to output
        # self.connect('analysis3.Np', 'Np_rated')
        # self.connect('analysis3.Tp', 'Tp_rated')
        # self.connect('analysis4.Np', 'Np_extreme')
        # self.connect('analysis4.Tp', 'Tp_extreme')




if __name__ == '__main__':


    import os

    # geometry
    # r_str = [1.5, 1.80135, 1.89975, 1.99815, 2.1027, 2.2011, 2.2995, 2.87145, 3.0006, 3.099, 5.60205, 6.9981, 8.33265, 10.49745, 11.75205, 13.49865, 15.84795, 18.4986, 19.95, 21.99795, 24.05205, 26.1, 28.14795, 32.25, 33.49845, 36.35205, 38.4984, 40.44795, 42.50205, 43.49835, 44.55, 46.49955, 48.65205, 52.74795, 56.16735, 58.89795, 61.62855, 63.]
    # chord_str = [3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.387, 3.39, 3.741, 4.035, 4.25, 4.478, 4.557, 4.616, 4.652, 4.543, 4.458, 4.356, 4.249, 4.131, 4.007, 3.748, 3.672, 3.502, 3.373, 3.256, 3.133, 3.073, 3.01, 2.893, 2.764, 2.518, 2.313, 2.086, 1.419, 1.085]
    # theta_str = [13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 12.53, 11.48, 10.63, 10.16, 9.59, 9.01, 8.4, 7.79, 6.54, 6.18, 5.36, 4.75, 4.19, 3.66, 3.4, 3.13, 2.74, 2.32, 1.53, 0.86, 0.37, 0.11, 0.0]
    # le_str = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    r_str = np.array([1.50142903976, 1.80306613137, 1.90155987557, 2.00005361977, 2.10470322299, 2.20319696719, 2.30169071139, 2.86802974054, 3.00345863882, 3.10195238302, 5.60738700113, 7.00476699698, 8.3405884027, 10.5074507751, 11.7632460137, 13.5115099732, 15.863048116, 18.5162233505, 19.9690060774, 22.0189071286, 24.0749640388, 26.12486509, 28.1747661411, 32.2807241025, 33.5303634821, 36.3866820639, 38.5350768593, 40.4864841663, 42.5425410764, 43.5397902365, 44.5924421276, 46.5438494346, 48.698400089, 52.7982021914, 56.2208598023, 58.9540612039, 61.6934184645, 63.0600191653])
    chord_str = np.array([3.2612, 3.30974143717, 3.32552109037, 3.34126379956, 3.35794850262, 3.37361093258, 3.38923246006, 3.47819767927, 3.49923810816, 3.51447938721, 3.88091952034, 4.06260542204, 4.21678354504, 4.41794897837, 4.50221292399, 4.57390637252, 4.57496258253, 4.49355793249, 4.44296484557, 4.36449116824, 4.27745048448, 4.18236399597, 4.07898355526, 3.84696806115, 3.7697498699, 3.58167705636, 3.42960605033, 3.28382135203, 3.12880081291, 3.05388779827, 2.97485038083, 2.82801258654, 2.66459194226, 2.34610078292, 2.06819798348, 1.83533535858, 1.58981373477, 1.4621])
    theta_str = np.array([13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 13.27752682, 12.626010184, 11.7718976268, 10.8375080565, 10.3384548031, 9.64877664692, 8.97337520037, 8.31552847131, 7.67242584923, 6.43503027652, 6.07433096971, 5.27792502023, 4.70461990261, 4.20301591557, 3.69421857347, 3.45471911559, 3.20707500727, 2.76233633354, 2.29860497622, 1.49730367007, 0.905122107237, 0.479162306686, 0.09155965266, -0.0878099])
    le_str = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    precurve_str = np.zeros_like(r_str)
    presweep_str = np.zeros_like(r_str)


    # -------- materials and composite layup  -----------------
    basepath = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(r_str)
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

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
    # --------------------------------------


    rs = RotorStruc()
    rs.replace('beam', PreComp())
    rs.replace('struc', RotorWithpBEAM())

    rs.theta = theta_str
    rs.precone = 2.5
    rs.precurve = precurve_str
    rs.presweep = presweep_str

    rs.aero_loads_defl.r = np.array([1.50142903976, 2.86943081405, 5.60533525332, 8.341239077, 11.7611942659, 15.8650998639, 19.9690060774, 24.072912291, 28.176817889, 32.2807241025, 36.3846303161, 40.4885359141, 44.5924421276, 48.6963483412, 52.8002539392, 56.220209128, 58.9561129517, 61.692017391, 63.0600191653])
    rs.aero_loads_defl.Px = np.array([0.0, 388.882106126, 484.447148982, 402.741351277, 3111.66541958, 3310.69291134, 4074.26486938, 4570.38064794, 5137.53869383, 5883.16246763, 6527.30546598, 7306.30332887, 8843.37366039, 9198.35713086, 9287.96756701, 9082.53707622, 8655.15257055, 7689.09458231, 0.0])
    rs.aero_loads_defl.Py = np.array([-0, 68.6533921102, 168.133937689, 209.369690377, -1076.87079678, -1055.70325825, -1306.55253118, -1506.95422869, -1633.24056079, -1690.6232744, -1843.22553237, -1883.76262587, -2010.28347412, -1917.6539151, -1776.49531767, -1610.58547474, -1424.92987638, -1051.3928091, -0])
    rs.aero_loads_defl.Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rs.aero_loads_defl.V = 11.649611668
    rs.aero_loads_defl.Omega = 12.1260909022
    rs.aero_loads_defl.pitch = 0.0
    rs.aero_loads_defl.azimuth = 180.0
    rs.aero_loads_defl.tilt = 5.0

    rs.aero_loads_strain.r = np.array([1.50142903976, 2.86943081405, 5.60533525332, 8.341239077, 11.7611942659, 15.8650998639, 19.9690060774, 24.072912291, 28.176817889, 32.2807241025, 36.3846303161, 40.4885359141, 44.5924421276, 48.6963483412, 52.8002539392, 56.220209128, 58.9561129517, 61.692017391, 63.0600191653])
    rs.aero_loads_strain.Px = np.array([0.0, 5275.91170102, 5954.97469367, 4581.21401614, 24317.5727315, 22887.0177512, 22273.0504313, 20337.8397148, 19170.3580565, 18331.9830106, 17757.7609212, 16526.853061, 15093.9066831, 13734.7900621, 12267.4308679, 10937.1620954, 9787.29281101, 8548.74476641, 0.0])
    rs.aero_loads_strain.Py = np.array([-0, -0, -0, -0, -8022.116732, -6577.57671572, -5839.69768021, -4922.9453766, -3742.72420205, -3141.49794777, -2329.22809287, -1842.52873914, -1357.22122192, -1019.59483393, -741.427649635, -550.033426097, -420.751366997, -310.90053355, -0])
    rs.aero_loads_strain.Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rs.aero_loads_strain.V = 70.0
    rs.aero_loads_strain.Omega = 0.0
    rs.aero_loads_strain.pitch = 0.0
    rs.aero_loads_strain.azimuth = 0.0
    rs.aero_loads_strain.tilt = 5.0

    rs.x_strain = np.array([1.62953277093, 1.65378760431, 1.6848244664, 1.69280026956, 1.70125331952, 1.70918845043, 1.71710285872, 1.7379700867, 1.77191347081, 1.77769022081, 1.66549686777, 1.46236108547, 1.24412860472, 0.946027322623, 0.855159937668, 0.793030786156, 0.741219389531, 0.678388044164, 0.656104367978, 0.626195492951, 0.594991331198, 0.535300387358, 0.49436378615, 0.466270531861, 0.437964716508, 0.374175959337, 0.35504758904, 0.343037547915, 0.30479534701, 0.285300267572, 0.270977493752, 0.256272902802, 0.243171569781, 0.215017502212, 0.190697407308, 0.170514014614, 0.149901126795, 0.13789004202])
    rs.y_strain = np.array([-0.00113112893698, -0.00114796526232, -0.0172978765403, -0.0173797630875, -0.0174665494665, -0.0175480184371, -0.0176292746511, -0.0015645000705, -0.02731068479, -0.0305932407033, -0.109340525932, -0.125949563315, -0.12117411121, -0.128218415335, -0.149327182633, -0.14656631794, -0.140266481404, -0.135652296602, -0.132851488119, -0.1287750113, -0.124992337883, -0.118409965897, -0.110552832125, -0.0979009471516, -0.0937777322172, -0.084620004667, -0.0782777410193, -0.0716491412175, -0.0584114110202, -0.0591831317106, -0.0558251490105, -0.0488944452438, -0.050399265575, -0.0549454662041, -0.0623708745921, -0.0713825225821, -0.0823128686813, -0.0695992262681])
    rs.z_strain = np.array([1.50142903976, 1.80306613137, 1.90155987557, 2.00005361977, 2.10470322299, 2.20319696719, 2.30169071139, 2.86802974054, 3.00345863882, 3.10195238302, 5.60738700113, 7.00476699698, 8.3405884027, 10.5074507751, 11.7632460137, 13.5115099732, 15.863048116, 18.5162233505, 19.9690060774, 22.0189071286, 24.0749640388, 26.12486509, 28.1747661411, 32.2807241025, 33.5303634821, 36.3866820639, 38.5350768593, 40.4864841663, 42.5425410764, 43.5397902365, 44.5924421276, 46.5438494346, 48.698400089, 52.7982021914, 56.2208598023, 58.9540612039, 61.6934184645, 63.0600191653])


    rs.beam.r = r_str
    rs.beam.chord = chord_str
    rs.beam.theta = theta_str
    rs.beam.leLoc = le_str
    rs.beam.theta = theta_str
    rs.beam.profile = profile
    rs.beam.materials = materials
    rs.beam.upperCS = upper
    rs.beam.lowerCS = lower
    rs.beam.websCS = webs

    rs.beam.sector_idx_buckling = [2]*ncomp

    rs.run()

    print rs.blade_mass
    print rs.freq
    print rs.tip_deflection
    print rs.strain
    print rs.beam.eps_crit

    # 17161.8168946
    # ???
    # 4.67929859612
    # [ -2.28358243e-04  -2.17393854e-04  -1.15914419e-04  -1.13759869e-04
    #   -1.11517615e-04  -1.09447311e-04  -1.12797115e-04  -1.14157204e-04
    #   -1.18455855e-04  -1.13027900e-04  -2.51873686e-04  -3.50517728e-04
    #   -4.65866370e-04  -1.05325851e-03  -3.21593957e-03  -3.28704328e-03
    #   -3.01119063e-03  -2.79448215e-03  -2.63896740e-03  -2.43410855e-03
    #   -2.25519422e-03  -2.28464035e-03  -2.25379703e-03  -1.88422309e-03
    #   -1.88487026e-03  -1.90956585e-03  -1.75147195e-03  -1.56748835e-03
    #   -1.53817853e-03  -1.56631711e-03  -1.56011825e-03  -1.43961146e-03
    #   -1.29415855e-03  -9.84150717e-04  -6.70424233e-04  -3.50877768e-04
    #   -3.74835571e-05  -3.89995434e-23]

 #    [-0.01987173 -0.01929312 -0.01911046 -0.0189308  -0.01874314 -0.01856951
 # -0.01663386 -0.00789359 -0.00751241 -0.00765203 -0.00725968 -0.00677502
 # -0.00653707 -0.00390532 -0.00414116 -0.00368076 -0.00329441 -0.00303573
 # -0.00292038 -0.0027829  -0.00267326 -0.0024449  -0.00225016 -0.00198843
 # -0.00190416 -0.00176648 -0.00168421 -0.00165569 -0.00159949 -0.001516
 # -0.00143407 -0.00127599 -0.00109732 -0.00080051 -0.00057373 -0.00043451
 # -0.00015101 -0.00010445]


