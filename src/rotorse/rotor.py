#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

from __future__ import print_function

import numpy as np
import math
from openmdao.api import IndepVarComp, Component, Group, ParallelFDGroup

from rotoraero import SetupRunVarSpeed, RegulatedPowerCurve, AEP, VarSpeedMachine, \
    RatedConditions, AeroLoads, RPM2RS, RS2RPM, RegulatedPowerCurveGroup, COE
from rotoraerodefaults import CCBladeGeometry, CCBlade, CSMDrivetrain, RayleighCDF, WeibullWithMeanCDF, CCBladeAirfoils, AirfoilSpline

from commonse.csystem import DirectionVector
from commonse.utilities import hstack, vstack, trapz_deriv, interp_with_deriv
from commonse.environment import PowerWind
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from akima import Akima, akima_interp_with_derivs
from enum import Enum
import _pBEAM
import _curvefem
import _bem  # TODO: move to rotoraero

import matlab.engine
import re
from scipy import stats
import os
import time
import matplotlib.pyplot as plt
import pickle

# Import AeroelasticSE
import sys

# AeroelasticSE
sys.path.insert(0, '../../../AeroelasticSE/src/AeroelasticSE/FAST_mdao')

# rainflow
sys.path.insert(0, '../../../AeroelasticSE/src/AeroelasticSE/rainflow')

# for creating FAST run directories
from distutils.dir_util import copy_tree

# ---------------------
# Base Components
# ---------------------

class BeamPropertiesBase(Component):
    def __init__(self, nstr):
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
    def __init__(self, nstr):
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
        self.add_param('N_damage', val=365*24*3600*20.0, desc='number of cycles used in fatigue analysis')

        self.add_param('Edg_max', shape=nstr, desc='FAST Edg_max')
        self.add_param('Flp_max', shape=nstr, desc='FAST Flp_max')



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

        self.add_output('calc_tip_def', shape=1, units='m', desc='Dynamically calculated maximum tip deflection')

# ---------------------
# Components
# ---------------------

class ResizeCompositeSection(Component):
    def __init__(self, nstr, num_airfoils, af_dof):
        super(ResizeCompositeSection, self).__init__()

        self.add_param('upperCSIn', shape=nstr, desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_param('lowerCSIn', shape=nstr, desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_param('websCSIn', shape=nstr, desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)

        self.add_param('capTriaxThk', shape=5, units='m', desc='spar cap TRIAX layer thickness')
        self.add_param('capCarbThk', shape=5, units='m', desc='spar cap carbon layer thickness')
        self.add_param('tePanelTriaxThk', shape=5, units='m', desc='trailing edge TRIAX layer thickness')
        self.add_param('tePanelFoamThk', shape=5, units='m', desc='trailing edge foam layer thickness')
        self.add_param('initial_str_grid', shape=nstr, desc='initial structural grid on unit radius')

        self.add_param('chord_str_ref', shape=nstr, units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness')
        self.add_param('thick_str_ref', shape=nstr, units='m', desc='thickness-to-chord distribution for reference section, thickness of structural layup scaled with reference thickness')
        self.add_param('af_str', val=np.zeros(nstr))
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

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        nstr = len(params['chord_str_ref'])
        upperCSIn = params['upperCSIn']
        lowerCSIn = params['lowerCSIn']
        websCSIn = params['websCSIn']

        # # radius location of control points
        x = [0.0, 0.18, 0.48, 0.75, 1.0]
        capTriaxAkima = Akima(x, params['capTriaxThk']*.01)
        capCarbAkima = Akima(x, params['capCarbThk']*.01)
        tePanelTriaxAkima = Akima(x, params['tePanelTriaxThk']*.01)
        tePanelFoamAkima = Akima(x, params['tePanelFoamThk']*.01)
        initial_str_grid = params['initial_str_grid']
        # import copy
        # lowerCSplot = copy.deepcopy(lowerCSIn)
        # for i in range(nstr):
        #     if (i < 8 or (i > 12 and i < 36)):
        #         # set thicknesses for UpperCS spar cap
        #         upperCSIn[i].t[2][1], _, _, _ = capTriaxAkima.interp(initial_str_grid[i]) # cap triax
        #         upperCSIn[i].t[2][3] = upperCSIn[i].t[2][1] # cap triax
        #         upperCSIn[i].t[2][2], _, _, _ = capCarbAkima.interp(initial_str_grid[i]) # cap carbon
        #
        #         # same for lowerCS spar cap
        #         lowerCSIn[i].t[2][1] = upperCSIn[i].t[2][1] # cap triax
        #         lowerCSIn[i].t[2][3] = upperCSIn[i].t[2][3] # cap triax
        #         lowerCSIn[i].t[2][2] = upperCSIn[i].t[2][2] # cap carbon
        #
        #         #set thicknesses for UpperCS te panel
        #         upperCSIn[i].t[3][1], _, _, _ = tePanelTriaxAkima.interp(initial_str_grid[i]) # triax
        #         upperCSIn[i].t[3][3] = upperCSIn[i].t[3][1] # triax
        #         upperCSIn[i].t[3][2], _, _, _ = tePanelFoamAkima.interp(initial_str_grid[i]) # foam
        #
        #         # same for lowerCS te panel
        #         lowerCSIn[i].t[3][1] = upperCSIn[i].t[3][1] # triax
        #         lowerCSIn[i].t[3][3] = upperCSIn[i].t[3][3] # triax
        #         lowerCSIn[i].t[3][2] = upperCSIn[i].t[3][2] # foam
        #     elif (i >= 8 and i <= 12):
        #         # set thicknesses for UpperCS spar cap
        #         upperCSIn[i].t[2][1], _, _, _ = capTriaxAkima.interp(initial_str_grid[i]) # cap triax
        #         upperCSIn[i].t[2][4] = upperCSIn[i].t[2][1] # cap triax
        #         upperCSIn[i].t[2][3], _, _, _ = capCarbAkima.interp(initial_str_grid[i]) # cap carbon
        #
        #         # same for lowerCS spar cap
        #         lowerCSIn[i].t[2][1] = upperCSIn[i].t[2][1] # cap triax
        #         lowerCSIn[i].t[2][4] = upperCSIn[i].t[2][4] # cap triax
        #         lowerCSIn[i].t[2][3] = upperCSIn[i].t[2][3] # cap carbon
        #
        #         #set thicknesses for UpperCS te panel
        #         upperCSIn[i].t[3][1], _, _, _ = tePanelTriaxAkima.interp(initial_str_grid[i]) # triax
        #         upperCSIn[i].t[3][4] = upperCSIn[i].t[3][1] # triax
        #         upperCSIn[i].t[3][3], _, _, _ = tePanelFoamAkima.interp(initial_str_grid[i]) # foam
        #
        #         # same for lowerCS te panel
        #         lowerCSIn[i].t[3][1] = upperCSIn[i].t[3][1] # triax
        #         lowerCSIn[i].t[3][4] = upperCSIn[i].t[3][4] # triax
        #         lowerCSIn[i].t[3][3] = upperCSIn[i].t[3][3] # foam
        #     elif (i >= 36):
        #         # set thicknesses for UpperCS spar cap
        #         upperCSIn[i].t[2][1], _, _, _ = capTriaxAkima.interp(initial_str_grid[i]) # cap triax
        #         upperCSIn[i].t[2][2] = upperCSIn[i].t[2][1] # cap triax
        #
        #         # same for lowerCS spar cap
        #         lowerCSIn[i].t[2][1] = upperCSIn[i].t[2][1] # cap triax
        #         lowerCSIn[i].t[2][2] = upperCSIn[i].t[2][2] # cap triax
        #
        #         #set thicknesses for UpperCS te panel
        #         upperCSIn[i].t[3][1], _, _, _ = tePanelTriaxAkima.interp(initial_str_grid[i]) # triax
        #         upperCSIn[i].t[3][2] = upperCSIn[i].t[3][1] # triax
        #
        #         # same for lowerCS te panel
        #         lowerCSIn[i].t[3][1] = upperCSIn[i].t[3][1] # triax
        #         lowerCSIn[i].t[3][2] = upperCSIn[i].t[3][2] # triax

        # import matplotlib.pylab as plt
        # t_before = []
        # t_after = []
        # x = range(len(lowerCSIn))
        # for z in range(4):
        #     for q in range(5):
        #         for k in range(len(lowerCSIn)):
        #             try:
        #                 if abs(lowerCSIn[k].t[q][z] - lowerCSplot[k].t[q][z]) > 0:
        #                     print(str(q) + ' ' + str(k) + ' ' + str(z) + str(abs(lowerCSIn[k].t[q][z] - lowerCSplot[k].t[q][z])))
        #             except:
        #                 pass
        # for z in range(4):
        #     for k in range(len(lowerCSIn)):
        #         t_before.append(lowerCSplot[k].t[2][z])
        #         t_after.append(lowerCSIn[k].t[2][z])
        # x = range(len(t_before))
        # plt.figure()
        # plt.plot(x, t_before)
        # plt.plot(x, t_after)
        # plt.show()
        # copy data across
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
            tc = params['af_str'][i].tc
            if tc is None:
                tc = params['thick_str_ref'][i]
            factor = params['chord_str'][i]/params['chord_str_ref'][i] * tc/params['thick_str_ref'][i] # scale by chord and then airfoil thickness

            for j in range(len(upper.t)):
                upper.t[j] *= factor

            for j in range(len(lower.t)):
                lower.t[j] *= factor

            for j in range(len(webs.t)):
                webs.t[j] *= factor


        # change spar and trailing edge thickness to specified values
        for i in range(nstr):

            idx_spar = params['sector_idx_strain_spar'][i]
            idx_te = params['sector_idx_strain_te'][i]
            upper = upperCSOut[i]
            lower = lowerCSOut[i]

            # upper and lower have same thickness for this design
            tspar = np.sum(upper.t[idx_spar])
            tte = np.sum(upper.t[idx_te])

            upper.t[idx_spar] *= params['sparT_str'][i]/tspar
            lower.t[idx_spar] *= params['sparT_str'][i]/tspar

            upper.t[idx_te] *= params['teT_str'][i]/tte
            lower.t[idx_te] *= params['teT_str'][i]/tte

        unknowns['upperCSOut'] = upperCSOut
        unknowns['lowerCSOut'] = lowerCSOut
        unknowns['websCSOut'] = websCSOut

class PreCompSections(BeamPropertiesBase):
    def __init__(self, nstr, num_airfoils, af_dof):
        super(PreCompSections, self).__init__(nstr)
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

        self.add_param('af_str', val=np.zeros(nstr), pass_by_obj=True)

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
        self.nstr = nstr
        self.num_airfoils = num_airfoils
        self.af_dof = af_dof


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
        self.chord, self.materials, self.r, self.profile, self.upperCS, self.lowerCS, \
        self.websCS, self.theta, self.leLoc, self.sector_idx_strain_spar, self.sector_idx_strain_te  = \
            params['chord'], params['materials'], params['r'], params['profile'], \
            params['upperCS'], params['lowerCS'], params['websCS'], params['theta'], params['leLoc'], \
            params['sector_idx_strain_spar'], params['sector_idx_strain_te']

        # radial discretization
        nsec = len(self.r)
        nstr = self.nstr

        # initialize variables
        self.properties_z = self.r
        self.properties_EA = np.zeros(nsec)
        self.properties_EIxx = np.zeros(nsec)
        self.properties_EIyy = np.zeros(nsec)
        self.properties_EIxy = np.zeros(nsec)
        self.properties_GJ = np.zeros(nsec)
        self.properties_rhoA = np.zeros(nsec)
        self.properties_rhoJ = np.zeros(nsec)

        # distance to elastic center from point about which structural properties are computed
        # using airfoil coordinate system
        self.properties_x_ec_str = np.zeros(nsec)
        self.properties_y_ec_str = np.zeros(nsec)

        # distance to elastic center from airfoil nose
        # using profile coordinate system
        x_ec_nose = np.zeros(nsec)
        y_ec_nose = np.zeros(nsec)

        profile = self.profile
        # update the profile coordinates if applicable
        for i in range(nstr):
            if params['af_str'][i].Saf is not None:
                x, y, xl, xu, yl, yu = params['af_str'][i].afanalysis.getCoordinates(form='all')
                # xu1 = np.zeros(len(xu))
                # xl1 = np.zeros(len(xl))
                # yu1 = np.zeros(len(xu))
                # yl1 = np.zeros(len(xl))
                # for k in range(len(xu)):
                #     xu1[k] = float(xu[k])
                #     yu1[k] = float(yu[k])
                # for k in range(len(xl)):
                #     xl1[k] = float(xl[k])
                #     yl1[k] = float(yl[k])
                # x = np.append(xu1, xl1)
                # y = np.append(yu1, yl1)
                # TODO : check if necessary
                profile[i] = Profile.initFromCoordinates(x, y, LEtoLE=False)
                self.profile = profile

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


            self.properties_EIxx[i] = results[1]  # EIedge
            self.properties_EIyy[i] = results[0]  # EIflat
            self.properties_GJ[i] = results[2]
            self.properties_EA[i] = results[3]
            self.properties_EIxy[i] = results[4]  # EIflapedge
            self.properties_x_ec_str[i] = results[12] - results[10]
            self.properties_y_ec_str[i] = results[13] - results[11]
            self.properties_rhoA[i] = results[14]
            self.properties_rhoJ[i] = results[15] + results[16]  # perpindicular axis theorem

            x_ec_nose[i] = results[13] + self.leLoc[i]*self.chord[i]
            y_ec_nose[i] = results[12]  # switch b.c of coordinate system used

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
        unknowns['beam:z'] = self.properties_z
        unknowns['beam:EIxx'] = self.properties_EIxx
        unknowns['beam:EIyy'] = self.properties_EIyy
        unknowns['beam:GJ'] = self.properties_GJ
        unknowns['beam:EA'] = self.properties_EA
        unknowns['beam:EIxy'] = self.properties_EIxy
        unknowns['beam:x_ec_str'] = self.properties_x_ec_str
        unknowns['beam:y_ec_str'] = self.properties_y_ec_str
        unknowns['beam:rhoA'] = self.properties_rhoA
        unknowns['beam:rhoJ'] = self.properties_rhoJ
        unknowns['eps_crit_spar'] = self.panelBucklingStrain(self.sector_idx_strain_spar)
        unknowns['eps_crit_te'] = self.panelBucklingStrain(self.sector_idx_strain_te)

class RotorWithpBEAM(StrucBase):

    def __init__(self, nstr, FASTinfo):
        super(RotorWithpBEAM, self).__init__(nstr)


        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

        self.use_FAST = FASTinfo['use_FAST']
        self.check_damage = FASTinfo['check_damage']

        self.use_FAST_struc = FASTinfo['use_struc_cons']


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

    def strain(self, blade, xu, yu, xl, yl, EI11, EI22, EA, ca, sa, Edg_max=0.0, Flp_max=0.0):

        Vx, Vy, Fz, Mx, My, Tz = blade.shearAndBending()

        # print(Mx)
        # print(My)
        # print(Fz)
        # quit()

        if self.use_FAST_struc:
            Mx = Edg_max
            My = Flp_max

            # print(Mx)
            # print(My)
            # quit()

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

        # print(strainU)
        # print(strainL)

        return strainU, strainL

    def damage(self, Mx, My, xu, yu, xl, yl, EI11, EI22, EA, ca, sa, emax=0.01, eta=1.755, m=10.0, N=365*24*3600*20):


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

        # print(strainU)
        # print(strainL)
        # print(strainU/emax)
        # print(strainL/emax)


        # damageU = math.log(N) - m*(math.log(emax) - math.log(eta) - np.log(np.abs(strainU)))
        # damageL = math.log(N) - m*(math.log(emax) - math.log(eta) - np.log(np.abs(strainL)))

        # print(damageU[0])
        # quit()

        return damageU, damageL

    def solve_nonlinear(self, params, unknowns, resids):

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
        p_loads = _pBEAM.Loads(nsec, params['Px_defl'], params['Py_defl'], params['Pz_defl'])
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        self.dx_defl, self.dy_defl, self.dz_defl, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

        p_loads = _pBEAM.Loads(nsec, params['Px_pc_defl'], params['Py_pc_defl'], params['Pz_pc_defl'])
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
        self.freq = blade.naturalFrequencies(params['nF'])


        # ----- strain -----
        EI11, EI22, EA, ca, sa = self.principalCS(params['beam:EIyy'], params['beam:EIxx'], params['beam:y_ec_str'], params['beam:x_ec_str'], params['beam:EA'], params['beam:EIxy'])

        p_loads = _pBEAM.Loads(nsec, params['Px_strain'], params['Py_strain'], params['Pz_strain'])
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)

        # If self.use_FAST_struc is True, max loads from either FAST or the surrogate model will be used to calculate the max strain

        if self.use_FAST_struc:
            self.strainU_spar, self.strainL_spar = self.strain(blade, params['xu_strain_spar'],
                params['yu_strain_spar'], params['xl_strain_spar'], params['yl_strain_spar'], EI11, EI22, EA, ca, sa, params['Edg_max'], params['Flp_max'])
            self.strainU_te, self.strainL_te = self.strain(blade, params['xu_strain_te'], params['yu_strain_te'],
                params['xl_strain_te'], params['yl_strain_te'], EI11, EI22, EA, ca, sa, params['Edg_max'], params['Flp_max'])

        else:
            self.strainU_spar, self.strainL_spar = self.strain(blade, params['xu_strain_spar'],
                params['yu_strain_spar'], params['xl_strain_spar'], params['yl_strain_spar'], EI11, EI22, EA, ca, sa)
            self.strainU_te, self.strainL_te = self.strain(blade, params['xu_strain_te'], params['yu_strain_te'],
                  params['xl_strain_te'], params['yl_strain_te'], EI11, EI22, EA, ca, sa)

        # ------ damage ------

        # print(EI11)
        # print(EI22)
        # quit()

        # print('damage calc')
        # print(params['Mx_damage'])
        # print(params['My_damage'])
        # quit()

        self.damageU_spar, self.damageL_spar = self.damage(params['Mx_damage'], params['My_damage'], params['xu_strain_spar'], params['yu_strain_spar'],
            params['xl_strain_spar'], params['yl_strain_spar'], EI11, EI22, EA, ca, sa,
            emax=params['strain_ult_spar'], eta=params['eta_damage'], m=params['m_damage'], N=params['N_damage'])
        self.damageU_te, self.damageL_te = self.damage(params['Mx_damage'], params['My_damage'], params['xu_strain_te'], params['yu_strain_te'],
            params['xl_strain_te'], params['yl_strain_te'], EI11, EI22, EA, ca, sa,
            emax=params['strain_ult_te'], eta=params['eta_damage'], m=params['m_damage'], N=params['N_damage'])

        # quit()

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

        # print('strainU_spar is')
        # print(self.strainU_spar)
        # print('strainL_spar is')
        # print(self.strainL_spar)
        # print('strainU_te is')
        # print(self.strainU_te)
        # print('strainL_te is')
        # print(self.strainL_te)
        # quit()

        unknowns['damageU_spar'] = self.damageU_spar
        unknowns['damageL_spar'] = self.damageL_spar
        unknowns['damageU_te'] = self.damageU_te
        unknowns['damageL_te'] = self.damageL_te


        if self.check_damage:
            print('damageU_spar is:')
            print(unknowns['damageU_spar'])

            print('damageL_spar is:')
            print(unknowns['damageL_spar'])

            print('damageU_te is:')
            print(unknowns['damageU_te'])

            print('damageL_te is:')
            print(unknowns['damageL_te'])

            quit()

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

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

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

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'
        self.naero = naero
        self.nstr = nstr

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
        r_aug = np.concatenate([[0.0], params['r_aero'], [1.0]])

        nstr = len(params['fraction'])
        unknowns['r_str'] = np.zeros(nstr)
        for i in range(nstr):
            j = params['idxj'][i]
            unknowns['r_str'][i] = r_aug[j] + params['fraction'][i]*(r_aug[j+1] - r_aug[j])

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
        self.add_param('idx_cylinder_aero', val=1, desc='first idx in r_aero_unit of non-cylindrical section', pass_by_obj=True)  # constant twist inboard of here
        self.add_param('idx_cylinder_str', val=1, desc='first idx in r_str_unit of non-cylindrical section', pass_by_obj=True)
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
        self.add_output('diameter', shape=1, units='m', desc='dimensional diameter of hub')

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        Rhub = params['hubFraction'] * params['bladeLength']
        Rtip = Rhub + params['bladeLength']

        # setup chord parameterization
        nc = len(params['chord_sub'])
        r_max_chord = Rhub + (Rtip-Rhub)*params['r_max_chord']
        rc = np.linspace(r_max_chord, Rtip, nc-1)
        rc = np.concatenate([[Rhub], rc])
        chord_spline = Akima(rc, params['chord_sub'])

        # setup theta parameterization
        nt = len(params['theta_sub'])
        idxc_aero = params['idx_cylinder_aero']
        idxc_str = params['idx_cylinder_str']
        r_cylinder = Rhub + (Rtip-Rhub)*params['r_aero_unit'][idxc_aero]
        rt = np.linspace(r_cylinder, Rtip, nt)
        theta_spline = Akima(rt, params['theta_sub'])

        # setup precurve parameterization
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

        # self.x_az, self.y_az, self.z_az, cone, s = \
        #     _bem.definecurvature(self.r, self.precurve, self.presweep, 0.0)

        n = len(params['r'])
        dx_dx = np.eye(3*n)

        unknowns['x_az'], x_azd, unknowns['y_az'], y_azd, unknowns['z_az'], z_azd, \
            cone, coned, s, sd = _bem.definecurvature_dv2(params['r'], dx_dx[:, :n],
                params['precurve'], dx_dx[:, n:2*n], params['presweep'], dx_dx[:, 2*n:], 0.0, np.zeros(3*n))

        unknowns['totalCone'] = params['precone'] + np.degrees(cone)
        unknowns['s'] = params['r'][0] + s

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

    def linearize(self, params, unknowns, resids):

        return self.J

        # n = len(self.r)
        # precone = self.precone


        # # azimuthal position
        # self.x_az = -self.r*sind(precone) + self.precurve*cosd(precone)
        # self.z_az = self.r*cosd(precone) + self.precurve*sind(precone)
        # self.y_az = self.presweep


        # # total precone angle
        # x = self.precurve  # compute without precone and add in rotation after
        # z = self.r
        # y = self.presweep

        # totalCone = np.zeros(n)
        # totalCone[0] = math.atan2(-(x[1] - x[0]), z[1] - z[0])
        # totalCone[1:n-1] = 0.5*(np.arctan2(-(x[1:n-1] - x[:n-2]), z[1:n-1] - z[:n-2])
        #     + np.arctan2(-(x[2:] - x[1:n-1]), z[2:] - z[1:n-1]))
        # totalCone[n-1] = math.atan2(-(x[n-1] - x[n-2]), z[n-1] - z[n-2])

        # self.totalCone = precone + np.degrees(totalCone)


        # # total path length of blade  (TODO: need to do something with this.  This should be a geometry preprocessing step just like rotoraero)
        # s = np.zeros(n)
        # s[0] = self.r[0]
        # for i in range(1, n):
        #     s[i] = s[i-1] + math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2 + (z[i] - z[i-1])**2)

        # self.s = s


class DamageLoads(Component):
    def __init__(self, naero, nstr):
        super(DamageLoads, self).__init__()
        self.add_param('rstar', shape=naero+1, desc='nondimensional radial locations of damage equivalent moments')
        self.add_param('Mxb', shape=naero+1, units='N*m', desc='damage equivalent moments about blade c.s. x-direction')
        self.add_param('Myb', shape=naero+1, units='N*m', desc='damage equivalent moments about blade c.s. y-direction')
        self.add_param('theta', shape=nstr, units='deg', desc='structural twist')
        self.add_param('r', shape=nstr, units='m', desc='structural radial locations')

        self.add_output('Mxa', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil c.s. x-direction')
        self.add_output('Mya', shape=nstr, units='N*m', desc='damage equivalent moments about airfoil c.s. y-direction')

    def solve_nonlinear(self, params, unknowns, resids):

        rstar_str = (params['r']-params['r'][0])/(params['r'][-1]-params['r'][0])

        Mxb_str, self.dMxbstr_drstarstr, self.dMxbstr_drstar, self.dMxbstr_dMxb = \
            akima_interp_with_derivs(params['rstar'], params['Mxb'], rstar_str)

        Myb_str, self.dMybstr_drstarstr, self.dMybstr_drstar, self.dMybstr_dMyb = \
            akima_interp_with_derivs(params['rstar'], params['Myb'], rstar_str)

        self.Ma = DirectionVector(Mxb_str, Myb_str, 0.0).bladeToAirfoil(params['theta'])

        # print(self.Ma.y)
        # print(self.Ma.x)
        # quit()

        unknowns['Mxa'] = self.Ma.x
        unknowns['Mya'] = self.Ma.y

    def linearize(self, params, unknowns, resids):
        J = {}

        n = len(params['r'])
        drstarstr_dr = np.zeros((n, n))
        for i in range(1, n-1):
            drstarstr_dr[i, i] = 1.0/(params['r'][-1] - params['r'][0])
        drstarstr_dr[1:, 0] = (params['r'][1:] - params['r'][-1])/(params['r'][-1] - params['r'][0])**2
        drstarstr_dr[:-1, -1] = -(params['r'][:-1] - params['r'][0])/(params['r'][-1] - params['r'][0])**2

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

        # totalCone = self.precone
        # z_az = self.r*cosd(self.precone)
        totalCone = params['totalCone']
        z_az = params['z_az']

        # keep all in blade c.s. then rotate all at end

        # --- aero loads ---

        # interpolate aerodynamic loads onto structural grid
        P_a = DirectionVector(0, 0, 0)
        P_a.x, self.dPax_dr, self.dPax_daeror, self.dPax_daeroPx = akima_interp_with_derivs(params['aeroLoads:r'], params['aeroLoads:Px'], params['r'])
        P_a.y, self.dPay_dr, self.dPay_daeror, self.dPay_daeroPy = akima_interp_with_derivs(params['aeroLoads:r'], params['aeroLoads:Py'], params['r'])
        P_a.z, self.dPaz_dr, self.dPaz_daeror, self.dPaz_daeroPz = akima_interp_with_derivs(params['aeroLoads:r'], params['aeroLoads:Pz'], params['r'])


        # --- weight loads ---

        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -params['rhoA']*params['g'])

        self.P_w = weight.yawToHub(params['tilt']).hubToAzimuth(params['aeroLoads:azimuth'])\
            .azimuthToBlade(totalCone)


        # --- centrifugal loads ---

        # azimuthal c.s.
        Omega = params['aeroLoads:Omega']*RPM2RS
        load = DirectionVector(0.0, 0.0, params['rhoA']*Omega**2*z_az)

        self.P_c = load.azimuthToBlade(totalCone)


        # --- total loads ---
        P = P_a + self.P_w + self.P_c

        # rotate to airfoil c.s.
        theta = np.array(params['theta']) + params['aeroLoads:pitch']
        self.P = P.bladeToAirfoil(theta)

        unknowns['Px_af'] = self.P.x
        unknowns['Py_af'] = self.P.y
        unknowns['Pz_af'] = self.P.z

    def linearize(self, params, unknowns, resids):

        dPwx, dPwy, dPwz = self.P_w.dx, self.P_w.dy, self.P_w.dz
        dPcx, dPcy, dPcz = self.P_c.dx, self.P_c.dy, self.P_c.dz
        dPx, dPy, dPz = self.P.dx, self.P.dy, self.P.dz
        Omega = params['aeroLoads:Omega']*RPM2RS
        z_az = params['z_az']


        dPx_dOmega = dPcx['dz']*params['rhoA']*z_az*2*Omega*RPM2RS
        dPy_dOmega = dPcy['dz']*params['rhoA']*z_az*2*Omega*RPM2RS
        dPz_dOmega = dPcz['dz']*params['rhoA']*z_az*2*Omega*RPM2RS

        dPx_dr = np.diag(self.dPax_dr)
        dPy_dr = np.diag(self.dPay_dr)
        dPz_dr = np.diag(self.dPaz_dr)

        dPx_dprecone = np.diag(dPwx['dprecone'] + dPcx['dprecone'])
        dPy_dprecone = np.diag(dPwy['dprecone'] + dPcy['dprecone'])
        dPz_dprecone = np.diag(dPwz['dprecone'] + dPcz['dprecone'])

        dPx_dzaz = np.diag(dPcx['dz']*params['rhoA']*Omega**2)
        dPy_dzaz = np.diag(dPcy['dz']*params['rhoA']*Omega**2)
        dPz_dzaz = np.diag(dPcz['dz']*params['rhoA']*Omega**2)

        dPx_drhoA = np.diag(-dPwx['dz']*params['g'] + dPcx['dz']*Omega**2*z_az)
        dPy_drhoA = np.diag(-dPwy['dz']*params['g'] + dPcy['dz']*Omega**2*z_az)
        dPz_drhoA = np.diag(-dPwz['dz']*params['g'] + dPcz['dz']*Omega**2*z_az)

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

        theta = params['theta'] + params['pitch']

        dr = DirectionVector(params['dx'], params['dy'], params['dz'])
        self.delta = dr.airfoilToBlade(theta).bladeToAzimuth(params['totalConeTip']) \
            .azimuthToHub(params['azimuth']).hubToYaw(params['tilt'])

        unknowns['tip_deflection'] = params['dynamicFactor'] * self.delta.x

    def linearize(self, params, unknowns, resids):
        J = {}
        dx = params['dynamicFactor'] * self.delta.dx['dx']
        dy = params['dynamicFactor'] * self.delta.dx['dy']
        dz = params['dynamicFactor'] * self.delta.dx['dz']
        dtheta = params['dynamicFactor'] * self.delta.dx['dtheta']
        dpitch = params['dynamicFactor'] * self.delta.dx['dtheta']
        dazimuth = params['dynamicFactor'] * self.delta.dx['dazimuth']
        dtilt = params['dynamicFactor'] * self.delta.dx['dtilt']
        dtotalConeTip = params['dynamicFactor'] * self.delta.dx['dprecone']

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

        theta = params['theta_str'] + params['pitch']

        dr = DirectionVector(params['dx'], params['dy'], params['dz'])
        self.delta = dr.airfoilToBlade(theta)

        precurve_str_out = params['precurve_str0'] + self.delta.x

        self.length0 = params['Rhub0'] + np.sum(np.sqrt((params['precurve_str0'][1:] - params['precurve_str0'][:-1])**2 +
                                            (params['r_str0'][1:] - params['r_str0'][:-1])**2))
        self.length = params['Rhub0'] + np.sum(np.sqrt((precurve_str_out[1:] - precurve_str_out[:-1])**2 +
                                           (params['r_str0'][1:] - params['r_str0'][:-1])**2))

        self.shortening = self.length0/self.length

        self.delta_bladeLength = params['bladeLength0'] * (self.shortening - 1)
        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        self.delta_precurve_sub, self.dpcs_drsubpc0, self.dpcs_drstr0, self.dpcs_ddeltax = \
            interp_with_deriv(params['r_sub_precurve0'], params['r_str0'], self.delta.x)

        unknowns['delta_bladeLength'] = self.delta_bladeLength
        unknowns['delta_precurve_sub'] = self.delta_precurve_sub

    def linearize(self, params, unknowns, resids):

        n = len(params['theta_str'])

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

        precurve_str_out = params['precurve_str0'] + self.delta.x


        for i in range(1, n-1):
            sm0 = math.sqrt((params['precurve_str0'][i] - params['precurve_str0'][i-1])**2 + (params['r_str0'][i] - params['r_str0'][i-1])**2)
            sm = math.sqrt((precurve_str_out[i] - precurve_str_out[i-1])**2 + (params['r_str0'][i] - params['r_str0'][i-1])**2)
            sp0 = math.sqrt((params['precurve_str0'][i+1] - params['precurve_str0'][i])**2 + (params['r_str0'][i+1] - params['r_str0'][i])**2)
            sp = math.sqrt((precurve_str_out[i+1] - precurve_str_out[i])**2 + (params['r_str0'][i+1] - params['r_str0'][i])**2)
            dl0_dprecurvestr0[i] = (params['precurve_str0'][i] - params['precurve_str0'][i-1]) / sm0 \
                - (params['precurve_str0'][i+1] - params['precurve_str0'][i]) / sp0
            dl_dprecurvestr0[i] = (precurve_str_out[i] - precurve_str_out[i-1]) / sm \
                - (precurve_str_out[i+1] - precurve_str_out[i]) / sp
            dl0_drstr0[i] = (params['r_str0'][i] - params['r_str0'][i-1]) / sm0 \
                - (params['r_str0'][i+1] - params['r_str0'][i]) / sp0
            dl_drstr0[i] = (params['r_str0'][i] - params['r_str0'][i-1]) / sm \
                - (params['r_str0'][i+1] - params['r_str0'][i]) / sp

        sfirst0 = math.sqrt((params['precurve_str0'][1] - params['precurve_str0'][0])**2 + (params['r_str0'][1] - params['r_str0'][0])**2)
        sfirst = math.sqrt((precurve_str_out[1] - precurve_str_out[0])**2 + (params['r_str0'][1] - params['r_str0'][0])**2)
        slast0 = math.sqrt((params['precurve_str0'][n-1] - params['precurve_str0'][n-2])**2 + (params['r_str0'][n-1] - params['r_str0'][n-2])**2)
        slast = math.sqrt((precurve_str_out[n-1] - precurve_str_out[n-2])**2 + (params['r_str0'][n-1] - params['r_str0'][n-2])**2)
        dl0_dprecurvestr0[0] = -(params['precurve_str0'][1] - params['precurve_str0'][0]) / sfirst0
        dl0_dprecurvestr0[n-1] = (params['precurve_str0'][n-1] - params['precurve_str0'][n-2]) / slast0
        dl_dprecurvestr0[0] = -(precurve_str_out[1] - precurve_str_out[0]) / sfirst
        dl_dprecurvestr0[n-1] = (precurve_str_out[n-1] - precurve_str_out[n-2]) / slast
        dl0_drstr0[0] = -(params['r_str0'][1] - params['r_str0'][0]) / sfirst0
        dl0_drstr0[n-1] = (params['r_str0'][n-1] - params['r_str0'][n-2]) / slast0
        dl_drstr0[0] = -(params['r_str0'][1] - params['r_str0'][0]) / sfirst
        dl_drstr0[n-1] = (params['r_str0'][n-1] - params['r_str0'][n-2]) / slast

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
        dbl_drhub0 = params['bladeLength0'] * dshort_drhub0
        dbl_dprecurvestr0 = params['bladeLength0'] * dshort_dprecurvestr0
        dbl_drstr0 = params['bladeLength0'] * dshort_drstr0
        dbl_ddx = params['bladeLength0'] * dshort_ddx
        dbl_ddy = params['bladeLength0'] * dshort_ddy
        dbl_ddz = params['bladeLength0'] * dshort_ddz
        dbl_dthetastr = params['bladeLength0'] * dshort_dthetastr
        dbl_dpitch = params['bladeLength0'] * dshort_dpitch

        m = len(params['r_sub_precurve0'])
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
        self.add_param('r_str', shape=nstr, units='m')
        self.add_param('aeroLoads:r', units='m', desc='radial positions along blade going toward tip')
        self.add_param('aeroLoads:Px', units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_param('aeroLoads:Py', units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_param('aeroLoads:Pz', units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_param('totalCone', shape=nstr, units='deg', desc='total cone angle from precone and curvature')
        self.add_param('x_az', shape=nstr, units='m', desc='location of blade in azimuth x-coordinate system')
        self.add_param('y_az', shape=nstr, units='m', desc='location of blade in azimuth y-coordinate system')
        self.add_param('z_az', shape=nstr, units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_param('s', shape=nstr, units='m', desc='cumulative path length along blade')

        self.add_output('root_bending_moment', shape=1, units='N*m', desc='total magnitude of bending moment at root of blade')
        self.add_output('Mxyz', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_output('Fxyz', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')

    def solve_nonlinear(self, params, unknowns, resids):

        r = params['r_str']
        x_az = params['x_az']
        y_az = params['y_az']
        z_az = params['z_az']

        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        Px, self.dPx_dr, self.dPx_dalr, self.dPx_dalPx = interp_with_deriv(r, params['aeroLoads:r'], params['aeroLoads:Px'])
        Py, self.dPy_dr, self.dPy_dalr, self.dPy_dalPy = interp_with_deriv(r, params['aeroLoads:r'], params['aeroLoads:Py'])
        Pz, self.dPz_dr, self.dPz_dalr, self.dPz_dalPz = interp_with_deriv(r, params['aeroLoads:r'], params['aeroLoads:Pz'])

        # print 'al.Pz: ', aL.Pz #check=0

        Fx = np.trapz(Px, params['s'])
        Fy = np.trapz(Py, params['s'])
        Fz = np.trapz(Pz, params['s'])


        # loads in azimuthal c.s.
        P = DirectionVector(Px, Py, Pz).bladeToAzimuth(params['totalCone'])

        # distributed bending load in azimuth coordinate ysstem
        az = DirectionVector(x_az, y_az, z_az)
        Mp = az.cross(P)

        # integrate
        Mx = np.trapz(Mp.x, params['s'])
        My = np.trapz(Mp.y, params['s'])
        Mz = np.trapz(Mp.z, params['s'])

        # get total magnitude
        unknowns['root_bending_moment'] = math.sqrt(Mx**2 + My**2 + Mz**2)

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

        dMx_dMpx, dMx_ds = trapz_deriv(self.Mp.x, params['s'])
        dMy_dMpy, dMy_ds = trapz_deriv(self.Mp.y, params['s'])
        dMz_dMpz, dMz_ds = trapz_deriv(self.Mp.z, params['s'])

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

        drbm_dr = (self.Mx*dMx_dr + self.My*dMy_dr + self.Mz*dMz_dr)/unknowns['root_bending_moment']
        drbm_dalr = (self.Mx*dMx_dalr + self.My*dMy_dalr + self.Mz*dMz_dalr)/unknowns['root_bending_moment']
        drbm_dalPx = (self.Mx*dMx_dalPx + self.My*dMy_dalPx + self.Mz*dMz_dalPx)/unknowns['root_bending_moment']
        drbm_dalPy = (self.Mx*dMx_dalPy + self.My*dMy_dalPy + self.Mz*dMz_dalPy)/unknowns['root_bending_moment']
        drbm_dalPz = (self.Mx*dMx_dalPz + self.My*dMy_dalPz + self.Mz*dMz_dalPz)/unknowns['root_bending_moment']
        drbm_dtotalcone = (self.Mx*dMx_dtotalcone + self.My*dMy_dtotalcone + self.Mz*dMz_dtotalcone)/unknowns['root_bending_moment']
        drbm_dazx = (self.Mx*dMx_dazx + self.My*dMy_dazx + self.Mz*dMz_dazx)/unknowns['root_bending_moment']
        drbm_dazy = (self.Mx*dMx_dazy + self.My*dMy_dazy + self.Mz*dMz_dazy)/unknowns['root_bending_moment']
        drbm_dazz = (self.Mx*dMx_dazz + self.My*dMy_dazz + self.Mz*dMz_dazz)/unknowns['root_bending_moment']
        drbm_ds = (self.Mx*dMx_ds + self.My*dMy_ds + self.Mz*dMz_ds)/unknowns['root_bending_moment']


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

        unknowns['mass_all_blades'] = params['nBlades'] * params['blade_mass']

        Ibeam = params['nBlades'] * params['blade_moment_of_inertia']

        Ixx = Ibeam
        Iyy = Ibeam/2.0  # azimuthal average for 2 blades, exact for 3+
        Izz = Ibeam/2.0
        Ixy = 0
        Ixz = 0
        Iyz = 0  # azimuthal average for 2 blades, exact for 3+

        # rotate to yaw c.s.
        I = DirectionVector(Ixx, Iyy, Izz).hubToYaw(params['tilt'])  # because off-diagonal components are all zero

        unknowns['I_all_blades'] = np.array([I.x, I.y, I.z, Ixy, Ixz, Iyz])
        self.Ivec = I

    def linearize(self, params, unknowns, resids):
        I = self.Ivec

        dIx_dmoi = params['nBlades']*(I.dx['dx'] + I.dx['dy']/2.0 + I.dx['dz']/2.0)
        dIy_dmoi = params['nBlades']*(I.dy['dx'] + I.dy['dy']/2.0 + I.dy['dz']/2.0)
        dIz_dmoi = params['nBlades']*(I.dz['dx'] + I.dz['dy']/2.0 + I.dz['dz']/2.0)

        J = {}
        J['mass_all_blades', 'blade_mass'] = params['nBlades']
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

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        self.turbine_class = params['turbine_class']

        if self.turbine_class == 'I':
            Vref = 50.0
        elif self.turbine_class == 'II':
            Vref = 42.5
        elif self.turbine_class == 'III':
            Vref = 37.5
        elif self.turbine_class == 'IV':
            Vref = 30.0

        unknowns['V_mean'] = 0.2*Vref
        unknowns['V_extreme'] = 1.4*Vref
        unknowns['V_extreme_full'][0] = 1.4*Vref # for extreme cases TODO: check if other way to do
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

        n = float(params['nBlades'])
        unknowns['T_extreme'] = (params['T'][1] + params['T'][1]*(n-1)) / n #changing to all feathered since the stuck case overestimating load
        #unknowns['Q_extreme'] = (self.Q[1] + self.Q[1]*(n-1)) / n #TODO - commenting out since extreme torque analysis is suspect
        unknowns['Q_extreme'] = 0.0 #TODO - temporary setting of survival torque to 0

    def linearize(self, params, unknowns, resids):
        n = float(params['nBlades'])

        J = {}
        J['T_extreme', 'T'] = np.array([[0.0, 1.0]])
        J['T_extreme', 'Q'] = np.array([[0.0, 0.0]])
        J['Q_extreme', 'T'] = np.array([[0.0, 0.0]])
        J['Q_extreme', 'Q'] = np.array([[0.0, 0.0]])

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

        if params['turbulence_class'] == 'A':
            Iref = 0.16
        elif params['turbulence_class'] == 'B':
            Iref = 0.14
        elif params['turbulence_class'] == 'C':
            Iref = 0.12

        c = 2.0

        self.sigma = c * Iref * (0.072*(params['V_mean']/c + 3)*(params['V_hub']/c - 4) + 10)
        unknowns['V_gust'] = params['V_hub'] + params['std']*self.sigma
        self.Iref = Iref
        self.c = c

    def linearize(self, params, unknowns, resids):
        Iref = self.Iref
        c = self.c

        J = {}
        J['V_gust', 'V_mean'] = params['std']*(c*Iref*0.072/c*(params['V_hub']/c - 4))
        J['V_gust', 'V_hub'] = 1.0 + params['std']*(c*Iref*0.072*(params['V_mean']/c + 3)/c)

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
        self.add_output('azimuth', shape=1, units='deg') #TODO: Check

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['Uhub'] = params['Vfactor'] * params['Vrated']
        unknowns['Omega'] = params['control:tsr']*unknowns['Uhub']/params['R']*RS2RPM
        unknowns['pitch'] = params['control:pitch']
        unknowns['azimuth'] = 0.0

    def linearize(self, params, unknowns, resids):

        J = {}
        J['Uhub', 'control:tsr'] = 0.0
        J['Uhub', 'Vrated'] = params['Vfactor']
        J['Uhub', 'R'] = 0.0
        J['Omega', 'control:tsr'] = unknowns['Uhub']/params['R']*RS2RPM
        J['Omega', 'Vrated'] = params['control:tsr']*params['Vfactor']/params['R']*RS2RPM
        J['Omega', 'R'] = -params['control:tsr']*unknowns['Uhub']/params['R']**2*RS2RPM
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
        # additional drivetrain moments output
        self.add_param('Mxyz_0_in', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_param('Mxyz_120_in', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_param('Mxyz_240_in', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_param('Fxyz_0_in', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')
        self.add_param('Fxyz_120_in', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')
        self.add_param('Fxyz_240_in', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')
        self.add_param('TotalCone_in', shape=1, units='rad', desc='total cone angle for blades at rated')
        self.add_param('Pitch_in', shape=1, units='rad', desc='pitch angle at rated')

        # self.add_param('max_tip_def_in', shape=1, units='m', desc='maximum tip deflection')
        self.add_param('max_tip_def_in', shape=1, desc='maximum tip deflection')

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
        # additional drivetrain moments output
        self.add_output('Mxyz_0', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_output('Mxyz_120', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_output('Mxyz_240', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_output('Fxyz_0', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')
        self.add_output('Fxyz_120', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')
        self.add_output('Fxyz_240', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')
        self.add_output('TotalCone', shape=1, units='rad', desc='total cone angle for blades at rated')
        self.add_output('Pitch', shape=1, units='rad', desc='pitch angle at rated')

        self.add_output('max_tip_deflection', shape=1, units='m', desc='maximum tip deflection')


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
        unknowns['delta_precurve_sub_out'] = params['delta_precurve_sub_out_in']

        unknowns['Mxyz_0'] = params['Mxyz_0_in']
        unknowns['Mxyz_120'] = params['Mxyz_120_in']
        unknowns['Mxyz_240'] = params['Mxyz_240_in']
        unknowns['Fxyz_0'] = params['Fxyz_0_in']
        unknowns['Fxyz_120'] = params['Fxyz_120_in']
        unknowns['Fxyz_240'] = params['Fxyz_240_in']
        unknowns['TotalCone'] = params['TotalCone_in']
        unknowns['Pitch'] = params['Pitch_in']

        unknowns['max_tip_deflection'] = params['max_tip_def_in']

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

        J['Mxyz_0', 'Mxyz_0_in'] = 1
        J['Mxyz_120', 'Mxyz_120_in'] = 1
        J['Mxyz_240', 'Mxyz_240_in'] = 1
        J['Fxyz_0', 'Fxyz_0_in'] = 1
        J['Fxyz_120', 'Fxyz_120_in'] = 1
        J['Fxyz_240', 'Fxyz_240'] = 1
        J['TotalCone', 'TotalCone_in'] = 1
        J['Pitch', 'Pitch_in'] = 1

        J['max_tip_deflection', 'max_tip_def_in'] = 1

        return J

class CreateFASTConstraints(Component):
    def __init__(self, naero, nstr, FASTinfo, WNDfile_List, caseids):
        super(CreateFASTConstraints, self).__init__()

        self.caseids = caseids
        self.WNDfile_List = WNDfile_List
        self.dT = FASTinfo['dT']
        self.description = FASTinfo['description']
        self.path = FASTinfo['path']
        self.opt_dir = FASTinfo['opt_dir']

        self.train_sm = FASTinfo['train_sm']
        if self.train_sm:
            self.sm_dir = FASTinfo['sm_dir']

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']
        self.Run_Once = FASTinfo['Run_Once']

        self.dir_saved_plots = FASTinfo['dir_saved_plots']

        self.check_results = FASTinfo['check_results']
        self.check_sgp_spline = FASTinfo['check_sgp_spline']
        self.check_peaks = FASTinfo['check_peaks']
        self.check_rainflow = FASTinfo['check_rainflow']
        self.check_rm_time = FASTinfo['check_rm_time']

        # only works if check_damage is also set as 'true
        self.check_nom_DEM_damage = FASTinfo['check_nom_DEM_damage']

        self.sgp = FASTinfo['sgp']

        self.wndfiletype = FASTinfo['wnd_type_list']
        self.Tmax_turb = FASTinfo['Tmax_turb']
        self.Tmax_nonturb = FASTinfo['Tmax_nonturb']
        self.turb_sf = FASTinfo['turb_sf']
        self.rm_time = FASTinfo['rm_time']

        self.m_value = FASTinfo['m_value']

        self.add_param('cfg_master', val=dict(), pass_by_obj=False)

        self.add_param('rstar_damage', shape=naero + 1, desc='nondimensional radial locations of damage equivalent moments')

        self.add_param('initial_str_grid', shape=nstr, desc='initial structural grid on unit radius')
        self.add_param('initial_aero_grid', shape=naero, desc='initial structural grid on unit radius')

        for i in range(0, len(caseids)):
            self.add_param(caseids[i], val=dict())

        # DEMs
        self.add_output('DEMx', val=np.zeros(naero+1))
        self.add_output('DEMy', val=np.zeros(naero+1))

        # Tip Deflection
        self.add_output('max_tip_def', val=0.0)

        # Structure
        self.add_output('Edg_max',val=np.zeros(nstr))
        self.add_output('Flp_max',val=np.zeros(nstr))

    def solve_nonlinear(self, params, unknowns, resids):

        # === Check Results === #
        resultsdict = params[self.caseids[0]]
        if self.check_results:
            # bm_param = 'Spn4MLxb1'
            bm_param = ['RootMyb1', 'OoPDefl1']
            bm_param_units = ['kN*m', 'm']
            for i in range(0, len(bm_param)):
                plt.figure()
                plt.plot(resultsdict[bm_param[i]])
                plt.xlabel('Simulation Step')
                plt.ylabel(bm_param[i] + ' (' + bm_param_units[i] + ')')
                plt.title(bm_param[i])
                plt.savefig(self.dir_saved_plots + '/plots/param_plots/' + bm_param[i] + '_test.png')

                plt.show()

            quit()

        # total number of virtual strain gages
        tot_BldGagNd = []
        for i in range(0, len(self.BldGagNd)):
            for j in range(0, self.NBlGages[i]):
                tot_BldGagNd.append(self.BldGagNd[i][j])

        total_num_bl_gages = 0
        max_gage = 0
        for i in range(0, len(self.NBlGages)):
            total_num_bl_gages += self.NBlGages[i]
            max_gage = max(max_gage, max(self.BldGagNd[i]))

        # === DEM / structural calculations === #

        DEMx_master_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])
        DEMy_master_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])

        # maxes of DEMx, DEMy, (will be promoted)
        DEMx_max = np.zeros([1 + total_num_bl_gages, 1])
        DEMy_max = np.zeros([1 + total_num_bl_gages, 1])

        Edg_max_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])
        Flp_max_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])

        # maxes of Edg, Flp, (will be promoted)
        Edg_max = np.zeros([1 + total_num_bl_gages, 1])
        Flp_max = np.zeros([1 + total_num_bl_gages, 1])

        # === extrapolated loads variables === #
        # peaks master

        peaks_wnd_x = dict()
        for j in range(len(self.WNDfile_List)):
            peaks_wnd_x[str(j+1)] = dict()
            peaks_wnd_x[str(j+1)]['root'] = []
            for i in range(0, total_num_bl_gages):
                peaks_wnd_x[str(j+1)]['bld_gage_' + str(tot_BldGagNd[i])] = []

        peaks_wnd_y = dict()
        for j in range(len(self.WNDfile_List)):
            peaks_wnd_y[str(j+1)] = dict()
            peaks_wnd_y[str(j+1)]['root'] = []
            for i in range(0, total_num_bl_gages):
                peaks_wnd_y[str(j+1)]['bld_gage_' + str(tot_BldGagNd[i])] = []

        # peaks max (will be promoted)
        peaks_max_x = dict()
        peaks_max_x['root'] = []
        for i in range(0, total_num_bl_gages):
            peaks_max_x['bld_gage_' + str(tot_BldGagNd[i])] = []

        peaks_max_y = dict()
        peaks_max_y['root'] = []
        for i in range(0, total_num_bl_gages):
            peaks_max_y['bld_gage_' + str(tot_BldGagNd[i])] = []

        # === cycle through each set of strain gages (k) and each wind file (i) === #
        if self.train_sm:
            FAST_opt_dir = self.sm_dir
        else:
            FAST_opt_dir = self.opt_dir


        for k in range(0, len(self.NBlGages)):

            for i in range(0 + 1, len(self.WNDfile_List) + 1):

                spec_caseid = k*len(self.WNDfile_List) + i - 1


                # === extrapolated loads variables === #
                # peaks master
                peaks_master_x = dict()
                peaks_master_x['root'] = []
                for i_index in range(0, total_num_bl_gages):
                    peaks_master_x['bld_gage_' + str(tot_BldGagNd[i_index])] = []

                peaks_master_y = dict()
                peaks_master_y['root'] = []
                for i_index in range(0, total_num_bl_gages):
                    peaks_master_y['bld_gage_' + str(tot_BldGagNd[i_index])] = []

                # === naming conventions === #

                spec_caseid = k*len(self.WNDfile_List)+(i-1)
                resultsdict = params[self.caseids[spec_caseid]]

                FAST_wnd_directory = FAST_opt_dir + '/' + 'sgp' + str(self.sgp[k]) + '/' + self.caseids[spec_caseid]

                # === rainflow calculation files === #

                # files = [FAST_wnd_directory + '/fst_runfile.outb']
                files = [FAST_wnd_directory + '/fst_runfile.out']

                # read titles of file, since they don't seem to be in order
                file_rainflow = open(files[0])
                line_rainflow = file_rainflow.readlines()

                # extract names fron non-binary FAST output file
                name_line = 6
                str_val = re.findall("\w+", line_rainflow[name_line])

                # create output_array (needed for rainflow calculation)
                output_array = []

                # make RootMxb1 first in output_array
                for j in range (0,len(str_val)):
                    if str_val[j] == 'RootMxb1':
                        output_array.append(j)

                # make Spn1MLxb1 next in output array
                for l in range(0,self.NBlGages[k]):
                    for j in range(0,len(str_val)):
                        if str_val[j] == 'Spn{0}MLxb1'.format(str(l+1)):
                            output_array.append(j)

                # make RootMyb1 next in output_array
                for j in range(0, len(str_val)):
                    if str_val[j] == 'RootMyb1':
                        output_array.append(j)

                # make Spn1MLyb1 next in output array
                for l in range(0, self.NBlGages[k]):
                    for j in range(0, len(str_val)):
                        if str_val[j] == 'Spn{0}MLyb1'.format(str(l+1)):
                            output_array.append(j)

                ## these are the powers that the cycles are raised to in order to get the final fatigue.
                ## they are properties of the materials of the corresponding fields (so they are tied
                ## to the indices in "output_array"

                SNslope = np.zeros([1,len(output_array)])
                for index in range(0,len(output_array)):
                    for j in range(0,1):
                        SNslope[j,index] = self.m_value

                if self.wndfiletype[i-1] == 'turb':
                    Tmax = self.Tmax_turb
                else:
                    Tmax = self.Tmax_nonturb

                # === perform rainflow calculations === #
                # print(files)
                from rainflow import do_rainflow
                allres, peaks_list, orig_data, rm_data, data_name = \
                    do_rainflow(files, output_array, SNslope, self.dir_saved_plots, Tmax, self.dT, self.rm_time, self.check_rm_time)

                allres = allres[0]
                a = allres

                # print(a)
                # quit()

                # === rainflow sanity check === #
                # TODO: make an option to choose which .wnd file to check
                if self.check_rainflow:
                    
                    n = 0;
                    for m in output_array:

                        FAST_b = orig_data[:,m]
                        FAST_b_time = orig_data[:,0]
                        FAST_rm = rm_data[:,m]
                        FAST_rm_time = rm_data[:,0]

                        plt.figure()
                        plt.plot(FAST_b_time,FAST_b,'--',label='all data output')
                        plt.plot(FAST_rm_time,FAST_rm,label='used data output')

                        plt.xlabel('Time Step (s)')
                        plt.ylabel('Data')
                        # plt.title(data_name[m] + '; DEM = ' + str(a[n][0]) + ' kN*m')
                        plt.title(data_name[m] + '; DEM = ' + str(a[n][0]) + ' kN*m')

                        plt.legend()
                        # plt.savefig(self.dir_saved_plots + '/rainflow_check/' + data_name[m] + '.eps')
                        plt.savefig(self.dir_saved_plots + '/plots/rainflow_check/' + data_name[m] + '.png')
                        # plt.show()
                        plt.close()

                        n = n+1
                    quit()

                # peaks info
                peaks_array = dict()
                # for i in range(0,2*(1+total_num_bl_gages)):
                # for m in range(0, 16 * len(self.BldGagNd)):
                #     peaks_array['value' + str(m)] = []

                # create peaks master file
                for j in range(0,len(output_array)):
                    l = output_array[j]
                    peaks_array[str_val[l]] = []
                    peaks_array[str_val[l]].append(peaks_list[j].tolist())

                for j in range(0, len(peaks_array['RootMxb1'])):
                    peaks_master_x['root'].append(peaks_array['RootMxb1'][j])
                for j in range(0, len(peaks_array['RootMyb1'])):
                    peaks_master_y['root'].append(peaks_array['RootMyb1'][j])
                for j in range(0,len(peaks_array)):
                    for l in range(0, self.NBlGages[k]):
                        for m in range(0, len(peaks_array['Spn{0}MLxb1'.format(str(l+1))])):
                            peaks_master_x['bld_gage_' + str(self.BldGagNd[k][l])].append(peaks_array['Spn{0}MLxb1'.format(str(l+1))][m])
                        for m in range(0, len(peaks_array['Spn{0}MLyb1'.format(str(l + 1))])):
                            peaks_master_y['bld_gage_' + str(self.BldGagNd[k][l])].append(peaks_array['Spn{0}MLyb1'.format(str(l+1))][m])

                # addition of turbulent safety factor
                # if self.wndfiletype[i - 1] == 'turb':
                if self.wndfiletype[spec_caseid] == 'turb':
                    a = a*self.turb_sf

                # create xRoot, xDEM, yRoot, and yDEM
                xRoot = a[0][0]

                xDEM = []
                for l in range(0,self.NBlGages[k]):
                    xDEM.append(a[l+1][0])

                yRoot = a[1+self.NBlGages[k]][0]
                yDEM = []
                for l in range(0,self.NBlGages[k]):
                    yDEM.append(a[l+2+self.NBlGages[k]][0])

                for j in range(0,self.NBlGages[k]):
                    if j == 0:
                        Edg_param = 'RootMxb1'
                        Flp_param = 'RootMyb1'
                    else:
                        Edg_param = 'Spn{0}MLxb1'.format(j)
                        Flp_param = 'Spn{0}MLyb1'.format(j)

                    Edg_max_val = abs(max(resultsdict[Edg_param]))
                    Edg_min_val = abs(min(resultsdict[Edg_param]))

                    Flp_max_val = abs(max(resultsdict[Flp_param]))
                    Flp_min_val = abs(min(resultsdict[Flp_param]))

                    if j == 0:
                        Edg_max_array[i-1][0] = max(Edg_max_val, Edg_min_val)
                        Flp_max_array[i-1][0] = max(Flp_max_val, Flp_min_val)
                    else:
                        Edg_max_array[i-1][self.BldGagNd[k][j]] = max(Edg_max_val, Edg_min_val)
                        Flp_max_array[i-1][self.BldGagNd[k][j]] = max(Flp_max_val, Flp_min_val)

                # take max at each position
                Edg_max[0] = max(Edg_max[0], Edg_max_array[i - 1][0])
                Flp_max[0] = max(Flp_max[0], Flp_max_array[i - 1][0])
                for j in range(1, len(self.BldGagNd[k])+1):

                    for l in range(0,len(tot_BldGagNd)):
                        if tot_BldGagNd[l] == self.BldGagNd[k][j - 1]:
                            max_it = l

                    Edg_max[max_it] = max(Edg_max[max_it], Edg_max_array[i - 1][self.BldGagNd[k][j - 1]])
                    Flp_max[max_it] = max(Flp_max[max_it], Flp_max_array[i - 1][self.BldGagNd[k][j - 1]])

                # Add DEMs to master arrays
                DEMx_master_array[i-1][0] = xRoot
                DEMy_master_array[i-1][0] = yRoot

                DEMx_master_array[i-1][self.BldGagNd[k]] = xDEM
                DEMy_master_array[i-1][self.BldGagNd[k]] = yDEM

                # take max at each position
                DEMx_max[0] = max(DEMx_max[0], DEMx_master_array[i - 1][0])
                DEMy_max[0] = max(DEMy_max[0], DEMy_master_array[i - 1][0])

                for j in range(1, len(self.BldGagNd[k]) + 1):

                    for l in range(0,len(tot_BldGagNd)):
                        if tot_BldGagNd[l] == self.BldGagNd[k][j - 1]:
                            max_it = l+1

                    DEMx_max[max_it] = max(DEMx_max[max_it], DEMx_master_array[i-1][self.BldGagNd[k][j-1]])
                    DEMy_max[max_it] = max(DEMy_max[max_it], DEMy_master_array[i-1][self.BldGagNd[k][j-1]])

                if self.Run_Once:

                    # save root values

                    # xRoot file
                    xRoot_file = FAST_wnd_directory + '/' + 'xRoot.txt'
                    file_xroot = open(xRoot_file, "w")

                    # yRoot file
                    yRoot_file = FAST_wnd_directory + '/' + 'yRoot.txt'
                    file_yroot = open(yRoot_file, "w")

                    # write to xDEM file
                    file_xroot.write(str(xRoot) + '\n')
                    file_xroot.close()

                    # write to yDEM file
                    file_yroot.write(str(yRoot) + '\n')
                    file_yroot.close()

                    # save xDEM, yDEM

                    # xDEM file
                    xDEM_file = FAST_wnd_directory + '/' + 'xDEM_' + str(self.BldGagNd[k][0]) + '.txt'
                    file_x = open(xDEM_file, "w")

                    # yDEM file
                    yDEM_file = FAST_wnd_directory + '/' + 'yDEM_' + str(self.BldGagNd[k][0]) + '.txt'
                    file_y = open(yDEM_file, "w")

                    for j in range(0,len(xDEM)):

                        # write to xDEM file
                        file_x.write(str(xDEM[j]) + '\n')

                        # write to yDEM file
                        file_y.write(str(yDEM[j]) + '\n')

                    file_x.close()
                    file_y.close()

                # === turbulent extreme moment extrapolation === #
                if self.wndfiletype[spec_caseid] == 'turb':
                # if self.wndfiletype[i - 1] == 'turb':
                    from scipy.stats import norm

                    for j_index in range(0, 2):  # for both x,y bending moments

                        if j_index == 1:
                            peaks_master = peaks_master_x
                            data_type = 'x'
                        else:
                            peaks_master = peaks_master_y
                            data_type = 'y'

                        for i_index in range(0, total_num_bl_gages+1): # +1 for root bending moment

                            if i_index == 0:
                                data_name = 'root'
                            else:
                                data_name = 'bld_gage_' + str(tot_BldGagNd[i_index-1])
                            root_peaks = peaks_master[data_name]

                            # get data
                            rp_list = []
                            for m_index in range(0, len(root_peaks)):
                                for j in range(0, len(root_peaks[m_index])):
                                    rp_list.append(root_peaks[m_index][j])

                            # two distributions
                            if data_type == 'x':

                                # normal distribution
                                norm_dist = True
                                if norm_dist:
                                    # get fit
                                    data = rp_list

                                    if len(data) == 0:
                                        pass
                                    else:

                                        data_min = min(data)
                                        data_max = max(data)

                                        data1_subset = []
                                        data2_subset = []

                                        for k_index in range(0, len(data)):
                                            if abs(data[k_index] - data_min) < abs(data[k_index] - data_max):
                                                data1_subset.append(data[k_index])
                                            else:
                                                data2_subset.append(data[k_index])

                                        for l in range(2):

                                            if l == 0:
                                                data = data1_subset
                                            else:
                                                data = data2_subset

                                            plt.figure()
                                            # Fit a normal distribution to the data:
                                            mu, std = norm.fit(data)

                                            # Plot the histogram.
                                            plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

                                            # Plot the PDF.
                                            xmin, xmax = plt.xlim()
                                            x = np.linspace(xmin, xmax, 100)
                                            p = norm.pdf(x, mu, std)
                                            plt.plot(x, p, 'k', linewidth=2)
                                            plt.title(data_name + data_type + ' Turbulent Peaks, Normal Dist. Fit, data subset: ' + str(l+1))
                                            plt.ylabel('Normalized Frequency')
                                            plt.xlabel('Load Bins (kN*m)')

                                            # add extrapolated, extreme moment
                                            spec_sd = 3.7*10.0**(-8.0)

                                            extreme_mom = max(abs(mu + std*norm.ppf(spec_sd)), abs(mu + std*norm.ppf(1.0-spec_sd)))

                                            # checks max, since we're doing it for both data subsets
                                            peaks_wnd_x[str(i)][data_name] = max(extreme_mom, peaks_wnd_x[str(i)][data_name])


                                            # show plot, quit routine
                                            if self.check_peaks:
                                                plt.savefig(
                                                    self.dir_saved_plots + '/plots/hist_' + str(data_name) + str(data_type) + '_' + str(l) + '.png')
                                                plt.show()
                                                # quit()
                                            plt.close()

                            # one distribution
                            elif data_type == 'y':

                                # normal distribution
                                norm_dist = True
                                if norm_dist:
                                    # get fit
                                    data = rp_list

                                    if len(data) == 0:
                                        pass
                                    else:

                                        plt.figure()
                                        # Fit a normal distribution to the data:
                                        mu, std = norm.fit(data)

                                        # Plot the histogram.
                                        plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

                                        # Plot the PDF.
                                        xmin, xmax = plt.xlim()
                                        x = np.linspace(xmin, xmax, 100)
                                        p = norm.pdf(x, mu, std)
                                        plt.plot(x, p, 'k', linewidth=2)
                                        plt.title(data_name + data_type + ' Turbulent Peaks, Normal Dist. Fit')
                                        plt.ylabel('Normalized Frequency')
                                        plt.xlabel('Load Bins (kN*m)')

                                        # add extrapolated, extreme moment
                                        spec_sd = 3.7 * 10.0 ** (-8.0)

                                        extreme_mom = max(abs(mu + std * norm.ppf(spec_sd)),
                                                          abs(mu + std * norm.ppf(1.0 - spec_sd)))

                                        peaks_wnd_y[str(i)][data_name] = extreme_mom

                                        # show plot, quit routine
                                        if self.check_peaks:
                                            plt.savefig(
                                                self.dir_saved_plots + '/plots/hist_' + str(data_name) + str(data_type) + '.png')
                                            plt.show()
                                            # quit()
                                        plt.close()

        # === determine maximum of extreme extrapolated turublent loads === #
        for j in range(len(Edg_max)):

            cur_var_x = []
            cur_var_y = []

            for i in range(len(self.WNDfile_List)):

                if j == 0:
                    cur_var_x.append(peaks_wnd_x[str(i+1)]['root'])
                    cur_var_y.append(peaks_wnd_y[str(i+1)]['root'])
                else:
                    cur_var_x.append(peaks_wnd_x[str(i+1)]['bld_gage_' + str(tot_BldGagNd[j-1])])
                    cur_var_y.append(peaks_wnd_y[str(i+1)]['bld_gage_' + str(tot_BldGagNd[j-1])])

            if j == 0:
                peaks_max_x['root'] = max(cur_var_x)
                peaks_max_y['root'] = max(cur_var_y)
            else:
                peaks_max_x['bld_gage_' + str(tot_BldGagNd[j-1])] = max(cur_var_x)
                peaks_max_y['bld_gage_' + str(tot_BldGagNd[j-1])] = max(cur_var_y)

        if self.check_peaks:
            quit()

        # compare peaks_max_x, peaks_max_y with Edg_max, Flp_max
        for i in range(0, len(Edg_max)):
            if i == 0:
                Edg_max[i] = max(Edg_max[i],peaks_max_x['root'])
                Flp_max[i] = max(Flp_max[i],peaks_max_y['root'])
            else:
                Edg_max[i] = max(Edg_max[i], peaks_max_x['bld_gage_' + str(tot_BldGagNd[i-1])])
                Flp_max[i] = max(Flp_max[i], peaks_max_y['bld_gage_' + str(tot_BldGagNd[i-1])])


        # === structural akima spline === #

        spline_extr = params['initial_str_grid']
        spline_pos = params['rstar_damage'][np.insert(tot_BldGagNd, 0, 0.0)]

        Edg_max_spline = Akima(spline_pos, Edg_max)

        unknowns['Edg_max'] = Edg_max_spline.interp(spline_extr)[0]*10.0**3.0 # kN*m to N*m

        Flp_max_spline = Akima(spline_pos, Flp_max)
        unknowns['Flp_max'] = Flp_max_spline.interp(spline_extr)[0]*10.0**3.0 # kN*m to N*m

        # === DEM akima spline === #

        spline_extr = params['rstar_damage']
        spline_pos = params['rstar_damage'][np.insert(tot_BldGagNd,0,0.0)]

        # print(spline_extr)
        # print([np.insert(tot_BldGagNd,0,0.0)])
        # print(spline_pos)
        # print(DEMy_max)
        # quit()

        DEMx_spline = Akima(spline_pos,DEMx_max)
        unknowns['DEMx'] = DEMx_spline.interp(spline_extr)[0]

        DEMy_spline = Akima(spline_pos,DEMy_max)
        unknowns['DEMy'] = DEMy_spline.interp(spline_extr)[0]

        # kN*m to N*m
        unknowns['DEMx'] *= 1000.0
        unknowns['DEMy'] *= 1000.0


        if self.check_nom_DEM_damage:
            unknowns['DEMx'] = 1e3 * np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
                                                  1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002,
                                                  2.4458E+002, 1.6339E+002,
                                                  1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000,
                                                  4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
            unknowns['DEMy'] = 1e3 * np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
                                                  1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002,
                                                  6.2449E+002, 4.5229E+002,
                                                  3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001,
                                                  1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction

        # TODO: create plot check to see if akima spline can't be fitted better with other strain gage positions
        if self.check_sgp_spline:

            # plot splines
            spline_plot = np.linspace(0,1,200)
            DEMx_spline_plot = Akima(spline_pos, DEMx_max)
            DEMx_plot = DEMx_spline_plot.interp(spline_plot)[0]

            DEMy_spline = Akima(spline_pos, DEMy_max)
            DEMy_plot = DEMy_spline.interp(spline_plot)[0]

            # DEMx
            plt.figure()
            plt.plot(spline_pos, DEMx_max, '*', label='points')
            # plt.plot(spline_extr, unknowns['DEMx'], label='spline')
            plt.plot(spline_plot, DEMx_plot, label='spline')

            plt.xlabel('Unit Radius of Blade')
            plt.ylabel('DEMx (kN*m)')
            plt.title('DEMx spline - ' + str(total_num_bl_gages) + ' strain gages')
            plt.legend()

            plt.savefig(self.dir_saved_plots + "/plots/DEM_plots/DEMx_nsg" + str(total_num_bl_gages) + ".png")
            print('saved at ')
            print(self.dir_saved_plots + "/plots/DEM_plots/DEMx_nsg" + str(total_num_bl_gages) + ".png")
            print(unknowns['DEMx'])
            print(spline_pos)

            # DEMy
            plt.figure()
            plt.plot(spline_pos, DEMy_max, '*', label='points')
            # plt.plot(spline_extr, unknowns['DEMy'], label='spline')
            plt.plot(spline_plot, DEMy_plot, label='spline')

            plt.xlabel('Unit Radius of Blade')
            plt.ylabel('DEMy (N*m)')
            plt.title('DEMy spline - ' + str(total_num_bl_gages) + ' strain gages')
            plt.legend()

            plt.savefig(self.dir_saved_plots + "/plots/DEM_plots/DEMy_nsg" + str(total_num_bl_gages) + ".png")
            print('saved at ')
            print(self.dir_saved_plots + "/plots/DEM_plots/DEMy_nsg" + str(total_num_bl_gages) + ".png")
            print(unknowns['DEMy'])
            print(spline_pos)

            plt.show()

            quit()

        # === Tip Deflection extraction === #

        max_tip_def_array = np.zeros([len(self.caseids), 1])

        # for i in range(0+1, len(self.caseids)+1):
        for i in range(0, len(self.caseids)):
            # resultsdict = params['WNDfile{0}'.format(i)]
            resultsdict = params[self.caseids[i]]

            maxdeflection = abs(max(resultsdict['OoPDefl1']))
            mindeflection = abs(min(resultsdict['OoPDefl1']))

            max_tip_def_array[i - 1] = max(maxdeflection, mindeflection)

        # for now, just create single constraint
        unknowns['max_tip_def'] = max(max_tip_def_array)


class Calculate_FAST_sm_training_points(Component):
    def __init__(self, FASTinfo, naero, nstr):
        super(Calculate_FAST_sm_training_points, self).__init__()

        # === design variables === #
        self.add_param('r_max_chord', val=0.0)
        self.add_param('chord_sub',  val=np.zeros(4))
        self.add_param('theta_sub', val=np.zeros(4))
        self.add_param('sparT', val=np.zeros(5))
        self.add_param('teT', val=np.zeros(5))

        # === outputs from createFASTconstraints === #
        self.add_param('DEMx', shape=18, desc='DEMx')
        self.add_param('DEMy', shape=18, desc='DEMy')

        self.add_param('Edg_max', shape=nstr, desc='FAST Edg_max')
        self.add_param('Flp_max', shape=nstr, desc='FAST Flp_max')

        self.add_param('max_tip_def', shape=1, desc='FAST calculated maximum tip deflection')

        # === surrogate model options === #
        self.FASTinfo = FASTinfo

        self.training_point_dist = FASTinfo['training_point_dist'] # 'lhs', 'linear'

        self.sm_var_spec = FASTinfo['sm_var_spec']
        self.sm_var_index = FASTinfo['sm_var_index']
        self.sm_var_names = FASTinfo['sm_var_names']

        if self.training_point_dist == 'lhs':
            self.num_pts = FASTinfo['num_pts']

        else:
            self.sm_var_max = FASTinfo['sm_var_max']

        self.sm_var_file = FASTinfo['sm_var_file']
        self.sm_DEM_file = FASTinfo['sm_DEM_file']
        self.sm_load_file = FASTinfo['sm_load_file']
        self.sm_def_file = FASTinfo['sm_def_file']

        self.opt_dir = FASTinfo['opt_dir']


        self.var_filename = self.opt_dir + '/' + self.sm_var_file
        self.DEM_filename = self.opt_dir + '/' + self.sm_DEM_file
        self.load_filename = self.opt_dir + '/' + self.sm_load_file
        self.def_filename = self.opt_dir + '/' + self.sm_def_file

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']

        total_num_bl_gages = 0
        for i in range(0, len(self.NBlGages)):
            total_num_bl_gages += self.NBlGages[i]

        # to nondimensionalize chord_sub
        self.add_param('bladeLength', shape=1, desc='Blade length')
        self.nondimensionalize_chord = FASTinfo['nondimensionalize_chord']

    def solve_nonlinear(self, params, unknowns, resids):

        # print('new surrogate model params:')
        # print('Edg_max')
        # print(params['Edg_max'])
        # print('Flp_max')
        # print(params['Flp_max'])
        # print('max_tip_def')
        # print(params['max_tip_def'])
        # quit()

        def replace_line(file_name, line_num, text):
            lines = open(file_name, 'r').readlines()
            lines[line_num] = text
            out = open(file_name, 'w')
            out.writelines(lines)
            out.close()

        # === variable and output files === #

        # if training points are laid out linearly
        if self.training_point_dist == 'linear':

            # total variations
            tv = []
            for i in range(0, len(self.sm_var_max)):
                for j in range(0, len(self.sm_var_max[i])):
                    tv.append(self.sm_var_max[i][j])

            # specific variation
            sv = []
            for i in range(0, len(self.sm_var_spec)):
                for j in range(0, len(self.sm_var_spec[i])):
                    sv.append(self.sm_var_spec[i][j])

            # check if output file exists (if it doesn't, create it)
            if not (os.path.isfile(self.DEM_filename)):
                # create file
                f = open(self.DEM_filename,"w+")

                # write a header
                header0 = 'variable points: '
                for i in range(0, len(self.sm_var_names)):

                    header0 += self.sm_var_names[i]
                    for j in range(0, len(self.sm_var_index[i])):
                        header0 += '_' + str(self.sm_var_index[i][j])

                    header0 += ' '

                    for j in range(0, len(self.sm_var_spec[i])):
                        header0 += str(self.sm_var_max[i][j]) + ' '

                f.write(header0+'\n')

                # total variation product
                n_tv = np.prod(tv)

                for i in range(0,n_tv):
                    f.write('-- place holder --'+'\n')

                f.close()

            # variable file
            if not (os.path.isfile(self.var_filename)):
                # create file
                f = open(self.var_filename, "w+")

                # write a header
                header0 = 'variable points: '
                for i in range(0, len(self.sm_var_names)):

                    header0 += self.sm_var_names[i]
                    for j in range(0, len(self.sm_var_index[i])):
                        header0 += '_' + str(self.sm_var_index[i][j])

                    header0 += ' '

                    for j in range(0, len(self.sm_var_spec[i])):
                        header0 += str(self.sm_var_max[i][j]) + ' '

                f.write(header0 + '\n')

                # total variation product
                n_tv = np.prod(tv)

                for i in range(0, n_tv):
                    f.write('-- place holder --' + '\n')

                f.close()

            # determine which line we should write to

            # get position
            def surr_model_pos(spec_var, max_var):
                pos = 0
                for i in range(0, len(max_var) - 1):
                    pos += (spec_var[i] - 1) * np.prod(max_var[i + 1:len(max_var)])
                pos += spec_var[len(spec_var) - 1] - 1

                return pos

            spec_pos = surr_model_pos(sv, tv)

            header_len = 1

            # write first entry to line as naming convention (ex. 1_2_2_0_1 if 5 variables are being used)
            DEM_text = 'var_'
            for i in range(0, len(sv)):
                DEM_text += str(sv[i])+'_'

            # put DEMx and DEMy as values on line
            for i in range(0, len(params['DEMx'])):
                DEM_text += ' ' + str(params['DEMx'][i])

            for i in range(0, len(params['DEMy'])):
                DEM_text += ' ' + str(params['DEMy'][i])

            replace_line(self.DEM_filename, spec_pos+header_len, DEM_text+'\n')

            # add for var_file
            var_text = 'var_'
            for i in range(0, len(sv)):
                var_text += str(sv[i])+'_'

            for i in range(0, len(self.sm_var_names)):

                # chord_sub, theta_sub
                if hasattr(params[self.sm_var_names[i]],'__len__'):
                    for j in range(0, len(params[self.sm_var_names[i]])):
                        if j in self.sm_var_index[i]:

                            # nondimensionalize chord_sub (replace c with c/r)
                            if self.sm_var_names[i] == 'chord_sub' and self.nondimensionalize_chord:
                                var_text += ' ' + str(params[self.sm_var_names[i]][j]/params['bladeLength'])
                            else:
                                var_text += ' ' + str(params[self.sm_var_names[i]][j])

                # r_max_chord
                else:
                    var_text += ' ' + str(params[self.sm_var_names[i]])

            replace_line(self.var_filename, spec_pos + header_len, var_text + '\n')

        # if training points are determined with latin hypercube sampling
        elif self.training_point_dist == 'lhs':

            header_len = 1

            # variable file
            if not (os.path.isfile(self.var_filename)):
                # create file

                f = open(self.var_filename, "w+")

                # header line
                header0 = 'variable points: '
                for i in range(0, len(self.sm_var_names)):

                    header0 += self.sm_var_names[i]
                    for j in range(0, len(self.sm_var_index[i])):
                        header0 += '_' + str(self.sm_var_index[i][j])

                    header0 += ' '

                f.write(header0 + '\n')

                for i in range(0, self.num_pts):
                    f.write('-- place holder --' + '\n')

                f.close

            # add for var_file
            var_text = 'num_pt_' + str(self.sm_var_spec)

            for i in range(0, len(self.sm_var_names)):

                # chord_sub, theta_sub
                if hasattr(params[self.sm_var_names[i]],'__len__'):
                    for j in range(0, len(params[self.sm_var_names[i]])):
                        if j in self.sm_var_index[i]:

                            # nondimensionalize chord_sub (replace c with c/r)
                            if self.sm_var_names[i] == 'chord_sub' and self.nondimensionalize_chord:
                                var_text += ' ' + str(params[self.sm_var_names[i]][j]/params['bladeLength'])
                            else:
                                var_text += ' ' + str(params[self.sm_var_names[i]][j])

                # r_max_chord
                else:
                    var_text += ' ' + str(params[self.sm_var_names[i]])

            # output, load, and def files
            # check if output file exists (if it doesn't, create it)
            file_list = [self.DEM_filename, self.load_filename, self.def_filename]

            for k in range(len(file_list)):
                if not (os.path.isfile(file_list[k])):
                    # create file
                    f = open(file_list[k], "w+")

                    # write a header
                    header0 = 'variable points: '
                    for i in range(0, len(self.sm_var_names)):

                        header0 += self.sm_var_names[i]
                        for j in range(0, len(self.sm_var_index[i])):
                            header0 += '_' + str(self.sm_var_index[i][j])

                        header0 += ' '

                    f.write(header0 + '\n')

                    for i in range(0, self.num_pts):
                        f.write('-- place holder --' + '\n')

                    f.close()

            # write first entry to line as naming convention (ex. 1_2_2_0_1 if 5 variables are being used)
            DEM_text = 'pt_' + str(self.sm_var_spec)

            # put DEMx and DEMy as values on line
            for i in range(0, len(params['DEMx'])):
                DEM_text += ' ' + str(params['DEMx'][i])

            for i in range(0, len(params['DEMy'])):
                DEM_text += ' ' + str(params['DEMy'][i])

            # load file
            load_text = 'pt_' + str(self.sm_var_spec)

            # put Edg_max and Flp_max as values on line
            for i in range(0, len(params['Edg_max'])):
                load_text += ' ' + str(params['Edg_max'][i])

            for i in range(0, len(params['Flp_max'])):
                load_text += ' ' + str(params['Flp_max'][i])

            # def file
            def_text = 'pt_' + str(self.sm_var_spec)

            def_text += ' ' + str(params['max_tip_def'])

            # replace lines
            replace_line(self.var_filename, self.sm_var_spec + header_len, var_text + '\n')

            replace_line(self.DEM_filename, self.sm_var_spec + header_len, DEM_text + '\n')

            replace_line(self.load_filename, self.sm_var_spec + header_len, load_text + '\n')

            replace_line(self.def_filename, self.sm_var_spec + header_len, def_text + '\n')

        else:
            raise Exception('Need to specify training point distribution.')

        return


class calc_FAST_sm_fit(Component):

    def __init__(self, FASTinfo, naero, nstr):
        super(calc_FAST_sm_fit, self).__init__()

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

        self.FASTinfo = FASTinfo

        self.approximation_model = FASTinfo['approximation_model']

        self.training_point_dist = FASTinfo['training_point_dist'] # 'linear', 'lhs'

        if self.training_point_dist == 'lhs':
            self.num_pts = FASTinfo['num_pts']

            self.sm_var_file = FASTinfo['sm_var_file_master']
            self.sm_DEM_file = FASTinfo['sm_DEM_file_master']
            self.sm_load_file = FASTinfo['sm_load_file_master']
            self.sm_def_file = FASTinfo['sm_def_file_master']

        else:
            self.sm_var_max = FASTinfo['sm_var_max']

            self.sm_var_file = FASTinfo['sm_var_file']
            self.sm_DEM_file = FASTinfo['sm_DEM_file']

        self.opt_dir = FASTinfo['opt_dir']

        self.var_filename = self.opt_dir + '/' + self.sm_var_file
        self.DEM_filename = self.opt_dir + '/' + self.sm_DEM_file
        self.load_filename = self.opt_dir + '/' + self.sm_load_file
        self.def_filename = self.opt_dir + '/' + self.sm_def_file

        self.dir_saved_plots = FASTinfo['dir_saved_plots']

        self.sm_var_index = FASTinfo['sm_var_index']
        self.var_index = FASTinfo['var_index']
        self.sm_var_names = FASTinfo['sm_var_names']

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']

        self.add_param('DEMx', shape=18, desc='DEMx')
        self.add_param('DEMy', shape=18, desc='DEMy')

        self.check_fit = FASTinfo['check_fit']

        self.do_cv_DEM = FASTinfo['do_cv_DEM']
        self.do_cv_Load = FASTinfo['do_cv_Load']
        self.do_cv_def = FASTinfo['do_cv_def']

        self.print_sm = FASTinfo['print_sm']

        if self.do_cv_DEM or self.do_cv_Load or self.do_cv_def:

            self.kfolds = FASTinfo['kfolds']
            self.num_folds = FASTinfo['num_folds']

        self.theta0_val = FASTinfo['theta0_val']


        self.nstr = nstr

    def solve_nonlinear(self, params, unknowns, resids):

        # === extract variables === #
        header_len = 1

        if self.training_point_dist == 'linear':
            # determine total number of combinations
            tot_var = 1
            for i in range(0, len(self.sm_var_max)):
                for j in range(0, len(self.sm_var_max[i])):
                    tot_var *= self.sm_var_max[i][j]

        elif self.training_point_dist == 'lhs':

            tot_var = self.num_pts

        else:
            raise Exception('Need to specify training point distribution.')

        # === determine total number of design variables === #
        num_var = 0
        for i in range(0, len(self.sm_var_index)):
            for j in range(0, len(self.sm_var_index[i])):
                num_var += 1

        # === placeholder for var_names === #
        var_names = []

        for i in range(0, num_var):
            var_names.append('var_' + str(i))

        # read in variables
        var_dict = dict()
        for i in range(0, len(var_names)):
            var_dict[var_names[i]] = []

        # open variable .txt file

        if self.training_point_dist == 'linear':

            f = open(self.var_filename, "r")

            # lines = f.readlines(1)
            lines = list(f)

            for i in range(header_len, tot_var+header_len):

                cur_line = lines[i].split()

                # for the case where only varied variables are recorded in sm_var.txt
                if len(cur_line) == num_var+1:
                    for j in range(1, len(cur_line)):
                        var_dict[var_names[j-1]].append(float(cur_line[j]))

            f.close()

            # === extract outputs === #
            # open output .txt file
            f = open(self.DEM_filename, "r")

            # print(list(f))
            lines = list(f)

            # first out line
            first_line = lines[1].split()

            if len(first_line) == 37:
                sgp_range = 17+1
            elif len(first_line) == 17:
                sgp_range = 7+1

            # all outputs (DEMs, loads, tip def) dictionary
            out_dict = dict()
            DEM_names = []

            out_dict['Rootx'] = []
            DEM_names.append('Rootx')
            for i in range(1, sgp_range):
                out_dict['DEMx_' + str(i)] = []
                DEM_names.append('DEMx_' + str(i))

            out_dict['Rooty'] = []
            DEM_names.append('Rooty')
            for i in range(1, sgp_range):
                out_dict['DEMy_' + str(i)] = []
                DEM_names.append('DEMy_' + str(i))

            for i in range(header_len, tot_var + header_len):

                cur_line = lines[i].split()
                for j in range(1, len(cur_line)):
                    out_dict[DEM_names[j - 1]].append(float(cur_line[j]))

            f.close()

        elif self.training_point_dist == 'lhs':


            # === get design variable values / calculated outputs === #
            f_var = open(self.var_filename, "r")
            lines_var = list(f_var)
            f_var.close()

            f_DEM = open(self.DEM_filename, "r")
            lines_DEM = list(f_DEM)
            f_DEM.close()

            f_def = open(self.def_filename, "r")
            lines_def = list(f_def)
            f_def.close()

            f_load = open(self.load_filename, "r")
            lines_load = list(f_load)
            f_load.close()

            # === create var_dict === #

            # iterate for every training point
            for k in range(self.num_pts):

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_var[i].split()

                    # for the case where only varied variables are recorded in sm_var.txt
                    if len(cur_line) == num_var + 1:
                        for j in range(1, len(cur_line)):
                            var_dict[var_names[j - 1]].append(float(cur_line[j]))


            # === DEMs, extreme loads, and tip deflection (oh my!) === #
            # first out line
            first_line = lines_DEM[1].split()

            if len(first_line) == 37:
                sgp_range = 17 + 1
            elif len(first_line) == 17:
                sgp_range = 7 + 1

            # all outputs (DEMs, loads, tip def) dictionary
            out_dict = dict()

            DEM_names = []

            out_dict['Rootx'] = []
            DEM_names.append('Rootx')
            for i in range(1, sgp_range):
                out_dict['DEMx_' + str(i)] = []
                DEM_names.append('DEMx_' + str(i))

            out_dict['Rooty'] = []
            DEM_names.append('Rooty')
            for i in range(1, sgp_range):
                out_dict['DEMy_' + str(i)] = []
                DEM_names.append('DEMy_' + str(i))

            load_names = []

            for i in range(self.nstr):
                out_dict['Edg' + str(i)] = []
                load_names.append('Edg' + str(i))
            for i in range(self.nstr):
                out_dict['Flp' + str(i)] = []
                load_names.append('Flp' + str(i))

            def_names = []

            for i in range(1):
                out_dict['tip' + str(i)] = []
                def_names.append('tip' + str(i))

            for k in range(self.num_pts):
                # DEMx, DEMy

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_DEM[i].split()
                    for j in range(1, len(cur_line)):
                        out_dict[DEM_names[j - 1]].append(float(cur_line[j]))

                # Edg, Flp

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_load[i].split()
                    for j in range(1, len(cur_line)):

                        out_dict[load_names[j - 1]].append(float(cur_line[j]))

                # tip deflection

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_def[i].split()
                    for j in range(1, len(cur_line)):

                        out_dict[def_names[j - 1]].append(float(cur_line[j]))

        # === Approximation Model === #
        from smt.surrogate_models import QP, LS, KRG, KPLS, KPLSK

        if self.approximation_model == 'second_order_poly':
            sm_x_fit = QP()
            sm_y_fit = QP()
            sm_x_load_fit = QP()
            sm_y_load_fit = QP()
            sm_def_fit = QP()

            sm_check_fit = QP()

            cv_x_fit = QP()
            cv_y_fit = QP()

        elif self.approximation_model == 'least_squares':
            sm_x_fit = LS()
            sm_y_fit = LS()
            sm_x_load_fit = LS()
            sm_y_load_fit = LS()
            sm_def_fit = LS()

            sm_check_fit = LS()

            cv_x_fit = LS()
            cv_y_fit = LS()

        elif self.approximation_model == 'kriging':

            # initial hyperparameters
            theta0_val = np.zeros([len(self.var_index), 1])
            for i in range(len(theta0_val)):
                theta0_val[i] = self.theta0_val[0]

            sm_x_fit = KRG(theta0=theta0_val)
            sm_y_fit = KRG(theta0=theta0_val)
            sm_x_load_fit = KRG(theta0=theta0_val)
            sm_y_load_fit = KRG(theta0=theta0_val)
            sm_def_fit = KRG(theta0=theta0_val)

            sm_check_fit = KRG(theta0=theta0_val)

            cv_x_fit = KRG(theta0=theta0_val)
            cv_y_fit = KRG(theta0=theta0_val)

        elif self.approximation_model == 'KPLS':
            theta0_val = self.theta0_val

            sm_x_fit = KPLS(theta0=theta0_val)
            sm_y_fit = KPLS(theta0=theta0_val)
            sm_x_load_fit = KPLS(theta0=theta0_val)
            sm_y_load_fit = KPLS(theta0=theta0_val)
            sm_def_fit = KPLS(theta0=theta0_val)

            sm_check_fit = KPLS(theta0=theta0_val)

            cv_x_fit = KPLS(theta0=theta0_val)
            cv_y_fit = KPLS(theta0=theta0_val)

        elif self.approximation_model == 'KPLSK':
            theta0_val = self.theta0_val

            sm_x_fit = KPLSK(theta0=theta0_val)
            sm_y_fit = KPLSK(theta0=theta0_val)
            sm_x_load_fit = KPLSK(theta0=theta0_val)
            sm_y_load_fit = KPLSK(theta0=theta0_val)
            sm_def_fit = KPLSK(theta0=theta0_val)

            sm_check_fit = KPLSK(theta0=theta0_val)

            cv_x_fit = KPLSK(theta0=theta0_val)
            cv_y_fit = KPLSK(theta0=theta0_val)


        else:
            raise Exception('Need to specify which approximation model will be used in surrogate model.')

        # === initialize predicted values === #

        # DEMs
        DEMx_sm = np.zeros([18, 1])
        DEMy_sm = np.zeros([18, 1])

        # loads
        Edg_sm = np.zeros([self.nstr, 1])
        Flp_sm = np.zeros([self.nstr, 1])

        # tip deflection
        def_sm = np.zeros([1, 1])

        # need to get training values: xt -
        num_pts = len(out_dict['Rootx'])
        num_vars = len(var_names)

        xt = np.zeros([num_vars, num_pts])

        # === DEMx_sm, DEMy_sm fit creation === #

        # design variable values; yt - outputs
        yt_x = np.zeros([len(DEMx_sm), num_pts])
        yt_y = np.zeros([len(DEMy_sm), num_pts])

        for i in range(0,len(DEMx_sm)):

            for j in range(0, num_pts):

                # design variable values
                for k in range(0,len(var_names)):
                    xt[k, j] = var_dict[var_names[k]][j]

                # output values
                yt_x[i, j] = out_dict[DEM_names[i]][j]
                yt_y[i, j] = out_dict[DEM_names[i + 18]][j]

        sm_x = sm_x_fit
        sm_x.set_training_values(np.transpose(xt),np.transpose(yt_x))
        sm_x.options['print_global'] = self.print_sm
        sm_x.train()

        # print('sm_x type:')
        # print(sm_x)
        # print(sm_x.__dict__)
        # quit()

        sm_y = sm_y_fit
        sm_y.set_training_values(np.transpose(xt),np.transpose(yt_y))
        sm_y.options['print_global'] = self.print_sm
        sm_y.train()

        # === Edg_sm, Flp_sm fit creation === #

        num_pts_load = len(out_dict['Edg0'])

        yt_x_load = np.zeros([len(Edg_sm), num_pts_load])
        yt_y_load = np.zeros([len(Flp_sm), num_pts_load])

        for i in range(0,len(Edg_sm)):

            for j in range(0, num_pts_load):

                # output values
                yt_x_load[i, j] = out_dict[load_names[i]][j]
                yt_y_load[i, j] = out_dict[load_names[i + self.nstr]][j]

        sm_x_load = sm_x_load_fit
        sm_x_load.set_training_values(np.transpose(xt),np.transpose(yt_x_load))
        sm_x_load.options['print_global'] = self.print_sm
        sm_x_load.train()

        sm_y_load = sm_y_load_fit
        sm_y_load.set_training_values(np.transpose(xt),np.transpose(yt_y_load))
        sm_y_load.options['print_global'] = self.print_sm
        sm_y_load.train()

        # === tip deflection fit creation === #

        num_pts_def = len(out_dict[def_names[0]])

        yt_def = np.zeros([len(def_sm), num_pts_def])

        for i in range(0, len(def_sm)):

            for j in range(0, num_pts_def):
                # output values
                yt_def[i, j] = out_dict[def_names[i]][j]

        sm_def = sm_def_fit
        sm_def.set_training_values(np.transpose(xt), np.transpose(yt_def))
        sm_def.options['print_global'] = self.print_sm
        sm_def.train()

        # === created surrogate models to .pkl files

        sm_list = [sm_x, sm_y, sm_x_load, sm_y_load, sm_def]
        sm_string_list = ['sm_x', 'sm_y', 'sm_x_load', 'sm_y_load', 'sm_def']

        for i in range(1):#len(sm_list)):
            pkl_file_name = self.opt_dir + '/' + sm_string_list[i] + '_' + self.approximation_model + '.pkl'
            file_handle = open(pkl_file_name, "w+")
            pickle.dump(sm_list[i], file_handle)

        quit()

        if self.check_fit:
            sm = sm_check_fit

            # sm.set_training_values(np.array(var_dict['r_max_chord']), np.array(out_dict['Rooty']))
            sm.set_training_values(np.array(var_dict['var_0']), np.array(out_dict['Rooty']))
            sm.train()

            # predicted value
            val_y = sm.predict_values(np.array(params['r_max_chord']))

            # predicted curve
            num = 100
            fit_x = np.linspace(0.1, 0.5, num)
            fit_y = sm.predict_values(np.array(fit_x))

            plt.figure()
            plt.title('r_max_chord')
            # plt.plot(var_dict['r_max_chord'], out_dict['Rooty'], 'o')
            # plt.plot(var_dict['var_0'], out_dict['Rooty'], 'o')
            plt.plot(params['r_max_chord'], val_y, 'x')
            plt.plot(fit_x, fit_y)
            plt.xlabel('r_max_chord')
            plt.ylabel('Root DEMx (kN*m)')
            # plt.legend(['Training data', 'Calculated Value', 'Prediction'])
            plt.legend(['Calculated Value', 'Prediction'])
            # plt.savefig(self.dir_saved_plots + '/sm_ex.eps')
            plt.savefig(self.dir_saved_plots + '/sm_ex.png')
            plt.show()

            quit()

        # === Do a cross validation, check for total error === #

        if self.do_cv_DEM:

            print('Running DEM cross validation...')

            # === initialize error === #
            DEM_error_x = np.zeros([len(DEMx_sm), self.num_folds])
            percent_DEM_error_x = np.zeros([len(DEMx_sm), self.num_folds])

            DEM_error_y = np.zeros([len(DEMy_sm), self.num_folds])
            percent_DEM_error_y = np.zeros([len(DEMy_sm), self.num_folds])

            for j in range(len(self.kfolds)):

                cur_DEM_error_x = np.zeros([len(DEMx_sm), len(self.kfolds[j])])
                cur_percent_DEM_error_x = np.zeros([len(DEMx_sm), len(self.kfolds[j])])

                cur_DEM_error_y = np.zeros([len(DEMx_sm), len(self.kfolds[j])])
                cur_percent_DEM_error_y = np.zeros([len(DEMx_sm), len(self.kfolds[j])])

                for k in range(len(self.kfolds[j])):

                    cur_kfold = self.kfolds[j]

                    # choose training point indices
                    train_pts = np.linspace(0, self.num_pts - 1, self.num_pts)  # -1 so it's zero-based
                    train_pts = train_pts.tolist()

                    for i in range(0, len(cur_kfold)):
                        train_pts.remove(cur_kfold[i])

                    # choose training point values

                    train_xt = xt[:, train_pts]
                    kfold_xt = xt[:, cur_kfold]

                    train_yt_x = yt_x[:, train_pts]
                    kfold_yt_x = yt_x[:, cur_kfold]

                    train_yt_y = yt_y[:, train_pts]
                    kfold_yt_y = yt_y[:, cur_kfold]

                    # using current design variable values, predict output

                    cv_x = cv_x_fit
                    cv_x.set_training_values(np.transpose(train_xt), np.transpose(train_yt_x))
                    cv_x.options['print_global'] = self.print_sm
                    cv_x.train()

                    cv_y = cv_y_fit
                    cv_y.set_training_values(np.transpose(train_xt), np.transpose(train_yt_y))
                    cv_y.options['print_global'] = self.print_sm
                    cv_y.train()

                    DEMx_cv = np.transpose(cv_x.predict_values(np.array([kfold_xt[:, k]])))
                    DEMy_cv = np.transpose(cv_y.predict_values(np.array([kfold_xt[:, k]])))

                    for i in range(len(DEM_error_x)):
                        cur_DEM_error_x[i][k] = DEMx_cv[i] - kfold_yt_x[:, 0][i]
                        cur_percent_DEM_error_x[i][k] = abs(DEMx_cv[i] - kfold_yt_x[:, k][i]) / kfold_yt_x[:, k][i]

                        cur_DEM_error_y[i][k] = DEMy_cv[i] - kfold_yt_y[:, 0][i]
                        cur_percent_DEM_error_y[i][k] = abs(DEMy_cv[i] - kfold_yt_y[:, k][i]) / kfold_yt_y[:, k][i]

                # average error for specific k-fold
                for i in range(len(DEM_error_x)):
                    DEM_error_x[i][j] = sum(cur_DEM_error_x[i, :]) / len(cur_DEM_error_x[i, :])
                    percent_DEM_error_x[i][j] = sum(cur_percent_DEM_error_x[i, :]) / len(cur_percent_DEM_error_x[i, :])

                    DEM_error_y[i][j] = sum(cur_DEM_error_y[i, :]) / len(cur_DEM_error_y[i, :])
                    percent_DEM_error_y[i][j] = sum(cur_percent_DEM_error_y[i, :]) / len(cur_percent_DEM_error_y[i, :])

            # average percent error over all k-folds
            avg_percent_DEM_error_x = np.zeros([len(DEM_error_x), 1])
            avg_percent_DEM_error_y = np.zeros([len(DEM_error_y), 1])

            rms_percent_DEM_error_x = np.zeros([len(DEM_error_x), 1])
            rms_percent_DEM_error_y = np.zeros([len(DEM_error_y), 1])

            # calculate root mean square error
            for i in range(len(DEM_error_x)):

                avg_percent_DEM_error_x[i] = sum(percent_DEM_error_x[i, :]) / len(percent_DEM_error_x[i, :])
                avg_percent_DEM_error_y[i] = sum(percent_DEM_error_y[i, :]) / len(percent_DEM_error_y[i, :])

                squared_total_x = 0.0
                squared_total_y = 0.0
                for index in range(len(percent_DEM_error_x[i, :])):
                    squared_total_x += percent_DEM_error_x[i, index] ** 2.0
                    squared_total_y += percent_DEM_error_y[i, index] ** 2.0

                rms_percent_DEM_error_x[i] = (squared_total_x / len(percent_DEM_error_x[i, :])) ** 0.5
                rms_percent_DEM_error_y[i] = (squared_total_y / len(percent_DEM_error_y[i, :])) ** 0.5

            # maximum percent error over all k-folds
            max_percent_DEM_error_x = np.zeros([len(DEM_error_x), 1])
            max_percent_DEM_error_y = np.zeros([len(DEM_error_y), 1])

            for i in range(len(DEM_error_x)):
                max_percent_DEM_error_x[i] = max(percent_DEM_error_x[i, :])
                max_percent_DEM_error_y[i] = max(percent_DEM_error_y[i, :])

            # root mean square error over all DEMx, DEMy points
            total_squared_total_x = 0.0
            total_squared_total_y = 0.0
            for index in range(len(rms_percent_DEM_error_x)):
                total_squared_total_x += rms_percent_DEM_error_x[index] ** 2.0
                total_squared_total_y += rms_percent_DEM_error_y[index] ** 2.0

            rms_DEM_error_x = (total_squared_total_x / len(rms_percent_DEM_error_x)) ** 0.5
            rms_DEM_error_y = (total_squared_total_y / len(rms_percent_DEM_error_y)) ** 0.5

            # root mean square error overall
            rms_error = (((rms_DEM_error_x ** 2.0 + rms_DEM_error_y ** 2.0)) / 2.0) ** 0.5

            # print('avg_percent_DEM_error_x')
            # print(avg_percent_DEM_error_x)
            # print('max_percent_DEM_error_x')
            # print(max_percent_DEM_error_x)
            # print('rms_percent_DEM_error_x')
            # print(rms_percent_DEM_error_x)
            # print('rms_DEM_error_x')
            # print(rms_DEM_error_x)

            # print('rms_percent_DEM_error_y')
            # print(rms_percent_DEM_error_y)
            # print('avg_percent_DEM_error_y')
            # print(avg_percent_DEM_error_y)
            # print('max_percent_DEM_error_y')
            # print(max_percent_DEM_error_y)
            # print('rms_DEM_error_y')
            # print(rms_DEM_error_y)

            print('rms_error')
            print(rms_error)
            # quit()


            # save error values in .txt file
            error_file_name = str(self.opt_dir) + '/error_' + self.approximation_model + '_' + str(
                self.num_pts) + '.txt'
            ferror = open(error_file_name, "w+")
            ferror.write(str(rms_DEM_error_x[0]) + '\n')
            ferror.write(str(rms_DEM_error_y[0]))
            ferror.close()

            # DEMx plot
            plt.figure()
            plt.title('DEMx k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_DEM_error_x * 100.0, 'x', label='avg error')
            plt.plot(max_percent_DEM_error_x * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMx_kfold.png')
            plt.show()

            # DEMx plot
            plt.figure()
            plt.title('DEMy k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_DEM_error_y * 100.0, 'x', label='avg error')
            plt.plot(max_percent_DEM_error_y * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMy_kfold.png')
            plt.show()

            quit()

        if self.do_cv_Load:

            print('Running extreme loads cross validation...')

            # === initialize error === #
            Edg_error = np.zeros([len(Edg_sm), self.num_folds])
            percent_Edg_error = np.zeros([len(Edg_sm), self.num_folds])

            Flp_error = np.zeros([len(Flp_sm), self.num_folds])
            percent_Flp_error = np.zeros([len(Flp_sm), self.num_folds])

            for j in range(len(self.kfolds)):

                cur_Edg_error = np.zeros([len(Edg_sm), len(self.kfolds[j])])
                cur_percent_Edg_error = np.zeros([len(Edg_sm), len(self.kfolds[j])])

                cur_Flp_error = np.zeros([len(Flp_sm), len(self.kfolds[j])])
                cur_percent_Flp_error = np.zeros([len(Flp_sm), len(self.kfolds[j])])

                for k in range(len(self.kfolds[j])):

                    cur_kfold = self.kfolds[j]

                    # choose training point indices
                    train_pts = np.linspace(0, self.num_pts - 1, self.num_pts)  # -1 so it's zero-based
                    train_pts = train_pts.tolist()

                    for i in range(0, len(cur_kfold)):
                        train_pts.remove(cur_kfold[i])

                    # choose training point values
                    train_xt = xt[:, train_pts]
                    kfold_xt = xt[:, cur_kfold]

                    train_yt_x = yt_x_load[:, train_pts]
                    kfold_yt_x = yt_x_load[:, cur_kfold]

                    train_yt_y = yt_y_load[:, train_pts]
                    kfold_yt_y = yt_y_load[:, cur_kfold]

                    # using current design variable values, predict output

                    cv_x = cv_x_fit
                    cv_x.set_training_values(np.transpose(train_xt), np.transpose(train_yt_x))
                    cv_x.options['print_global'] = self.print_sm
                    cv_x.train()

                    cv_y = cv_y_fit
                    cv_y.set_training_values(np.transpose(train_xt), np.transpose(train_yt_y))
                    cv_y.options['print_global'] = self.print_sm
                    cv_y.train()

                    Edg_cv = np.transpose(cv_x.predict_values(np.array([kfold_xt[:, k]])))
                    Flp_cv = np.transpose(cv_y.predict_values(np.array([kfold_xt[:, k]])))

                    for i in range(len(Edg_error)):
                        cur_Edg_error[i][k] = Edg_cv[i] - kfold_yt_x[:, 0][i]
                        cur_percent_Edg_error[i][k] = abs(Edg_cv[i] - kfold_yt_x[:, k][i]) / kfold_yt_x[:, k][i]

                        cur_Flp_error[i][k] = Flp_cv[i] - kfold_yt_y[:, 0][i]
                        cur_percent_Flp_error[i][k] = abs(Flp_cv[i] - kfold_yt_y[:, k][i]) / kfold_yt_y[:, k][i]

                # average error for specific k-fold
                for i in range(len(Edg_error)):
                    Edg_error[i][j] = sum(cur_Edg_error[i, :]) / len(cur_Edg_error[i, :])
                    percent_Edg_error[i][j] = sum(cur_percent_Edg_error[i, :]) / len(cur_percent_Edg_error[i, :])

                    Flp_error[i][j] = sum(cur_Flp_error[i, :]) / len(cur_Flp_error[i, :])
                    percent_Flp_error[i][j] = sum(cur_percent_Flp_error[i, :]) / len(cur_percent_Flp_error[i, :])

            # average percent error over all k-folds
            avg_percent_Edg_error = np.zeros([len(Edg_error), 1])
            avg_percent_Flp_error = np.zeros([len(Flp_error), 1])

            rms_percent_Edg_error = np.zeros([len(Edg_error), 1])
            rms_percent_Flp_error = np.zeros([len(Flp_error), 1])

            # calculate root mean square error
            for i in range(len(Edg_error)):

                avg_percent_Edg_error[i] = sum(percent_Edg_error[i, :]) / len(percent_Edg_error[i, :])
                avg_percent_Flp_error[i] = sum(percent_Flp_error[i, :]) / len(percent_Flp_error[i, :])

                squared_total_x = 0.0
                squared_total_y = 0.0
                for index in range(len(percent_Edg_error[i, :])):
                    squared_total_x += percent_Edg_error[i, index] ** 2.0
                    squared_total_y += percent_Flp_error[i, index] ** 2.0

                rms_percent_Edg_error[i] = (squared_total_x / len(percent_Edg_error[i, :])) ** 0.5
                rms_percent_Flp_error[i] = (squared_total_y / len(percent_Flp_error[i, :])) ** 0.5

            # maximum percent error over all k-folds
            max_percent_Edg_error = np.zeros([len(Edg_error), 1])
            max_percent_Flp_error = np.zeros([len(Flp_error), 1])

            for i in range(len(Edg_error)):
                max_percent_Edg_error[i] = max(percent_Edg_error[i, :])
                max_percent_Flp_error[i] = max(percent_Flp_error[i, :])

            # root mean square error over all DEMx, DEMy points
            total_squared_total_x = 0.0
            total_squared_total_y = 0.0
            for index in range(len(rms_percent_Edg_error)):
                total_squared_total_x += rms_percent_Edg_error[index] ** 2.0
                total_squared_total_y += rms_percent_Flp_error[index] ** 2.0

            rms_Edg_error = (total_squared_total_x / len(rms_percent_Edg_error)) ** 0.5
            rms_Flp_error = (total_squared_total_y / len(rms_percent_Flp_error)) ** 0.5

            # root mean square error overall
            rms_error = (((rms_Edg_error ** 2.0 + rms_Flp_error ** 2.0)) / 2.0) ** 0.5

            print('rms_error')
            print(rms_error)
            # quit()


            # save error values in .txt file
            error_file_name = str(self.opt_dir) + '/error_' + self.approximation_model + '_' + str(
                self.num_pts) + '.txt'
            ferror = open(error_file_name, "w+")
            ferror.write(str(rms_Edg_error[0]) + '\n')
            ferror.write(str(rms_Flp_error[0]))
            ferror.close()

            # DEMx plot
            plt.figure()
            plt.title('DEMx k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_Edg_error * 100.0, 'x', label='avg error')
            plt.plot(max_percent_Edg_error * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMx_kfold.png')
            plt.show()

            # DEMx plot
            plt.figure()
            plt.title('DEMy k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_Flp_error * 100.0, 'x', label='avg error')
            plt.plot(max_percent_Flp_error * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMy_kfold.png')
            plt.show()

            quit()

        if self.do_cv_def:

            print('Running tip deflection cross validation...')

            # === initialize error === #
            def_error = np.zeros([len(def_sm), self.num_folds])
            percent_def_error = np.zeros([len(def_sm), self.num_folds])

            for j in range(len(self.kfolds)):

                cur_def_error = np.zeros([len(def_sm), len(self.kfolds[j])])
                cur_percent_def_error = np.zeros([len(def_sm), len(self.kfolds[j])])

                for k in range(len(self.kfolds[j])):

                    cur_kfold = self.kfolds[j]

                    # choose training point indices
                    train_pts = np.linspace(0, self.num_pts - 1, self.num_pts)  # -1 so it's zero-based
                    train_pts = train_pts.tolist()

                    for i in range(0, len(cur_kfold)):
                        train_pts.remove(cur_kfold[i])

                    # choose training point values
                    train_xt = xt[:, train_pts]
                    kfold_xt = xt[:, cur_kfold]

                    train_yt_x = yt_def[:, train_pts]
                    kfold_yt_x = yt_def[:, cur_kfold]

                    # using current design variable values, predict output

                    cv_x = cv_x_fit
                    cv_x.set_training_values(np.transpose(train_xt), np.transpose(train_yt_x))
                    cv_x.options['print_global'] = self.print_sm
                    cv_x.train()

                    def_cv = np.transpose(cv_x.predict_values(np.array([kfold_xt[:, k]])))

                    for i in range(len(def_error)):
                        cur_def_error[i][k] = def_cv[i] - kfold_yt_x[:, 0][i]
                        cur_percent_def_error[i][k] = abs(def_cv[i] - kfold_yt_x[:, k][i]) / kfold_yt_x[:, k][i]

                # average error for specific k-fold
                for i in range(len(def_error)):
                    def_error[i][j] = sum(cur_def_error[i, :]) / len(cur_def_error[i, :])
                    percent_def_error[i][j] = sum(cur_percent_def_error[i, :]) / len(cur_percent_def_error[i, :])

            # average percent error over all k-folds
            avg_percent_def_error = np.zeros([len(def_error), 1])

            rms_percent_def_error = np.zeros([len(def_error), 1])

            # calculate root mean square error
            for i in range(len(def_error)):

                avg_percent_def_error[i] = sum(percent_def_error[i, :]) / len(percent_def_error[i, :])

                squared_total_x = 0.0
                for index in range(len(percent_def_error[i, :])):
                    squared_total_x += percent_def_error[i, index] ** 2.0

                rms_percent_def_error[i] = (squared_total_x / len(percent_def_error[i, :])) ** 0.5

            # maximum percent error over all k-folds
            max_percent_def_error = np.zeros([len(def_error), 1])

            for i in range(len(def_error)):
                max_percent_def_error[i] = max(percent_def_error[i, :])

            # root mean square error over all DEMx, DEMy points
            total_squared_total_x = 0.0
            for index in range(len(rms_percent_def_error)):
                total_squared_total_x += rms_percent_def_error[index] ** 2.0

            rms_def_error = (total_squared_total_x / len(rms_percent_def_error)) ** 0.5

            # root mean square error overall
            rms_error = rms_def_error

            print('rms_error')
            print(rms_error)
            quit()


class use_FAST_surr_model(Component):
    def __init__(self, FASTinfo, naero, nstr):
        super(use_FAST_surr_model, self).__init__()

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

        self.FASTinfo = FASTinfo

        self.add_param('r_max_chord', val=0.0)
        self.add_param('chord_sub', val=np.zeros(4))
        self.add_param('theta_sub', val=np.zeros(4))
        self.add_param('sparT', val=np.zeros(5))
        self.add_param('teT', val=np.zeros(5))

        self.approximation_model = FASTinfo['approximation_model']

        self.training_point_dist = FASTinfo['training_point_dist']  # 'linear', 'lhs'

        if self.training_point_dist == 'lhs':
            self.num_pts = FASTinfo['num_pts']

            self.sm_var_file = FASTinfo['sm_var_file_master']
            self.sm_DEM_file = FASTinfo['sm_DEM_file_master']
            self.sm_load_file = FASTinfo['sm_load_file_master']
            self.sm_def_file = FASTinfo['sm_def_file_master']

        else:
            self.sm_var_max = FASTinfo['sm_var_max']

            self.sm_var_file = FASTinfo['sm_var_file']
            self.sm_DEM_file = FASTinfo['sm_DEM_file']

        self.opt_dir = FASTinfo['opt_dir']

        self.var_filename = self.opt_dir + '/' + self.sm_var_file
        self.DEM_filename = self.opt_dir + '/' + self.sm_DEM_file
        self.load_filename = self.opt_dir + '/' + self.sm_load_file
        self.def_filename = self.opt_dir + '/' + self.sm_def_file

        self.dir_saved_plots = FASTinfo['dir_saved_plots']

        self.sm_var_index = FASTinfo['sm_var_index']
        self.var_index = FASTinfo['var_index']
        self.sm_var_names = FASTinfo['sm_var_names']

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']

        self.add_param('DEMx', shape=18, desc='DEMx')
        self.add_param('DEMy', shape=18, desc='DEMy')

        self.check_fit = FASTinfo['check_fit']

        self.do_cv_DEM = FASTinfo['do_cv_DEM']
        self.do_cv_Load = FASTinfo['do_cv_Load']
        self.do_cv_def = FASTinfo['do_cv_def']

        self.print_sm = FASTinfo['print_sm']

        if self.do_cv_DEM or self.do_cv_Load or self.do_cv_def:
            self.kfolds = FASTinfo['kfolds']
            self.num_folds = FASTinfo['num_folds']

            self.theta0_val = FASTinfo['theta0_val']

        self.add_output('DEMx_sm', val=np.zeros(18))  # , pass_by_obj=False)
        self.add_output('DEMy_sm', val=np.zeros(18))  # , pass_by_obj=False)

        self.add_output('Edg_sm', val=np.zeros(nstr))  # , pass_by_obj=False)
        self.add_output('Flp_sm', val=np.zeros(nstr))  # , pass_by_obj=False)

        self.add_output('def_sm', val=0.0)  # , pass_by_obj=False)

        self.nstr = nstr

        # to nondimensionalize chord_sub
        self.add_param('bladeLength', shape=1, desc='Blade length')
        self.nondimensionalize_chord = FASTinfo['nondimensionalize_chord']

    def solve_nonlinear(self, params, unknowns, resids):


        # === load surrogate model fits === #
        sm_name_list = ['sm_x', 'sm_y', 'sm_x_load', 'sm_y_load', 'sm_def']

        for i in range(len(sm_name_list)):
            pkl_file_name = self.opt_dir + '/' + sm_name_list[i] + '_' + self.approximation_model + '.pkl'

            file_handle = open(pkl_file_name, "r")

            if sm_name_list[i] == 'sm_x':
                sm_x = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_y':
                sm_y = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_x_load':
                sm_x_load = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_y_load':
                sm_y_load = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_def':
                sm_def = pickle.load(file_handle)

        # === estimate outputs === #

        # current design variable values
        sv = []
        for i in range(0, len(self.sm_var_names)):

            # chord_sub, theta_sub
            if hasattr(params[self.sm_var_names[i]], '__len__'):
                for j in range(0, len(self.sm_var_names[i])):

                    if j in self.sm_var_index[i]:

                        # nondimensionalize chord_sub
                        if self.sm_var_names[i] == 'chord_sub' and self.nondimensionalize_chord:
                            sv.append(params[self.sm_var_names[i]][j]/params['bladeLength'])
                        else:
                            sv.append(params[self.sm_var_names[i]][j])
            # chord_sub
            else:
                sv.append(params[self.sm_var_names[i]])

        # === predict values === #

        int_sv = np.zeros([len(sv), 1])
        for i in range(0, len(int_sv)):
            int_sv[i] = sv[i]

        # DEMs
        DEMx_sm = np.transpose(sm_x.predict_values(np.transpose(int_sv)))
        DEMy_sm = np.transpose(sm_y.predict_values(np.transpose(int_sv)))

        # extreme loads
        Edg_sm = np.transpose(sm_x_load.predict_values(np.transpose(int_sv)))
        Flp_sm = np.transpose(sm_y_load.predict_values(np.transpose(int_sv)))

        # tip deflections
        def_sm = np.transpose(sm_def.predict_values(np.transpose(int_sv)))

        # === === #

        def_sm = def_sm[0][0]

        unknowns['DEMx_sm'] = DEMx_sm
        unknowns['DEMy_sm'] = DEMy_sm
        unknowns['Edg_sm'] = Edg_sm
        unknowns['Flp_sm'] = Flp_sm
        unknowns['def_sm'] = def_sm

class CreateFASTConfig(Component):
    def __init__(self, naero, nstr, FASTinfo, WNDfile_List, caseids):
        super(CreateFASTConfig, self).__init__()

        self.caseids = caseids
        self.WNDfile_List = WNDfile_List

        self.dT = FASTinfo['dT']
        self.description = FASTinfo['description']
        self.path = FASTinfo['path']
        self.NBlGages = FASTinfo['NBlGages']
        # self.BldGagNd = FASTinfo['BldGagNd']
        self.BldGagNd = FASTinfo['BldGagNd_config']
        self.sgp = FASTinfo['sgp']

        self.nonturb_dir = FASTinfo['nonturb_wnd_dir']
        self.turb_dir = FASTinfo['turb_wnd_dir']
        self.wndfiletype = FASTinfo['wnd_type_list']
        self.parked_type = FASTinfo['parked']

        self.Tmax_turb = FASTinfo['Tmax_turb']
        self.Tmax_nonturb = FASTinfo['Tmax_nonturb']

        self.FAST_opt_directory = FASTinfo['opt_dir']
        self.template_dir = FASTinfo['template_dir']

        self.train_sm = FASTinfo['train_sm']
        if self.train_sm:
            self.sm_dir = FASTinfo['sm_dir']

        self.check_stif_spline = FASTinfo['check_stif_spline']

        self.output_list = FASTinfo['output_list']


        # add necessary parameters
        self.add_param('nBlades', val=0)

        self.add_param('EIyy', val=np.zeros(nstr))

        self.add_param('r_max_chord', val=0.0)
        self.add_param('chord_sub', val=np.zeros(4))
        self.add_param('theta_sub', val=np.zeros(4))
        self.add_param('idx_cylinder_aero', val=0)
        self.add_param('initial_aero_grid', val=np.zeros(naero))

        self.add_param('rho', val=0.0)
        self.add_param('control:tsr', val=0.0)
        self.add_param('g', val=0.0)
        self.add_param('hubHt', np.zeros(1))
        self.add_param('mu', val=0.0)
        self.add_param('precone', val=0.0)
        self.add_param('tilt', val=0.0)
        self.add_param('hubFraction', val=0.0)
        self.add_param('leLoc', val=np.zeros(nstr))

        self.add_param('FAST_Chord_Aero', val=np.zeros(naero))
        self.add_param('FAST_Theta_Aero', val=np.zeros(naero))

        self.add_param('FAST_Chord_Str', val=np.zeros(nstr))
        self.add_param('FAST_Theta_Str', val=np.zeros(nstr))

        self.add_param('FAST_r_Aero', val=np.zeros(naero))
        self.add_param('FAST_precurve_Aero', val=np.zeros(naero))
        self.add_param('FAST_precurve_Str', val=np.zeros(nstr))
        self.add_param('FAST_Rhub', val=0.0)
        self.add_param('FAST_Rtip', val=0.0)
        self.add_param('V', val=np.zeros(200))

        self.add_param('FlpStff', val=np.zeros(nstr))
        self.add_param('EdgStff', val=np.zeros(nstr))
        self.add_param('GJStff', val=np.zeros(nstr))
        self.add_param('EAStff', val=np.zeros(nstr))
        self.add_param('BMassDen', val=np.zeros(nstr))

        self.add_param('af_idx', val=np.zeros(naero))
        self.add_param('airfoil_types', val=np.zeros(8))

        # Set all constraints to be calculated using finite difference method
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-6

        # add_output
        self.add_output('cfg_master', val=dict(),pass_by_obj=False)

    def solve_nonlinear(self, params, unknowns, resids):

        # # Placeholders for unconnected parameters
        WindSpeed = 11.4  # m/s

        # exposed parameters without connections
        BldFlDmp1 = 0.477465
        BldFlDmp2 = 0.477465
        BldEdDmp1 = 0.477465
        FlStTunr1 = 1.0
        FlStTunr2 = 1.0
        AdjBlMs = 1.04536
        AdjFlSt = 1.0
        AdjEdSt = 1.0

        SysUnits = 'SI'
        StallMod = 'BEDDOES'
        UseCm = 'NO_CM'
        InfModel = 'EQUIL'
        AToler = 0.005
        TwrShad = 0.0
        ShadHWid = 9999.9
        T_Shad_Refpt = 9999.9

        IndModel = 'WAKE'
        TLModel = 'PRANDtl'
        HLModel = 'PRANDtl'
        # #

        airfoil_types_FAST = ['AeroData/Cylinder1.dat',
                              'AeroData/Cylinder2.dat',
                              'AeroData/DU40_A17.dat',
                              'AeroData/DU35_A17.dat',
                              'AeroData/DU30_A17.dat',
                              'AeroData/DU25_A17.dat',
                              'AeroData/DU21_A17.dat',
                              'AeroData/NACA64_A17.dat']

        # TODO: change to FoilNm=params['airfoil_types']
        # Will need work; files headers need to be different

        # create file directory for each surrogate model training point
        if self.train_sm:
            FAST_opt_directory = self.sm_dir
        else:
            FAST_opt_directory = self.FAST_opt_directory

        # needs to be created just once for optimization
        if os.path.isdir(FAST_opt_directory):
            # placeholder
            # print('optimization directory already created')
            pass
        else:
            os.mkdir(FAST_opt_directory)

        # === create config === #

        # Setup input config dictionary of dictionaries.
        caseids = self.caseids
        cfg_master = {}  # master config dictionary (dictionary of dictionaries)

        for sgp in range(0,len(self.sgp)):

            sgp_dir = FAST_opt_directory + '/' + 'sgp' + str(self.sgp[sgp])

            for wnd_file in range(0, len(self.WNDfile_List)):

                spec_caseid = sgp*len(self.WNDfile_List) + wnd_file

                FAST_sgp_directory = sgp_dir

                FAST_wnd_directory = sgp_dir + '/' + caseids[spec_caseid]

                # needs to be created for each DLC
                if os.path.isdir(FAST_sgp_directory):
                    # print('sgp specific directory already created')
                    pass
                else:
                    os.mkdir(FAST_sgp_directory)


                if os.path.isdir(FAST_wnd_directory):
                    # placeholder
                    print('.wnd specific directory already created')
                else:
                    os.mkdir(FAST_wnd_directory)

                    copy_tree(self.template_dir, FAST_wnd_directory)

                # Create dictionary for this particular index
                cfg = {}

                # === run files/directories === #
                cfg['fst_masterfile'] = 'NRELOffshrBsline5MW_Onshore.fst'

                cfg['fst_runfile'] = 'fst_runfile.fst'

                cfg['fst_masterdir'] = FAST_wnd_directory

                cfg['fst_rundir'] = cfg['fst_masterdir']
                cfg['fst_exe'] = ''.join((self.path, 'FAST_glin64'))
                cfg['fst_file_type'] = 0
                cfg['ad_file_type'] = 1

                def replace_line(file_name, line_num, text):
                    lines = open(file_name, 'r').readlines()
                    lines[line_num] = text
                    out = open(file_name, 'w')
                    out.writelines(lines)
                    out.close()

                # exposed parameters (no corresponding RotorSE parameter)
                # if self.wndfiletype[wnd_file] == 'turb':
                if self.wndfiletype[spec_caseid] == 'turb':
                    cfg['TMax'] = self.Tmax_turb
                else:
                    cfg['TMax'] = self.Tmax_nonturb
                cfg['DT'] = self.dT

                # === Add .wnd file location to Aerodyn.ipt file === #
                # turbulent/nonturbulent wind file locations

                # if self.wndfiletype[wnd_file] == 'turb':
                if self.wndfiletype[spec_caseid] == 'turb':
                    wnd_file_path = self.path + self.turb_dir + self.WNDfile_List[wnd_file]
                else:
                    wnd_file_path = self.path + self.nonturb_dir + self.WNDfile_List[wnd_file]

                aerodyn_file_name = cfg['fst_masterdir'] + '/' + 'NRELOffshrBsline5MW_AeroDyn.ipt'
                replace_line(aerodyn_file_name, 9, wnd_file_path + '\n')

                # === general parameters === #
                cfg['NumBl'] = params['nBlades']

                if hasattr(params['g'], "__len__"):
                    cfg['Gravity'] = params['g'][0]
                else:
                    cfg['Gravity'] = params['g']

                cfg['RotSpeed'] = params['control:tsr']
                cfg['TipRad'] = params['FAST_Rtip']
                cfg['HubRad'] = params['FAST_Rhub']
                cfg['ShftTilt'] = params['tilt']
                cfg['PreCone1'] = params['precone']
                cfg['PreCone2'] = params['precone']
                cfg['PreCone3'] = params['precone']

                # === parked configuration === #
                if self.parked_type[wnd_file] == 'yes':
                    cfg['TimGenOn'] = 9999.9

                cfg['OutFileFmt'] = 3  # text and binary output files

                # strain gage placement for bending moment
                cfg['NBlGages'] = self.NBlGages[sgp]
                cfg['BldGagNd'] = self.BldGagNd[sgp]

                # print('in create fast config')
                # print(cfg['BldGagNd'])
                # print(self.BldGagNd[sgp])
                # print('---------------------')

                #  parameters we'll eventually want to connect

                # #

                # # Aerodyn File

                # Add DLC .wnd file name to Aerodyn.ipt input file
                cfg['HH'] = params['hubHt'][0]

                if hasattr(params['rho'], "__len__"):
                    cfg['AirDens'] = params['rho'][0]
                    cfg['KinVisc'] = params['mu'][0] / params['rho'][0]
                else:
                    cfg['AirDens'] = params['rho']
                    cfg['KinVisc'] = params['mu'] / params['rho']

                # cfg['FoilNm'] = FoilNm
                cfg['NFoil'] = (params['af_idx'] + np.ones(np.size(params['af_idx']))).astype(int)
                cfg['BldNodes'] = np.size(params['af_idx'])

                # Make akima splines of RNodes/AeroTwst and RNodes/Chord
                theta_sub_spline = Akima(params['FAST_r_Aero'], params['FAST_Theta_Aero'])
                chord_sub_spline = Akima(params['FAST_r_Aero'], params['FAST_Chord_Aero'])

                # Redefine RNodes so that DRNodes can be calculated using AeroSubs
                RNodes = params['FAST_r_Aero']
                RNodes = np.linspace(RNodes[0], RNodes[-1], len(RNodes))

                cfg['RNodes'] = RNodes
                # Find new values of AeroTwst and Chord using redefined RNodes

                FAST_Theta = theta_sub_spline.interp(RNodes)[0]
                FAST_Chord = chord_sub_spline.interp(RNodes)[0]

                cfg['Chord'] = FAST_Chord
                cfg['AeroTwst'] = FAST_Theta

                DRNodes = np.zeros(np.size(params['af_idx']))
                for i in range(0, np.size(params['af_idx'])):
                    if i == 0:
                        DRNodes[i] = 2.0 * (RNodes[0] - params['FAST_Rhub'])
                    else:
                        DRNodes[i] = 2.0 * (RNodes[i] - RNodes[i - 1]) - DRNodes[i - 1]

                cfg['DRNodes'] = DRNodes

                # # exposed parameters (no corresponding RotorSE parameter)
                cfg['SysUnits'] = 'SI'
                cfg['StallMod'] = 'BEDDOES'
                cfg['UseCm'] = 'NO_CM'
                cfg['InfModel'] = 'DYNIN'
                cfg['AToler'] = 0.005
                cfg['TLModel'] = 'PRANDtl'
                cfg['HLModel'] = 'NONE'
                cfg['TwrShad'] = 0.0
                cfg['ShadHWid'] = 9999.9
                cfg['T_Shad_Refpt'] = 9999.9
                cfg['DTAero'] = 0.02479

                # #

                # # Blade File

                cfg['NBlInpSt'] = len(params['FlpStff'])
                cfg['BlFract'] = np.linspace(0, 1, len(params['FlpStff']))
                cfg['AeroCent'] = params['leLoc']
                cfg['StrcTwst'] = params['FAST_Theta_Str']
                cfg['BMassDen'] = params['BMassDen']

                cfg['FlpStff'] = params['FlpStff']
                cfg['EdgStff'] = params['EdgStff']
                cfg['GJStff'] = params['GJStff']
                cfg['EAStff'] = params['EAStff']

                # exposed parameters (no corresponding RotorSE parameter)
                cfg['CalcBMode'] = 'False'
                cfg['BldFlDmp1'] = 2.477465
                cfg['BldFlDmp2'] = 2.477465
                cfg['BldEdDmp1'] = 2.477465
                cfg['FlStTunr1'] = 1.0
                cfg['FlStTunr2'] = 1.0
                cfg['AdjBlMs'] = 1.04536
                cfg['AdjFlSt'] = 1.0
                cfg['AdjEdSt'] = 1.0

                # unused parameters (not used by FAST)
                alpha = 0.5 * np.arctan2(2 * params['EAStff'], params['FlpStff'] - params['EAStff'])
                for i in range(0, len(alpha)):
                    alpha[i] = min(0.99999, alpha[i])
                cfg['Alpha'] = alpha

                cfg['PrecrvRef'] = np.zeros(len(params['FlpStff']))
                cfg['PreswpRef'] = np.zeros(len(params['FlpStff']))
                cfg['FlpcgOf'] = np.zeros(len(params['FlpStff']))
                cfg['Edgcgof'] = np.zeros(len(params['FlpStff']))
                cfg['FlpEAOf'] = np.zeros(len(params['FlpStff']))
                cfg['EdgEAOf'] = np.zeros(len(params['FlpStff']))

                # #

                # set EI stiffness
                # TODO: just get this from the blade input file
                BladeAerodynamicProperties = np.loadtxt('FAST_Files/RotorSE_InputFiles/BladeAerodynamicProperties.txt')
                BladeStructureProperties = np.loadtxt('FAST_Files/RotorSE_InputFiles/BladeStructureProperties.txt')

                # Blade Structural Properties
                #0 BlFract
                #1 AeroCent
                #2 StrcTwst
                #3 BMassDen
                #4 FlpStff
                #5 EdgStff
                #6 GJStff
                #7 EAStff
                #8 Alpha
                #9 FlpIner
                #10 EdgIner
                #11 PrecrvRef
                #12 PreswpRef
                #13 FlpcgOf
                #14 EdgcgOf
                #15 FlpEAOf
                #16 EdgEAOf

                # FlpStff, EdgStff, GJStff, EAStff
                EI_flp_spline = Akima(params['FAST_precurve_Str'], params['FlpStff'])
                EI_flp = EI_flp_spline.interp(BladeStructureProperties[:, 0])[0]

                EI_edge_spline = Akima(params['FAST_precurve_Str'], params['EdgStff'])
                EI_edge = EI_edge_spline.interp(BladeStructureProperties[:, 0])[0]

                EI_gj_spline = Akima(params['FAST_precurve_Str'], params['GJStff'])
                EI_gj = EI_gj_spline.interp(BladeStructureProperties[:, 0])[0]

                EI_ea_spline = Akima(params['FAST_precurve_Str'], params['EAStff'])
                EI_ea = EI_ea_spline.interp(BladeStructureProperties[:, 0])[0]

                if self.check_stif_spline:

                    # plots
                    BlFract = BladeStructureProperties[:, 0]
                    FlpStff = BladeStructureProperties[:, 4]
                    EdgStff = BladeStructureProperties[:, 5]
                    GJStff = BladeStructureProperties[:, 6]
                    EAStff = BladeStructureProperties[:, 7]

                    plt.figure()
                    plt.plot(BlFract, EI_flp, label='RotorSE spline')
                    plt.plot(BlFract, FlpStff, label='FAST nominal value')
                    plt.legend()
                    plt.title('FlpStff')

                    plt.figure()
                    plt.plot(BlFract, EI_edge, label='RotorSE spline')
                    plt.plot(BlFract, EdgStff, label='FAST nominal value')
                    plt.legend()
                    plt.title('EdgStff')

                    plt.figure()
                    plt.plot(BlFract, EI_gj, label='RotorSE spline')
                    plt.plot(BlFract, GJStff, label='FAST nominal value')
                    plt.legend()
                    plt.title('GJStff Stiffness')

                    plt.figure()
                    plt.plot(BlFract, EI_ea, label='RotorSE spline')
                    plt.plot(BlFract, EAStff, label='FAST nominal value')
                    plt.legend()
                    plt.title('EAStff Stiffness')

                    plt.show()

                    quit()

                cfg_master[self.caseids[spec_caseid]] = cfg

        unknowns['cfg_master'] = cfg_master

class ObjandCons(Component):
    def __init__(self, nstr, npower, num_airfoils, af_dof, FASTinfo):
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
        self.add_param('airfoil_parameterization', val=np.zeros((num_airfoils,af_dof)))
        self.add_param('power', val=np.zeros(npower))
        self.add_param('control:ratedPower', val=0.0)
        self.add_param('ratedConditions:T', val=0.0)

        self.add_param('damageU_spar', val=np.zeros(nstr))
        self.add_param('damageL_spar', val=np.zeros(nstr))
        self.add_param('damageU_te', val=np.zeros(nstr))
        self.add_param('damageL_te', val=np.zeros(nstr))

        self.add_param('max_tip_deflection', val=0.0)

        self.useFAST = FASTinfo['use_FAST']
        self.use_tip_def_cons = FASTinfo['use_tip_def_cons']

        self.add_output('obj', val=1.0)
        self.add_output('con_strain_spar', val=np.zeros(7))
        self.add_output('con_strainU_te', val=np.zeros(8))
        self.add_output('con_strainL_te', val=np.zeros(8))
        self.add_output('con_eps_spar', val=np.zeros(8))
        self.add_output('con_eps_te', val=np.zeros(7))
        self.add_output('con_freq', val=np.zeros(2))

        self.add_output('con_damageU_spar', val=np.zeros(8))
        self.add_output('con_damageL_spar', val=np.zeros(8))
        self.add_output('con_damageU_te', val=np.zeros(8))
        self.add_output('con_damageL_te', val=np.zeros(8))

        self.add_output('con_max_tip_def', val=0.0)

        if af_dof == 8:
            self.add_output('con_afp', val=np.zeros((num_airfoils,af_dof/2)))
        elif af_dof == 2 or af_dof == 1:
            self.add_output('con_afp', val=np.zeros(5))
        self.add_output('con_power', val=0.0)
        self.add_output('con_thrust', val=0.0)
        self.af_dof = af_dof

        # Set all constraints to be calculated using finite difference method
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-6

    def solve_nonlinear(self, params, unknowns, resids):
        self.eta_strain = 1.35*1.3*1.0
        self.con1_indices = [0, 12, 14, 18, 22, 28, 34]
        self.con2_indices = [0, 8, 12, 14, 18, 22, 28, 34]
        self.con3_indices = self.con2_indices
        self.con4_indices = [10, 12, 14, 20, 23, 27, 31, 33]
        self.con5_indices = [10, 12, 13, 14, 21, 28, 33]
        unknowns['obj'] = params['COE']*100.0
        unknowns['con_strain_spar'] = params['strainU_spar'][self.con1_indices]*self.eta_strain/params['strain_ult_spar']

        #

        # print('------------------')
        # print(unknowns['con_strain_spar'])
        # print('------------------')

        unknowns['con_strainU_te'] = params['strainU_te'][self.con2_indices]*self.eta_strain/params['strain_ult_te']
        unknowns['con_strainL_te'] = params['strainL_te'][self.con3_indices]*self.eta_strain/params['strain_ult_te']
        unknowns['con_eps_spar'] = (params['eps_crit_spar'][self.con4_indices] - params['strainU_spar'][self.con4_indices]) / params['strain_ult_spar']
        unknowns['con_eps_te'] = (params['eps_crit_te'][self.con5_indices] - params['strainU_te'][self.con5_indices]) / params['strain_ult_te']
        unknowns['con_freq'] = params['freq_curvefem'][0:2] - params['nBlades']*params['ratedConditions:Omega']/60.0*1.1
        if self.af_dof == 8:
            unknowns['con_afp'] = params['airfoil_parameterization'][:, [4, 5, 6, 7]] - params['airfoil_parameterization'][:, [0, 1, 2, 3]]
        if self.af_dof == 2:
            for i in range(5):
                unknowns['con_afp'][i] = params['airfoil_parameterization'][i][0] - params['airfoil_parameterization'][i+1][0]
        elif self.af_dof == 1:
            for i in range(5):
                unknowns['con_afp'][i] = params['airfoil_parameterization'][i] - params['airfoil_parameterization'][i+1]

        unknowns['con_power'] = (params['power'][-1] - params['control:ratedPower']) / 1.e6
        unknowns['con_thrust'] = params['ratedConditions:T'] / 1.e6

        unknowns['con_damageU_spar'] = params['damageU_spar'][self.con2_indices]#*0.0
        unknowns['con_damageL_spar'] = params['damageL_spar'][self.con2_indices]#*0.0
        unknowns['con_damageU_te'] = params['damageU_te'][self.con2_indices]#*0.0
        unknowns['con_damageL_te'] = params['damageL_te'][self.con2_indices]#*0.0

        # print('---FAST calculated tip deflection check---')
        # print(params['max_tip_deflection'])
        # print('--- ---')
        # quit()

        if self.use_tip_def_cons:
            unknowns['con_max_tip_def'] = params['max_tip_deflection']

        # print(params['damageU_spar'])
        # print(params['damageL_spar'])
        # print(params['damageU_te'])
        # print(params['damageL_te'])
        #
        # print(unknowns['con_damageU_spar'])
        # print(unknowns['con_damageL_spar'])
        # print(unknowns['con_damageU_te'])
        # print(unknowns['con_damageL_te'])

        # print(unknowns['con_strainU_te'])
        # print(unknowns['con_strainL_te'])

        # quit()

    def linearize(self, params, unknowns, resids):
        J = {}

        return J


class Misc(Component):
    def __init__(self):
        super(Misc, self).__init__()
        # Added so can do finite difference across group instead of across individual components
        self.add_param('precurve_sub', val=np.zeros(3))
        self.add_param('delta_precurve_sub', val=np.zeros(3))
        self.add_param('bladeLength', val=0.0)
        self.add_param('delta_bladeLength', val=0.0)

        self.add_output('spline_precurve_sub', val=np.zeros(3))
        self.add_output('spline_bladeLength', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['spline_precurve_sub'] = params['precurve_sub'] + params['delta_precurve_sub']
        unknowns['spline_bladeLength'] = params['bladeLength'] + params['delta_bladeLength']

    def linearize(self, params, unknowns, resids):
        J = {}
        self.n = 3
        J['spline_precurve_sub', 'precurve_sub'] = np.eye(self.n)
        J['spline_precurve_sub', 'delta_precurve_sub'] = np.eye(self.n)
        J['spline_bladeLength', 'bladeLength'] = 1
        J['spline_bladeLength', 'delta_bladeLength'] = 1
        return J

class StructureGroup(Group):
    def __init__(self, naero, nstr, num_airfoils, af_dof):
        super(StructureGroup, self).__init__()
        # Added so can do finite difference across group instead of across individual components
        self.add('spline1', GeometrySpline(naero, nstr))
        self.add('resize', ResizeCompositeSection(nstr, num_airfoils, af_dof))
        self.add('beam', PreCompSections(nstr, num_airfoils, af_dof))
        self.add('curvefem', CurveFEM(nstr))

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

class RotorSE(Group):
    def __init__(self, FASTinfo, naero=17, nstr=38, npower=20, num_airfoils=6, af_dof=1, nspline=200):
        super(RotorSE, self).__init__()
        """rotor model"""

        # --- geometry inputs ---
        self.add('initial_aero_grid', IndepVarComp('initial_aero_grid', np.zeros(naero)), promotes=['*'])
        self.add('initial_str_grid', IndepVarComp('initial_str_grid', np.zeros(nstr)), promotes=['*'])
        self.add('idx_cylinder_aero', IndepVarComp('idx_cylinder_aero', 1, pass_by_obj=True), promotes=['*'])
        self.add('idx_cylinder_str', IndepVarComp('idx_cylinder_str', 1, pass_by_obj=True), promotes=['*'])
        self.add('hubFraction', IndepVarComp('hubFraction', 0.0), promotes=['*'])
        self.add('r_aero', IndepVarComp('r_aero', np.zeros(naero)), promotes=['*'])
        self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
        self.add('chord_sub', IndepVarComp('chord_sub', np.zeros(4),units='m'), promotes=['*'])
        self.add('theta_sub', IndepVarComp('theta_sub', np.zeros(4), units='deg'), promotes=['*'])
        self.add('precurve_sub', IndepVarComp('precurve_sub', np.zeros(3), units='m'), promotes=['*'])
        self.add('delta_precurve_sub', IndepVarComp('delta_precurve_sub', np.zeros(3)), promotes=['*'])
        self.add('bladeLength', IndepVarComp('bladeLength', 0.0, units='m'), promotes=['*'])
        self.add('delta_bladeLength', IndepVarComp('delta_bladeLength', 0.0, units='m', desc='adjustment to blade length to account for curvature from loading'), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0, units='deg'), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0, units='deg'), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0, units='deg'), promotes=['*'])
        self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])

        # --- airfoil shapes ---
        self.add('airfoil_types', IndepVarComp('airfoil_types', val=np.zeros(8), pass_by_obj=True), promotes=['*'])
        self.add('airfoil_parameterization', IndepVarComp('airfoil_parameterization', val=np.zeros((num_airfoils, af_dof))), promotes=['*'])
        self.add('afOptions', IndepVarComp('afOptions', {}, pass_by_obj=True), promotes=['*'])
        self.add('af_idx', IndepVarComp('af_idx', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])
        self.add('af_str_idx', IndepVarComp('af_str_idx', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])

        # --- atmosphere inputs ---
        self.add('rho', IndepVarComp('rho', val=1.225, units='kg/m**3', desc='density of air', pass_by_obj=True), promotes=['*'])
        self.add('mu', IndepVarComp('mu', val=1.81206e-5, units='kg/m/s', desc='dynamic viscosity of air', pass_by_obj=True), promotes=['*'])
        self.add('shearExp', IndepVarComp('shearExp', val=0.2, desc='shear exponent', pass_by_obj=True), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', val=np.zeros(1), units='m', desc='hub height'), promotes=['*'])
        self.add('turbine_class', IndepVarComp('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
        self.add('turbulence_class', IndepVarComp('turbulence_class', val=Enum('B', 'A', 'C'), desc='IEC turbulence class class', pass_by_obj=True), promotes=['*'])
        self.add('g', IndepVarComp('g', val=9.81, units='m/s**2', desc='acceleration of gravity', pass_by_obj=True), promotes=['*'])
        self.add('cdf_reference_height_wind_speed', IndepVarComp('cdf_reference_height_wind_speed', val=0.0, units='m', desc='reference hub height for IEC wind speed (used in CDF calculation)'), promotes=['*'])
        # cdf_reference_mean_wind_speed = Float(iotype='in')
        self.add('weibull_shape', IndepVarComp('weibull_shape', val=0.0), promotes=['*'])
        self.add('VfactorPC', IndepVarComp('VfactorPC', val=0.7, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation'), promotes=['*'])

        # --- composite sections ---
        self.add('sparT', IndepVarComp('sparT', val=np.zeros(5), units='m', desc='spar cap thickness parameters'), promotes=['*'])
        self.add('teT', IndepVarComp('teT', val=np.zeros(5), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])
        self.add('chord_str_ref', IndepVarComp('chord_str_ref', val=np.zeros(nstr), units='m', desc='chord distribution for reference section, thickness of structural layup scaled with reference thickness'), promotes=['*'])
        self.add('thick_str_ref', IndepVarComp('thick_str_ref', val=np.zeros(nstr), units='m', desc='thickness-to-chord distribution for reference section, thickness of structural layup scaled with reference thickness', pass_by_obj=True), promotes=['*'])
        self.add('leLoc', IndepVarComp('leLoc', val=np.zeros(nstr), desc='array of leading-edge positions from a reference blade axis \
            (usually blade pitch axis). locations are normalized by the local chord length.  \
            e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.   \
            positive in -x direction for airfoil-aligned coordinate system'), promotes=['*'])
        self.add('capTriaxThk', IndepVarComp('capTriaxThk', val=np.zeros(5), units='m', desc='spar cap TRIAX layer thickness'), promotes=['*'])
        self.add('capCarbThk', IndepVarComp('capCarbThk', val=np.zeros(5), units='m', desc='spar cap carbon layer thickness'), promotes=['*'])
        self.add('tePanelTriaxThk', IndepVarComp('tePanelTriaxThk', val=np.zeros(5), units='m', desc='trailing edge TRIAX layer thickness'), promotes=['*'])
        self.add('tePanelFoamThk', IndepVarComp('tePanelFoamThk', val=np.zeros(5), units='m', desc='trailing edge foam layer thickness'), promotes=['*'])
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
        self.add('rstar_damage', IndepVarComp('rstar_damage', val=np.zeros(naero+1), desc='nondimensional radial locations of damage equivalent moments'), promotes=['*'])
        self.add('Mxb_damage', IndepVarComp('Mxb_damage', val=np.zeros(naero+1), units='N*m', desc='damage equivalent moments about blade c.s. x-direction'), promotes=['*'])
        self.add('Myb_damage', IndepVarComp('Myb_damage', val=np.zeros(naero+1), units='N*m', desc='damage equivalent moments about blade c.s. y-direction'), promotes=['*'])
        self.add('strain_ult_spar', IndepVarComp('strain_ult_spar', val=0.01, desc='ultimate strain in spar cap'), promotes=['*'])
        self.add('strain_ult_te', IndepVarComp('strain_ult_te', val=2500*1e-6, desc='ultimate strain in trailing-edge panels'), promotes=['*'])
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

        # add components
        self.add('turbineclass', TurbineClass())
        self.add('gridsetup', GridSetup(naero, nstr))
        self.add('grid', RGrid(naero, nstr))
        self.add('spline0', GeometrySpline(naero, nstr))
        self.add('spline', GeometrySpline(naero, nstr))
        self.add('geom', CCBladeGeometry())
        # self.add('tipspeed', MaxTipSpeed())
        self.add('setup', SetupRunVarSpeed(npower))
        self.add('airfoil_analysis', CCBladeAirfoils(naero, nstr, num_airfoils, af_dof))
        self.add('airfoil_spline', AirfoilSpline(naero, nstr, num_airfoils, af_dof))
        self.add('analysis', CCBlade('power', naero, npower, num_airfoils, af_dof))

        self.add('dt', CSMDrivetrain(npower))
        self.add('powercurve', RegulatedPowerCurveGroup(npower, nspline))
        self.add('wind', PowerWind(1))
        # self.add('cdf', WeibullWithMeanCDF(nspline))
        self.add('cdf', RayleighCDF(nspline))
        self.add('aep', AEP(nspline))
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

        # connections to spline1
        self.connect('r_aero', 'spline1.r_aero_unit')
        self.connect('grid.r_str', 'spline1.r_str_unit')
        self.connect('r_max_chord', 'spline1.r_max_chord')
        self.connect('chord_sub', 'spline1.chord_sub')
        self.connect('theta_sub', 'spline1.theta_sub')
        self.connect('precurve_sub', 'spline1.precurve_sub')
        self.connect('bladeLength', 'spline1.bladeLength')
        self.connect('idx_cylinder_aero', 'spline1.idx_cylinder_aero')
        self.connect('idx_cylinder_str', 'spline1.idx_cylinder_str')
        self.connect('hubFraction', 'spline1.hubFraction')
        self.connect('sparT', 'spline1.sparT')
        self.connect('teT', 'spline1.teT')

        # connections to spline
        self.connect('r_aero', 'spline.r_aero_unit')
        self.connect('grid.r_str', 'spline.r_str_unit')
        self.connect('r_max_chord', 'spline.r_max_chord')
        self.connect('chord_sub', 'spline.chord_sub')
        self.connect('theta_sub', 'spline.theta_sub')
        self.add('spline_misc', Misc(), promotes=['precurve_sub', 'delta_precurve_sub', 'delta_bladeLength', 'bladeLength'])
        self.connect('spline_misc.spline_precurve_sub', 'spline.precurve_sub')
        self.connect('spline_misc.spline_bladeLength', 'spline.bladeLength')
        self.connect('idx_cylinder_aero', 'spline.idx_cylinder_aero')
        self.connect('idx_cylinder_str', 'spline.idx_cylinder_str')
        self.connect('hubFraction', 'spline.hubFraction')
        self.connect('sparT', 'spline.sparT')
        self.connect('teT', 'spline.teT')

        # connections to geom
        # self.spline['precurve_str'] = np.zeros(1)
        # self.connect('spline.precurve_str', 'geom.precurveTip', src_indices=[naero-1]) # not needed as default is zero.
        self.connect('spline.Rtip', 'geom.Rtip')
        self.connect('precone', 'geom.precone')

        # # connectiosn to tipspeed
        # self.connect('geom.R', 'tipspeed.R')
        # self.connect('max_tip_speed', 'tipspeed.Vtip_max')
        # self.connect('tipspeed.Omega_max', 'control:maxOmega')

        # connections to setup
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

        # airfoils
        self.connect('airfoil_types', 'airfoil_analysis.airfoil_types')
        self.connect('af_idx', 'airfoil_spline.af_idx')
        self.connect('af_idx', 'airfoil_analysis.af_idx')
        self.connect('af_str_idx', 'airfoil_analysis.af_str_idx')
        self.connect('airfoil_analysis.af_str', 'beam.af_str')
        self.connect('af_str_idx', 'airfoil_spline.af_str_idx')
        self.connect('airfoil_parameterization', 'airfoil_spline.airfoil_parameterization')
        self.connect('airfoil_parameterization', 'airfoil_analysis.airfoil_parameterization')
        self.connect('afOptions', 'airfoil_analysis.afOptions')
        self.connect('idx_cylinder_str', 'airfoil_spline.idx_cylinder_str')
        self.connect('idx_cylinder_aero', 'airfoil_spline.idx_cylinder_aero')
        self.connect('airfoil_spline.airfoil_parameterization_full', 'analysis.airfoil_parameterization')
        self.connect('afOptions', 'analysis.afOptions')
        self.connect('airfoil_analysis.af', 'analysis.af')
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

        # connections to wind
        # self.wind.z = np.zeros(1)
        # self.wind.U = np.zeros(1)
        # self.connect('cdf_reference_mean_wind_speed', 'wind.Uref')
        self.connect('turbineclass.V_mean', 'wind.Uref')
        self.connect('cdf_reference_height_wind_speed', 'wind.zref')
        self.connect('hubHt', 'wind.z', src_indices=[0])
        self.connect('shearExp', 'wind.shearExp')

        # connections to cdf
        self.connect('powercurve.V', 'cdf.x')
        self.connect('wind.U', 'cdf.xbar', src_indices=[0])
        # self.connect('weibull_shape', 'cdf.k')

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
        self.add('structure_group', StructureGroup(naero, nstr, num_airfoils, af_dof), promotes=['*']) # resize, beam, curvefem
        self.add('curvature', BladeCurvature(nstr))
        self.add('gust', GustETM())
        self.add('setuppc',  SetupPCModVarSpeed())
        self.add('aero_rated', CCBlade('loads', naero, 1, num_airfoils, af_dof))
        self.add('aero_extrm', CCBlade('loads', naero,  1, num_airfoils, af_dof))
        self.add('aero_extrm_forces', CCBlade('power', naero, 2, num_airfoils, af_dof))
        self.add('aero_defl_powercurve', CCBlade('loads', naero,  1, num_airfoils, af_dof))
        self.add('loads_defl', TotalLoads(nstr))
        self.add('loads_pc_defl', TotalLoads(nstr))
        self.add('loads_strain', TotalLoads(nstr))
        self.add('damage', DamageLoads(naero, nstr))
        # self.add('struc', RotorWithpBEAM(nstr))
        self.add('struc', RotorWithpBEAM(nstr,FASTinfo))
        self.add('tip', TipDeflection())
        self.add('root_moment', RootMoment(nstr))
        self.add('mass', MassProperties())
        self.add('extreme', ExtremeLoads())
        self.add('blade_defl', BladeDeflection(nstr))
        self.add('aero_0', CCBlade('loads', naero,  1, num_airfoils, af_dof))
        self.add('aero_120', CCBlade('loads', naero,  1, num_airfoils, af_dof))
        self.add('aero_240', CCBlade('loads', naero,  1, num_airfoils, af_dof))
        self.add('root_moment_0', RootMoment(nstr))
        self.add('root_moment_120', RootMoment(nstr))
        self.add('root_moment_240', RootMoment(nstr))
        self.add('output_struc', OutputsStructures(nstr), promotes=['*'])

        # === Turbine Dynamic Response Incorporation - FAST === #

        # === use surrogate model of FAST outputs === #
        if FASTinfo['Use_FAST_sm']:

            # create fit - can check to see if files already created either here or in component
            pkl_file_name = FASTinfo['opt_dir'] + '/' + 'sm_x' + '_' + FASTinfo['approximation_model'] + '.pkl'
            if not os.path.isfile(pkl_file_name):
                self.add('FAST_sm_fit', calc_FAST_sm_fit(FASTinfo, naero, nstr))

            # use fit
            self.add('use_FAST_sm_fit', use_FAST_surr_model(FASTinfo, naero, nstr), promotes=['DEMx_sm','DEMy_sm', 'Flp_sm', 'Edg_sm', 'def_sm'])

        if FASTinfo['use_FAST']:

            WND_File_List = FASTinfo['wnd_list']

            # create WNDfile case ids
            caseids = FASTinfo['caseids']

            self.add('FASTconfig', CreateFASTConfig(naero, nstr, FASTinfo, WND_File_List, caseids), promotes=['cfg_master'])

            from FST7_aeroelasticsolver import FST7Workflow, FST7AeroElasticSolver

            self.add('ParallelFASTCases', FST7AeroElasticSolver(caseids, FASTinfo['Tmax_turb'],
                FASTinfo['Tmax_nonturb'],FASTinfo['wnd_type_list'], FASTinfo['dT'], FASTinfo['output_list']))

            self.connect('cfg_master', 'ParallelFASTCases.cfg_master')

            self.add('FASTConstraints', CreateFASTConstraints(naero, nstr, FASTinfo, WND_File_List, caseids),
                     promotes=['DEMx', 'DEMy', 'max_tip_def', 'Edg_max', 'Flp_max'])

            # for j in range(0, len(FASTinfo['sgp'])):
            #     for i in range(0 + 1, len(WND_File_List) + 1):
            #
            #         self.connect('ParallelFASTCases.WNDfile{0}'.format(i)+ '_sgp' + str(FASTinfo['sgp'][j]),
            #                      'FASTConstraints.WNDfile{0}'.format(i)+ '_sgp' + str(FASTinfo['sgp'][j]))
            for i in range(len(caseids)):
                self.connect('ParallelFASTCases.' + caseids[i], 'FASTConstraints.' + caseids[i])


            self.connect('cfg_master', 'FASTConstraints.cfg_master')

            if FASTinfo['calc_surr_model']:
                self.add('calc_FAST_sm_training_points', Calculate_FAST_sm_training_points(FASTinfo, naero,nstr))

        # outputs
        self.add('coe', COE(), promotes=['*'])
        self.add('obj_cons', ObjandCons(nstr, npower, num_airfoils, af_dof, FASTinfo), promotes=['*'])

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
        self.connect('thick_str_ref', 'resize.thick_str_ref')
        self.connect('sector_idx_strain_spar', 'resize.sector_idx_strain_spar')
        self.connect('sector_idx_strain_te', 'resize.sector_idx_strain_te')
        self.connect('spline1.chord_str', 'resize.chord_str')
        self.connect('spline1.sparT_str', 'resize.sparT_str')
        self.connect('spline1.teT_str', 'resize.teT_str')
        self.connect('idx_cylinder_str', 'resize.idx_cylinder_str')
        self.connect('capTriaxThk', 'resize.capTriaxThk')
        self.connect('capCarbThk', 'resize.capCarbThk')
        self.connect('tePanelTriaxThk', 'resize.tePanelTriaxThk')
        self.connect('tePanelFoamThk', 'resize.tePanelFoamThk')
        self.connect('initial_str_grid', 'resize.initial_str_grid')
        self.connect('airfoil_analysis.af_str', 'resize.af_str')

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
        self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_rated.airfoil_parameterization')
        self.connect('afOptions', 'aero_rated.afOptions')
        self.connect('airfoil_analysis.af', 'aero_rated.af')
        self.connect('nBlades', 'aero_rated.B')
        self.connect('rho', 'aero_rated.rho')
        self.connect('mu', 'aero_rated.mu')
        self.connect('shearExp', 'aero_rated.shearExp')
        self.connect('nSector', 'aero_rated.nSector')
        # self.connect('powercurve.ratedConditions:V + 3*gust.sigma', 'aero_rated.V_load')  # OpenMDAO bug
        self.connect('gust.V_gust', 'aero_rated.V_load')
        self.connect('powercurve.ratedConditions:Omega', 'aero_rated.Omega_load')
        self.connect('powercurve.ratedConditions:pitch', 'aero_rated.pitch_load')
        self.connect('powercurve.azimuth', 'aero_rated.azimuth_load') # self.aero_rated.azimuth_load = 180.0  # closest to tower

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
        self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_extrm.airfoil_parameterization')
        self.connect('afOptions', 'aero_extrm.afOptions')
        self.connect('airfoil_analysis.af', 'aero_extrm.af')
        self.connect('nBlades', 'aero_extrm.B')
        self.connect('rho', 'aero_extrm.rho')
        self.connect('mu', 'aero_extrm.mu')
        self.connect('shearExp', 'aero_extrm.shearExp')
        self.connect('nSector', 'aero_extrm.nSector')
        self.connect('turbineclass.V_extreme', 'aero_extrm.V_load')
        self.connect('pitch_extreme', 'aero_extrm.pitch_load')
        self.connect('azimuth_extreme', 'aero_extrm.azimuth_load')
        self.connect('Omega_load', 'aero_extrm.Omega_load') # self.aero_extrm.Omega_load = 0.0  # parked case

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
        self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_extrm_forces.airfoil_parameterization')
        self.connect('afOptions', 'aero_extrm_forces.afOptions')
        self.connect('airfoil_analysis.af', 'aero_extrm_forces.af')
        self.connect('nBlades', 'aero_extrm_forces.B')
        self.connect('rho', 'aero_extrm_forces.rho')
        self.connect('mu', 'aero_extrm_forces.mu')
        self.connect('shearExp', 'aero_extrm_forces.shearExp')
        self.connect('nSector', 'aero_extrm_forces.nSector')
        # self.aero_extrm_forces.Uhub = np.zeros(2)
        # self.aero_extrm_forces.Omega = np.zeros(2)  # parked case
        # self.aero_extrm_forces.pitch = np.zeros(2)
        self.connect('turbineclass.V_extreme_full', 'aero_extrm_forces.Uhub')
        self.connect('pitch_extreme_full', 'aero_extrm_forces.pitch') # self.aero_extrm_forces.pitch[1] = 90  # feathered
        # self.aero_extrm_forces.T = np.zeros(2)
        # self.aero_extrm_forces.Q = np.zeros(2) #TODO: check effect

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
        self.connect('airfoil_spline.airfoil_parameterization_full', 'aero_defl_powercurve.airfoil_parameterization')
        self.connect('afOptions', 'aero_defl_powercurve.afOptions')
        self.connect('airfoil_analysis.af', 'aero_defl_powercurve.af')
        self.connect('nBlades', 'aero_defl_powercurve.B')
        self.connect('rho', 'aero_defl_powercurve.rho')
        self.connect('mu', 'aero_defl_powercurve.mu')
        self.connect('shearExp', 'aero_defl_powercurve.shearExp')
        self.connect('nSector', 'aero_defl_powercurve.nSector')
        self.connect('setuppc.Uhub', 'aero_defl_powercurve.V_load')
        self.connect('setuppc.Omega', 'aero_defl_powercurve.Omega_load')
        self.connect('setuppc.pitch', 'aero_defl_powercurve.pitch_load')
        self.connect('setuppc.azimuth', 'aero_defl_powercurve.azimuth_load') # self.aero_defl_powercurve.azimuth_load = 0.0

        # connections to beam
        self.connect('spline1.r_str', 'beam.r')
        self.connect('spline1.chord_str', 'beam.chord')
        self.connect('spline1.theta_str', 'beam.theta')
        self.connect('leLoc', 'beam.leLoc')
        self.connect('profile', 'beam.profile')
        self.connect('materials', 'beam.materials')
        self.connect('resize.upperCSOut', 'beam.upperCS')
        self.connect('resize.lowerCSOut', 'beam.lowerCS')
        self.connect('resize.websCSOut', 'beam.websCS')
        self.connect('sector_idx_strain_spar', 'beam.sector_idx_strain_spar')
        self.connect('sector_idx_strain_te', 'beam.sector_idx_strain_te')


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

        # use FAST surrogate model
        if FASTinfo['Use_FAST_sm']:
            self.connect('DEMx_sm', 'damage.Mxb')
            self.connect('DEMy_sm', 'damage.Myb')

        # use FAST @ every iteration
        if FASTinfo['use_FAST']:
            if FASTinfo['use_fatigue_cons']:
                self.connect('DEMx', 'damage.Mxb')
                self.connect('DEMy', 'damage.Myb')
            else:
                self.connect('Mxb_damage', 'damage.Mxb')
                self.connect('Myb_damage', 'damage.Myb')

        # use FAST precalculated/fixed DEMs OR not using FAST
        if not FASTinfo['use_FAST'] and not FASTinfo['Use_FAST_sm']:
            self.connect('Mxb_damage', 'damage.Mxb')
            self.connect('Myb_damage', 'damage.Myb')

        self.connect('spline.theta_str', 'damage.theta')
        self.connect('beam.beam:z', 'damage.r')

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
        self.connect('spline1.theta_str', 'curvefem.theta_str')
        self.connect('spline1.precurve_str', 'curvefem.precurve_str')
        self.connect('spline1.presweep_str', 'curvefem.presweep_str')
        self.connect('nF', 'curvefem.nF')

        # connections to tip
        # self.struc.dx_defl = np.zeros(1)
        # self.struc.dy_defl = np.zeros(1)
        # self.struc.dz_defl = np.zeros(1)
        # self.spline.theta_str = np.zeros(1)
        # self.curvature.totalCone = np.zeros(1) # TODO : Check effect
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
        self.connect('analysis.P', 'power')
        # TODO: add more outputs

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
        self.connect('yaw', ['aero_0.yaw', 'aero_120.yaw', 'aero_240.yaw'])
        self.connect('airfoil_analysis.af', ['aero_0.af', 'aero_120.af', 'aero_240.af'])
        self.connect('nBlades', ['aero_0.B','aero_120.B', 'aero_240.B'])
        self.connect('rho', ['aero_0.rho', 'aero_120.rho', 'aero_240.rho'])
        self.connect('mu', ['aero_0.mu','aero_120.mu' ,'aero_240.mu'])
        self.connect('shearExp', ['aero_0.shearExp','aero_120.shearExp','aero_240.shearExp'])
        self.connect('nSector', ['aero_0.nSector','aero_120.nSector','aero_240.nSector'])
        # self.connect('powercurve.ratedConditions.V + 3*gust.sigma', 'aero_0.V_load')  # OpenMDAO bug
        self.connect('gust.V_gust', ['aero_0.V_load','aero_120.V_load','aero_240.V_load'])
        self.connect('powercurve.ratedConditions:Omega', ['aero_0.Omega_load','aero_120.Omega_load','aero_240.Omega_load'])
        self.add('pitch_load89', IndepVarComp('pitch_load89', val=89.0, units='deg'), promotes=['*'])
        self.add('azimuth_load0', IndepVarComp('azimuth_load0', val=0.0, units='deg'), promotes=['*'])
        self.add('azimuth_load120', IndepVarComp('azimuth_load120', val=120.0, units='deg'), promotes=['*'])
        self.add('azimuth_load240', IndepVarComp('azimuth_load240', val=240.0, units='deg'), promotes=['*'])
        self.connect('pitch_load89', 'aero_0.pitch_load')
        self.connect('pitch_load89', 'aero_120.pitch_load')
        self.connect('pitch_load89', 'aero_240.pitch_load')
        self.connect('azimuth_load0', 'aero_0.azimuth_load')
        self.connect('azimuth_load120', 'aero_120.azimuth_load')
        self.connect('azimuth_load240', 'aero_240.azimuth_load')

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
        self.connect('root_moment_0.Mxyz','Mxyz_0_in')
        self.connect('root_moment_120.Mxyz','Mxyz_120_in')
        self.connect('root_moment_240.Mxyz','Mxyz_240_in')
        self.connect('curvature.totalCone','TotalCone_in', src_indices=[nstr-1])
        self.connect('aero_0.pitch_load','Pitch_in')
        self.connect('root_moment_0.Fxyz', 'Fxyz_0_in')
        self.connect('root_moment_120.Fxyz', 'Fxyz_120_in')
        self.connect('root_moment_240.Fxyz', 'Fxyz_240_in')
        #azimuths not passed. assumed 0,120,240 in drivese function

        # FAST surrogate model
        if FASTinfo['Use_FAST_sm']:
            # design variables
            self.connect('r_max_chord', 'use_FAST_sm_fit.r_max_chord')
            self.connect('chord_sub', 'use_FAST_sm_fit.chord_sub')
            self.connect('theta_sub', 'use_FAST_sm_fit.theta_sub')
            self.connect('sparT', 'use_FAST_sm_fit.sparT')
            self.connect('teT', 'use_FAST_sm_fit.teT')

            # Loads
            self.connect('Edg_sm', 'struc.Edg_max')
            self.connect('Flp_sm', 'struc.Flp_max')

            # nondimensionalize chord_sub
            self.connect('bladeLength', 'use_FAST_sm_fit.bladeLength')

            # Tip deflection
            if FASTinfo['use_tip_def_cons']:
                self.connect('def_sm', 'max_tip_def_in')


        # Top Level Connections for Call FAST (in ObjandCons)
        if FASTinfo['use_FAST']:

            # FAST config
            self.connect('nBlades', 'FASTconfig.nBlades')

            self.connect('beam.beam:EIyy', 'FASTconfig.EIyy')

            self.connect('r_max_chord', 'FASTconfig.r_max_chord')
            self.connect('chord_sub', 'FASTconfig.chord_sub')
            self.connect('theta_sub', 'FASTconfig.theta_sub')
            self.connect('idx_cylinder_aero', 'FASTconfig.idx_cylinder_aero')
            self.connect('initial_aero_grid', 'FASTconfig.initial_aero_grid')

            self.connect('rho', 'FASTconfig.rho')
            self.connect('control:tsr', 'FASTconfig.control:tsr')
            self.connect('g', 'FASTconfig.g')
            self.connect('hubHt', 'FASTconfig.hubHt')
            self.connect('mu', 'FASTconfig.mu')
            self.connect('precone', 'FASTconfig.precone')
            self.connect('tilt', 'FASTconfig.tilt')
            self.connect('hubFraction', 'FASTconfig.hubFraction')
            self.connect('leLoc', 'FASTconfig.leLoc')

            self.connect('af_idx', 'FASTconfig.af_idx')
            self.connect('airfoil_types', 'FASTconfig.airfoil_types')


            self.connect('beam.beam:EIxx', 'FASTconfig.EdgStff')
            self.connect('beam.beam:EIyy', 'FASTconfig.FlpStff')
            self.connect('beam.beam:rhoA', 'FASTconfig.BMassDen')
            self.connect('beam.beam:GJ', 'FASTconfig.GJStff')
            self.connect('beam.beam:EA', 'FASTconfig.EAStff')

            self.connect('spline1.theta_str', 'FASTconfig.FAST_Theta_Str')
            self.connect('spline1.chord_str', 'FASTconfig.FAST_Chord_Str')
            self.connect('spline.chord_aero', 'FASTconfig.FAST_Chord_Aero')
            self.connect('spline.theta_aero', 'FASTconfig.FAST_Theta_Aero')

            self.connect('spline.r_aero', 'FASTconfig.FAST_r_Aero')
            self.connect('spline.precurve_aero', 'FASTconfig.FAST_precurve_Aero')
            self.connect('initial_str_grid', 'FASTconfig.FAST_precurve_Str')
            self.connect('spline.Rhub', 'FASTconfig.FAST_Rhub')
            self.connect('spline.Rtip', 'FASTconfig.FAST_Rtip')

            # FAST Constraints
            self.connect('initial_aero_grid', 'FASTConstraints.initial_aero_grid')
            self.connect('initial_str_grid', 'FASTConstraints.initial_str_grid')
            self.connect('rstar_damage', 'FASTConstraints.rstar_damage')

            # loads
            self.connect('Edg_max', 'struc.Edg_max')
            self.connect('Flp_max', 'struc.Flp_max')

            # Tip deflection
            if FASTinfo['use_tip_def_cons']:
                self.connect('max_tip_def', 'max_tip_def_in')

            # create FAST surrogate model
            if FASTinfo['train_sm']:

                # FAST outputs
                self.connect('Flp_max', 'calc_FAST_sm_training_points.Flp_max')
                self.connect('Edg_max', 'calc_FAST_sm_training_points.Edg_max')
                self.connect('DEMx', 'calc_FAST_sm_training_points.DEMx')
                self.connect('DEMy', 'calc_FAST_sm_training_points.DEMy')
                self.connect('max_tip_def', 'calc_FAST_sm_training_points.max_tip_def')

                # design variables
                self.connect('r_max_chord', 'calc_FAST_sm_training_points.r_max_chord')
                self.connect('chord_sub', 'calc_FAST_sm_training_points.chord_sub')
                self.connect('theta_sub', 'calc_FAST_sm_training_points.theta_sub')
                self.connect('sparT', 'calc_FAST_sm_training_points.sparT')
                self.connect('teT', 'calc_FAST_sm_training_points.teT')

                # nondimensionalize chord
                self.connect('bladeLength', 'calc_FAST_sm_training_points.bladeLength')

if __name__ == '__main__':

    # === import and instantiate ===
    import os
    import matplotlib.pyplot as plt
    from openmdao.api import Problem
    # from rotorse.rotor import RotorSE  (include this line)

    bl_case = 20 #int(sys.argv[1])

    bl_nom = 61.5
    bl_min = bl_nom * 0.9
    bl_max = bl_nom * 1.1

    bl_cases = np.linspace(bl_min, bl_max, 21)

    bladelength = bl_cases[bl_case]

    # -------------------
    rotor = Problem()

    FASTinfo = dict()
    FASTinfo['use_FAST'] = False

    rotor.root = RotorSE(FASTinfo=FASTinfo, naero=17, nstr=38, npower=20)  # , af_dof=2)
    #rotor.root = RotorSE(naero=17, nstr=38, npower=20, num_airfoils=6, af_dof=2)
    rotor.setup(check=False)

    # === blade grid ===
    rotor['initial_aero_grid'] = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
        0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): initial aerodynamic grid on unit radius
    rotor['initial_str_grid'] = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
        0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
        0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
        0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
        0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
        1.0])  # (Array): initial structural grid on unit radius
    rotor['idx_cylinder_aero'] = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
    rotor['idx_cylinder_str'] = 14  # (Int): first idx in r_str_unit of non-cylindrical section
    rotor['hubFraction'] = 0.025  # (Float): hub location as fraction of radius
    # ------------------

    # === blade geometry ===
    rotor['r_aero'] = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): new aerodynamic grid on unit radius
    rotor['r_max_chord'] = 0.23577  # (Float): location of max chord on unit radius
    rotor['chord_sub'] = np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_sub'] = np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_sub'] = np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['delta_precurve_sub'] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
    rotor['sparT'] = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT'] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    rotor['bladeLength'] = bladelength  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    rotor['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    rotor['precone'] = 2.5  # (Float, deg): precone angle
    rotor['tilt'] = 5.0  # (Float, deg): shaft tilt
    rotor['yaw'] = 0.0  # (Float, deg): yaw error
    rotor['nBlades'] = 3  # (Int): number of blades
    # ------------------

    # === airfoil files ===
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_AFFiles')

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
    rotor['af_idx'] = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7])
    rotor['af_str_idx'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7])
    # rotor['airfoil_parameterization'] = np.asarray([[0.404458, 0.0], [0.349012, 0.0], [0.29892, 0.0], [0.251105, 0.0], [0.211299, 0.0], [0.179338, 1.0]])
    # rotor['afOptions'] = dict(AnalysisMethod='XFOIL', AirfoilParameterization='Precomputational', SplineOptions=dict(maxDirectAoA=0), PrecomputationalOptions=dict())#airfoilShapeMethod='NACA'))  # (Dict): dictionary of options for airfoil shape parameterization and analysis
    #
    # baseAirfoilsCoordindates0 = [0]*5
    # baseAirfoilsCoordindates0[0] = os.path.join(basepath, 'DU40.dat')
    # baseAirfoilsCoordindates0[1] = os.path.join(basepath, 'DU35.dat')
    # baseAirfoilsCoordindates0[2] = os.path.join(basepath, 'DU30.dat')
    # baseAirfoilsCoordindates0[3] = os.path.join(basepath, 'DU25.dat')
    # baseAirfoilsCoordindates0[4] = os.path.join(basepath, 'DU21.dat')
    # # Corresponding to blended airfoil family factor of 1.0
    # baseAirfoilsCoordindates1 = [0]*1
    # baseAirfoilsCoordindates1[0] = os.path.join(basepath, 'NACA64.dat')
    # rotor['afOptions']['PrecomputationalOptions']['BaseAirfoilsCoordinates0'] = baseAirfoilsCoordindates0
    # rotor['afOptions']['PrecomputationalOptions']['BaseAirfoilsCoordinates1'] = baseAirfoilsCoordindates1
    rotor['airfoil_types'] = airfoil_types  # (List): names of airfoil file or initialized CCAirfoils

    # ----------------------

    # === atmosphere ===
    rotor['rho'] = 1.225  # (Float, kg/m**3): density of air
    rotor['mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['shearExp'] = 0.25  # (Float): shear exponent
    rotor['hubHt'] = np.array([90.0])  # (Float, m): hub height
    rotor['turbine_class'] = 'I'  # (Enum): IEC turbine class
    rotor['turbulence_class'] = 'B'  # (Enum): IEC turbulence class class
    rotor['cdf_reference_height_wind_speed'] = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    rotor['g'] = 9.81  # (Float, m/s**2): acceleration of gravity
    # ----------------------

    # === control ===
    rotor['control:Vin'] = 3.0  # (Float, m/s): cut-in wind speed
    rotor['control:Vout'] = 25.0  # (Float, m/s): cut-out wind speed
    rotor['control:ratedPower'] = 5e6  # (Float, W): rated power
    rotor['control:minOmega'] = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control:maxOmega'] = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control:tsr'] = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control:pitch'] = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor['pitch_extreme'] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    rotor['azimuth_extreme'] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    rotor['VfactorPC'] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['npts_coarse_power_curve'] = 20  # (Int): number of points to evaluate aero analysis at
    rotor['npts_spline_power_curve'] = 200  # (Int): number of points to use in fitting spline to power curve
    rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType'] = 'geared'  # (Enum)
    rotor['nF'] = 5  # (Int): number of natural frequencies to compute
    rotor['dynamic_amplication_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------

    # === materials and composite layup  ===
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(rotor['initial_str_grid'])
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    rotor['leLoc'] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
        0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
    rotor['sector_idx_strain_spar'] = [2]*ncomp  # (Array): index of sector for spar (PreComp definition of sector)
    rotor['sector_idx_strain_te'] = [3]*ncomp  # (Array): index of sector for trailing-edge (PreComp definition of sector)
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    rotor['chord_str_ref'] = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
        3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
         4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
         4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
         3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
         2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness
    rotor['thick_str_ref'] = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.404457084248, 0.404457084248,
                                   0.349012780126, 0.349012780126, 0.349012780126, 0.349012780126, 0.29892003076, 0.29892003076, 0.25110545018, 0.25110545018, 0.25110545018, 0.25110545018,
                                   0.211298863564, 0.211298863564, 0.211298863564, 0.211298863564, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591,
                                   0.17933792591, 0.17933792591])  # (Array, m): airfoil thickness distribution for reference section, thickness of structural layup scaled with reference thickness

    rotor['capTriaxThk'] = np.array([0.30, 0.29, 0.28, 0.275, 0.27])
    rotor['capCarbThk'] = np.array([4.2, 2.5, 1.0, 0.90, 0.658])
    rotor['tePanelTriaxThk'] = np.array([0.30, 0.29, 0.28, 0.275, 0.27])
    rotor['tePanelFoamThk'] = np.array([9.00, 7.00, 5.00, 3.00, 2.00])

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

    rotor['materials'] = materials  # (List): list of all Orthotropic2DMaterial objects used in defining the geometry
    rotor['upperCS'] = upper  # (List): list of CompositeSection objections defining the properties for upper surface
    rotor['lowerCS'] = lower  # (List): list of CompositeSection objections defining the properties for lower surface
    rotor['websCS'] = webs  # (List): list of CompositeSection objections defining the properties for shear webs
    rotor['profile'] = profile  # (List): airfoil shape at each radial position
    # --------------------------------------


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
    rotor['N_damage'] = 365*24*3600*20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------

    # from myutilities import plt

    # === run and outputs ===
    rotor.run()
    #quit()
    print("================== RotorSE Outputs ==================")
    print('COE =', rotor['COE']*100, 'cents/kWh')
    print('AEP =', rotor['AEP'])
    print('diameter =', rotor['diameter'])
    print('ratedConditions.V =', rotor['ratedConditions:V'])
    print('ratedConditions.Omega =', rotor['ratedConditions:Omega'])
    print('ratedConditions.pitch =', rotor['ratedConditions:pitch'])
    print('ratedConditions.T =', rotor['ratedConditions:T'])
    print('ratedConditions.Q =', rotor['ratedConditions:Q'])
    print('mass_one_blade =', rotor['mass_one_blade'])
    print('mass_all_blades =', rotor['mass_all_blades'])
    print('I_all_blades =', rotor['I_all_blades'])
    print('freq =', rotor['freq'])
    print('tip_deflection =', rotor['tip_deflection'])
    print('root_bending_moment =', rotor['root_bending_moment'])
    print('totalCone =', rotor['TotalCone'])

    quit()

    plt.figure()
    plt.plot(rotor['V'], rotor['P']/1e6)
    plt.xlabel('wind speed (m/s)')
    plt.xlabel('power (W)')

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
