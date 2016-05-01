#!/usr/bin/env python
# encoding: utf-8

"""
airfoilprep_free.py

Created by Andrew Ning on 2012-04-16.
Copyright (c) NREL. All rights reserved.


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from math import pi, sin, cos, radians, degrees, tan, ceil, floor, factorial
import numpy as np
import copy
import pyXLIGHT
from naca_generator import naca4, naca5
import cmath
import mpmath
from copy import deepcopy
import os
import sys
import subprocess
from scipy.interpolate import RectBivariateSpline, bisplev
from akima import Akima
import os, sys, shutil, copy
import csv

global lexitflag_counter
lexitflag_counter = 0

class Polar(object):
    """
    Defines section lift, drag, and pitching moment coefficients as a
    function of angle of attack at a particular Reynolds number.

    """

    def __init__(self, Re, alpha, cl, cd, cm):
        """Constructor

        Parameters
        ----------
        Re : float
            Reynolds number
        alpha : ndarray (deg)
            angle of attack
        cl : ndarray
            lift coefficient
        cd : ndarray
            drag coefficient
        cm : ndarray
            moment coefficient
        """

        self.Re = Re
        self.alpha = np.array(alpha)
        self.cl = np.array(cl)
        self.cd = np.array(cd)
        self.cm = np.array(cm)

    def blend(self, other, weight):
        """Blend this polar with another one with the specified weighting

        Parameters
        ----------
        other : Polar
            another Polar object to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        polar : Polar
            a blended Polar

        """

        # generate merged set of angles of attack - get unique values
        alpha = np.union1d(self.alpha, other.alpha)

        # truncate (TODO: could also have option to just use one of the polars for values out of range)
        min_alpha = max(self.alpha.min(), other.alpha.min())
        max_alpha = min(self.alpha.max(), other.alpha.max())
        alpha = alpha[np.logical_and(alpha >= min_alpha, alpha <= max_alpha)]
        # alpha = np.array([a for a in alpha if a >= min_alpha and a <= max_alpha])

        # interpolate to new alpha
        cl1 = np.interp(alpha, self.alpha, self.cl)
        cl2 = np.interp(alpha, other.alpha, other.cl)
        cd1 = np.interp(alpha, self.alpha, self.cd)
        cd2 = np.interp(alpha, other.alpha, other.cd)
        cm1 = np.interp(alpha, self.alpha, self.cm)
        cm2 = np.interp(alpha, other.alpha, other.cm)

        # linearly blend
        Re = self.Re + weight*(other.Re-self.Re)
        cl = cl1 + weight*(cl2-cl1)
        cd = cd1 + weight*(cd2-cd1)
        cm = cm1 + weight*(cm2-cm1)

        return type(self)(Re, alpha, cl, cd, cm)



    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """Applies 3-D corrections for rotating sections from the 2-D data.

        Parameters
        ----------
        r_over_R : float
            local radial position / rotor radius
        chord_over_r : float
            local chord length / local radial location
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        polar : Polar
            A new Polar object corrected for 3-D effects

        Notes
        -----
        The Du-Selig method :cite:`Du1998A-3-D-stall-del` is used to correct lift, and
        the Eggers method :cite:`Eggers-Jr2003An-assessment-o` is used to correct drag.


        """

        # rename and convert units for convenience
        alpha = np.radians(self.alpha)
        cl_2d = self.cl
        cd_2d = self.cd
        alpha_max_corr = radians(alpha_max_corr)
        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        # parameters in Du-Selig model
        a = 1
        b = 1
        d = 1
        lam = tsr/(1+tsr**2)**0.5  # modified tip speed ratio
        expon = d/lam/r_over_R

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)
        p = np.polyfit(alpha[idx], cl_2d[idx], 1)
        m = p[0]
        alpha0 = -p[1]/m

        # correction factor
        fcl = 1.0/m*(1.6*chord_over_r/0.1267*(a-chord_over_r**expon)/(b+chord_over_r**expon)-1)

        # not sure where this adjustment comes from (besides AirfoilPrep spreadsheet of course)
        adj = ((pi/2-alpha)/(pi/2-alpha_max_corr))**2
        adj[alpha <= alpha_max_corr] = 1.0

        # Du-Selig correction for lift
        cl_linear = m*(alpha-alpha0)
        cl_3d = cl_2d + fcl*(cl_linear-cl_2d)*adj

        # Eggers 2003 correction for drag
        delta_cl = cl_3d-cl_2d

        delta_cd = delta_cl*(np.sin(alpha) - 0.12*np.cos(alpha))/(np.cos(alpha) + 0.12*np.sin(alpha))
        cd_3d = cd_2d + delta_cd

        return type(self)(self.Re, np.degrees(alpha), cl_3d, cd_3d, self.cm)



    def extrapolate(self, cdmax, AR=None, cdmin=0.001, nalpha=15):
        """Extrapolates force coefficients up to +/- 180 degrees using Viterna's method
        :cite:`Viterna1982Theoretical-and`.

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            aspect ratio = (rotor radius / chord_75% radius)
            if provided, cdmax is computed from AR
        cdmin: float, optional
            minimum drag coefficient.  used to prevent negative values that can sometimes occur
            with this extrapolation method
        nalpha: int, optional
            number of points to add in each segment of Viterna method

        Returns
        -------
        polar : Polar
            a new Polar object

        Notes
        -----
        If the current polar already supplies data beyond 90 degrees then
        this method cannot be used in its current form and will just return itself.

        If AR is provided, then the maximum drag coefficient is estimated as

        >>> cdmax = 1.11 + 0.018*AR


        """

        if cdmin < 0:
            raise Exception('cdmin cannot be < 0')

        # lift coefficient adjustment to account for assymetry
        cl_adj = 0.7

        # estimate CD max
        if AR is not None:
            cdmax = 1.11 + 0.018*AR
        self.cdmax = max(max(self.cd), cdmax)

        # extract matching info from ends
        alpha_high = radians(self.alpha[-1])
        cl_high = self.cl[-1]
        cd_high = self.cd[-1]
        cm_high = self.cm[-1]

        alpha_low = radians(self.alpha[0])
        cl_low = self.cl[0]
        cd_low = self.cd[0]

        if alpha_high > pi/2:
            raise Exception('alpha[-1] > pi/2')
            return self
        if alpha_low < -pi/2:
            raise Exception('alpha[0] < -pi/2')
            return self

        # parameters used in model
        sa = sin(alpha_high)
        ca = cos(alpha_high)
        self.A = (cl_high - self.cdmax*sa*ca)*sa/ca**2
        self.B = (cd_high - self.cdmax*sa*sa)/ca

        # alpha_high <-> 90
        alpha1 = np.linspace(alpha_high, pi/2, nalpha)
        alpha1 = alpha1[1:]  # remove first element so as not to duplicate when concatenating
        cl1, cd1 = self.__Viterna(alpha1, 1.0)

        # 90 <-> 180-alpha_high
        alpha2 = np.linspace(pi/2, pi-alpha_high, nalpha)
        alpha2 = alpha2[1:]
        cl2, cd2 = self.__Viterna(pi-alpha2, -cl_adj)

        # 180-alpha_high <-> 180
        alpha3 = np.linspace(pi-alpha_high, pi, nalpha)
        alpha3 = alpha3[1:]
        cl3, cd3 = self.__Viterna(pi-alpha3, 1.0)
        cl3 = (alpha3-pi)/alpha_high*cl_high*cl_adj  # override with linear variation

        if alpha_low <= -alpha_high:
            alpha4 = []
            cl4 = []
            cd4 = []
            alpha5max = alpha_low
        else:
            # -alpha_high <-> alpha_low
            # Note: this is done slightly differently than AirfoilPrep for better continuity
            alpha4 = np.linspace(-alpha_high, alpha_low, nalpha)
            alpha4 = alpha4[1:-2]  # also remove last element for concatenation for this case
            cl4 = -cl_high*cl_adj + (alpha4+alpha_high)/(alpha_low+alpha_high)*(cl_low+cl_high*cl_adj)
            cd4 = cd_low + (alpha4-alpha_low)/(-alpha_high-alpha_low)*(cd_high-cd_low)
            alpha5max = -alpha_high

        # -90 <-> -alpha_high
        alpha5 = np.linspace(-pi/2, alpha5max, nalpha)
        alpha5 = alpha5[1:]
        cl5, cd5 = self.__Viterna(-alpha5, -cl_adj)

        # -180+alpha_high <-> -90
        alpha6 = np.linspace(-pi+alpha_high, -pi/2, nalpha)
        alpha6 = alpha6[1:]
        cl6, cd6 = self.__Viterna(alpha6+pi, cl_adj)

        # -180 <-> -180 + alpha_high
        alpha7 = np.linspace(-pi, -pi+alpha_high, nalpha)
        cl7, cd7 = self.__Viterna(alpha7+pi, 1.0)
        cl7 = (alpha7+pi)/alpha_high*cl_high*cl_adj  # linear variation

        alpha = np.concatenate((alpha7, alpha6, alpha5, alpha4, np.radians(self.alpha), alpha1, alpha2, alpha3))
        cl = np.concatenate((cl7, cl6, cl5, cl4, self.cl, cl1, cl2, cl3))
        cd = np.concatenate((cd7, cd6, cd5, cd4, self.cd, cd1, cd2, cd3))

        cd = np.maximum(cd, cdmin)  # don't allow negative drag coefficients


        # Setup alpha and cm to be used in extrapolation
        cm1_alpha = floor(self.alpha[0] / 10.0) * 10.0
        cm2_alpha = ceil(self.alpha[-1] / 10.0) * 10.0
        alpha_num = abs(int((-180.0-cm1_alpha)/10.0 - 1))
        alpha_cm1 = np.linspace(-180.0, cm1_alpha, alpha_num)
        alpha_cm2 = np.linspace(cm2_alpha, 180.0, int((180.0-cm2_alpha)/10.0 + 1))
        alpha_cm = np.concatenate((alpha_cm1, self.alpha, alpha_cm2))  # Specific alpha values are needed for cm function to work
        cm1 = np.zeros(len(alpha_cm1))
        cm2 = np.zeros(len(alpha_cm2))
        cm_ext = np.concatenate((cm1, self.cm, cm2))
        if np.count_nonzero(self.cm) > 0:
            cmCoef = self.__CMCoeff(cl_high, cd_high, cm_high)  # get cm coefficient
            cl_cm = np.interp(alpha_cm, np.degrees(alpha), cl)  # get cl for applicable alphas
            cd_cm = np.interp(alpha_cm, np.degrees(alpha), cd)  # get cd for applicable alphas
            alpha_low_deg = self.alpha[0]
            alpha_high_deg = self.alpha[-1]
            for i in range(len(alpha_cm)):
                cm_new = self.__getCM(i, cmCoef, alpha_cm, cl_cm, cd_cm, alpha_low_deg, alpha_high_deg)
                if cm_new is None:
                    pass  # For when it reaches the range of cm's that the user provides
                else:
                    cm_ext[i] = cm_new
        try:
            cm = np.interp(np.degrees(alpha), alpha_cm, cm_ext)
        except:
            cm = np.zeros(len(cl))
        return type(self)(self.Re, np.degrees(alpha), cl, cd, cm)




    def __Viterna(self, alpha, cl_adj):
        """private method to perform Viterna extrapolation"""

        alpha = np.maximum(alpha, 0.0001)  # prevent divide by zero

        cl = self.cdmax/2*np.sin(2*alpha) + self.A*np.cos(alpha)**2/np.sin(alpha)
        cl = cl*cl_adj

        cd = self.cdmax*np.sin(alpha)**2 + self.B*np.cos(alpha)

        return cl, cd

    def __CMCoeff(self, cl_high, cd_high, cm_high):
        """private method to obtain CM0 and CMCoeff"""

        found_zero_lift = False

        for i in range(len(self.cm)):
            # try:
            if abs(self.alpha[i]) < 20.0 and self.cl[i] <= 0 and self.cl[i+1] >= 0:
                p = -self.cl[i] / (self.cl[i + 1] - self.cl[i])
                cm0 = self.cm[i] + p * (self.cm[i+1] - self.cm[i])
                found_zero_lift = True
                break
            # except:
            #     pass
        if not found_zero_lift:
            p = -self.cl[0] / (self.cl[1] - self.cl[0])
            cm0 = self.cm[0] + p * (self.cm[1] - self.cm[0])
        self.cm0 = cm0
        alpha_high = radians(self.alpha[-1])
        XM = (-cm_high + cm0) / (cl_high * cos(alpha_high) + cd_high * sin(alpha_high))
        cmCoef = (XM - 0.25) / tan((alpha_high - pi/2))
        return cmCoef

    def __getCM(self, i, cmCoef, alpha, cl_ext, cd_ext, alpha_low_deg, alpha_high_deg):
        """private method to extrapolate Cm"""

        cm_new = 0
        if alpha[i] >= alpha_low_deg and alpha[i] <= alpha_high_deg:
            return
        if alpha[i] > -165 and alpha[i] < 165:
            if abs(alpha[i]) < 0.01:
                cm_new = self.cm0
            else:
                if alpha[i] > 0:
                    x = cmCoef * tan(radians(alpha[i]) - pi/2) + 0.25
                    cm_new = self.cm0 - x * (cl_ext[i] * cos(radians(alpha[i])) + cd_ext[i] * sin(radians(alpha[i])))
                else:
                    x = cmCoef * tan(-radians(alpha[i]) - pi/2) + 0.25
                    cm_new = -(self.cm0 - x * (-cl_ext[i] * cos(-radians(alpha[i])) + cd_ext[i] * sin(-radians(alpha[i]))))
        else:
            if alpha[i] == 165:
                cm_new = -0.4
            elif alpha[i] == 170:
                cm_new = -0.5
            elif alpha[i] == 175:
                cm_new = -0.25
            elif alpha[i] == 180:
                cm_new = 0
            elif alpha[i] == -165:
                cm_new = 0.35
            elif alpha[i] == -170:
                cm_new = 0.4
            elif alpha[i] == -175:
                cm_new = 0.2
            elif alpha[i] == -180:
                cm_new = 0
            else:
                print "Angle encountered for which there is no CM table value (near +/-180 deg). Program will stop."
        return cm_new

    def unsteadyparam(self, alpha_linear_min=-5, alpha_linear_max=5):
        """compute unsteady aero parameters used in AeroDyn input file

        Parameters
        ----------
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        aerodynParam : tuple of floats
            (control setting, stall angle, alpha for 0 cn, cn slope,
            cn at stall+, cn at stall-, alpha for min CD, min(CD))

        """

        alpha = np.radians(self.alpha)
        cl = self.cl
        cd = self.cd

        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        cn = cl*np.cos(alpha) + cd*np.sin(alpha)

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)

        # checks for inppropriate data (like cylinders)
        if len(idx) < 10 or len(np.unique(cl)) < 10:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        # linear fit
        p = np.polyfit(alpha[idx], cn[idx], 1)
        m = p[0]
        alpha0 = -p[1]/m

        # find cn at stall locations
        alphaUpper = np.radians(np.arange(40.0))
        alphaLower = np.radians(np.arange(5.0, -40.0, -1))
        cnUpper = np.interp(alphaUpper, alpha, cn)
        cnLower = np.interp(alphaLower, alpha, cn)
        cnLinearUpper = m*(alphaUpper - alpha0)
        cnLinearLower = m*(alphaLower - alpha0)
        deviation = 0.05  # threshold for cl in detecting stall

        alphaU = np.interp(deviation, cnLinearUpper-cnUpper, alphaUpper)
        alphaL = np.interp(deviation, cnLower-cnLinearLower, alphaLower)

        # compute cn at stall according to linear fit
        cnStallUpper = m*(alphaU-alpha0)
        cnStallLower = m*(alphaL-alpha0)

        # find min cd
        minIdx = cd.argmin()

        # return: control setting, stall angle, alpha for 0 cn, cn slope,
        #         cn at stall+, cn at stall-, alpha for min CD, min(CD)
        return (0.0, degrees(alphaU), degrees(alpha0), m,
                cnStallUpper, cnStallLower, alpha[minIdx], cd[minIdx])

    def plot(self):
        """plot cl/cd/cm polar

        Returns
        -------
        figs : list of figure handles

        """
        import matplotlib.pyplot as plt

        p = self

        figs = []

        # plot cl
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        plt.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('lift coefficient')
        ax.legend(loc='best')

        # plot cd
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('drag coefficient')
        ax.legend(loc='best')

        # plot cm
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('moment coefficient')
        ax.legend(loc='best')

        return figs

class Airfoil(object):
    """A collection of Polar objects at different Reynolds numbers

    """

    def __init__(self, polars, failure=False):
        """Constructor

        Parameters
        ----------
        polars : list(Polar)
            list of Polar objects

        """

        # sort by Reynolds number
        self.polars = sorted(polars, key=lambda p: p.Re)

        # save type of polar we are using
        self.polar_type = polars[0].__class__

        self.failure = failure


    @classmethod
    def initFromAerodynFile(cls, aerodynFile, polarType=Polar):
        """Construct Airfoil object from AeroDyn file

        Parameters
        ----------
        aerodynFile : str
            path/name of a properly formatted Aerodyn file

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        # open aerodyn file
        f = open(aerodynFile, 'r')

        # skip through header
        f.readline()
        description = f.readline().rstrip()  # remove newline
        f.readline()
        numTables = int(f.readline().split()[0])

        # loop through tables
        for i in range(numTables):

            # read Reynolds number
            Re = float(f.readline().split()[0])*1e6

            # read Aerodyn parameters
            param = [0]*8
            for j in range(8):
                param[j] = float(f.readline().split()[0])

            alpha = []
            cl = []
            cd = []
            cm = []

            # read polar information line by line
            while True:
                line = f.readline()
                if 'EOT' in line:
                    break
                data = [float(s) for s in line.split()]
                alpha.append(data[0])
                cl.append(data[1])
                cd.append(data[2])
                cm.append(data[3])

            polars.append(polarType(Re, alpha, cl, cd, cm))

        f.close()

        return cls(polars)

    @classmethod
    def initFromCoordinateFile(cls, CoordinateFile, alphas, Re, polarType=Polar):
        """Construct Airfoil object from airfoil coordinate file

        Parameters
        ----------
        CoordinateFile : array of str
            paths/names of properly formatted airfoil coordinate files

        alphas : array of floats
            array of angles of attack

        Re : float
            Reynolds number

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        for i in range(len(CoordinateFile)):
            # read in coordinate file
            # with suppress_stdout_stderr():
            airfoil = pyXLIGHT.xfoilAnalysis(CoordinateFile[i])
            airfoil.re = self.Re #Re
            airfoil.mach = 0.00
            airfoil.iter = 100

            cl = np.zeros(len(alphas))
            cd = np.zeros(len(alphas))
            cm = np.zeros(len(alphas))
            to_delete = np.zeros(0)

            for j in range(len(alphas)):
                angle = alphas[j]
                cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(angle)
                if lexitflag:
                    cl[j] = -10.0
                    cd[j] = 0.0
            # error handling in case of XFOIL failure
            for k in range(len(cl)):
                if cl[k] == -10.0:
                    if k == 0:
                        cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
                        cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
                    elif k == len(cl)-1:
                        cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
                        cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
                    else:
                        cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
                        cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
                if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
                    to_delete = np.append(to_delete, k)
            cl = np.delete(cl, to_delete)
            cd = np.delete(cd, to_delete)
            alphas = np.delete(alphas, to_delete)

            polars.append(polarType(Re, alphas, cl, cd, cm))

        return cls(polars)

    @classmethod
    def initFromNACA(cls, NACA, alphas, Re, polarType=Polar):
        """Construct Airfoil object from airfoil coordinate file

        Parameters
        ----------
        NACA : array of str
            paths/names of properly formatted airfoil coordinate files

        alphas : array of floats
            array of angles of attack

        Re : float
            Reynolds number

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        for i in range(len(NACA)):
            x = []
            y = []
            if len(NACA[i]) == 4:
                pts = naca4(NACA[i], 60)
            if len(NACA[i]) == 5:
                pts = naca5(NACA[i], 60)
            else:
                'Please input only NACA 4 or 5 series airfoils'
            for j in range(len(pts)):
                x.append(pts[j][0])
                y.append(pts[j][1])

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
            airfoil_shape_file = basepath + os.path.sep + 'naca_coordinates.dat'

            coord_file = open(airfoil_shape_file, 'w')

            print >> coord_file, 'naca' + NACA[i]
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

            coord_file.close()

            # read in coordinate file
            # with suppress_stdout_stderr():
            airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file)
            airfoil.re = self.Re #Re
            airfoil.mach = 0.00
            airfoil.iter = 100

            cl = np.zeros(len(alphas))
            cd = np.zeros(len(alphas))
            cm = np.zeros(len(alphas))
            to_delete = np.zeros(0)
            for j in range(len(alphas)):
                angle = alphas[j]
                cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(angle)
                if lexitflag:
                    cl[j] = -10.0
                    cd[j] = 0.0
            # error handling in case of XFOIL failure
            for k in range(len(cl)):
                if cl[k] == -10.0:
                    if k == 0:
                        cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
                        cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
                    elif k == len(cl)-1:
                        cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
                        cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
                    else:
                        cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
                        cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
                if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
                    to_delete = np.append(to_delete, k)
            cl = np.delete(cl, to_delete)
            cd = np.delete(cd, to_delete)
            alphas = np.delete(alphas, to_delete)

            polars.append(polarType(Re, alphas, cl, cd, cm))

        return cls(polars)


    @classmethod
    def initFromCST(cls, CST, airfoil_analysis_options, polarType=Polar):
        """Construct Airfoil object from airfoil coordinate file

        Parameters
        ----------
        NACA : array of str
            paths/names of properly formatted airfoil coordinate files

        alphas : array of floats
            array of angles of attack

        Re : float
            Reynolds number

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []
        Re = airfoil_analysis_options['Re']
        alphas = airfoil_analysis_options['alphas']
        cl = np.zeros(len(alphas))
        cd = np.zeros(len(alphas))
        cm = np.zeros(len(alphas))
        failure = False
        if airfoil_analysis_options['BEMSpline'] == 'XFOIL':
            [x, y] = cst_to_coordinates_full(CST)
            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
            airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates.dat'
            coord_file = open(airfoil_shape_file, 'w')
            print >> coord_file, 'CST'
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])
            coord_file.close()

            # read in coordinate file
            airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 100


            to_delete = np.zeros(0)
            for j in range(len(alphas)):
                cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(alphas[j])
                if lexitflag:
                    global lexitflag_counter
                    lexitflag_counter += 1
                if lexitflag:
                    cl[j] = -10.0
                    cd[j] = 0.0
            cl_diff = np.diff(np.asarray(cl))
            cd_diff = np.diff(np.asarray(cd))
            for zz in range(len(cl_diff)):
                if abs(cd_diff[zz]) > 0.02 or abs(cl_diff[zz]) > 0.5:
                    to_delete = np.append(to_delete, zz)
            # error handling in case of XFOIL failure
            for k in range(len(cl)):
                if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 1.0 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
                    to_delete = np.append(to_delete, k)
            cl = np.delete(cl, to_delete)
            cd = np.delete(cd, to_delete)
            cm = np.delete(cm, to_delete)
            if not cl.size or len(cl) < 3 or max(cl) < 0.0:
                print "XFOIL Failure! Using default backup airfoil.", CST
                # for CST = [-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25]
                cl = [-1.11249573, -1.10745928, -1.10242437, -1.10521061, -1.03248528, -0.9272929, -0.81920516, -0.70843745, -0.58962942, -0.45297636, -0.34881162, -0.26194, -0.17375163, -0.09322158, -0.01072867,  0.07232111,
                      0.15326737,  0.22932199, 0.29657574,  0.36818004,  0.45169576,  0.55640456 , 0.68532189,  0.81592085, 0.93355555,  1.04754944,  1.06513144,  1.07821432 , 1.09664777,  1.11425611]
                cd = [ 0.03966997,  0.03289554,  0.02783541,  0.02418726,  0.02120267,  0.01849611,  0.01623273,  0.01424686,  0.0124225 ,  0.01083306,  0.00973778,  0.00908278, 0.00867001,  0.00838171,  0.00823596,  0.00820681,
                       0.00828496 , 0.00842328,  0.00867177,  0.00921659,  0.01004469,  0.01129231,  0.01306175 , 0.01509252, 0.01731396,  0.01986422,  0.02234169 , 0.02555122,  0.02999641 , 0.03574208]
                #cl = [-1.1227906,  -0.55726515, -0.30884085, -0.02638192,  0.19234127,  0.40826801,  0.67141856,  0.95384527,  1.28095228]
                #cd = [ 0.05797574,  0.01721584,  0.01167788,  0.01055452,  0.0102769,   0.01022808,  0.01051864,  0.01179746, 0.0337189 ]
                cm = np.zeros(len(cl))
                alphas = np.linspace(-15, 15, len(cl))
                failure = True

            else:
                alphas = np.delete(alphas, to_delete)
        elif airfoil_analysis_options['BEMSpline'] == 'CFD':
            import time
            if airfoil_analysis_options['ParallelAirfoils']:
                time0 = time.time()
                cl, cd = cfdDirectSolveParallel(np.radians(alphas), Re, CST, airfoil_analysis_options)
                print "PARALLEL TIME", time.time() - time0
            else:
                time0 = time.time()
                for j in range(len(alphas)):
                    if j == 0:
                        mesh = True
                    else:
                        mesh = False

                    cl[j], cd[j] = cfdDirectSolve(np.radians(alphas[j]), Re, CST, airfoil_analysis_options, GenerateMESH=mesh)
                print "SERIAL TIME", time.time() - time0
        else:
            print "Please choose CFD or XFOIL."
            raise ValueError

        polars.append(polarType(Re, alphas, cl, cd, cm))

        return cls(polars, failure)




    def getPolar(self, Re):
        """Gets a Polar object for this airfoil at the specified Reynolds number.

        Parameters
        ----------
        Re : float
            Reynolds number

        Returns
        -------
        obj : Polar
            a Polar object

        Notes
        -----
        Interpolates as necessary. If Reynolds number is larger than or smaller than
        the stored Polars, it returns the Polar with the closest Reynolds number.

        """

        p = self.polars

        if Re <= p[0].Re:
            return copy.deepcopy(p[0])

        elif Re >= p[-1].Re:
            return copy.deepcopy(p[-1])

        else:
            Relist = [pp.Re for pp in p]
            i = np.searchsorted(Relist, Re)
            weight = (Re - Relist[i-1]) / (Relist[i] - Relist[i-1])
            return p[i-1].blend(p[i], weight)



    def blend(self, other, weight):
        """Blend this Airfoil with another one with the specified weighting.


        Parameters
        ----------
        other : Airfoil
            other airfoil to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        obj : Airfoil
            a blended Airfoil object

        Notes
        -----
        First finds the unique Reynolds numbers.  Evaluates both sets of polars
        at each of the Reynolds numbers, then blends at each Reynolds number.

        """

        # combine Reynolds numbers
        Relist1 = [p.Re for p in self.polars]
        Relist2 = [p.Re for p in other.polars]
        Relist = np.union1d(Relist1, Relist2)

        # blend polars
        n = len(Relist)
        polars = [0]*n
        for i in range(n):
            p1 = self.getPolar(Relist[i])
            p2 = other.getPolar(Relist[i])
            polars[i] = p1.blend(p2, weight)


        return Airfoil(polars)


    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """apply 3-D rotational corrections to each polar in airfoil

        Parameters
        ----------
        r_over_R : float
            radial position / rotor radius
        chord_over_r : float
            local chord / local radius
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        airfoil : Airfoil
            airfoil with 3-D corrections

        See Also
        --------
        Polar.correction3D : apply 3-D corrections for a Polar

        """

        n = len(self.polars)
        polars = [0]*n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.correction3D(r_over_R, chord_over_r, tsr, alpha_max_corr, alpha_linear_min, alpha_linear_max)

        return Airfoil(polars)


    def extrapolate(self, cdmax, AR=None, cdmin=0.001):
        """apply high alpha extensions to each polar in airfoil

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            blade aspect ratio (rotor radius / chord at 75% radius).  if included
            it is used to estimate cdmax
        cdmin: minimum drag coefficient

        Returns
        -------
        airfoil : Airfoil
            airfoil with +/-180 degree extensions

        See Also
        --------
        Polar.extrapolate : extrapolate a Polar to high angles of attack

        """

        n = len(self.polars)
        polars = [0]*n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.extrapolate(cdmax, AR, cdmin)

        return Airfoil(polars)



    def interpToCommonAlpha(self, alpha=None):
        """Interpolates all polars to a common set of angles of attack

        Parameters
        ----------
        alpha : ndarray, optional
            common set of angles of attack to use.  If None a union of
            all angles of attack in the polars is used.

        """

        if alpha is None:
            # union of angle of attacks
            alpha = []
            for p in self.polars:
                alpha = np.union1d(alpha, p.alpha)

        # interpolate each polar to new alpha
        n = len(self.polars)
        polars = [0]*n
        if n == 1:
            polars[0] = self.polar_type(p.Re, alpha, p.cl, p.cd, p.cm)
            return Airfoil(polars)
        for idx, p in enumerate(self.polars):
            cl = np.interp(alpha, p.alpha, p.cl)
            cd = np.interp(alpha, p.alpha, p.cd)
            cm = np.interp(alpha, p.alpha, p.cm)
            polars[idx] = self.polar_type(p.Re, alpha, cl, cd, cm)

        return Airfoil(polars)

    def writeToAerodynFile(self, filename):
        """Write the airfoil section data to a file using AeroDyn input file style.

        Parameters
        ----------
        filename : str
            name (+ relative path) of where to write file

        """

        # aerodyn and wtperf require common set of angles of attack
        af = self.interpToCommonAlpha()

        f = open(filename, 'w')

        print >> f, 'AeroDyn airfoil file.'
        print >> f, 'Compatible with AeroDyn v13.0.'
        print >> f, 'Generated by airfoilprep_free.py'
        print >> f, '{0:<10d}\t\t{1:40}'.format(len(af.polars), 'Number of airfoil tables in this file')
        for p in af.polars:
            print >> f, '{0:<10f}\t{1:40}'.format(p.Re/1e6, 'Reynolds number in millions.')
            param = p.unsteadyparam()
            print >> f, '{0:<10f}\t{1:40}'.format(param[0], 'Control setting')
            print >> f, '{0:<10f}\t{1:40}'.format(param[1], 'Stall angle (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[2], 'Angle of attack for zero Cn for linear Cn curve (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[3], 'Cn slope for zero lift for linear Cn curve (1/rad)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[4], 'Cn at stall value for positive angle of attack for linear Cn curve')
            print >> f, '{0:<10f}\t{1:40}'.format(param[5], 'Cn at stall value for negative angle of attack for linear Cn curve')
            print >> f, '{0:<10f}\t{1:40}'.format(param[6], 'Angle of attack for minimum CD (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[7], 'Minimum CD value')
            for a, cl, cd, cm in zip(p.alpha, p.cl, p.cd, p.cm):
                print >> f, '{:<10f}\t{:<10f}\t{:<10f}\t{:<10f}'.format(a, cl, cd, cm)
            print >> f, 'EOT'
        f.close()


    def createDataGrid(self):
        """interpolate airfoil data onto uniform alpha-Re grid.

        Returns
        -------
        alpha : ndarray (deg)
            a common set of angles of attack (union of all polars)
        Re : ndarray
            all Reynolds numbers defined in the polars
        cl : ndarray
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : ndarray
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """
        if len(self.polars) > 1:
            af = self.interpToCommonAlpha()
            polarList = af.polars
        else:
            polarList = self.polars
        # angle of attack is already same for each polar
        alpha = polarList[0].alpha

        # all Reynolds numbers
        Re = [p.Re for p in polarList]

        # fill in cl, cd grid
        cl = np.zeros((len(alpha), len(Re)))
        cd = np.zeros((len(alpha), len(Re)))
        cm = np.zeros((len(alpha), len(Re)))

        for (idx, p) in enumerate(polarList):
            cl[:, idx] = p.cl
            cd[:, idx] = p.cd
            cm[:, idx] = p.cm


        return alpha, Re, cl, cd, cm

    def plot(self, single_figure=True):
        """plot cl/cd/cm polars

        Parameters
        ----------
        single_figure : bool
            True  : plot all cl on the same figure (same for cd,cm)
            False : plot all cl/cd/cm on separate figures

        Returns
        -------
        figs : list of figure handles

        """

        import matplotlib.pyplot as plt

        figs = []

        # if in single figure mode (default)
        if single_figure:
            # generate figure handles
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            figs.append(fig1)

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            figs.append(fig2)

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            figs.append(fig3)

            # loop through polars and plot
            for p in self.polars:
                # plot cl
                ax1.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
                ax1.set_xlabel('angle of attack (deg)')
                ax1.set_ylabel('lift coefficient')
                ax1.legend(loc='best')

                # plot cd
                ax2.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
                ax2.set_xlabel('angle of attack (deg)')
                ax2.set_ylabel('drag coefficient')
                ax2.legend(loc='best')

                # plot cm
                ax3.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
                ax3.set_xlabel('angle of attack (deg)')
                ax3.set_ylabel('moment coefficient')
                ax3.legend(loc='best')

        # otherwise, multi figure mode -- plot all on separate figures
        else:
            for p in self.polars:
                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('lift coefficient')
                ax.legend(loc='best')

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('drag coefficient')
                ax.legend(loc='best')

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('moment coefficient')
                ax.legend(loc='best')
        plt.show()
        return figs


class CCAirfoil:
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""
    # implements(AirfoilInterface)


    def __init__(self, alpha, Re, cl, cd, cm, afp=None, airfoil_analysis_options=None, airfoilNum=0, failure=False):
        """Setup CCAirfoil from raw airfoil data on a grid.

        Parameters
        ----------
        alpha : array_like (deg)
            angles of attack where airfoil data are defined
            (should be defined from -180 to +180 degrees)
        Re : array_like
            Reynolds numbers where airfoil data are defined
            (can be empty or of length one if not Reynolds number dependent)
        cl : array_like
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : array_like
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """
        alpha = np.radians(alpha)
        self.Re = Re
        if not all(np.diff(alpha)):
            to_delete = np.zeros(0)
            diff = np.diff(alpha)
            for i in range(len(alpha)-1):
                if not diff[i] > 0.0:
                    to_delete = np.append(to_delete, i)
            alpha = np.delete(alpha, to_delete)
            cl = np.delete(cl, to_delete)
            cd = np.delete(cd, to_delete)

        self.one_Re = False

        # special case if zero or one Reynolds number (need at least two for bivariate spline)
        if len(Re) < 2:
            Re = [1e1, 1e15]
            cl = np.c_[cl, cl]
            cd = np.c_[cd, cd]
            self.one_Re = True

        kx = min(len(alpha)-1, 3)
        ky = min(len(Re)-1, 3)

        # a small amount of smoothing is used to prevent spurious multiple solutions
        self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky)#, s=0.1)#, s=0.1)
        self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky)#, s=0.001) #, s=0.001)

        self.failure = failure
        if failure:
            afp = np.asarray([-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25])
        if afp is not None:
            # To check spline
            # n = 2000
            # import matplotlib.pylab as plt
            # alphas2 = np.linspace(-30, 30, n)
            # Re = 1e6
            # cl2 = np.zeros(n)
            # cd2 = np.zeros(n)
            # for i in range(n):
            #     cl2[i] = self.cl_spline.ev(np.radians(alphas2[i]), Re)
            #     cd2[i] = self.cd_spline.ev(np.radians(alphas2[i]), Re)
            # plt.figure()
            # plt.plot(alphas2, cl2)
            # plt.plot(np.degrees(alpha), cl[:,0], 'rx')
            # plt.xlim(xmin=-30, xmax=30)
            #
            # plt.figure()
            # plt.plot(alphas2, cd2)
            # plt.plot(np.degrees(alpha), cd[:,0], 'rx')
            # plt.xlim(xmin=-30, xmax=30)
            # plt.show()

            self.afp = afp
            self.airfoilNum = airfoilNum
            self.airfoil_analysis_options = airfoil_analysis_options
            self.cl_storage = []
            self.cd_storage = []
            self.alpha_storage = []
            self.dcl_storage = []
            self.dcd_storage = []
            self.dalpha_storage = []
            self.dcl_dafp_storage = []
            self.dcd_dafp_storage = []
            self.dalpha_dafp_storage = []

            if airfoil_analysis_options['FreeFormDesign'] and airfoil_analysis_options['ComputeGradient']:
                self.cl_splines_new = [0]*8
                self.cd_splines_new = [0]*8
                for i in range(8):
                    fd_step = 1e-6
                    afp_new = deepcopy(self.afp)
                    afp_new[i] += fd_step
                    af = Airfoil.initFromCST(afp_new, airfoil_analysis_options)
                    af_extrap = af.extrapolate(1.5)
                    alphas_new, Re_new, cl_new, cd_new, cm_new = af_extrap.createDataGrid()
                    alphas_new = np.radians(alphas_new)

                    if not all(np.diff(alphas_new)):
                        to_delete = np.zeros(0)
                        diff = np.diff(alphas_new)
                        for j in range(len(alphas_new)-1):
                            if not diff[j] > 0.0:
                                to_delete = np.append(to_delete, j)
                        alphas_new = np.delete(alphas_new, to_delete)
                        cl_new = np.delete(cl_new, to_delete)
                        cd_new = np.delete(cd_new, to_delete)

                    # special case if zero or one Reynolds number (need at least two for bivariate spline)
                    if len(Re_new) < 2:
                        Re2 = [1e1, 1e15]
                        cl_new = np.c_[cl_new, cl_new]
                        cd_new = np.c_[cd_new, cd_new]
                    kx = min(len(alphas_new)-1, 3)
                    ky = min(len(Re2)-1, 3)
                    # a small amount of smoothing is used to prevent spurious multiple solutions
                    self.cl_splines_new[i] = RectBivariateSpline(alphas_new, Re2, cl_new, kx=kx, ky=ky)#, s=0.1)#, s=0.1)
                    self.cd_splines_new[i] = RectBivariateSpline(alphas_new, Re2, cd_new, kx=kx, ky=ky)#, s=0.001) #, s=0.001)
            # if airfoil_analysis_options['AnalysisMethod'] == 'CFD' and airfoil_analysis_options['maxDirectAoA']>0.0:
            #     # Create restart file and generate mesh
            #     cl, cd = self.cfdSolve(np.radians(5.0), Re, ComputeGradients=False, GenerateMESH=True, airfoilNum=airfoilNum)



        else:
            self.afp = None


    @classmethod
    def initFromAerodynFile(cls, aerodynFile):
        """convenience method for initializing with AeroDyn formatted files

        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file

        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object

        """

        af = Airfoil.initFromAerodynFile(aerodynFile)
        alpha, Re, cl, cd, cm = af.createDataGrid()
        return cls(alpha, Re, cl, cd, cm)

    @classmethod
    def initFromCST(cls, CST, airfoil_analysis_options, airfoilNum=0):
        """convenience method for initializing with AeroDyn formatted files

        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file

        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object

        """
        af = Airfoil.initFromCST(CST, airfoil_analysis_options)
        failure = af.failure
        # For 3D Correction TODO
        # r_over_R = 0.5
        # chord_over_r = 0.15
        # tsr = 7.55
        # af3D = af.correction3D(r_over_R, chord_over_r, tsr)
        cd_max = 1.5
        af_extrap1 = af.extrapolate(cd_max)
        alpha, Re, cl, cd, cm = af_extrap1.createDataGrid()

        return cls(alpha, Re, cl, cd, cm, afp=CST, airfoil_analysis_options=airfoil_analysis_options, airfoilNum=airfoilNum, failure=failure)

    @classmethod
    def initFromNACA(cls, NACA, alphas, Re, airfoil_analysis_options, ComputeGradient=False):
        """convenience method for initializing with AeroDyn formatted files

        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file

        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object

        """
        af = Airfoil.initFromNACA(NACA, alphas, [Re], airfoil_analysis_options, ComputeGradient=ComputeGradient)

        r_over_R = 0.5
        chord_over_r = 0.15
        tsr = 7.55
        cd_max = 1.5
        # af3D = af.correction3D(r_over_R, chord_over_r, tsr)
        # af_extrap1 = af3D.extrapolate(cd_max)
        af_extrap1 = af.extrapolate(cd_max)
        alpha, Re, cl, cd, cm = af_extrap1.createDataGrid()

        return cls(alpha, Re, cl, cd, cm, CST=NACA, cl_2D=af.polars[0].cl, cd_2D=af.polars[0].cd, alpha_2D=af.polars[0].alpha, dcl_dafp=af.polars[0].dcl_dafp, dcd_dafp=af.polars[0].dcd_dafp, alphas_freeform=af.polars[0].alphas_freeform)

    @classmethod
    def initFromInput(cls, alpha, Re, cl, cd, cm=None):
        """convenience method for initializing with AeroDyn formatted files

        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file

        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object

        """
        return cls(alpha, [Re], cl, cd, cm)

    def evaluate(self, alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number.

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        This method uses a spline so that the output is continuously differentiable, and
        also uses a small amount of smoothing to help remove spurious multiple solutions.

        """
        cl = self.cl_spline.ev(alpha, Re)
        cd = self.cd_spline.ev(alpha, Re)

        return cl, cd

    def evaluate_direct(self, alpha, Re, computeAlphaGradient=False, computeAFPGradient=False):
        if self.afp is not None and abs(np.degrees(alpha)) < self.airfoil_analysis_options['maxDirectAoA']:
            if alpha in self.alpha_storage and alpha in self.dalpha_storage:
                index = self.alpha_storage.index(alpha)
                cl = self.cl_storage[index]
                cd = self.cd_storage[index]
                if computeAlphaGradient:
                    index = self.dalpha_storage.index(alpha)
                    dcl_dalpha = self.dcl_storage[index]
                    dcd_dalpha = self.dcd_storage[index]
                if computeAFPGradient and alpha in self.dalpha_dafp_storage:
                    index = self.dalpha_dafp_storage.index(alpha)
                    dcl_dafp = self.dcl_dafp_storage[index]
                    dcd_dafp = self.dcd_dafp_storage[index]
                else:
                    dcl_dafp = np.zeros(8)
                    dcd_dafp = np.zeros(8)
                dcl_dRe = 0.0
                dcd_dRe = 0.0
            else:
                airfoil_shape_file = None
                x, y = self.afp_to_coordinates()
                if self.airfoil_analysis_options['AnalysisMethod'] == 'XFOIL':
                    airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
                    airfoil.re = self.Re[0]
                    airfoil.mach = 0.00
                    airfoil.iter = 100
                    if computeAlphaGradient:
                        cs_step = 1e-20
                        angle = complex(np.degrees(alpha), cs_step)
                        cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
                        dcl_dalpha, dcd_dalpha = 180.0/np.pi*np.imag(deepcopy(cl))/ cs_step, 180.0/np.pi*np.imag(deepcopy(cd)) / cs_step
                        cl, cd = np.real(np.asscalar(cl)), np.real(np.asscalar(cd))
                        #cl2, cd2, cm2, lexitflag = airfoil.solveAlpha(np.degrees(alpha))
                        #cl2, cd2 = np.asscalar(cl2), np.asscalar(cd2)
                        #print "Diff", cl2 - cl, cd2-cd
                        #if lexitflag or abs(cl) > 2.5 or cd < 0.000001 or cd > 1.5 or not np.isfinite(cd) or not np.isfinite(cl):
                        #    print "flag", self.afp, "alpha", np.degrees(alpha)
                        if abs(dcl_dalpha) > 10.0 or abs(dcd_dalpha) > 10.0:
                        #    print "ERROR dcl", self.afp, "alpha", np.degrees(alpha)
                            #airfoil.iter = 500
                            fd_step = self.airfoil_analysis_options['fd_step']
                            cl, cd, cm, lexitflag = airfoil.solveAlpha(np.degrees(alpha))
                            if lexitflag or abs(cl) > 2.5 or cd < 0.000001 or cd > 1.5 or not np.isfinite(cd) or not np.isfinite(cl):
                                print "flag1", self.afp, "alpha", np.degrees(alpha)
                            cl, cd = np.asscalar(cl), np.asscalar(cd)
                            angle2 = np.degrees(alpha + fd_step)
                            cl2, cd2, cm2, lexitflag = airfoil.solveAlpha(angle2)
                            if lexitflag or abs(cl) > 2.5 or cd < 0.000001 or cd > 1.5 or not np.isfinite(cd) or not np.isfinite(cl):
                                print "flag2", self.afp, "alpha", np.degrees(alpha)
                            cl2, cd2 = np.asscalar(cl2), np.asscalar(cd2)
                            dcl_dalpha, dcd_dalpha = (cl2-cl)/ fd_step, (cd2-cd)/ fd_step
                    else:
                        cl, cd, cm, lexitflag = airfoil.solveAlpha(np.degrees(alpha))
                        cl, cd = np.asscalar(cl), np.asscalar(cd)
                    if computeAFPGradient:
                        dcl_dafp, dcd_dafp = self.xfoilGradients(alpha, Re)
                        if np.any(abs(dcl_dafp)) > 100.0 or np.any(abs(dcd_dafp) > 100.0):
                            print "ERROR 2"

                else:
                    if computeAFPGradient or computeAlphaGradient:
                        cl, cd, dcl_dafp, dcd_dafp, dcl_dalpha2, dcd_dalpha2 = self.cfdSolve(alpha, Re, ComputeGradients=True, GenerateMESH=True, airfoilNum=self.airfoilNum)
                    else:
                        cl, cd = self.cfdSolve(alpha, Re, ComputeGradients=False, GenerateMESH=True, airfoilNum=self.airfoilNum)
                    if computeAlphaGradient:
                        fd_step = 1e-4
                        cl2, cd2 = self.cfdSolve(alpha+fd_step, Re, ComputeGradients=False, GenerateMESH=True, airfoilNum=self.airfoilNum)
                        dcl_dalpha, dcd_dalpha = (cl2-cl)/fd_step, (cd2-cd)/fd_step
                    lexitflag = 0
                if lexitflag or abs(cl) > 2.5 or cd < 0.000001 or cd > 1.5 or not np.isfinite(cd) or not np.isfinite(cl):
                    print "error cl"
                    cl = self.cl_spline.ev(alpha, Re)
                    cd = self.cd_spline.ev(alpha, Re)
                    tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
                    tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees
                    dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
                    dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)
                dcl_dRe = 0.0
                dcd_dRe = 0.0
                self.cl_storage.append(cl)
                self.cd_storage.append(cd)
                self.alpha_storage.append(alpha)
                if computeAlphaGradient:
                    self.dcl_storage.append(dcl_dalpha)
                    self.dcd_storage.append(dcd_dalpha)
                    self.dalpha_storage.append(alpha)
                if computeAFPGradient:
                    self.dcl_dafp_storage.append(dcl_dafp)
                    self.dcd_dafp_storage.append(dcd_dafp)
                    self.dalpha_dafp_storage.append(alpha)
        else:
            cl = self.cl_spline.ev(alpha, Re)
            cd = self.cd_spline.ev(alpha, Re)
            tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
            tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees

            dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
            dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

            if self.one_Re:
                dcl_dRe = 0.0
                dcd_dRe = 0.0
            else:
                dcl_dRe = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
                dcd_dRe = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
            if computeAFPGradient and self.afp is not None:
                dcl_dafp, dcd_dafp = self.splineFreeFormGrad(alpha, Re)
            else:
                dcl_dafp, dcd_dafp = 0.0, 0.0
        if computeAFPGradient:
            try:
                return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp
            except:
                raise
        elif computeAlphaGradient:
            return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe
        else:
            return cl, cd

    def derivatives(self, alpha, Re):

        # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
        tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
        tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees

        dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
        dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

        if self.one_Re:
            dcl_dRe = 0.0
            dcd_dRe = 0.0
        else:
            dcl_dRe = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
            dcd_dRe = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    def afp_to_coordinates(self):
        if self.airfoil_analysis_options['AirfoilParameterization'] == 'CST':
            [x, y] = cst_to_coordinates_full(self.afp)
        return x, y

    def splineFreeFormGrad(self, alpha, Re):
        dcl_dafp, dcd_dafp = np.zeros(8), np.zeros(8)
        fd_step = self.airfoil_analysis_options['fd_step']
        cl_cur = self.cl_spline.ev(alpha, Re)
        cd_cur = self.cd_spline.ev(alpha, Re)
        for i in range(8):
            cl_new_fd = self.cl_splines_new[i].ev(alpha, self.Re)
            cd_new_fd = self.cd_splines_new[i].ev(alpha, self.Re)
            dcl_dafp[i] = (cl_new_fd - cl_cur) / fd_step
            dcd_dafp[i] = (cd_new_fd - cd_cur) / fd_step
        return dcl_dafp, dcd_dafp

    def xfoilGradients(self, alpha, Re):
        alpha = np.degrees(alpha)
        wl, wu, N, dz = CST_to_kulfan(self.afp)
        step_size = 1e-20
        cs_step = complex(0, step_size)
        dcl_dafp, dcd_dafp = np.zeros(8), np.zeros(8)
        lexitflag = np.zeros(8)
        for i in range(len(wl)):
            wl_complex = deepcopy(wl.astype(complex))
            wu_complex = deepcopy(wu.astype(complex))
            wl_complex[i] += cs_step
            cl_complex, cd_complex, lexitflag[i] = self.xfoilFlowComplex(alpha, wl_complex, wu_complex, N, dz)
            dcl_dafp[i], dcd_dafp[i] = np.imag(cl_complex)/step_size, np.imag(cd_complex)/step_size
            wl_complex = deepcopy(wl.astype(complex))
            wu_complex = deepcopy(wu.astype(complex))
            wu_complex[i] += cs_step
            cl_complex, cd_complex, lexitflag[i+4] = self.xfoilFlowComplex(alpha, wl_complex, wu_complex, N, dz)
            dcl_dafp[i+4], dcd_dafp[i+4] = np.imag(cl_complex)/step_size, np.imag(cd_complex)/step_size
            if lexitflag[i] or lexitflag[i+4] or abs(dcl_dafp[i+4]) > 100.0 or abs(dcd_dafp[i+4]) > 100.0 or abs(dcl_dafp[i]) > 100.0 or abs(dcd_dafp[i]) > 100.0:
                print "ERROR"
                fd_step = self.airfoil_analysis_options['fd_step']
                wl_fd1 = np.real(deepcopy(wl))
                wl_fd2 = np.real(deepcopy(wl))
                wl_fd1[i] -= 0.0#fd_step
                wl_fd2[i] += fd_step
                cl_fd1, cd_fd1, flag1 = self.xfoilFlowReal(alpha, wl_fd1, wu, N, dz)
                cl_fd2, cd_fd2, flag2 = self.xfoilFlowReal(alpha, wl_fd2, wu, N, dz)
                lexitflag[i] = np.logical_or(flag1, flag2)
                dcl_dafp[i] = (cl_fd2 - cl_fd1)/fd_step #(2.*fd_step)
                dcd_dafp[i] = (cd_fd2 - cd_fd1)/fd_step #(2.*fd_step)
                wu_fd1 = np.real(deepcopy(wu))
                wu_fd2 = np.real(deepcopy(wu))
                wu_fd1[i] -= 0.0#fd_step
                wu_fd2[i] += fd_step
                cl_fd1, cd_fd1, flag1 = self.xfoilFlowReal(alpha, wl, wu_fd1, N, dz)
                cl_fd2, cd_fd2, flag2 = self.xfoilFlowReal(alpha, wl, wu_fd2, N, dz)
                lexitflag[i+4] = np.logical_or(flag1, flag2)
                dcl_dafp[i+4] = (cl_fd2 - cl_fd1)/fd_step #(2.*fd_step)
                dcd_dafp[i+4] = (cd_fd2 - cd_fd1)/fd_step #(2.*fd_step)
                if lexitflag[i] or lexitflag[i+4] or abs(dcl_dafp[i+4]) > 100.0 or abs(dcd_dafp[i+4]) > 100.0 or abs(dcl_dafp[i]) > 100.0 or abs(dcd_dafp[i]) > 100.0:
                    cl_cur = self.cl_spline.ev(np.radians(alpha), self.Re)
                    cd_cur = self.cd_spline.ev(np.radians(alpha), self.Re)
                    cl_new_fd = self.cl_splines_new[i].ev(np.radians(alpha), self.Re)
                    cd_new_fd = self.cd_splines_new[i].ev(np.radians(alpha), self.Re)
                    dcl_dafp[i] = (cl_new_fd - cl_cur) / fd_step
                    dcd_dafp[i] = (cd_new_fd - cd_cur) / fd_step
                    cl_new_fd = self.cl_splines_new[i+4].ev(np.radians(alpha), self.Re)
                    cd_new_fd = self.cd_splines_new[i+4].ev(np.radians(alpha), self.Re)
                    dcl_dafp[i+4] = (cl_new_fd - cl_cur) / fd_step
                    dcd_dafp[i+4] = (cd_new_fd - cd_cur) / fd_step
                #print "derivative CST fail", alpha
        for i in range(8):
            if lexitflag[i]:
                af1 = Airfoil.initFromCST(self.afp, self.airfoil_analysis_options)
                af_extrap11 = af1.extrapolate(1.5)
                alphas_cur, Re_cur, cl_cur, cd_cur, cm_cur = af_extrap11.createDataGrid()
                cl_spline_cur = Akima(alphas_cur, cl_cur)
                cd_spline_cur = Akima(alphas_cur, cd_cur)
                cl_fd_cur, _, _, _ = cl_spline_cur.interp(alpha)
                cd_fd_cur, _, _, _ = cd_spline_cur.interp(alpha)
                afp_new = deepcopy(self.afp)
                afp_new[i] += fd_step
                af = Airfoil.initFromCST(afp_new, self.airfoil_analysis_options)
                af_extrap1 = af.extrapolate(1.5)
                alphas_new, Re_new, cl_new, cd_new, cm_new = af_extrap1.createDataGrid()
                cl_spline = Akima(alphas_new, cl_new)
                cd_spline = Akima(alphas_new, cd_new)
                cl_fd_new, _, _, _ = cl_spline.interp(alpha)
                cd_fd_new, _, _, _ = cd_spline.interp(alpha)
                dcl_dafp[i] = (cl_fd_new - cl_fd_cur) / fd_step
                dcd_dafp[i] = (cd_fd_new - cd_fd_cur) / fd_step
        return dcl_dafp, dcd_dafp

    def xfoilFlowComplex(self, alpha, wl_complex, wu_complex, N, dz):
        airfoil_shape_file = None
        [x, y] = cst_to_coordinates_complex(wl_complex, wu_complex, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.Re
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl_complex, cd_complex, cm_complex, lexitflag = airfoil.solveAlphaComplex(alpha)
        return deepcopy(cl_complex), deepcopy(cd_complex), deepcopy(lexitflag)

    def xfoilFlowReal(self, alpha, wl, wu, N, dz):
        airfoil_shape_file = None
        [x, y] = cst_to_coordinates(wl, wu, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.Re
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl, cd, cm, lexitflag = airfoil.solveAlpha(alpha)
        return np.asscalar(cl), np.asscalar(cd), deepcopy(lexitflag)

    def cfdSolve(self, alpha, Re, ComputeGradients=False, GenerateMESH=True, airfoilNum=0):

        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
        config_filename = basepath + os.path.sep + self.airfoil_analysis_options['cfdConfigFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = self.airfoil_analysis_options['CFDprocessors']
        config.EXT_ITER    = self.airfoil_analysis_options['CFDiterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = 'mesh_AIRFOIL'+str(airfoilNum+1)+'.su2'
        config.CONSOLE = 'QUIET'

        if GenerateMESH:

            # Create airfoil coordinate file for SU2
            [x, y] = cst_to_coordinates_full(self.afp)
            airfoilFile = 'airfoil_shape.dat'
            coord_file = open(airfoilFile, 'w')
            print >> coord_file, 'Airfoil ' + str(airfoilNum+1)
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])
            coord_file.close()
            oldstdout = sys.stdout
            sys.stdout = open('output_meshes_stdout.txt', 'w')
            konfig = copy.deepcopy(config)
            konfig.MESH_OUT_FILENAME = meshFileName
            konfig.DV_KIND = 'AIRFOIL'
            tempname = 'config_DEF.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            # must run with rank 1
            processes = konfig['NUMBER_PART']
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = 'SU2_DEF ' + tempname
            the_Command = base_Command % the_Command
            sys.stdout.flush()
            proc = subprocess.Popen( the_Command, shell=True    ,
                             stdout=sys.stdout      ,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
            proc.stderr.close()
            #proc.stdin.write(airfoilFile+'\n')
            proc.stdin.write('airfoil_shape.dat\n')
            proc.stdin.write('Selig\n')
            proc.stdin.write('1.0\n')
            proc.stdin.write('Yes\n')
            proc.stdin.write('clockwise\n')
            proc.stdin.close()
            return_code = proc.wait()

            restart = False
            sys.stdout = oldstdout
        else:
            restart = True

        config.MESH_FILENAME = meshFileName #'mesh_out.su2' # basepath + os.path.sep + 'mesh_AIRFOIL.su2'
        state.FILES.MESH = config.MESH_FILENAME
        config.AoA = np.degrees(alpha)
        Uinf = 10.0
        Ma = Uinf / 340.29  # Speed of sound at sea level
        x_vel = Uinf * cos(alpha)
        y_vel = Uinf * sin(alpha)
        config.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
        config.MACH_NUMBER = Ma
        config.REYNOLDS_NUMBER = 5e5 #Re

        if restart:
            config.RESTART_SOL = 'YES'
            config.RESTART_FLOW_FILENAME = 'solution_flow_AIRFOIL' + str(airfoilNum+1) +'.dat'
            config.SOLUTION_FLOW_FILENAME = 'solution_flow_SOLVED_AIRFOIL' + str(airfoilNum+1) + '.dat'
        else:
            config.RESTART_SOL = 'NO'
            config.SOLUTION_FLOW_FILENAME = 'solution_flow_AIRFOIL' + str(airfoilNum+1) + '.dat'
        oldstdout = sys.stdout
        sys.stdout = open('output_cfd_direct.txt', 'w')
        info = SU2.run.direct(config)
        sys.stdout = oldstdout
        state.update(info)
        cl = state.FUNCTIONS.LIFT
        cd = state.FUNCTIONS.DRAG
        #cd = SU2.eval.func('DRAG', config, state)
        #cl = SU2.eval.func('LIFT', config, state)

        if ComputeGradients:
            config.RESTART_SOL = 'NO'
            mesh_data = SU2.mesh.tools.read(config.MESH_FILENAME)
            points_sorted, loop_sorted = SU2.mesh.tools.sort_airfoil(mesh_data, marker_name='airfoil')
            sys.stdout = open('output_cfd_adjoint.txt', 'w')
            SU2.io.restart2solution(config, state)
            # RUN FOR DRAG GRADIENTS

            config.OBJECTIVE_FUNCTION = 'DRAG'
            info = SU2.run.adjoint(config)
            state.update(info)
            dcd_dx, xl, xu = self.su2Gradient(loop_sorted)
            dcd_dalpha = state.HISTORY.ADJOINT_DRAG.Sens_AoA[-1]
            config.OBJECTIVE_FUNCTION = 'LIFT'
            info = SU2.run.adjoint(config)
            state.update(info)
            dcl_dx, xl, xu = self.su2Gradient(loop_sorted)
            dcl_dalpha = state.HISTORY.ADJOINT_LIFT.Sens_AoA[-1]

            dz = 0
            n = 8
            #fd_step = self.airfoil_analysis_options['fd_step']
            m = 200
            dx_dafp = np.zeros((n, m))
            wl_original, wu_original, N, dz = CST_to_kulfan(self.afp)
            #yl_old, yu_old = cst_to_y_coordinates_given_x(wl_original, wu_original, N, dz, xl, xu)
            # grad_type = self.airfoil_analysis_options['grad_type']
            # if grad_type == 'FD':
            #     for i in range(0, n):
            #         wl_new = deepcopy(wl_original)
            #         wu_new = deepcopy(wu_original)
            #         if i < n/2:
            #             wl_new[i] += fd_step
            #         else:
            #             wu_new[i-4] += fd_step
            #
            #         yl_new, yu_new = cst_to_y_coordinates_given_x(wl_new, wu_new, N, dz, xl, xu)
            #         for j in range(m):
            #             if i < n/2:
            #                 if j < len(yl_new):
            #                     dx_dafp[i][j] = (yl_new[j] - yl_old[j]) / fd_step
            #                 else:
            #                     dx_dafp[i][j] = 0.0
            #             else:
            #                 if j > m - len(yu_new):
            #                     dx_dafp[i][j] = (yu_new[j- (m-len(yu_new))] - yu_old[j-(m-len(yu_new))]) / fd_step
            #                 else:
            #                     dx_dafp[i][j] = 0.0
            # elif grad_type == 'CS':
            step_size = self.airfoil_analysis_options['cs_step']
            cs_step = complex(0, step_size)

            for i in range(0, n):
                wl_new = deepcopy(wl_original.astype(complex))
                wu_new = deepcopy(wu_original.astype(complex))
                if i < n/2:
                    wl_new[i] += cs_step
                else:
                    wu_new[i-4] += cs_step
                yl_new, yu_new = cst_to_y_coordinates_given_x_Complexx(wl_new, wu_new, N, dz, xl, xu)
                for j in range(m):
                    if i < n/2:
                        if j < len(yl_new):
                            dx_dafp[i][j] = (np.imag(yl_new[j])) / step_size
                        else:
                            dx_dafp[i][j] = 0.0
                    else:
                        if j > m - len(yu_new):
                            dx_dafp[i][j] = -np.imag(yu_new[j- (m-len(yu_new))]) / step_size
                        else:
                            dx_dafp[i][j] = 0.0
            dy_dafp = cst_to_y_coordinates_derivatives(wl_original, wu_original, N, dz, xl, xu)
            dy_dafp = np.matrix(dy_dafp)
            dafp_dx = np.matrix(dx_dafp)
            dcl_dx = np.matrix(dcl_dx)
            dcd_dx = np.matrix(dcd_dx)
            dcl_dafp = dy_dafp * dcl_dx.T
            dcd_dafp = dy_dafp * dcd_dx.T

            dcl_dafp2 = dafp_dx * dcl_dx.T
            dcd_dafp2 = dafp_dx * dcd_dx.T
            sys.stdout = oldstdout
            print cl, cd, np.asarray(dcl_dafp).reshape(8), np.asarray(dcd_dafp).reshape(8), np.asarray(dcl_dafp2).reshape(8), np.asarray(dcd_dafp2).reshape(8)
            return cl, cd, np.asarray(dcl_dafp).reshape(8), np.asarray(dcd_dafp).reshape(8), dcl_dalpha, dcd_dalpha
        print "CL, CD", cl, cd
        return cl, cd

    def su2Gradient(self, loop_sorted):
        data = np.zeros([500, 8])
        with open('surface_adjoint.csv', 'rb') as f1:
            reader = csv.reader(f1, dialect='excel', quotechar='|')
            i = 0
            for row in reader:
                if i > 0:
                    data[i, :] = row[0:8]
                i += 1
            f1.close()
        N = 200
        dobj_dx_raw = data[:, 1][1:N+1].reshape(N,1)
        point_ids = data[:, 0][1:N+1].reshape(N,1)
        x = data[:, 6][1:N+1].reshape(N,1)
        y = data[:, 7][1:N+1].reshape(N,1)

        xu, xl, yu, yl, dobj_dxl, dobj_dxu = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),  np.zeros(0), np.zeros(0) #TODO: Initalize
        for i in range(N):
            index = np.where(point_ids == loop_sorted[i])[0][0]
            if i < N/2:
                xl = np.append(xl, x[index])
                yl = np.append(yl, y[index])
                dobj_dxl = np.append(dobj_dxl, dobj_dx_raw[index])
            else:
                xu = np.append(xu, x[index])
                yu = np.append(yu, y[index])
                dobj_dxu = np.append(dobj_dxu, dobj_dx_raw[index])
        return np.concatenate([dobj_dxl, dobj_dxu]), xl, xu




def cst_to_y_coordinates_given_x(wl, wu, N, dz, xl, xu):

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1
    yl = ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates
    return yl, yu

def cst_to_y_coordinates_derivatives(wl, wu, N, dz, xl, xu):

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1
    dyl = ClassShapeDerivative(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    dyu = ClassShapeDerivative(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates
    dyl_dzeros = np.zeros((len(wl), N-len(xl)))
    dyu_dzeros = np.zeros((len(wu), N-len(xu)))
    dyl_dw = np.hstack((dyl, dyl_dzeros))
    dyu_dw = np.hstack((dyu_dzeros, dyu))
    dy_dafp = np.vstack((dyl_dw, dyu_dw))

    return dy_dafp

def cst_to_y_coordinates_given_x_Complexx(wl, wu, N, dz, xl, xu):

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1
    yl = ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates
    return yl, yu

def cstComplex(alpha, Re, wl, wu, N, dz, Uinf):

    N = N
    dz = dz

    # Populate x coordinates
    x = np.ones((N, 1), dtype=complex)
    zeta = np.zeros((N, 1)) #, dtype=complex)
    for z in range(0, N):
        zeta[z] = 2.0 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cmath.cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind, dtype=complex)
    xu = np.zeros(N-zerind, dtype=complex)

    for z in range(len(xl)):
        xl[z] = x[z][0]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind][0]   # Upper surface x-coordinates

    yl = ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x), dtype=complex)
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1

    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
    airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates_complex.dat'

    coord_file = open(airfoil_shape_file, 'w')

    print >> coord_file, 'CST'
    for i in range(len(x)):
        print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

    coord_file.close()

    # read in coordinate file
    # with suppress_stdout_stderr():
    airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
    airfoil.re = self.Re
    airfoil.mach = 0.0 # Uinf / 340.29
    airfoil.iter = 100

    angle = deepcopy(alpha)
    cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
    if lexitflag:
        cl = -10.0
        cd = 0.0
        print "XFOIL FAILURE"
    return cl, cd, lexitflag

def cstReal(alpha, Re, wl, wu, N, dz, Uinf):

    # Populate x coordinates
    x = np.ones((N, 1))
    zeta = np.zeros((N, 1))
    for z in range(0, N):
        zeta[z] = 2 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind)
    xu = np.zeros(N-zerind)

    for z in range(len(xl)):
        xl[z] = x[z]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind]   # Upper surface x-coordinates

    yl = ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x))
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1

    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
    airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates.dat'

    coord_file = open(airfoil_shape_file, 'w')

    print >> coord_file, 'CST'
    for i in range(len(x)):
        print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

    coord_file.close()

    # read in coordinate file
    # with suppress_stdout_stderr():
    airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
    airfoil.re = self.Re
    airfoil.mach = 0.0 #Uinf / 340.29
    airfoil.iter = 100

    angle = alpha
    cl, cd, cm, lexitflag = airfoil.solveAlpha(angle)
    if lexitflag:
        cl = -10.0
        cd = 0.0
    return cl, cd, lexitflag

def CST_to_kulfan(CST):
    n1 = len(CST)/2
    wu = np.zeros(n1)
    wl = np.zeros(n1)
    for j in range(n1):
        wu[j] = CST[j]
        wl[j] = CST[j + n1]
    w1 = np.average(wl)
    w2 = np.average(wu)
    if w1 < w2:
        pass
    else:
        higher = wl
        lower = wu
        wl = lower
        wu = higher
    N = 200
    dz = 0.
    return wl, wu, N, dz

def cst_to_coordinates_full(CST):
    wl, wu, N, dz = CST_to_kulfan(CST)
    x = np.ones((N, 1))
    zeta = np.zeros((N, 1))
    for z in range(0, N):
        zeta[z] = 2 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind)
    xu = np.zeros(N-zerind)

    for z in range(len(xl)):
        xl[z] = x[z]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind]   # Upper surface x-coordinates

    yl = ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x))
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1
    return [x, y]

def cst_to_coordinates(wl, wu, N, dz):
    x = np.ones((N, 1))
    zeta = np.zeros((N, 1))
    for z in range(0, N):
        zeta[z] = 2 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind)
    xu = np.zeros(N-zerind)

    for z in range(len(xl)):
        xl[z] = x[z]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind]   # Upper surface x-coordinates

    yl = ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x))
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1
    return [x, y]

def cst_to_coordinates_complex(wl, wu, N, dz):
    # Populate x coordinates
    x = np.ones((N, 1), dtype=complex)
    zeta = np.zeros((N, 1)) #, dtype=complex)
    for z in range(0, N):
        zeta[z] = 2.0 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cmath.cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind, dtype=complex)
    xu = np.zeros(N-zerind, dtype=complex)

    for z in range(len(xl)):
        xl[z] = x[z][0]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind][0]   # Upper surface x-coordinates

    yl = ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x), dtype=complex)
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1
    return [x, y]


def ClassShape(w, x, N1, N2, dz):

    # Class function; taking input of N1 and N2
    C = np.zeros(len(x))
    for i in range(len(x)):
        C[i] = x[i]**N1*((1-x[i])**N2)

    # Shape function; using Bernstein Polynomials
    n = len(w) - 1  # Order of Bernstein polynomials

    K = np.zeros(n+1)
    for i in range(0, n+1):
        K[i] = factorial(n)/(factorial(i)*(factorial((n)-(i))))

    S = np.zeros(len(x))
    for i in range(len(x)):
        S[i] = 0
        for j in range(0, n+1):
            S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

    # Calculate y output
    y = np.zeros(len(x))
    for i in range(len(y)):
        y[i] = C[i] * S[i] + x[i] * dz

    return y

def ClassShapeComplex(w, x, N1, N2, dz):

    # Class function; taking input of N1 and N2
    C = np.zeros(len(x), dtype=complex)
    for i in range(len(x)):
        C[i] = x[i]**N1*((1-x[i])**N2)

    # Shape function; using Bernstein Polynomials
    n = len(w) - 1  # Order of Bernstein polynomials

    K = np.zeros(n+1, dtype=complex)
    for i in range(0, n+1):
        K[i] = mpmath.factorial(n)/(mpmath.factorial(i)*(mpmath.factorial((n)-(i))))

    S = np.zeros(len(x), dtype=complex)
    for i in range(len(x)):
        S[i] = 0
        for j in range(0, n+1):
            S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

    # Calculate y output
    y = np.zeros(len(x), dtype=complex)
    for i in range(len(y)):
        y[i] = C[i] * S[i] + x[i] * dz

    return y

def ClassShapeDerivative(w, x, N1, N2, dz):
    n = len(w) - 1
    dy_dw = np.zeros((n+1, len(x)))
    for i in range(len(x)):
        for j in range(0, n+1):
            dy_dw[j][i] = x[i]**N1*((1-x[i])**N2) * factorial(n)/(factorial(j)*(factorial((n)-(j)))) * x[i]**(j) * ((1-x[i])**(n-(j)))
    y = ClassShape(w, x, N1, N2, dz)

    dy_total = np.zeros_like(dy_dw)
    for i in range(len(y)):
        if i == 0 or i == len(y) - 1:
            norm_y = 0
        else:
            # normal vector of forward line adjacent point
            dx1 = x[i+1] - x[i]
            dy1 = y[i+1] - y[i]
            dnormy1 = dx1 - -dx1
            dnormx1 = -dy1 - dy1

            # normal vector of backward line with adjacent point
            dx2 = x[i] - x[i-1]
            dy2 = y[i] - y[i-1]
            dnormy2 = dx2 - -dx2
            dnormx2 = -dy2 - dy2

            dnormx = dnormx1 + dnormx2
            dnormy = dnormy1 + dnormy2

            norm_y = -dnormy / np.sqrt(dnormy**2 + dnormx**2)
            print norm_y, x[i], y[i]
            if norm_y > 1.0:
                print "NORM", norm_y

        for j in range(0, n+1):
            dy_total[j][i] = dy_dw[j][i] * norm_y
    return dy_total

def getCoordinates(CST):
    try:
        wl, wu, N, dz = CST_to_kulfan(CST[0])
    except:
        wl, wu, N, dz = CST_to_kulfan(CST)
    x = np.ones((N, 1))
    zeta = np.zeros((N, 1))
    for z in range(0, N):
        zeta[z] = 2 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind)
    xu = np.zeros(N-zerind)

    for z in range(len(xl)):
        xl[z] = x[z]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind]   # Upper surface x-coordinates

    yl = ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x))
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1

    return xl, xu, yl, yu

def cfdDirectSolve(alpha, Re, afp, airfoil_analysis_options, GenerateMESH=False, airfoilNum=0):
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
        config_filename = basepath + os.path.sep + airfoil_analysis_options['cfdConfigFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = airfoil_analysis_options['CFDprocessors']
        config.EXT_ITER    = airfoil_analysis_options['CFDiterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = 'mesh_AIRFOIL'+str(airfoilNum+1)+'.su2'
        config.CONSOLE = 'QUIET'

        if GenerateMESH:

            # Create airfoil coordinate file for SU2
            [x, y] = cst_to_coordinates_full(afp)
            airfoilFile = 'airfoil_shape'+str(airfoilNum+1)+'.dat'
            coord_file = open(airfoilFile, 'w')
            print >> coord_file, 'Airfoil ' + str(airfoilNum+1)
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])
            coord_file.close()
            oldstdout = sys.stdout
            sys.stdout = open('output_meshes_stdout.txt', 'w')
            konfig = copy.deepcopy(config)
            konfig.MESH_OUT_FILENAME = meshFileName
            konfig.DV_KIND = 'AIRFOIL'
            tempname = 'config_DEF.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            # must run with rank 1
            processes = konfig['NUMBER_PART']
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = 'SU2_DEF ' + tempname
            the_Command = base_Command % the_Command
            sys.stdout.flush()
            proc = subprocess.Popen( the_Command, shell=True    ,
                             stdout=sys.stdout      ,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.write(airfoilFile+'\n')
            #proc.stdin.write('airfoil_shape.dat\n')
            proc.stdin.write('Selig\n')
            proc.stdin.write('1.0\n')
            proc.stdin.write('Yes\n')
            proc.stdin.write('clockwise\n')
            proc.stdin.close()
            return_code = proc.wait()

            restart = False
            sys.stdout = oldstdout
        else:
            restart = True

        config.MESH_FILENAME = meshFileName #'mesh_out.su2' # basepath + os.path.sep + 'mesh_AIRFOIL.su2'
        state.FILES.MESH = config.MESH_FILENAME
        config.AoA = np.degrees(alpha)
        Uinf = 10.0
        Ma = Uinf / 340.29  # Speed of sound at sea level
        x_vel = Uinf * cos(alpha)
        y_vel = Uinf * sin(alpha)
        config.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
        config.MACH_NUMBER = Ma
        config.REYNOLDS_NUMBER = 5e5 #Re

        if restart:
            config.RESTART_SOL = 'YES'
            config.RESTART_FLOW_FILENAME = 'solution_flow_AIRFOIL' + str(airfoilNum+1) +'.dat'
            config.SOLUTION_FLOW_FILENAME = 'solution_flow_SOLVED_AIRFOIL' + str(airfoilNum+1) + '.dat'
        else:
            config.RESTART_SOL = 'NO'
            config.SOLUTION_FLOW_FILENAME = 'solution_flow_AIRFOIL' + str(airfoilNum+1) + '.dat'

        cd = SU2.eval.func('DRAG', config, state)
        cl = SU2.eval.func('LIFT', config, state)
        print cl, cd
        return cl, cd

def cfdDirectSolveParallel(alphas, Re, afp, airfoil_analysis_options):
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CFDFiles')

        config_filename = basepath + os.path.sep + airfoil_analysis_options['cfdConfigFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = airfoil_analysis_options['CFDprocessors']
        config.EXT_ITER    = airfoil_analysis_options['CFDiterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = basepath + os.path.sep + 'mesh_AIRFOIL_parallel.su2'
        config.CONSOLE = 'QUIET'

        # Create airfoil coordinate file for SU2
        [x, y] = cst_to_coordinates_full(afp)
        airfoilFile = basepath + os.path.sep + 'airfoil_shape_parallel.dat'
        coord_file = open(airfoilFile, 'w')
        print >> coord_file, 'Airfoil Parallel'
        for i in range(len(x)):
            print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])
        coord_file.close()
        oldstdout = sys.stdout
        sys.stdout = open(basepath + os.path.sep + 'output_meshes_stdout.txt', 'w')
        konfig = copy.deepcopy(config)
        konfig.MESH_OUT_FILENAME = meshFileName
        konfig.DV_KIND = 'AIRFOIL'
        tempname = 'config_DEF.cfg'
        konfig.dump(tempname)
        SU2_RUN = os.environ['SU2_RUN']
        base_Command = os.path.join(SU2_RUN,'%s')
        the_Command = 'SU2_DEF ' + tempname
        the_Command = base_Command % the_Command
        sys.stdout.flush()
        proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=sys.stdout      ,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
        proc.stderr.close()
        proc.stdin.write(airfoilFile+'\n')
        proc.stdin.write('Selig\n')
        proc.stdin.write('1.0\n')
        proc.stdin.write('Yes\n')
        proc.stdin.write('clockwise\n')
        proc.stdin.close()
        return_code = proc.wait()

        restart = False
        sys.stdout = oldstdout

        config.MESH_FILENAME = meshFileName
        state.FILES.MESH = config.MESH_FILENAME
        Uinf = 10.0
        Ma = Uinf / 340.29  # Speed of sound at sea level
        config.MACH_NUMBER = Ma
        config.REYNOLDS_NUMBER = Re

        if restart:
            config.RESTART_SOL = 'YES'
            config.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_AIRFOIL_parallel.dat'
            config.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_SOLVED_AIRFOIL_parallel.dat'
        else:
            config.RESTART_SOL = 'NO'
            config.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_AIRFOIL_parallel.dat'
        cl = np.zeros(len(alphas))
        cd = np.zeros(len(alphas))
        alphas = np.degrees(alphas)
        procTotal = []
        konfigTotal = []
        for i in range(len(alphas)):
            x_vel = Uinf * cos(np.radians(alphas[i]))
            y_vel = Uinf * sin(np.radians(alphas[i]))
            config.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
            config.AoA = alphas[i]
            config.CONV_FILENAME = basepath + os.path.sep + 'history_'+str(int(alphas[i]))
            state = SU2.io.State(state)
            konfig = copy.deepcopy(config)
            # setup direct problem
            konfig['MATH_PROBLEM']  = 'DIRECT'
            konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'

            # Run Solution
            tempname = basepath + os.path.sep + 'config_CFD'+str(int(alphas[i]))+'.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            sys.path.append( SU2_RUN )

            mpi_Command = 'mpirun -n %i %s'

            processes = konfig['NUMBER_PART']

            the_Command = 'SU2_CFD ' + tempname
            the_Command = base_Command % the_Command
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError , 'could not find an mpi interface'
            the_Command = mpi_Command % (processes,the_Command)
            sys.stdout.flush()
            cfd_output = open(basepath + os.path.sep + 'cfd_output'+str(i+1)+'.txt', 'w')
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=cfd_output      ,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.close()
            procTotal.append(deepcopy(proc))
            konfigTotal.append(deepcopy(konfig))

        for i in range(len(alphas)):
            while procTotal[i].poll() is None:
                pass
            konfig = konfigTotal[i]
            konfig['SOLUTION_FLOW_FILENAME'] = konfig['RESTART_FLOW_FILENAME']
            oldstdout = sys.stdout
            sys.stdout = oldstdout
            plot_format      = konfig['OUTPUT_FORMAT']
            plot_extension   = SU2.io.get_extension(plot_format)
            history_filename = konfig['CONV_FILENAME'] + plot_extension
            special_cases    = SU2.io.get_specialCases(konfig)

            final_avg = config.get('ITER_AVERAGE_OBJ',0)
            aerodynamics = SU2.io.read_aerodynamics( history_filename , special_cases, final_avg )
            config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM']  })
            info = SU2.io.State()
            info.FUNCTIONS.update( aerodynamics )
            state.update(info)

            cl[i], cd[i] = info.FUNCTIONS['LIFT'], info.FUNCTIONS['DRAG']
        return cl, cd

if __name__ == '__main__':

    import os
    from argparse import ArgumentParser, RawTextHelpFormatter

    # setup command line arguments
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                            description='Preprocessing airfoil data for wind turbine applications.')
    parser.add_argument('src_file', type=str, help='source file')
    parser.add_argument('--stall3D', type=str, nargs=3, metavar=('r/R', 'c/r', 'tsr'), help='2D data -> apply 3D corrections')
    parser.add_argument('--extrap', type=str, nargs=1, metavar=('cdmax'), help='3D data -> high alpha extrapolations')
    parser.add_argument('--blend', type=str, nargs=2, metavar=('otherfile', 'weight'), help='blend 2 files weight 0: sourcefile, weight 1: otherfile')
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--plot', action='store_true', help='plot data using matplotlib')
    parser.add_argument('--common', action='store_true', help='interpolate the data at different Reynolds numbers to a common set of angles of attack')


    # parse command line arguments
    args = parser.parse_args()
    fileOut = args.out

    if args.plot:
        import matplotlib.pyplot as plt

    # perform actions
    if args.stall3D is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_3D' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)
        floats = [float(var) for var in args.stall3D]
        af3D = af.correction3D(*floats)

        if args.common:
            af3D = af3D.interpToCommonAlpha()

        af3D.writeToAerodynFile(fileOut)

        if args.plot:

            for p, p3D in zip(af.polars, af3D.polars):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                plt.plot(p.alpha, p.cl, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cl, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend(loc='lower right')

                # plt.subplot(122)
                plt.figure()
                plt.plot(p.alpha, p.cd, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cd, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend(loc='upper center')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/stall3d.pdf')

            plt.show()


    elif args.extrap is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_extrap' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)

        afext = af.extrapolate(float(args.extrap[0]))

        if args.common:
            afext = afext.interpToCommonAlpha()

        afext.writeToAerodynFile(fileOut)

        if args.plot:

            for p, pext in zip(af.polars, afext.polars):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cl, 'r')
                p2, = plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='upper right')

                # plt.subplot(122)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cd, 'r')
                p2, = plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='lower right')

                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cm, 'r')
                p2, = plt.plot(p.alpha, p.cm, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('moment coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='upper right')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/extrap.pdf')

            plt.show()


    elif args.blend is not None:

        if fileOut is None:
            name1, ext = os.path.splitext(args.src_file)
            name2, ext = os.path.splitext(os.path.basename(args.blend[0]))
            fileOut = name1 + '+' + name2 + '_blend' + args.blend[1] + ext

        af1 = Airfoil.initFromAerodynFile(args.src_file)
        af2 = Airfoil.initFromAerodynFile(args.blend[0])
        afOut = af1.blend(af2, float(args.blend[1]))

        if args.common:
            afOut = afOut.interpToCommonAlpha()

        afOut.writeToAerodynFile(fileOut)



        if args.plot:

            for p in afOut.polars:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.text(0.6, 0.2, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.text(0.2, 0.8, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cm, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('moment coefficient')
                plt.text(0.2, 0.8, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

            plt.show()


