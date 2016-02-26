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



def run_command( Command ):
    """ runs os command with subprocess
        checks for errors from command
    """

    sys.stdout.flush()

    proc = subprocess.Popen( Command, shell=True    ,
                             stdout=sys.stdout      ,
                             stderr=subprocess.PIPE  )
    return_code = proc.wait()
    message = proc.stderr.read()

    if return_code < 0:
        message = "SU2 process was terminated by signal '%s'\n%s" % (-return_code,message)
        raise SystemExit , message
    elif return_code > 0:
        message = "Path = %s\nCommand = %s\nSU2 process returned error '%s'\n%s" % (os.path.abspath(','),Command,return_code,message)
        if return_code in return_code_map.keys():
            exception = return_code_map[return_code]
        else:
            exception = RuntimeError
        raise exception , message
    else:
        sys.stdout.write(message)

    return return_code

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
            if abs(self.alpha[i]) < 20.0 and self.cl[i] <= 0 and self.cl[i+1] >= 0:
                p = -self.cl[i] / (self.cl[i + 1] - self.cl[i])
                cm0 = self.cm[i] + p * (self.cm[i+1] - self.cm[i])
                found_zero_lift = True
                break

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

    def __init__(self, polars):
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
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 1000

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
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 1000

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
    def initFromCST(cls, CST, alphas, Re, CFDorXFOIL, processors=0, iterations=1000, polarType=Polar):
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

        try:
            n1 = len(CST[0])/2
            n2 = len(CST)
            CST = CST[0]
        except:
            n2 = 1
            n1 = len(CST)/2

        for i in range(n2):

            if CFDorXFOIL == 'XFOIL':
                wl, wu, N, dz = CST_to_kulfan(CST)

                [x, y] = cst_to_coordinates_from_kulfan(wl, wu, N, dz)

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
                airfoil.re = Re
                airfoil.mach = 0.00
                airfoil.iter = 100

                cl = np.zeros(len(alphas))
                cd = np.zeros(len(alphas))
                cm = np.zeros(len(alphas))
                to_delete = np.zeros(0)
                for j in range(len(alphas)):
                    cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(alphas[j])
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
                print CST
                polars.append(polarType(Re, alphas, cl, cd, cm))

            else:
                cl = np.zeros(len(alphas))
                cd = np.zeros(len(alphas))
                cm = np.zeros(len(alphas))
                for j in range(len(alphas)):
                    if j == 0:
                        mesh = True
                    else:
                        mesh = False
                    cl[j], cd[j],  = Airfoil.cfdGradients(CST, alphas[j], Re, iterations, processors, 'CS', Uinf=10.0, ComputeGradients=False, GenerateMESH=mesh)
                print "RESULT:, ", cl, cd, alphas, CST
                polars.append(polarType(Re, alphas, cl, cd, cm))


        return cls(polars)


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

    def __ClassShape(self, w, x, N1, N2, dz):

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

    def __ClassShapeComplex(self, w, x, N1, N2, dz):

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

    @classmethod
    def xfoilFlowGradients(self, CST, alpha, Re, FDorCS):

        x, y = cst_to_coordinates(CST)
        # read in coordinate file
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
        airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates.dat'
        # with suppress_stdout_stderr():
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = Re
        airfoil.mach = 0.00
        airfoil.iter = 100

        ## TODO Fix CS alpha and Re
        FDorCS = 'FD'
        if FDorCS == 'CS':
            alpha = np.degrees(alpha)
            step_size = 1e-20
            cs_step = complex(0, step_size)
            angle = alpha+cs_step
            cl_alpha, cd_alpha, cm, lexitflag = airfoil.solveAlphaComplex(angle)
            if lexitflag:
                cl_alpha = -10.0
                cd_alpha = 0.0
            dcl_dalpha = np.imag(cl_alpha)/np.imag(cs_step)
            dcd_dalpha = np.imag(cd_alpha)/np.imag(cs_step)

            airfoil.re = Re[0][0] + cs_step
            cl_Re, cd_Re, cm, lexitflag = airfoil.solveAlphaComplex(alpha)
            if lexitflag:
                cl_Re = -10.0
                cd_Re = 0.0
            dcl_dRe = np.imag(cl_Re)/np.imag(cs_step)
            dcd_dRe = np.imag(cd_Re)/np.imag(cs_step)
        else:
            angle = np.degrees(alpha)
            cl, cd, cm, lexitflag = airfoil.solveAlpha(angle)
            step_size = 1
            fd_step = step_size
            angle = alpha+fd_step
            angle = np.degrees(angle)
            cl_alpha, cd_alpha, cm, lexitflag = airfoil.solveAlpha(angle)
            if lexitflag:
                cl_alpha = -10.0
                cd_alpha = 0.0
            dcl_dalpha = (cl_alpha - cl) / fd_step
            dcd_dalpha = (cd_alpha - cd) / fd_step

            airfoil.re = Re[0][0] + fd_step
            cl_Re, cd_Re, cm, lexitflag = airfoil.solveAlpha(alpha)
            if lexitflag:
                cl_Re = -10.0
                cd_Re = 0.0
            dcl_dRe = (cl_Re - cl) / fd_step
            dcd_dRe = (cd_Re - cl) / fd_step

        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    @classmethod
    def xfoilGradients(self, CST, alpha, Re, FDorCS):
        alpha = np.degrees(alpha)
        def cstComplex(alpha, Re, wl, wu, N, dz, Uinf):
            # wl = self.wl
            # wu = self.wu
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
            airfoil.re = Re
            airfoil.mach = 0.0 # Uinf / 340.29
            airfoil.iter = 100

            angle = deepcopy(alpha)
            cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
            if lexitflag:
                cl = -10.0
                cd = 0.0
                print "XFOIL FAILURE"
            return cl, cd
            # error handling in case of XFOIL failure
            # for k in range(len(cl)):
            #     if cl[k] == -10.0:
            #         if k == 0:
            #             cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
            #             cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
            #         elif k == len(cl)-1:
            #             cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
            #             cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
            #         else:
            #             cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
            #             cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
            #     if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
            #         to_delete = np.append(to_delete, k)
            # cl = np.delete(cl, to_delete)
            # cd = np.delete(cd, to_delete)
            # alphas = np.delete(alphas, to_delete)

            # polars.append(polarType(Re, alphas, cl, cd, cm))

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
            airfoil.re = Re
            airfoil.mach = 0.0 #Uinf / 340.29
            airfoil.iter = 1000

            angle = alpha
            cl, cd, cm, lexitflag = airfoil.solveAlpha(angle)
            if lexitflag:
                cl = -10.0
                cd = 0.0
            return cl, cd



        n2 = 1
        n1 = len(CST)/2
        CST = np.array([CST])
        for i in range(n2):
            wu = np.zeros(n1, dtype=complex)
            wl = np.zeros(n1, dtype=complex)
            for j in range(n1):
                wu[j] = CST[i][j]
                wl[j] = CST[i][j + n1]
            # wu, wl = np.split(af_parameters[i], 2)
            w1 = np.average(wl)
            w2 = np.average(wu)
            if w1 < w2:
                pass
            else:
                higher = wl
                lower = wu
                wl = lower
                wu = higher
            N = 120
            dz = 0.
        self.wl = wl
        self.wu = wu
        self.N = N
        self.dz = dz


        dcl_dcst, dcd_dcst = np.zeros(8), np.zeros(8)
        cl, cd = cstReal(alpha, Re, np.real(wl), np.real(wu), N, dz, Uinf=10.0)

        if FDorCS == 'CS':
            step_size = 1e-20
            cs_step = complex(0, step_size)
            for i in range(len(wl)):

                wl_complex = deepcopy(wl)
                wl_complex[i] += cs_step
                cl_complex, cd_complex = cstComplex(alpha, Re, wl_complex, wu, N, dz, Uinf=10.0)
                dcl_dcst[i] = np.imag(cl_complex)/np.imag(cs_step)
                dcd_dcst[i] = np.imag(cd_complex)/np.imag(cs_step)
                wu_complex = deepcopy(wu)
                wu_complex[i] += cs_step
                cl_complex, cd_complex = cstComplex(alpha, Re, wl, wu_complex, N, dz, Uinf=10.0)
                dcl_dcst[i+4] = np.imag(cl_complex)/np.imag(cs_step)
                dcd_dcst[i+4] = np.imag(cd_complex)/np.imag(cs_step)
        else:
            step_size = 1e-6
            fd_step = step_size
            for i in range(len(wl)):
                wl_fd1 = np.real(deepcopy(wl))
                wl_fd2 = np.real(deepcopy(wl))
                wl_fd1[i] -= fd_step
                wl_fd2[i] += fd_step
                cl_fd1, cd_fd1 = cstReal(alpha, Re, wl_fd1, np.real(wu), N, dz, Uinf=10.0)
                cl_fd1, cd_fd1 = deepcopy(cl_fd1), deepcopy(cd_fd1)
                cl_fd2, cd_fd2 = cstReal(alpha, Re, wl_fd2, np.real(wu), N, dz, Uinf=10.0)
                cl_fd2, cd_fd2 = deepcopy(cl_fd2), deepcopy(cd_fd2)
                dcl_dcst[i] = (cl_fd2 - cl_fd1)/(2.*fd_step)
                dcd_dcst[i] = (cd_fd2 - cd_fd1)/(2.*fd_step)
                wu_fd1 = np.real(deepcopy(wu))
                wu_fd2 = np.real(deepcopy(wu))
                wu_fd1[i] -= fd_step
                wu_fd2[i] += fd_step
                cl_fd1, cd_fd1 = cstReal(alpha, Re, np.real(wl), wu_fd1, N, dz, Uinf=10.0)
                cl_fd1, cd_fd1 = deepcopy(cl_fd1), deepcopy(cd_fd1)
                cl_fd2, cd_fd2 = cstReal(alpha, Re, np.real(wl), wu_fd2, N, dz, Uinf=10.0)
                cl_fd2, cd_fd2 = deepcopy(cl_fd2), deepcopy(cd_fd2)
                dcl_dcst[i+4] = (cl_fd2 - cl_fd1)/(2.*fd_step)
                dcd_dcst[i+4] = (cd_fd2 - cd_fd1)/(2.*fd_step)

        return cl, cd, dcl_dcst, dcd_dcst

    @classmethod
    def cfdGradients(self, CST, alpha, Re, iterations, processors, FDorCS, Uinf, ComputeGradients, GenerateMESH=True):

        import os, sys, shutil, copy
        sys.path.append(os.environ['SU2_RUN'])
        # sys.path.append("/usr/local/bin")
        import SU2

        wl, wu, N, dz = CST_to_kulfan(CST)

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

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SU2_EDU/bin')
        airfoil_shape_file = basepath + os.path.sep + 'airfoil_shape.dat'

        coord_file = open(airfoil_shape_file, 'w')

        print >> coord_file, 'CST'
        for i in range(len(x)):
            print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

        coord_file.close()

        # if GenerateMESH:
        #
        #     ## Update mesh for specific airfoil (mesh deformation in SU2_EDU)
        #     basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SU2_EDU/bin')
        #     su2_file_execute = basepath + os.path.sep + 'SU2_EDU'
        #
        #     savedPath = os.getcwd()
        #     os.chdir(basepath)
        #     subprocess.call([su2_file_execute])
        #     os.chdir(savedPath)

        partitions = processors
        compute = True
        step = 1e-4
        iterations = iterations

        # Config and state
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
        # filename = basepath + os.path.sep + 'inv_NACA0012.cfg'
        # filename = basepath + os.path.sep + 'test_incomp_rans.cfg'
        # filename = basepath + os.path.sep + 'turb_nasa.cfg'
        filename = basepath + os.path.sep + 'su2_incomp_rans.cfg'

        config = SU2.io.Config(filename)
        state  = SU2.io.State()
        config.NUMBER_PART = 0
        config.EXT_ITER    = iterations
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SU2_EDU/bin')
        mesh_filename = basepath + os.path.sep + 'mesh_AIRFOIL.su2'
        config.MESH_FILENAME = mesh_filename
        if GenerateMESH:
            import os

            konfig = copy.deepcopy(config)

            tempname = 'config_DEF.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            # must run with rank 1
            processes = konfig['NUMBER_PART']
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = 'SU2_DEF ' + tempname
            the_Command = base_Command % the_Command
            # the_Command = build_command( the_Command , processes )

            sys.stdout.flush()

            proc = subprocess.Popen( the_Command, shell=True    ,
                             stdout=sys.stdout      ,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.write('airfoil_shape.dat\n')
            proc.stdin.write('Selig\n')
            proc.stdin.write('1.0\n')
            proc.stdin.write('Yes\n')
            proc.stdin.write('clockwise\n')

            # return_code = proc.wait()
            # message = proc.stderr.read()
            proc.stdin.close()
            # proc.stdout.close()
            return_code = proc.wait()
            # run_command( the_Command )

            # config.DV_VALUE_NEW = config.DV_VALUE
            from subprocess import Popen, PIPE, STDOUT
            # p = Popen(['myapp'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
            # konfig = copy.deepcopy(config)
            # tempname = 'config_DEF.cfg'
            # konfig.dump(tempname)
            # su2_file_execute = 'SU2_DEF ' + filename
            # subprocess.call(su2_file_execute)
            #
            # # info = SU2.run.DEF(config)
            #
            # stdout_data = p.communicate(input='data_to_write')[0]
            # sys.stdout.write('airfoil_shape.dat')
            # sys.stdout.write('Selig')
            # sys.stdout.write('1.0')
            # sys.stdout.write('Yes')
            # sys.stdout.write('clockwise')
            # state.update(info)

        config.NUMBER_PART = partitions
        config.WRT_CSV_SOL = 'YES'

        config.AoA = alpha #np.degrees(alpha)
        Ma = Uinf / 340.29  # Speed of sound at sea level
        x_vel = Uinf * cos(np.radians(alpha))
        y_vel = Uinf * sin(np.radians(alpha))
        config.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
        config.MACH_NUMBER = Ma
        config.REYNOLDS_NUMBER = Re
        config.FREESTREAM_DENSITY = 1.225
        config.FREESTREAM_VISCOSITY = 1.81206e-5


        # find solution files if they exist
        # state.find_files(config)
        ffdtag = []
        kind = []
        marker = []
        param = []
        scale = []
        for i in range(len(x)):
            ffdtag.append([])
            kind.append('HICKS_HENNE')
            marker.append(['airfoil'])
            if i < len(x) / 2.0:
                param.append([0.0, x[i]])
            else:
                param.append([1.0, x[i]])
            scale.append(1.0)
        config.DEFINITION_DV = dict(FFDTAG=ffdtag, KIND=kind, MARKER=marker, PARAM=param, SCALE=scale)

        restart = True

        if restart:
            print "restart"
            config.RESTART_SOL = 'YES'
            basepath2 = os.path.dirname(os.path.realpath(__file__))
            config.VOLUME_FLOW_FILENAME = basepath2 + os.path.sep + 'DIRECT/flow_' + str(wu[0]) + '_' + str(alpha)
            restart_file = basepath2 + os.path.sep + 'solution_flow_' + str(wu[0]) + '_' + str(alpha) + '.dat'
            if os.path.isfile(restart_file):
                config.RESTART_FLOW_FILENAME = basepath2 + os.path.sep + 'solution_flow_' + str(wu[0]) + '_' + str(alpha) + '.dat'
                config.EXT_ITER = iterations / 200
            else:
                config.RESTART_FLOW_FILENAME = basepath2 + os.path.sep + 'solution_flow.dat'
            config.SOLUTION_FLOW_FILENAME = basepath2 + os.path.sep + 'solution_flow_' + str(wu[0]) + '_' + str(alpha) + '.dat'
            state.find_files(config)
        else:
            basepath2 = os.path.dirname(os.path.realpath(__file__))
            # restart_file = basepath2 + os.path.sep + 'solution_flow_' + str(wu[0]) + '_' + str(alpha) + '.dat'
            # config.RESTART_FLOW_FILENAME = basepath2 + os.path.sep + 'solution_flow.dat'
            #config.SOLUTION_FLOW_FILENAME = basepath2 + os.path.sep + 'solution_flow_' + str(wu[0]) + '_' + str(alpha) + '.dat'
            #state.find_files(config)
            state.FILES.MESH = config.MESH_FILENAME

        cd = SU2.eval.func('DRAG', config, state)
        cl = SU2.eval.func('LIFT', config, state)
        cm = SU2.eval.func('MOMENT_Z', config, state)
        if ComputeGradients:
            # RUN FOR DRAG GRADIENTS
            info = SU2.run.adjoint(config)
            state.update(info)
            #SU2.io.restart2solution(config,state)
            # Gradient Projection
            info = SU2.run.projection(config, step)
            state.update(info)
            get_gradients = info.get('GRADIENTS')
            dcd_dx = get_gradients.get('DRAG')

            # RUN FOR LIFT GRADIENTS
            config.OBJECTIVE_FUNCTION = 'LIFT'
            info = SU2.run.adjoint(config)
            state.update(info)
            #SU2.io.restart2solution(config,state)
            # Gradient Projection
            info = SU2.run.projection(config, step)
            state.update(info)
            get_gradients = info.get('GRADIENTS')
            dcl_dx = get_gradients.get('LIFT')

            where_are_NaNs_cl = np.isnan(dcl_dx)
            dcl_dx[where_are_NaNs_cl] = 0.0

            where_are_NaNs_cl = np.isnan(dcl_dx)
            dcd_dx[where_are_NaNs_cl] = 0.0

            n = len(CST)
            m = len(dcd_dx)
            dcst_dx = np.zeros((n, m))

            wl_original, wu_original = wu, wl
            dz = 0.0
            N = 200
            coord_old = cst_to_coordinates_from_kulfan(wl_original, wu_original, N, dz)

            # design = [85, 79, 74, 70, 67, 63, 60, 56, 53, 50, 47, 43, 40, 37, 33, 29, 25, 21, 14, 115, 121, 126, 130, 133, 137, 140, 144, 147, 150, 153, 157, 160, 163, 167, 171, 175, 179, 186]
            design = range(0, len(x))
            # Gradients
            FDorCS = 'FD'
            if FDorCS == 'FD':
                fd_step = 1e-6
                for i in range(0, n):
                    wl_new = deepcopy(wl_original)
                    wu_new = deepcopy(wu_original)
                    if i < n/2:
                        wl_new[i] += fd_step
                    else:
                        wu_new[i-4] += fd_step
                    coor_new = cst_to_coordinates_from_kulfan(wl_new, wu_new, N, dz)
                    j = 0
                    for coor_d in design:
                        if (coor_new[1][coor_d] - coord_old[1][coor_d]).real == 0:
                            dcst_dx[i][j] = 0
                        else:
                            dcst_dx[i][j] = 1/((coor_new[1][coor_d] - coord_old[1][coor_d]).real / fd_step)
                        j += 1

            elif FDorCS == 'CS':
                step_size = 1e-20
                cs_step = complex(0, step_size)

                for i in range(0, n):
                    wl_new = deepcopy(wl_original.astype(complex))
                    wu_new = deepcopy(wu_original.astype(complex))
                    if i >= n/2:
                        wl_new[i-4] += cs_step
                    else:
                        wu_new[i] += cs_step
                    coor_new = cst_to_coordinates_complex(wl_new, wu_new, N, dz)
                    j = 0
                    for coor_d in design:
                        if coor_new[1][coor_d].imag == 0:
                            dcst_dx[i][j] = 0
                        else:
                            dcst_dx[i][j] = 1/(coor_new[1][coor_d].imag / np.imag(cs_step))
                        j += 1
            else:
                print 'Warning. FDorCS needs to be set to either FD or CS'
            dcst_dx = np.matrix(dcst_dx)
            dcl_dx = np.matrix(dcl_dx)
            dcd_dx = np.matrix(dcd_dx)

            dcl_dcst = dcst_dx * dcl_dx.T
            dcd_dcst = dcst_dx * dcd_dx.T

            # print cl, cd, dcl_dcst, dcd_dcst
            return cl, cd, dcl_dcst, dcd_dcst

        return cl, cd

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

def CST_to_kulfan(CST):
    n1 = len(CST)/2
    # CST = np.array([CST])

    wu = np.zeros(n1)
    wl = np.zeros(n1)
    for j in range(n1):
        wu[j] = CST[j]
        wl[j] = CST[j + n1]
    # wu, wl = np.split(af_parameters[i], 2)
    w1 = np.average(wl)
    w2 = np.average(wu)
    if w1 < w2:
        pass
    else:
        higher = wl
        lower = wu
        wl = lower
        wu = higher
    N = 120
    dz = 0.
    return wl, wu, N, dz

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

def cst_to_coordinates(CST):
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

def cst_to_coordinates_from_kulfan(wl, wu, N, dz):

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


def getCoordinates(CST):
    wl, wu, N, dz = CST_to_kulfan(CST[0])
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
    return xl, xu, yl, xu



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


class CCAirfoil:
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""
    # implements(AirfoilInterface)


    def __init__(self, alpha, Re, cl, cd, cm, CST=None):
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
        self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky, s=0.1)
        self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky, s=0.001)
        # test = self.cd_spline.ev(0.0, 1e6)
        # if test != 0.5 or test != 0.35:
        #     n = 1000
        #     alpha = np.linspace(-180, 180, n)
        #     cl = np.zeros(n)
        #     cd = np.zeros(n)
        #     for i in range(n):
        #         cl[i] = self.cl_spline.ev(np.radians(alpha[i]), 1e6)
        #         cd[i] = self.cd_spline.ev(np.radians(alpha[i]), 1e6)
        #     import matplotlib.pylab as plt
        #     import test
        #
        #     plt.figure()
        #     plt.plot(alpha, cl, label='XFOIL')
        #     plt.plot(test.alphas_wind, test.cl_wind, label='WIND')
        #     plt.legend()
        #
        #     plt.figure()
        #     plt.plot(alpha, cd, label='XFOIL')
        #     plt.plot(test.alphas_wind, test.cd_wind, label='WIND')
        #
        #     plt.show()
        if CST is not None:
            self.CST = CST


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
    def initFromCST(cls, CST, CFDorXFOIL, processors=0, iterations=1000,):
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

        alphas = np.linspace(-15, 15, 5)
        alphas = np.insert(alphas, 2, -4.5)
        alphas = np.insert(alphas, 4, 4.5)
        alphas = np.insert(alphas, 3, -2.)
        alphas = np.insert(alphas, 5, 2.)
        Re = 1e6
        af = Airfoil.initFromCST(CST, alphas, [Re], CFDorXFOIL, processors=processors, iterations=iterations)
        r_over_R = 0.5
        chord_over_r = 0.15
        tsr = 7.55
        cd_max = 1.5
        af3D = af.correction3D(r_over_R, chord_over_r, tsr)
        af_extrap1 = af3D.extrapolate(cd_max)
        alpha, Re, cl, cd, cm = af_extrap1.createDataGrid()

        return cls(alpha, Re, cl, cd, cm, CST=CST)


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
        # print dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe
        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    def cstderivatives(self, alpha, Re, CST, airfoil_analysis_options):
        Uinf = 10.0
        ComputeGradients = True
        if airfoil_analysis_options['CFDorXFOIL'] == 'XFOIL':
            cl, cd, dcl_dCST, dcd_dCST = xfoilGradients(CST, alpha, Re, airfoil_analysis_options['FDorCS'])
        else:
            cl, cd, dcl_dCST, dcd_dCST = cfdGradients(CST, alpha, Re, airfoil_analysis_options['iterations'], airfoil_analysis_options['processors'], airfoil_analysis_options['FDorCS'], Uinf, ComputeGradients, GenerateMESH=True)

        return dcl_dCST, dcd_dCST


def cfdGradients(CST, alpha, Re, iterations, processors, FDorCS, Uinf, ComputeGradients, GenerateMESH=True):

    import os, sys, shutil, copy
    sys.path.append(os.environ['SU2_RUN'])
    import SU2

    wl, wu, N, dz = CST_to_kulfan(CST)

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

    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SU2_EDU/bin')
    airfoil_shape_file = basepath + os.path.sep + 'airfoil_shape.dat'

    coord_file = open(airfoil_shape_file, 'w')

    print >> coord_file, 'CST'
    for i in range(len(x)):
        print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

    coord_file.close()

    if GenerateMESH:
        ## Update mesh for specific airfoil (mesh deformation in SU2_EDU)
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SU2_EDU/bin')
        su2_file_execute = basepath + os.path.sep + 'SU2_EDU'

        savedPath = os.getcwd()
        os.chdir(basepath)
        subprocess.call([su2_file_execute])
        os.chdir(savedPath)

    partitions = processors
    compute = True
    step = 1e-4
    iterations = iterations

    # Config and state
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
    # filename = basepath + os.path.sep + 'inv_NACA0012.cfg'
    filename = basepath + os.path.sep + 'test_incomp_rans.cfg'
    # filename = basepath + os.path.sep + 'turb_nasa.cfg'

    config = SU2.io.Config(filename)
    state  = SU2.io.State()
    config.NUMBER_PART = partitions
    config.EXT_ITER    = iterations

    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SU2_EDU/bin')
    mesh_filename = basepath + os.path.sep + 'mesh_AIRFOIL.su2'

    config.WRT_CSV_SOL = 'YES'
    config.MESH_FILENAME = mesh_filename
    config.AoA = alpha #np.degrees(alpha)
    Ma = Uinf / 340.29  # Speed of sound at sea level
    x_vel = Uinf * cos(np.radians(alpha))
    y_vel = Uinf * sin(np.radians(alpha))
    config.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
    config.MACH_NUMBER = Ma
    config.REYNOLDS_NUMBER = Re
    restart = True
    if restart:
        config.RESTART_SOL = 'YES'
        basepath2 = os.path.dirname(os.path.realpath(__file__))
        config.RESTART_FLOW_FILENAME = basepath2 + os.path.sep + 'solution_flow.dat' #_' + str(int(wu[0]) + '_' + str(alpha) + '.dat'
        config.SOLUTION_FLOW_FILENAME = basepath2 + os.path.sep + 'solution_flow_' + str(wu[0]) + '_' + str(alpha) + '.dat'
    info = SU2.run.DEF(config)
    state.update(info)
    # find solution files if they exist
    # state.find_files(config)
    ffdtag = []
    kind = []
    marker = []
    param = []
    scale = []
    for i in range(len(x)):
        ffdtag.append([])
        kind.append('HICKS_HENNE')
        marker.append(['airfoil'])
        if i < len(x) / 2.0:
            param.append([0.0, x[i]])
        else:
            param.append([1.0, x[i]])
        scale.append(1.0)
    config.DEFINITION_DV = dict(FFDTAG=ffdtag, KIND=kind, MARKER=marker, PARAM=param, SCALE=scale)
    state.FILES.MESH = config.MESH_FILENAME


    cd = SU2.eval.func('DRAG', config, state)
    cl = SU2.eval.func('LIFT', config, state)
    cm = SU2.eval.func('MOMENT_Z', config, state)
    if ComputeGradients:
        # RUN FOR DRAG GRADIENTS
        info = SU2.run.adjoint(config)
        state.update(info)
        #SU2.io.restart2solution(config,state)
        # Gradient Projection
        info = SU2.run.projection(config, step)
        state.update(info)
        get_gradients = info.get('GRADIENTS')
        dcd_dx = get_gradients.get('DRAG')

        # RUN FOR LIFT GRADIENTS
        config.OBJECTIVE_FUNCTION = 'LIFT'
        info = SU2.run.adjoint(config)
        state.update(info)
        #SU2.io.restart2solution(config,state)
        # Gradient Projection
        info = SU2.run.projection(config, step)
        state.update(info)
        get_gradients = info.get('GRADIENTS')
        dcl_dx = get_gradients.get('LIFT')

        where_are_NaNs_cl = np.isnan(dcl_dx)
        dcl_dx[where_are_NaNs_cl] = 0.0

        where_are_NaNs_cl = np.isnan(dcl_dx)
        dcd_dx[where_are_NaNs_cl] = 0.0

        n = len(CST)
        m = len(dcd_dx)
        dcst_dx = np.zeros((n, m))

        wl_original, wu_original = wu, wl
        dz = 0.0
        N = 200
        coord_old = cst_to_coordinates_from_kulfan(wl_original, wu_original, N, dz)

        # design = [85, 79, 74, 70, 67, 63, 60, 56, 53, 50, 47, 43, 40, 37, 33, 29, 25, 21, 14, 115, 121, 126, 130, 133, 137, 140, 144, 147, 150, 153, 157, 160, 163, 167, 171, 175, 179, 186]
        design = range(0, len(x))
        # Gradients
        FDorCS = 'FD'
        if FDorCS == 'FD':
            fd_step = 1e-6
            for i in range(0, n):
                wl_new = deepcopy(wl_original)
                wu_new = deepcopy(wu_original)
                if i < n/2:
                    wl_new[i] += fd_step
                else:
                    wu_new[i-4] += fd_step
                coor_new = cst_to_coordinates_from_kulfan(wl_new, wu_new, N, dz)
                j = 0
                for coor_d in design:
                    if (coor_new[1][coor_d] - coord_old[1][coor_d]).real == 0:
                        dcst_dx[i][j] = 0
                    else:
                        dcst_dx[i][j] = 1/((coor_new[1][coor_d] - coord_old[1][coor_d]).real / fd_step)
                    j += 1

        elif FDorCS == 'CS':
            step_size = 1e-20
            cs_step = complex(0, step_size)

            for i in range(0, n):
                wl_new = deepcopy(wl_original.astype(complex))
                wu_new = deepcopy(wu_original.astype(complex))
                if i >= n/2:
                    wl_new[i-4] += cs_step
                else:
                    wu_new[i] += cs_step
                coor_new = cst_to_coordinates_complex(wl_new, wu_new, N, dz)
                j = 0
                for coor_d in design:
                    if coor_new[1][coor_d].imag == 0:
                        dcst_dx[i][j] = 0
                    else:
                        dcst_dx[i][j] = 1/(coor_new[1][coor_d].imag / np.imag(cs_step))
                    j += 1
        else:
            print 'Warning. FDorCS needs to be set to either FD or CS'
        dcst_dx = np.matrix(dcst_dx)
        dcl_dx = np.matrix(dcl_dx)
        dcd_dx = np.matrix(dcd_dx)

        dcl_dcst = dcst_dx * dcl_dx.T
        dcd_dcst = dcst_dx * dcd_dx.T

        # print cl, cd, dcl_dcst, dcd_dcst
        return cl, cd, dcl_dcst, dcd_dcst

    return cl, cd


def xfoilGradients(CST, alpha, Re, FDorCS):
    alpha = np.degrees(alpha)
    def cstComplex(alpha, Re, wl, wu, N, dz, Uinf):
        # wl = self.wl
        # wu = self.wu
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
        airfoil.re = Re
        airfoil.mach = 0.0 # Uinf / 340.29
        airfoil.iter = 100

        angle = alpha
        cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
        if lexitflag:
            cl = -10.0
            cd = 0.0
            print "XFOIL FAILURE"
        return cl, cd
        # error handling in case of XFOIL failure
        # for k in range(len(cl)):
        #     if cl[k] == -10.0:
        #         if k == 0:
        #             cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
        #             cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
        #         elif k == len(cl)-1:
        #             cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
        #             cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
        #         else:
        #             cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
        #             cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
        #     if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
        #         to_delete = np.append(to_delete, k)
        # cl = np.delete(cl, to_delete)
        # cd = np.delete(cd, to_delete)
        # alphas = np.delete(alphas, to_delete)

        # polars.append(polarType(Re, alphas, cl, cd, cm))

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
        airfoil.re = Re
        airfoil.mach = 0.0 #Uinf / 340.29
        airfoil.iter = 1000

        angle = alpha
        cl, cd, cm, lexitflag = airfoil.solveAlpha(angle)
        if lexitflag:
            cl = -10.0
            cd = 0.0
        return cl, cd



    n2 = 1
    try:
        n1 = len(CST)/2
    except:
        n1 = 4
    if n1 == 0:
        n1 = 4
        CST = CST[0]
    wu = np.zeros(n1, dtype=complex)
    wl = np.zeros(n1, dtype=complex)
    for j in range(n1):
        wu[j] = CST[j]
        wl[j] = CST[j + n1]
    # wu, wl = np.split(af_parameters[i], 2)
    w1 = np.average(wl)
    w2 = np.average(wu)
    if w1 < w2:
        pass
    else:
        higher = wl
        lower = wu
        wl = lower
        wu = higher
    N = 120
    dz = 0.
    dcl_dcst, dcd_dcst = np.zeros(8), np.zeros(8)
    cl, cd = cstReal(alpha, Re, np.real(wl), np.real(wu), N, dz, Uinf=10.0)

    if FDorCS == 'CS':
        step_size = 1e-20
        cs_step = complex(0, step_size)
        for i in range(len(wl)):

            wl_complex = deepcopy(wl)
            wl_complex[i] += cs_step
            cl_complex, cd_complex = cstComplex(alpha, Re, wl_complex, wu, N, dz, Uinf=10.0)
            dcl_dcst[i] = np.imag(cl_complex)/np.imag(cs_step)
            dcd_dcst[i] = np.imag(cd_complex)/np.imag(cs_step)
            wu_complex = deepcopy(wu)
            wu_complex[i] += cs_step
            cl_complex, cd_complex = cstComplex(alpha, Re, wl, wu_complex, N, dz, Uinf=10.0)
            dcl_dcst[i+4] = np.imag(cl_complex)/np.imag(cs_step)
            dcd_dcst[i+4] = np.imag(cd_complex)/np.imag(cs_step)
    else:
        step_size = 1e-6
        fd_step = step_size
        for i in range(len(wl)):
            wl_fd1 = np.real(deepcopy(wl))
            wl_fd2 = np.real(deepcopy(wl))
            wl_fd1[i] -= fd_step
            wl_fd2[i] += fd_step
            cl_fd1, cd_fd1 = cstReal(alpha, Re, wl_fd1, np.real(wu), N, dz, Uinf=10.0)
            cl_fd1, cd_fd1 = deepcopy(cl_fd1), deepcopy(cd_fd1)
            cl_fd2, cd_fd2 = cstReal(alpha, Re, wl_fd2, np.real(wu), N, dz, Uinf=10.0)
            cl_fd2, cd_fd2 = deepcopy(cl_fd2), deepcopy(cd_fd2)
            dcl_dcst[i] = (cl_fd2 - cl_fd1)/(2.*fd_step)
            dcd_dcst[i] = (cd_fd2 - cd_fd1)/(2.*fd_step)
            wu_fd1 = np.real(deepcopy(wu))
            wu_fd2 = np.real(deepcopy(wu))
            wu_fd1[i] -= fd_step
            wu_fd2[i] += fd_step
            cl_fd1, cd_fd1 = cstReal(alpha, Re, np.real(wl), wu_fd1, N, dz, Uinf=10.0)
            cl_fd1, cd_fd1 = deepcopy(cl_fd1), deepcopy(cd_fd1)
            cl_fd2, cd_fd2 = cstReal(alpha, Re, np.real(wl), wu_fd2, N, dz, Uinf=10.0)
            cl_fd2, cd_fd2 = deepcopy(cl_fd2), deepcopy(cd_fd2)
            dcl_dcst[i+4] = (cl_fd2 - cl_fd1)/(2.*fd_step)
            dcd_dcst[i+4] = (cd_fd2 - cd_fd1)/(2.*fd_step)

    return cl, cd, dcl_dcst, dcd_dcst
