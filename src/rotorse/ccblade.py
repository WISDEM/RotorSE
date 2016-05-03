#!/usr/bin/env python
# encoding: utf-8
"""
ccblade.py

Created by S. Andrew Ning on 5/11/2012
Copyright (c) NREL. All rights reserved.

A blade element momentum method using theory detailed in [1]_.  Has the
advantages of guaranteed convergence and at a superlinear rate, and
continuously differentiable output.

.. [1] S. Andrew Ning, "A simple solution method for the blade element momentum
equations with guaranteed convergence", Wind Energy, 2013.



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

import numpy as np
from math import pi, radians, sin, cos
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline, bisplev
from zope.interface import Interface, implements
import warnings
import time
# from airfoilprep import Airfoil
from airfoilprep_free import Airfoil
import _bem
import pyXLIGHT
from copy import deepcopy
from airfoil_parameterization import AirfoilAnalysis

class CCAirfoil:
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""

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

        self.freeform = False
        if afp is not None:
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
                self.freeform = True
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
        else:
            self.afp = afp
            self.airfoil_analysis_options = airfoil_analysis_options



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
    def initFromFreeForm(cls, afp, airfoil_analysis_options, airfoilNum=0):
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
        af = Airfoil.initFromCST(afp, airfoil_analysis_options)
        failure = af.failure
        # For 3D Correction TODO
        # r_over_R = 0.5
        # chord_over_r = 0.15
        # tsr = 7.55
        # af3D = af.correction3D(r_over_R, chord_over_r, tsr)
        cd_max = 1.5
        af_extrap1 = af.extrapolate(cd_max)
        alpha, Re, cl, cd, cm = af_extrap1.createDataGrid()

        return cls(alpha, Re, cl, cd, cm, afp=afp, airfoil_analysis_options=airfoil_analysis_options, airfoilNum=airfoilNum, failure=failure)

    @classmethod
    def initFromPrecomputational(cls, t_c, airfoil_analysis_options, airfoilNum=0):

        af = Airfoil.initFromThicknesses(t_c, airfoil_analysis_options)
        failure = af.failure
        # For 3D Correction TODO
        # r_over_R = 0.5
        # chord_over_r = 0.15
        # tsr = 7.55
        # af3D = af.correction3D(r_over_R, chord_over_r, tsr)
        cd_max = 1.5
        af_extrap1 = af.extrapolate(cd_max)
        alpha, Re, cl, cd, cm = af_extrap1.createDataGrid()

        return cls(alpha, Re, cl, cd, cm, afp=t_c, airfoil_analysis_options=airfoil_analysis_options, airfoilNum=airfoilNum, failure=failure)

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
                afanalysis = AirfoilAnalysis(self.afp, self.airfoil_analysis_options)
                cl, cd, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dafp, dcd_dafp, lexitflag = afanalysis.computeDirect(alpha, Re)
                print cl, cd, alpha
                if lexitflag or abs(cl) > 2.5 or cd < 0.000001 or cd > 1.5 or not np.isfinite(cd) or not np.isfinite(cl):
                    cl, cd = self.evaluate(alpha, Re)
                    dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe = self.derivatives(alpha, Re)
                self.cl_storage.append(cl)
                self.cd_storage.append(cd)
                self.alpha_storage.append(alpha)
                if self.airfoil_analysis_options['ComputeGradient']:
                    self.dcl_storage.append(dcl_dalpha)
                    self.dcd_storage.append(dcd_dalpha)
                    self.dalpha_storage.append(alpha)
                    if self.airfoil_analysis_options['FreeFormDesign']:
                        self.dcl_dafp_storage.append(dcl_dafp)
                        self.dcd_dafp_storage.append(dcd_dafp)
                        self.dalpha_dafp_storage.append(alpha)
        else:
            cl, cd = self.evaluate(alpha, Re)
            dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = self.derivatives(alpha, Re)
            dcl_dafp, dcd_dafp = self.splineFreeFormGrad(alpha, Re)

        return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp

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

    def splineFreeFormGrad(self, alpha, Re):
        dcl_dafp, dcd_dafp = np.zeros(8), np.zeros(8)
        if self.freeform and self.afp is not None:
            fd_step = self.airfoil_analysis_options['fd_step']
            cl_cur = self.cl_spline.ev(alpha, Re)
            cd_cur = self.cd_spline.ev(alpha, Re)
            for i in range(8):
                cl_new_fd = self.cl_splines_new[i].ev(alpha, self.Re)
                cd_new_fd = self.cd_splines_new[i].ev(alpha, self.Re)
                dcl_dafp[i] = (cl_new_fd - cl_cur) / fd_step
                dcd_dafp[i] = (cd_new_fd - cd_cur) / fd_step
        return dcl_dafp, dcd_dafp

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
            dcd_dx, xl, xu = self.su2Gradient(loop_sorted, config.SURFACE_ADJ_FILENAME + '.csv')
            dcd_dalpha = state.HISTORY.ADJOINT_DRAG.Sens_AoA[-1]
            config.OBJECTIVE_FUNCTION = 'LIFT'
            info = SU2.run.adjoint(config)
            state.update(info)
            dcl_dx, xl, xu = self.su2Gradient(loop_sorted, config.SURFACE_ADJ_FILENAME + '.csv')
            dcl_dalpha = state.HISTORY.ADJOINT_LIFT.Sens_AoA[-1]

            dz = 0
            n = 8
            #fd_step = self.airfoil_analysis_options['fd_step']
            m = 200
            dx_dafp = np.zeros((n, m))
            wl_original, wu_original, N, dz = CST_to_kulfan(self.afp)
            step_size = self.airfoil_analysis_options['cs_step']
            cs_step = complex(0, step_size)
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

def evaluate_direct_parallel2(alphas, Res, afs, computeAlphaGradient=False, computeAFPGradient=False):
        indices_to_compute = []
        n = len(alphas)
        airfoil_analysis_options = afs[-1].airfoil_analysis_options
        cl = np.zeros(n)
        cd = np.zeros(n)
        dcl_dalpha = [0]*n
        dcd_dalpha = [0]*n
        dcl_dafp = [0]*n
        dcd_dafp = [0]*n
        dcl_dRe = [0]*n
        dcd_dRe = [0]*n

        for i in range(len(alphas)):
            alpha = alphas[i]
            Re = Res[i]
            af = afs[i]
            if af.afp is not None and abs(np.degrees(alpha)) < af.airfoil_analysis_options['maxDirectAoA']:
                if alpha in af.alpha_storage and alpha in af.dalpha_storage:
                    index = af.alpha_storage.index(alpha)
                    cl[i] = af.cl_storage[index]
                    cd[i] = af.cd_storage[index]
                    if computeAlphaGradient:
                        index = af.dalpha_storage.index(alpha)
                        dcl_dalpha[i] = af.dcl_storage[index]
                        dcd_dalpha[i] = af.dcd_storage[index]
                    if computeAFPGradient and alpha in af.dalpha_dafp_storage:
                        index = af.dalpha_dafp_storage.index(alpha)
                        dcl_dafp[i] = af.dcl_dafp_storage[index]
                        dcd_dafp[i] = af.dcd_dafp_storage[index]
                    dcl_dRe[i] = 0.0
                    dcd_dRe[i] = 0.0
                else:
                    indices_to_compute.append(i)
            else:
                cl[i] = af.cl_spline.ev(alpha, Re)
                cd[i] = af.cd_spline.ev(alpha, Re)
                tck_cl = af.cl_spline.tck[:3] + af.cl_spline.degrees  # concatenate lists
                tck_cd = af.cd_spline.tck[:3] + af.cd_spline.degrees

                dcl_dalpha[i] = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
                dcd_dalpha[i] = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

                if af.one_Re:
                    dcl_dRe[i] = 0.0
                    dcd_dRe[i] = 0.0
                else:
                    dcl_dRe[i] = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
                    dcd_dRe[i] = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
                if computeAFPGradient and af.afp is not None:
                    dcl_dafp[i], dcd_dafp[i] = af.splineFreeFormGrad(alpha, Re)
                else:
                    dcl_dafp[i], dcd_dafp[i] = np.zeros(8), np.zeros(8)
        if indices_to_compute is not None:
            alphas_to_compute = [alphas[i] for i in indices_to_compute]
            Res_to_compute = [Res[i] for i in indices_to_compute]
            afps_to_compute = [afs[i].afp for i in indices_to_compute]
            if airfoil_analysis_options['ComputeGradient']:
                cls, cds, dcls_dalpha, dcls_dRe, dcds_dalpha, dcds_dRe, dcls_dafp, dcds_dafp = cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, afps_to_compute, airfoil_analysis_options)
                for j in range(len(indices_to_compute)):
                    dcl_dalpha[indices_to_compute[j]] = dcls_dalpha[j]
                    dcl_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcd_dalpha[indices_to_compute[j]] = dcds_dalpha[j]
                    dcd_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcl_dafp[indices_to_compute[j]] = dcls_dafp[j]
                    dcd_dafp[indices_to_compute[j]] = dcds_dafp[j]

            else:
                cls, cds = cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, afps_to_compute, airfoil_analysis_options)
            for j in range(len(indices_to_compute)):
                cl[indices_to_compute[j]] = cls[j]
                cd[indices_to_compute[j]] = cds[j]

        if computeAFPGradient:
            try:
                return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp
            except:
                raise
        elif computeAlphaGradient:
            return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe
        else:
            return cl, cd

# ------------------
#  Main Class: CCBlade
# ------------------


class CCBlade:

    def __init__(self, r, chord, theta, af, Rhub, Rtip, B=3, rho=1.225, mu=1.81206e-5,
                 precone=0.0, tilt=0.0, yaw=0.0, shearExp=0.2, hubHt=80.0,
                 nSector=8, precurve=None, precurveTip=0.0, presweep=None, presweepTip=0.0,
                 tiploss=True, hubloss=True, wakerotation=True, usecd=True, iterRe=1, derivatives=False,
                 airfoil_parameterization=None, airfoil_options=None):
        """Constructor for aerodynamic rotor analysis

        Parameters
        ----------
        r : array_like (m)
            locations defining the blade along z-axis of :ref:`blade coordinate system <azimuth_blade_coord>`
            (values should be increasing).
        chord : array_like (m)
            corresponding chord length at each section
        theta : array_like (deg)
            corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---
            positive twist decreases angle of attack.
        af : list(AirfoilInterface)
            list of :ref:`AirfoilInterface <airfoil-interface-label>` objects at each section
        Rhub : float (m)
            location of hub
        Rtip : float (m)
            location of tip
        B : int, optional
            number of blades
        rho : float, optional (kg/m^3)
            freestream fluid density
        mu : float, optional (kg/m/s)
            dynamic viscosity of fluid
        precone : float, optional (deg)
            :ref:`hub precone angle <azimuth_blade_coord>`
        tilt : float, optional (deg)
            nacelle :ref:`tilt angle <yaw_hub_coord>`
        yaw : float, optional (deg)
            nacelle :ref:`yaw angle<wind_yaw_coord>`
        shearExp : float, optional
            shear exponent for a power-law wind profile across hub
        hubHt : float, optional
            hub height used for power-law wind profile.
            U = Uref*(z/hubHt)**shearExp
        nSector : int, optional
            number of azimuthal sectors to descretize aerodynamic calculation.  automatically set to
            ``1`` if tilt, yaw, and shearExp are all 0.0.  Otherwise set to a minimum of 4.
        precurve : array_like, optional (m)
            location of blade pitch axis in x-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
        precurveTip : float, optional (m)
            location of blade pitch axis in x-direction at the tip (analogous to Rtip)
        presweep : array_like, optional (m)
            location of blade pitch axis in y-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
        presweepTip : float, optional (m)
            location of blade pitch axis in y-direction at the tip (analogous to Rtip)
        tiploss : boolean, optional
            if True, include Prandtl tip loss model
        hubloss : boolean, optional
            if True, include Prandtl hub loss model
        wakerotation : boolean, optional
            if True, include effect of wake rotation (i.e., tangential induction factor is nonzero)
        usecd : boolean, optional
            If True, use drag coefficient in computing induction factors
            (always used in evaluating distributed loads from the induction factors).
            Note that the default implementation may fail at certain points if drag is not included
            (see Section 4.2 in :cite:`Ning2013A-simple-soluti`).  This can be worked around, but has
            not been implemented.
        iterRe : int, optional
            The number of iterations to use to converge Reynolds number.  Generally iterRe=1 is sufficient,
            but for high accuracy in Reynolds number effects, iterRe=2 iterations can be used.  More than that
            should not be necessary.  Gradients have only been implemented for the case iterRe=1.
        derivatives : boolean, optional
            if True, derivatives along with function values will be returned for the various methods

        """

        self.r = np.array(r)
        self.chord = np.array(chord)
        self.theta = np.radians(theta)
        self.af = af
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.B = B
        self.rho = rho
        self.mu = mu
        self.precone = radians(precone)
        self.tilt = radians(tilt)
        self.yaw = radians(yaw)
        self.shearExp = shearExp
        self.hubHt = hubHt
        self.bemoptions = dict(usecd=usecd, tiploss=tiploss, hubloss=hubloss, wakerotation=wakerotation)
        self.iterRe = iterRe
        self.derivatives = derivatives

        self.airfoil_parameterization = airfoil_parameterization
        self.airfoil_analysis_options = airfoil_options
        if airfoil_parameterization is None:
            self.freeform = False
        else:
            self.freeform = airfoil_options['FreeFormDesign']

        # check if no precurve / presweep
        if precurve is None:
            precurve = np.zeros(len(r))
            precurveTip = 0.0

        if presweep is None:
            presweep = np.zeros(len(r))
            presweepTip = 0.0

        self.precurve = precurve
        self.precurveTip = precurveTip
        self.presweep = presweep
        self.presweepTip = presweepTip

        # rotor radius
        if self.precurveTip != 0 and self.precone != 0.0:
            warnings.warn('rotor diameter may be modified in unexpected ways if tip precurve and precone are both nonzero')

        self.rotorR = Rtip*cos(self.precone) + self.precurveTip*sin(self.precone)


        # azimuthal discretization
        if self.tilt == 0.0 and self.yaw == 0.0 and self.shearExp == 0.0:
            self.nSector = 1  # no more are necessary
        else:
            self.nSector = max(4, nSector)  # at least 4 are necessary

    # residual
    def __runBEM(self, phi, r, chord, theta, af, Vx, Vy):
        """residual of BEM method and other corresponding variables"""

        a = 0.0
        ap = 0.0
        for i in range(self.iterRe):

            alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, self.pitch,
                                             chord, theta, self.rho, self.mu)
            cl, cd = af.evaluate(alpha, Re)

            fzero, a, ap = _bem.inductionfactors(r, chord, self.Rhub, self.Rtip, phi,
                                                 cl, cd, self.B, Vx, Vy, **self.bemoptions)

        return fzero, a, ap


    def __errorFunction(self, phi, r, chord, theta, af, Vx, Vy):
        """strip other outputs leaving only residual for Brent's method"""

        fzero, a, ap = self.__runBEM(phi, r, chord, theta, af, Vx, Vy)

        return fzero



    def __residualDerivatives(self, phi, r, chord, theta, af, Vx, Vy, airfoil_parameterization=None):
        """derivatives of fzero, a, ap"""

        if self.iterRe != 1:
            ValueError('Analytic derivatives not supplied for case with iterRe > 1')

        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)

        # alpha, Re (analytic derivaives)
        a = 0.0
        ap = 0.0
        alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, self.pitch,
                                         chord, theta, self.rho, self.mu)

        dalpha_dx = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        dRe_dx = np.array([0.0, Re/chord, 0.0, Re*Vx/W**2, Re*Vy/W**2, 0.0, 0.0, 0.0, 0.0])

        cl, cd = af.evaluate(alpha, Re)
        dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af.derivatives(alpha, Re)

        # chain rule
        dcl_dx = dcl_dalpha*dalpha_dx + dcl_dRe*dRe_dx
        dcd_dx = dcd_dalpha*dalpha_dx + dcd_dRe*dRe_dx

        # residual, a, ap (Tapenade)
        dx_dx = np.eye(9)

        fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(r, chord, self.Rhub, self.Rtip,
            phi, cl, cd, self.B, Vx, Vy, dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
            dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **self.bemoptions)


        if self.freeform and af.afp is not None:
            fzero_cl, dR_dcl, a, ap,  = _bem.coefficients_dv(r, chord, self.Rhub, self.Rtip,
                phi, cl, 1, cd, 0, self.B, Vx, Vy, **self.bemoptions)
            fzero_cd, dR_dcd, a, ap,  = _bem.coefficients_dv(r, chord, self.Rhub, self.Rtip,
                phi, cl, 0, cd, 1, self.B, Vx, Vy, **self.bemoptions)
            dcl_dafp_R, dcd_dafp_R = af.splineFreeFormGrad(alpha, Re)
            dR_dafp = dR_dcl*dcl_dafp_R + dR_dcd*dcd_dafp_R
        else:
            dR_dafp = np.zeros(8)
        return dR_dx, da_dx, dap_dx, dR_dafp


    def __loads(self, phi, rotating, r, chord, theta, af, Vx, Vy, airfoil_parameterization=None):
        """normal and tangential loads at one section (and optionally derivatives)"""

        cphi = cos(phi)
        sphi = sin(phi)


        if rotating:
            _, a, ap = self.__runBEM(phi, r, chord, theta, af, Vx, Vy)
        else:
            a = 0.0
            ap = 0.0

        alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, self.pitch,
                                         chord, theta, self.rho, self.mu)
        if rotating:
            cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp = af.evaluate_direct(alpha, Re, computeAlphaGradient=True, computeAFPGradient=True)
        else:
            cl, cd = af.evaluate(alpha, Re)
            dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af.derivatives(alpha, Re)
            dcl_dafp, dcd_dafp = af.splineFreeFormGrad(alpha, Re)

        cn = cl*cphi + cd*sphi  # these expressions should always contain drag
        ct = cl*sphi - cd*cphi
        q = 0.5*self.rho*W**2
        Np = cn*q*chord
        Tp = ct*q*chord

        if not self.derivatives:
            return Np, Tp, 0.0, 0.0, 0.0, np.zeros(8), np.zeros(8), np.zeros(8)

        # derivative of residual function
        if rotating:
            dR_dx, da_dx, dap_dx, dR_dafp = self.__residualDerivatives(phi, r, chord, theta, af, Vx, Vy, airfoil_parameterization)
            dphi_dx = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            dR_dx = np.zeros(9)
            dR_dx[0] = 1.0  # just to prevent divide by zero
            da_dx = np.zeros(9)
            dap_dx = np.zeros(9)
            dphi_dx = np.zeros(9)
            dR_dafp, dcl_dafp, dcd_dafp = np.zeros(8), np.zeros(8), np.zeros(8)


        # x = [phi, chord,  theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
        dx_dx = np.eye(9)
        dchord_dx = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # alpha, W, Re (Tapenade)

        alpha, dalpha_dx, W, dW_dx, Re, dRe_dx = _bem.relativewind_dv(phi, dx_dx[0, :],
            a, da_dx, ap, dap_dx, Vx, dx_dx[3, :], Vy, dx_dx[4, :],
            self.pitch, dx_dx[8, :], chord, dx_dx[1, :], theta, dx_dx[2, :],
            self.rho, self.mu)

        # chain rule
        dcl_dx = dcl_dalpha*dalpha_dx + dcl_dRe*dRe_dx
        dcd_dx = dcd_dalpha*dalpha_dx + dcd_dRe*dRe_dx


        # cn, cd
        dcn_dx = dcl_dx*cphi - cl*sphi*dphi_dx + dcd_dx*sphi + cd*cphi*dphi_dx
        dct_dx = dcl_dx*sphi + cl*cphi*dphi_dx - dcd_dx*cphi + cd*sphi*dphi_dx

        # Np, Tp
        dNp_dx = Np*(1.0/cn*dcn_dx + 2.0/W*dW_dx + 1.0/chord*dchord_dx)
        dTp_dx = Tp*(1.0/ct*dct_dx + 2.0/W*dW_dx + 1.0/chord*dchord_dx)

        # freeform design
        dphi_dafp = 0.0
        dcn_dafp = dcl_dafp*cphi - cl*sphi*dphi_dafp + dcd_dafp*sphi + cd*cphi*dphi_dafp
        dct_dafp = dcl_dafp*sphi + cl*cphi*dphi_dafp - dcd_dafp*cphi + cd*sphi*dphi_dafp
        dNp_dafp = Np*(1.0/cn*dcn_dafp)
        dTp_dafp = Tp*(1.0/ct*dct_dafp)

        return Np, Tp, dNp_dx, dTp_dx, dR_dx, dNp_dafp, dTp_dafp, dR_dafp

    def __loads_parallel(self, phi, rotating, r, chord, theta, af, Vx, Vy, airfoil_parameterization=None):
        """normal and tangential loads at one section (and optionally derivatives)"""
        alphas = []
        Ws = []
        Res = []
        a_s = []
        ap_s = []
        for i in range(len(r)):


            if rotating:
                _, a, ap = self.__runBEM(phi[i], r[i], chord[i], theta[i], af[i], Vx[i], Vy[i])
            else:
                a = 0.0
                ap = 0.0

            alpha, W, Re = _bem.relativewind(phi[i], a, ap, Vx[i], Vy[i], self.pitch,
                                             chord[i], theta[i], self.rho, self.mu)
            alphas.append(alpha)
            Res.append(Re)
            Ws.append(W)
            a_s.append(a)
            ap_s.append(ap)
        if rotating and self.derivatives:
            cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp = evaluate_direct_parallel(alphas, Res, af, computeAlphaGradient=True, computeAFPGradient=True)
        elif rotating:
            cl, cd = evaluate_direct_parallel(alphas, Res, af, computeAlphaGradient=False, computeAFPGradient=False)
        else:
            cl = np.zeros(len(r))
            cd = np.zeros(len(r))
            dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = [], [], [], []

            for i in range(len(alphas)):
                cl[i], cd[i] = af[i].evaluate(alphas[i], Res[i])
                dcl_dalpha1, dcl_dRe1, dcd_dalpha1, dcd_dRe1 = af[i].derivatives(alphas[i], Res[i])
                dcl_dalpha.append(dcl_dalpha1)
                dcl_dRe.append(dcl_dRe1)
                dcd_dalpha.append(dcd_dalpha1)
                dcd_dRe.append(dcd_dRe1)
            dcl_dafp = np.zeros((len(r),8))
            dcd_dafp = np.zeros((len(r), 8))
        Nps = np.zeros(len(r))
        Tps = np.zeros(len(r))
        n = len(r)
        dRs_dx = []
        dRs_dafp = []
        dNps_dafp = []
        dTps_dafp = []
        dcl_dafp_total = []
        dcd_dafp_total = []
        dNps_dx = []
        dTps_dx = []
        for i in range(len(r)):
            cphi = cos(phi[i])
            sphi = sin(phi[i])
            cn = cl[i]*cos(phi[i]) + cd[i]*sin(phi[i])  # these expressions should always contain drag
            ct = cl[i]*sin(phi[i]) - cd[i]*cos(phi[i])
            q = 0.5*self.rho*Ws[i]**2
            Nps[i] = cn*q*chord[i]
            Tps[i] = ct*q*chord[i]
            if self.derivatives:

                # derivative of residual function
                if rotating:
                    if self.freeform and af[i].afp is not None:
                        dR_dx, da_dx, dap_dx, dR_dafp = self.__residualDerivatives(phi[i], r[i], chord[i], theta[i], af[i], Vx[i], Vy[i], airfoil_parameterization[i])
                    else:
                        dR_dx, da_dx, dap_dx = self.__residualDerivatives(phi[i], r[i], chord[i], theta[i], af[i], Vx[i], Vy[i])
                    dphi_dx = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    dR_dx = np.zeros(9)
                    dR_dx[0] = 1.0  # just to prevent divide by zero
                    da_dx = np.zeros(9)
                    dap_dx = np.zeros(9)
                    dphi_dx = np.zeros(9)
                    dR_dafp = np.zeros(8)#,  dcl_dafp, dcd_dafp = np.zeros(8), np.zeros(8), np.zeros(8)


                # x = [phi, chord,  theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
                dx_dx = np.eye(9)
                dchord_dx = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                # alpha, W, Re (Tapenade)

                alpha, dalpha_dx, W, dW_dx, Re, dRe_dx = _bem.relativewind_dv(phi[i], dx_dx[0, :],
                    a_s[i], da_dx, ap_s[i], dap_dx, Vx, dx_dx[3, :], Vy[i], dx_dx[4, :],
                    self.pitch, dx_dx[8, :], chord[i], dx_dx[1, :], theta[i], dx_dx[2, :],
                    self.rho, self.mu)
                # cl, cd (spline derivatives)

                # chain rule
                dcl_dx = dcl_dalpha[i]*dalpha_dx + dcl_dRe[i]*dRe_dx
                dcd_dx = dcd_dalpha[i]*dalpha_dx + dcd_dRe[i]*dRe_dx


                # cn, cd
                dcn_dx = dcl_dx*cphi - cl[i]*sphi*dphi_dx + dcd_dx*sphi + cd[i]*cphi*dphi_dx
                dct_dx = dcl_dx*sphi + cl[i]*cphi*dphi_dx - dcd_dx*cphi + cd[i]*sphi*dphi_dx

                # Np, Tp
                dNp_dx = Nps[i]*(1.0/cn*dcn_dx + 2.0/Ws[i]*dW_dx + 1.0/chord[i]*dchord_dx)
                dTp_dx = Tps[i]*(1.0/ct*dct_dx + 2.0/Ws[i]*dW_dx + 1.0/chord[i]*dchord_dx)

                if self.freeform and af[i].afp is not None:
                    dphi_dafp = 0.0
                    # try:
                    dcn_dafp = dcl_dafp[i]*cphi - cl[i]*sphi*dphi_dafp + dcd_dafp[i]*sphi + cd[i]*cphi*dphi_dafp
                    # except:
                    #     pass
                    dct_dafp = dcl_dafp[i]*sphi + cl[i]*cphi*dphi_dafp - dcd_dafp[i]*cphi + cd[i]*sphi*dphi_dafp
                    dNp_dafp = Nps[i]*(1.0/cn*dcn_dafp)
                    dTp_dafp = Tps[i]*(1.0/ct*dct_dafp)
                    dNps_dafp.append(dNp_dafp)
                    dTps_dafp.append(dTp_dafp)
                    dRs_dafp.append(dR_dafp)
                elif self.freeform:
                    dNps_dafp.append(np.zeros(8))
                    dTps_dafp.append(np.zeros(8))
                    dRs_dafp.append(np.zeros(8))
                dNps_dx.append(dNp_dx)
                dTps_dx.append(dTp_dx)
                dRs_dx.append(dR_dx)

        if not self.derivatives:
            return Nps, Tps, [0.0]*n, [0.0]*n, [0.0]*n
        elif self.freeform and rotating:
            return Nps, Tps, dNps_dx, dTps_dx, dRs_dx, dNps_dafp, dTps_dafp, dRs_dafp
        else:
            return Nps, Tps, dNps_dx, dTps_dx, dRs_dx


    def __windComponents(self, Uinf, Omega, azimuth):
        """x, y components of wind in blade-aligned coordinate system"""

        Vx, Vy = _bem.windcomponents(self.r, self.precurve, self.presweep,
            self.precone, self.yaw, self.tilt, azimuth, Uinf, Omega, self.hubHt, self.shearExp)

        if not self.derivatives:
            return Vx, Vy, 0.0, 0.0, 0.0, 0.0

        # y = [r, precurve, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]  (derivative order)
        n = len(self.r)
        dy_dy = np.eye(3*n+7)

        _, Vxd, _, Vyd = _bem.windcomponents_dv(self.r, dy_dy[:, :n], self.precurve, dy_dy[:, n:2*n],
            self.presweep, dy_dy[:, 2*n:3*n], self.precone, dy_dy[:, 3*n], self.yaw, dy_dy[:, 3*n+3],
            self.tilt, dy_dy[:, 3*n+1], azimuth, dy_dy[:, 3*n+4], Uinf, dy_dy[:, 3*n+5],
            Omega, dy_dy[:, 3*n+6], self.hubHt, dy_dy[:, 3*n+2], self.shearExp)

        dVx_dr = np.diag(Vxd[:n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dr = np.diag(Vyd[:n, :])

        dVx_dcurve = Vxd[n:2*n, :]  # tri-diagonal  (note: dVx_j / dcurve_i  i==row)
        dVy_dcurve = Vyd[n:2*n, :]  # off-diagonal are actually all zero, but leave for convenience

        dVx_dsweep = np.diag(Vxd[2*n:3*n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dsweep = np.diag(Vyd[2*n:3*n, :])

        # w = [r, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]
        dVx_dw = np.vstack((dVx_dr, dVx_dsweep, Vxd[3*n:, :]))
        dVy_dw = np.vstack((dVy_dr, dVy_dsweep, Vyd[3*n:, :]))

        return Vx, Vy, dVx_dw, dVy_dw, dVx_dcurve, dVy_dcurve




    def distributedAeroLoads(self, Uinf, Omega, pitch, azimuth):
        """Compute distributed aerodynamic loads along blade.

        Parameters
        ----------
        Uinf : float or array_like (m/s)
            hub height wind speed (float).  If desired, an array can be input which specifies
            the velocity at each radial location along the blade (useful for analyzing loads
            behind tower shadow for example).  In either case shear corrections will be applied.
        Omega : float (RPM)
        Omega : float (RPM)
            rotor rotation speed
        pitch : float (deg)
            blade pitch in same direction as :ref:`twist <blade_airfoil_coord>`
            (positive decreases angle of attack)
        azimuth : float (deg)
            the :ref:`azimuth angle <hub_azimuth_coord>` where aerodynamic loads should be computed at

        Returns
        -------
        Np : ndarray (N/m)
            force per unit length normal to the section on downwind side
        Tp : ndarray (N/m)
            force per unit length tangential to the section in the direction of rotation
        dNp : dictionary containing arrays (present if ``self.derivatives = True``)
            derivatives of normal loads.  Each item in the dictionary a 2D Jacobian.
            The array sizes and keys are (where n = number of stations along blade):

            n x n (diagonal): 'dr', 'dchord', 'dtheta', 'dpresweep'

            n x n (tridiagonal): 'dprecurve'

            n x 1: 'dRhub', 'dRtip', 'dprecone', 'dtilt', 'dhubHt', 'dyaw', 'dazimuth', 'dUinf', 'dOmega', 'dpitch'

            for example dNp_dr = dNp['dr']  (where dNp_dr is an n x n array)
            and dNp_dr[i, j] = dNp_i / dr_j
        dTp : dictionary (present if ``self.derivatives = True``)
            derivatives of tangential loads.  Same keys as dNp.
        """

        self.pitch = radians(pitch)
        azimuth = radians(azimuth)

        # component of velocity at each radial station
        Vx, Vy, dVx_dw, dVy_dw, dVx_dcurve, dVy_dcurve = self.__windComponents(Uinf, Omega, azimuth)


        # initialize
        n = len(self.r)
        Np = np.zeros(n)
        Tp = np.zeros(n)

        dNp_dVx = np.zeros(n)
        dTp_dVx = np.zeros(n)
        dNp_dVy = np.zeros(n)
        dTp_dVy = np.zeros(n)
        dNp_dz = np.zeros((6, n))
        dTp_dz = np.zeros((6, n))
        DNp_Dafp = np.zeros((17, 8))
        DTp_Dafp = np.zeros((17, 8))

        errf = self.__errorFunction
        rotating = (Omega != 0)

        # ---------------- loop across blade ------------------
        for i in range(n):

            # index dependent arguments
            args = (self.r[i], self.chord[i], self.theta[i], self.af[i], Vx[i], Vy[i])
            if not rotating:  # non-rotating

                phi_star = pi/2.0

            else:

                # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------

                # set standard limits
                epsilon = 1e-6
                phi_lower = epsilon
                phi_upper = pi/2

                if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # an uncommon but possible case

                    if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
                        phi_lower = -pi/4
                        phi_upper = -epsilon
                    else:
                        phi_lower = pi/2
                        phi_upper = pi - epsilon

                try:
                    phi_star = brentq(errf, phi_lower, phi_upper, args=args)

                except ValueError:

                    warnings.warn('error.  check input values.')
                    phi_star = 0.0

                # ----------------------------------------------------------------

            # derivatives of residual
            Np[i], Tp[i], dNp_dx, dTp_dx, dR_dx, dNp_dafp, dTp_dafp, dR_dafp = self.__loads(phi_star, rotating, *args)

            if self.derivatives:
                # separate state vars from design vars
                # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
                dNp_dy = dNp_dx[0]
                dNp_dx = dNp_dx[1:]
                dTp_dy = dTp_dx[0]
                dTp_dx = dTp_dx[1:]
                dR_dy = dR_dx[0]
                dR_dx = dR_dx[1:]

                # direct (or adjoint) total derivatives
                DNp_Dx = dNp_dx - dNp_dy/dR_dy*dR_dx
                DTp_Dx = dTp_dx - dTp_dy/dR_dy*dR_dx

                DNp_Dafp[i, :] = dNp_dafp - dNp_dy/dR_dy*dR_dafp
                DTp_Dafp[i, :] = dTp_dafp - dTp_dy/dR_dy*dR_dafp

                # parse components
                # z = [r, chord, theta, Rhub, Rtip, pitch]
                zidx = [4, 0, 1, 5, 6, 7]
                dNp_dz[:, i] = DNp_Dx[zidx]
                dTp_dz[:, i] = DTp_Dx[zidx]

                dNp_dVx[i] = DNp_Dx[2]
                dTp_dVx[i] = DTp_Dx[2]

                dNp_dVy[i] = DNp_Dx[3]
                dTp_dVy[i] = DTp_Dx[3]



        if not self.derivatives:
            return Np, Tp

        else:

            # chain rule
            dNp_dw = dNp_dVx*dVx_dw + dNp_dVy*dVy_dw
            dTp_dw = dTp_dVx*dVx_dw + dTp_dVy*dVy_dw

            dNp_dprecurve = dNp_dVx*dVx_dcurve + dNp_dVy*dVy_dcurve
            dTp_dprecurve = dTp_dVx*dVx_dcurve + dTp_dVy*dVy_dcurve

            # stack
            # z = [r, chord, theta, Rhub, Rtip, pitch]
            # w = [r, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]
            # X = [r, chord, theta, Rhub, Rtip, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega, pitch]
            dNp_dz[0, :] += dNp_dw[0, :]  # add partial w.r.t. r
            dTp_dz[0, :] += dTp_dw[0, :]

            dNp_dX = np.vstack((dNp_dz[:-1, :], dNp_dw[1:, :], dNp_dz[-1, :]))
            dTp_dX = np.vstack((dTp_dz[:-1, :], dTp_dw[1:, :], dTp_dz[-1, :]))

            # add chain rule for conversion to radians
            ridx = [2, 6, 7, 9, 10, 13]
            dNp_dX[ridx, :] *= pi/180.0
            dTp_dX[ridx, :] *= pi/180.0

            # save these values as the packing in one matrix is convenient for evaluate
            # (intended for internal use only.  not to be accessed by user)
            self._dNp_dX = dNp_dX
            self._dTp_dX = dTp_dX
            self._dNp_dprecurve = dNp_dprecurve
            self._dTp_dprecurve = dTp_dprecurve

            # pack derivatives into dictionary
            dNp = {}
            dTp = {}

            # n x n (diagonal)
            dNp['dr'] = np.diag(dNp_dX[0, :])
            dTp['dr'] = np.diag(dTp_dX[0, :])
            dNp['dchord'] = np.diag(dNp_dX[1, :])
            dTp['dchord'] = np.diag(dTp_dX[1, :])
            dNp['dtheta'] = np.diag(dNp_dX[2, :])
            dTp['dtheta'] = np.diag(dTp_dX[2, :])
            dNp['dpresweep'] = np.diag(dNp_dX[5, :])
            dTp['dpresweep'] = np.diag(dTp_dX[5, :])

            # n x n (tridiagonal)
            dNp['dprecurve'] = dNp_dprecurve.T
            dTp['dprecurve'] = dTp_dprecurve.T

            # n x 1
            dNp['dRhub'] = dNp_dX[3, :].reshape(n, 1)
            dTp['dRhub'] = dTp_dX[3, :].reshape(n, 1)
            dNp['dRtip'] = dNp_dX[4, :].reshape(n, 1)
            dTp['dRtip'] = dTp_dX[4, :].reshape(n, 1)
            dNp['dprecone'] = dNp_dX[6, :].reshape(n, 1)
            dTp['dprecone'] = dTp_dX[6, :].reshape(n, 1)
            dNp['dtilt'] = dNp_dX[7, :].reshape(n, 1)
            dTp['dtilt'] = dTp_dX[7, :].reshape(n, 1)
            dNp['dhubHt'] = dNp_dX[8, :].reshape(n, 1)
            dTp['dhubHt'] = dTp_dX[8, :].reshape(n, 1)
            dNp['dyaw'] = dNp_dX[9, :].reshape(n, 1)
            dTp['dyaw'] = dTp_dX[9, :].reshape(n, 1)
            dNp['dazimuth'] = dNp_dX[10, :].reshape(n, 1)
            dTp['dazimuth'] = dTp_dX[10, :].reshape(n, 1)
            dNp['dUinf'] = dNp_dX[11, :].reshape(n, 1)
            dTp['dUinf'] = dTp_dX[11, :].reshape(n, 1)
            dNp['dOmega'] = dNp_dX[12, :].reshape(n, 1)
            dTp['dOmega'] = dTp_dX[12, :].reshape(n, 1)
            dNp['dpitch'] = dNp_dX[13, :].reshape(n, 1)
            dTp['dpitch'] = dTp_dX[13, :].reshape(n, 1)

            dNp['dafp'] = np.zeros((n, 17*8))
            dTp['dafp'] = np.zeros((n, 17*8))
            for z in range(n):
                dNp_zeros = np.zeros((17,8))
                dTp_zeros = np.zeros((17,8))
                dNp_zeros[z, :] = DNp_Dafp[z]
                dTp_zeros[z, :] = DTp_Dafp[z]
                dNp['dafp'][z] = dNp_zeros.flatten()
                dTp['dafp'][z] = dTp_zeros.flatten()

            return Np, Tp, dNp, dTp

    def distributedAeroLoadsParallel(self, Uinf, Omega, pitch, azimuth):
        """Compute distributed aerodynamic loads along blade.

        Parameters
        ----------
        Uinf : float or array_like (m/s)
            hub height wind speed (float).  If desired, an array can be input which specifies
            the velocity at each radial location along the blade (useful for analyzing loads
            behind tower shadow for example).  In either case shear corrections will be applied.
        Omega : float (RPM)
        Omega : float (RPM)
            rotor rotation speed
        pitch : float (deg)
            blade pitch in same direction as :ref:`twist <blade_airfoil_coord>`
            (positive decreases angle of attack)
        azimuth : float (deg)
            the :ref:`azimuth angle <hub_azimuth_coord>` where aerodynamic loads should be computed at

        Returns
        -------
        Np : ndarray (N/m)
            force per unit length normal to the section on downwind side
        Tp : ndarray (N/m)
            force per unit length tangential to the section in the direction of rotation
        dNp : dictionary containing arrays (present if ``self.derivatives = True``)
            derivatives of normal loads.  Each item in the dictionary a 2D Jacobian.
            The array sizes and keys are (where n = number of stations along blade):

            n x n (diagonal): 'dr', 'dchord', 'dtheta', 'dpresweep'

            n x n (tridiagonal): 'dprecurve'

            n x 1: 'dRhub', 'dRtip', 'dprecone', 'dtilt', 'dhubHt', 'dyaw', 'dazimuth', 'dUinf', 'dOmega', 'dpitch'

            for example dNp_dr = dNp['dr']  (where dNp_dr is an n x n array)
            and dNp_dr[i, j] = dNp_i / dr_j
        dTp : dictionary (present if ``self.derivatives = True``)
            derivatives of tangential loads.  Same keys as dNp.
        """

        self.pitch = radians(pitch)
        azimuth = radians(azimuth)

        # component of velocity at each radial station
        Vx, Vy, dVx_dw, dVy_dw, dVx_dcurve, dVy_dcurve = self.__windComponents(Uinf, Omega, azimuth)


        # initialize
        n = len(self.r)
        Np = np.zeros(n)
        Tp = np.zeros(n)
        phi = np.zeros(n)

        dNp_dVx = np.zeros(n)
        dTp_dVx = np.zeros(n)
        dNp_dVy = np.zeros(n)
        dTp_dVy = np.zeros(n)
        dNp_dz = np.zeros((6, n))
        dTp_dz = np.zeros((6, n))
        DNp_Dafp = np.zeros((17, 8))
        DTp_Dafp = np.zeros((17, 8))

        errf = self.__errorFunction
        rotating = (Omega != 0)
        args_all = (self.r, self.chord, self.theta, self.af, Vx, Vy)
        # ---------------- loop across blade ------------------
        for i in range(n):

            # index dependent arguments
            args = (self.r[i], self.chord[i], self.theta[i], self.af[i], Vx[i], Vy[i])
            if not rotating:  # non-rotating

                phi_star = pi/2.0

            else:

                # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------

                # set standard limits
                epsilon = 1e-6
                phi_lower = epsilon
                phi_upper = pi/2

                if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # an uncommon but possible case

                    if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
                        phi_lower = -pi/4
                        phi_upper = -epsilon
                    else:
                        phi_lower = pi/2
                        phi_upper = pi - epsilon

                try:
                    phi_star = brentq(errf, phi_lower, phi_upper, args=args)

                except ValueError:

                    warnings.warn('error.  check input values.')
                    phi_star = 0.0
            phi[i] = phi_star
                # ----------------------------------------------------------------
        if self.freeform and rotating and self.derivatives:
            Nps, Tps, dNps_dx, dTps_dx, dRs_dx, dNps_dafp, dTps_dafp, dRs_dafp = self.__loads_parallel(phi, rotating, *args_all, airfoil_parameterization=self.airfoil_parameterization)
        elif self.derivatives:
            Nps, Tps, dNps_dx, dTps_dx, dRs_dx = self.__loads_parallel(phi, rotating, *args_all, airfoil_parameterization=self.airfoil_parameterization)
        else:
            Nps, Tps, dNps_dx, dTps_dx, dRs_dx = self.__loads_parallel(phi, rotating, *args_all, airfoil_parameterization=self.airfoil_parameterization)

        if self.derivatives:
            for i in range(n):
                Np, Tp, dNp_dx, dTp_dx, dR_dx = Nps[i], Tps[i], dNps_dx[i], dTps_dx[i], dRs_dx[i]
                # separate state vars from design vars
                # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
                dNp_dy = dNp_dx[0]
                dNp_dx = dNp_dx[1:]
                dTp_dy = dTp_dx[0]
                dTp_dx = dTp_dx[1:]
                dR_dy = dR_dx[0]
                dR_dx = dR_dx[1:]

                # direct (or adjoint) total derivatives
                DNp_Dx = dNp_dx - dNp_dy/dR_dy*dR_dx
                DTp_Dx = dTp_dx - dTp_dy/dR_dy*dR_dx

                if self.freeform and rotating:
                    dNp_dafp, dTp_dafp, dR_dafp = dNps_dafp[i], dTps_dafp[i], dRs_dafp[i]
                    DNp_Dafp[i, :] = dNp_dafp - dNp_dy/dR_dy*dR_dafp
                    DTp_Dafp[i, :] = dTp_dafp - dTp_dy/dR_dy*dR_dafp

                # parse components
                # z = [r, chord, theta, Rhub, Rtip, pitch]
                zidx = [4, 0, 1, 5, 6, 7]
                dNp_dz[:, i] = DNp_Dx[zidx]
                dTp_dz[:, i] = DTp_Dx[zidx]

                dNp_dVx[i] = DNp_Dx[2]
                dTp_dVx[i] = DTp_Dx[2]

                dNp_dVy[i] = DNp_Dx[3]
                dTp_dVy[i] = DTp_Dx[3]



        if not self.derivatives:
            return Nps, Tps

        else:

            # chain rule
            dNp_dw = dNp_dVx*dVx_dw + dNp_dVy*dVy_dw
            dTp_dw = dTp_dVx*dVx_dw + dTp_dVy*dVy_dw

            dNp_dprecurve = dNp_dVx*dVx_dcurve + dNp_dVy*dVy_dcurve
            dTp_dprecurve = dTp_dVx*dVx_dcurve + dTp_dVy*dVy_dcurve

            # stack
            # z = [r, chord, theta, Rhub, Rtip, pitch]
            # w = [r, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]
            # X = [r, chord, theta, Rhub, Rtip, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega, pitch]
            dNp_dz[0, :] += dNp_dw[0, :]  # add partial w.r.t. r
            dTp_dz[0, :] += dTp_dw[0, :]

            dNp_dX = np.vstack((dNp_dz[:-1, :], dNp_dw[1:, :], dNp_dz[-1, :]))
            dTp_dX = np.vstack((dTp_dz[:-1, :], dTp_dw[1:, :], dTp_dz[-1, :]))

            # add chain rule for conversion to radians
            ridx = [2, 6, 7, 9, 10, 13]
            dNp_dX[ridx, :] *= pi/180.0
            dTp_dX[ridx, :] *= pi/180.0

            # save these values as the packing in one matrix is convenient for evaluate
            # (intended for internal use only.  not to be accessed by user)
            self._dNp_dX = dNp_dX
            self._dTp_dX = dTp_dX
            self._dNp_dprecurve = dNp_dprecurve
            self._dTp_dprecurve = dTp_dprecurve

            # pack derivatives into dictionary
            dNp = {}
            dTp = {}

            # n x n (diagonal)
            dNp['dr'] = np.diag(dNp_dX[0, :])
            dTp['dr'] = np.diag(dTp_dX[0, :])
            dNp['dchord'] = np.diag(dNp_dX[1, :])
            dTp['dchord'] = np.diag(dTp_dX[1, :])
            dNp['dtheta'] = np.diag(dNp_dX[2, :])
            dTp['dtheta'] = np.diag(dTp_dX[2, :])
            dNp['dpresweep'] = np.diag(dNp_dX[5, :])
            dTp['dpresweep'] = np.diag(dTp_dX[5, :])

            # n x n (tridiagonal)
            dNp['dprecurve'] = dNp_dprecurve.T
            dTp['dprecurve'] = dTp_dprecurve.T

            # n x 1
            dNp['dRhub'] = dNp_dX[3, :].reshape(n, 1)
            dTp['dRhub'] = dTp_dX[3, :].reshape(n, 1)
            dNp['dRtip'] = dNp_dX[4, :].reshape(n, 1)
            dTp['dRtip'] = dTp_dX[4, :].reshape(n, 1)
            dNp['dprecone'] = dNp_dX[6, :].reshape(n, 1)
            dTp['dprecone'] = dTp_dX[6, :].reshape(n, 1)
            dNp['dtilt'] = dNp_dX[7, :].reshape(n, 1)
            dTp['dtilt'] = dTp_dX[7, :].reshape(n, 1)
            dNp['dhubHt'] = dNp_dX[8, :].reshape(n, 1)
            dTp['dhubHt'] = dTp_dX[8, :].reshape(n, 1)
            dNp['dyaw'] = dNp_dX[9, :].reshape(n, 1)
            dTp['dyaw'] = dTp_dX[9, :].reshape(n, 1)
            dNp['dazimuth'] = dNp_dX[10, :].reshape(n, 1)
            dTp['dazimuth'] = dTp_dX[10, :].reshape(n, 1)
            dNp['dUinf'] = dNp_dX[11, :].reshape(n, 1)
            dTp['dUinf'] = dTp_dX[11, :].reshape(n, 1)
            dNp['dOmega'] = dNp_dX[12, :].reshape(n, 1)
            dTp['dOmega'] = dTp_dX[12, :].reshape(n, 1)
            dNp['dpitch'] = dNp_dX[13, :].reshape(n, 1)
            dTp['dpitch'] = dTp_dX[13, :].reshape(n, 1)

            if self.freeform:
                dNp['dafp'] = np.zeros((n, 17*8))
                dTp['dafp'] = np.zeros((n, 17*8))
                for z in range(n):
                    dNp_zeros = np.zeros((17,8))
                    dTp_zeros = np.zeros((17,8))
                    dNp_zeros[z, :] = DNp_Dafp[z]
                    dTp_zeros[z, :] = DTp_Dafp[z]
                    dNp['dafp'][z] = dNp_zeros.flatten()
                    dTp['dafp'][z] = dTp_zeros.flatten()

            return Nps, Tps, dNp, dTp

    def TEST(self, Uinf, Omega, pitch, azimuth, i):

        self.pitch = radians(pitch)
        azimuth = radians(azimuth)

        # component of velocity at each radial station
        Vx, Vy, dVx_dw, dVy_dw, dVx_dcurve, dVy_dcurve = self.__windComponents(Uinf, Omega, azimuth)


        # initialize
        n = len(self.r)
        Np = np.zeros(n)
        Tp = np.zeros(n)

        dNp_dVx = np.zeros(n)
        dTp_dVx = np.zeros(n)
        dNp_dVy = np.zeros(n)
        dTp_dVy = np.zeros(n)
        dNp_dz = np.zeros((6, n))
        dTp_dz = np.zeros((6, n))
        DNp_Dafp = np.zeros((17, 8))
        DTp_Dafp = np.zeros((17, 8))

        errf = self.__errorFunction
        rotating = (Omega != 0)



        # index dependent arguments
        args = (self.r[i], self.chord[i], self.theta[i], self.af[i], Vx[i], Vy[i])
        if not rotating:  # non-rotating

            phi_star = pi/2.0

        else:

            # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------

            # set standard limits
            epsilon = 1e-6
            phi_lower = epsilon
            phi_upper = pi/2

            if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # an uncommon but possible case

                if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
                    phi_lower = -pi/4
                    phi_upper = -epsilon
                else:
                    phi_lower = pi/2
                    phi_upper = pi - epsilon

            try:
                phi_star = brentq(errf, phi_lower, phi_upper, args=args)

            except ValueError:

                warnings.warn('error.  check input values.')
                phi_star = 0.0

            # ----------------------------------------------------------------

        # derivatives of residual
        airfoils = [False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        if self.freeform and rotating and self.derivatives:
            if airfoils[i]:
                self.freeform_gradient = True
                Np[i], Tp[i], dNp_dx, dTp_dx, dR_dx, dNp_dafp, dTp_dafp, dR_dafp = self.__loads(phi_star, rotating, *args, airfoil_parameterization=self.airfoil_parameterization[i])
            else:
                self.freeform_gradient = False
                Np[i], Tp[i], dNp_dx, dTp_dx, dR_dx = self.__loads(phi_star, rotating, *args)
                dNp_dafp, dTp_dafp, dR_dafp = 0.0, 0.0, 0.0
        else:
            self.freeform_gradient = False
            Np[i], Tp[i], dNp_dx, dTp_dx, dR_dx = self.__loads(phi_star, rotating, *args)

        if self.derivatives:
            # separate state vars from design vars
            # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
            dNp_dy = dNp_dx[0]
            dNp_dx = dNp_dx[1:]
            dTp_dy = dTp_dx[0]
            dTp_dx = dTp_dx[1:]
            dR_dy = dR_dx[0]
            dR_dx = dR_dx[1:]

            # direct (or adjoint) total derivatives
            DNp_Dx = dNp_dx - dNp_dy/dR_dy*dR_dx
            DTp_Dx = dTp_dx - dTp_dy/dR_dy*dR_dx

            if self.freeform and rotating:
                print dTp_dafp - dTp_dy/dR_dy*dR_dafp
                DNp_Dafp[i, :] = dNp_dafp - dNp_dy/dR_dy*dR_dafp
                DTp_Dafp[i, :] = dTp_dafp - dTp_dy/dR_dy*dR_dafp

            # parse components
            # z = [r, chord, theta, Rhub, Rtip, pitch]
            zidx = [4, 0, 1, 5, 6, 7]
            dNp_dz[:, i] = DNp_Dx[zidx]
            dTp_dz[:, i] = DTp_Dx[zidx]

            dNp_dVx[i] = DNp_Dx[2]
            dTp_dVx[i] = DTp_Dx[2]

            dNp_dVy[i] = DNp_Dx[3]
            dTp_dVy[i] = DTp_Dx[3]

        return Tp[i]

    def evaluate(self, Uinf, Omega, pitch, coefficient=False):
        """Run the aerodynamic analysis at the specified conditions.

        Parameters
        ----------
        Uinf : array_like (m/s)
            hub height wind speed
        Omega : array_like (RPM)
            rotor rotation speed
        pitch : array_like (deg)
            blade pitch setting
        coefficient : bool, optional
            if True, results are returned in nondimensional form

        Returns
        -------
        P or CP : ndarray (W)
            power or power coefficient
        T or CT : ndarray (N)
            thrust or thrust coefficient (magnitude)
        Q or CQ : ndarray (N*m)
            torque or torque coefficient (magnitude)
        dP or dCP : dictionary of arrays (present only if derivatives==True)
            derivatives of power or power coefficient.  Each item in dictionary is a Jacobian.
            The array sizes and keys are below
            npts is the number of conditions (len(Uinf)),
            n = number of stations along blade (len(r))

            npts x 1: 'dprecone', 'dtilt', 'dhubHt', 'dRhub', 'dRtip', 'dprecurveTip', 'dpresweepTip', 'dyaw'

            npts x npts: 'dUinf', 'dOmega', 'dpitch'

            npts x n: 'dr', 'dchord', 'dtheta', 'dprecurve', 'dpresweep'

            for example dP_dr = dP['dr']  (where dP_dr is an npts x n array)
            and dP_dr[i, j] = dP_i / dr_j
        dT or dCT : dictionary of arrays (present only if derivatives==True)
            derivative of thrust or thrust coefficient.  Same format as dP and dCP
        dQ or dCQ : dictionary of arrays (present only if derivatives==True)
            derivative of torque or torque coefficient.  Same format as dP and dCP


        Notes
        -----

        CP = P / (q * Uinf * A)

        CT = T / (q * A)

        CQ = Q / (q * A * R)

        The rotor radius R, may not actually be Rtip if precone and precurve are both nonzero
        ``R = Rtip*cos(precone) + precurveTip*sin(precone)``

        """

        # rename
        args = (self.r, self.precurve, self.presweep, self.precone,
            self.Rhub, self.Rtip, self.precurveTip, self.presweepTip)
        nsec = self.nSector

        # initialize
        Uinf = np.array(Uinf)
        Omega = np.array(Omega)
        pitch = np.array(pitch)

        npts = len(Uinf)
        T = np.zeros(npts)
        Q = np.zeros(npts)
        P = np.zeros(npts)

        if self.derivatives:
            dT_ds = np.zeros((npts, 11))
            dQ_ds = np.zeros((npts, 11))
            dT_dv = np.zeros((npts, 5, len(self.r)))
            dQ_dv = np.zeros((npts, 5, len(self.r)))
            dT_dafp = np.zeros((npts, 17*8))
            dQ_dafp = np.zeros((npts, 17*8))

        for i in range(npts):  # iterate across conditions

            for j in range(nsec):  # integrate across azimuth
                azimuth = 360.0*float(j)/nsec

                if not self.derivatives:
                    # contribution from this azimuthal location
                    Np, Tp = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], azimuth)

                else:

                    Np, Tp, dNp, dTp = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], azimuth)

                    if self.freeform:
                        dT_ds_sub, dQ_ds_sub, dT_dv_sub, dQ_dv_sub, dT_dafp_sub, dQ_dafp_sub = self.__thrustTorqueDeriv(
                            Np, Tp, self._dNp_dX, self._dTp_dX, self._dNp_dprecurve, self._dTp_dprecurve, *args, dNp_dafp=dNp['dafp'], dTp_dafp=dTp['dafp'])
                        dT_dafp[i, :] += self.B * dT_dafp_sub.reshape(17*8) / nsec
                        dQ_dafp[i, :] += self.B * dQ_dafp_sub.reshape(17*8) / nsec
                    else:
                        dT_ds_sub, dQ_ds_sub, dT_dv_sub, dQ_dv_sub = self.__thrustTorqueDeriv(
                            Np, Tp, self._dNp_dX, self._dTp_dX, self._dNp_dprecurve, self._dTp_dprecurve, *args)

                    dT_ds[i, :] += self.B * dT_ds_sub / nsec
                    dQ_ds[i, :] += self.B * dQ_ds_sub / nsec
                    dT_dv[i, :, :] += self.B * dT_dv_sub / nsec
                    dQ_dv[i, :, :] += self.B * dQ_dv_sub / nsec



                Tsub, Qsub = _bem.thrusttorque(Np, Tp, *args)

                T[i] += self.B * Tsub / nsec
                Q[i] += self.B * Qsub / nsec


        # Power
        P = Q * Omega*pi/30.0  # RPM to rad/s

        # normalize if necessary
        if coefficient:
            q = 0.5 * self.rho * Uinf**2
            A = pi * self.rotorR**2
            CP = P / (q * A * Uinf)
            CT = T / (q * A)
            CQ = Q / (q * self.rotorR * A)

            if self.derivatives:

                # s = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, Uinf, Omega, pitch]

                dR_ds = np.array([-self.Rtip*sin(self.precone)*pi/180.0 + self.precurveTip*cos(self.precone)*pi/180.0,
                    0.0, 0.0, 0.0, cos(self.precone), sin(self.precone), 0.0, 0.0, 0.0, 0.0, 0.0])
                dR_ds = np.dot(np.ones((npts, 1)), np.array([dR_ds]))  # same for each operating condition

                dA_ds = 2*pi*self.rotorR*dR_ds

                dU_ds = np.zeros((npts, 11))
                dU_ds[:, 8] = 1.0

                dOmega_ds = np.zeros((npts, 11))
                dOmega_ds[:, 9] = 1.0

                dq_ds = (self.rho*Uinf*dU_ds.T).T

                dCT_ds = (CT * (dT_ds.T/T - dA_ds.T/A - dq_ds.T/q)).T
                dCT_dv = (dT_dv.T / (q*A)).T

                dCQ_ds = (CQ * (dQ_ds.T/Q - dA_ds.T/A - dq_ds.T/q - dR_ds.T/self.rotorR)).T
                dCQ_dv = (dQ_dv.T / (q*self.rotorR*A)).T

                dCP_ds = (CP * (dQ_ds.T/Q + dOmega_ds.T/Omega - dA_ds.T/A - dq_ds.T/q - dU_ds.T/Uinf)).T
                dCP_dv = (dQ_dv.T * CP/Q).T

                if self.freeform:
                    dCT_dafp = (dT_dafp.T / (q*A)).T
                    dCQ_dafp = (dQ_dafp.T / (q*self.rotorR*A)).T
                    dCP_dafp = (dQ_dafp.T * CP/Q).T
                    dCT, dCQ, dCP = self.__thrustTorqueDictionary(dCT_ds, dCQ_ds, dCP_ds, dCT_dv, dCQ_dv, dCP_dv, npts, dCT_dafp, dCQ_dafp, dCP_dafp)
                # pack derivatives into dictionary
                else:
                    dCT, dCQ, dCP = self.__thrustTorqueDictionary(dCT_ds, dCQ_ds, dCP_ds, dCT_dv, dCQ_dv, dCP_dv, npts)


                return CP, CT, CQ, dCP, dCT, dCQ

            else:
                return CP, CT, CQ


        if self.derivatives:
            # scalars = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, Uinf, Omega, pitch]
            # vectors = [r, chord, theta, precurve, presweep]

            dP_ds = (dQ_ds.T * Omega*pi/30.0).T
            dP_ds[:, 9] += Q*pi/30.0
            dP_dv = (dQ_dv.T * Omega*pi/30.0).T

            # pack derivatives into dictionary
            if self.freeform:
                dP_dafp = (dQ_dafp.T * Omega*pi/30.0).T
                dT, dQ, dP = self.__thrustTorqueDictionary(dT_ds, dQ_ds, dP_ds, dT_dv, dQ_dv, dP_dv, npts, dT_dafp, dQ_dafp, dP_dafp)
            else:
                dT, dQ, dP = self.__thrustTorqueDictionary(dT_ds, dQ_ds, dP_ds, dT_dv, dQ_dv, dP_dv, npts)
            return P, T, Q, dP, dT, dQ

        else:
            return P, T, Q

    def evaluateParallel(self, Uinf, Omega, pitch, coefficient=False):
        """Run the aerodynamic analysis at the specified conditions.

        Parameters
        ----------
        Uinf : array_like (m/s)
            hub height wind speed
        Omega : array_like (RPM)
            rotor rotation speed
        pitch : array_like (deg)
            blade pitch setting
        coefficient : bool, optional
            if True, results are returned in nondimensional form

        Returns
        -------
        P or CP : ndarray (W)
            power or power coefficient
        T or CT : ndarray (N)
            thrust or thrust coefficient (magnitude)
        Q or CQ : ndarray (N*m)
            torque or torque coefficient (magnitude)
        dP or dCP : dictionary of arrays (present only if derivatives==True)
            derivatives of power or power coefficient.  Each item in dictionary is a Jacobian.
            The array sizes and keys are below
            npts is the number of conditions (len(Uinf)),
            n = number of stations along blade (len(r))

            npts x 1: 'dprecone', 'dtilt', 'dhubHt', 'dRhub', 'dRtip', 'dprecurveTip', 'dpresweepTip', 'dyaw'

            npts x npts: 'dUinf', 'dOmega', 'dpitch'

            npts x n: 'dr', 'dchord', 'dtheta', 'dprecurve', 'dpresweep'

            for example dP_dr = dP['dr']  (where dP_dr is an npts x n array)
            and dP_dr[i, j] = dP_i / dr_j
        dT or dCT : dictionary of arrays (present only if derivatives==True)
            derivative of thrust or thrust coefficient.  Same format as dP and dCP
        dQ or dCQ : dictionary of arrays (present only if derivatives==True)
            derivative of torque or torque coefficient.  Same format as dP and dCP


        Notes
        -----

        CP = P / (q * Uinf * A)

        CT = T / (q * A)

        CQ = Q / (q * A * R)

        The rotor radius R, may not actually be Rtip if precone and precurve are both nonzero
        ``R = Rtip*cos(precone) + precurveTip*sin(precone)``

        """

        # rename
        args = (self.r, self.precurve, self.presweep, self.precone,
            self.Rhub, self.Rtip, self.precurveTip, self.presweepTip)
        nsec = self.nSector

        # initialize
        Uinf = np.array(Uinf)
        Omega = np.array(Omega)
        pitch = np.array(pitch)

        npts = len(Uinf)
        T = np.zeros(npts)
        Q = np.zeros(npts)
        P = np.zeros(npts)

        if self.derivatives:
            dT_ds = np.zeros((npts, 11))
            dQ_ds = np.zeros((npts, 11))
            dT_dv = np.zeros((npts, 5, len(self.r)))
            dQ_dv = np.zeros((npts, 5, len(self.r)))
            dT_dafp = np.zeros((npts, 17*8))
            dQ_dafp = np.zeros((npts, 17*8))

        for i in range(npts):  # iterate across conditions

            for j in range(nsec):  # integrate across azimuth
                azimuth = 360.0*float(j)/nsec

                if not self.derivatives:
                    # contribution from this azimuthal location
                    Np, Tp = self.distributedAeroLoadsParallel(Uinf[i], Omega[i], pitch[i], azimuth)

                else:

                    Np, Tp, dNp, dTp = self.distributedAeroLoadsParallel(Uinf[i], Omega[i], pitch[i], azimuth)

                    if self.freeform:
                        dT_ds_sub, dQ_ds_sub, dT_dv_sub, dQ_dv_sub, dT_dafp_sub, dQ_dafp_sub = self.__thrustTorqueDeriv(
                            Np, Tp, self._dNp_dX, self._dTp_dX, self._dNp_dprecurve, self._dTp_dprecurve, *args, dNp_dafp=dNp['dafp'], dTp_dafp=dTp['dafp'])
                        dT_dafp[i, :] += self.B * dT_dafp_sub.reshape(17*8) / nsec
                        dQ_dafp[i, :] += self.B * dQ_dafp_sub.reshape(17*8) / nsec
                    else:
                        dT_ds_sub, dQ_ds_sub, dT_dv_sub, dQ_dv_sub = self.__thrustTorqueDeriv(
                            Np, Tp, self._dNp_dX, self._dTp_dX, self._dNp_dprecurve, self._dTp_dprecurve, *args)

                    dT_ds[i, :] += self.B * dT_ds_sub / nsec
                    dQ_ds[i, :] += self.B * dQ_ds_sub / nsec
                    dT_dv[i, :, :] += self.B * dT_dv_sub / nsec
                    dQ_dv[i, :, :] += self.B * dQ_dv_sub / nsec



                Tsub, Qsub = _bem.thrusttorque(Np, Tp, *args)

                T[i] += self.B * Tsub / nsec
                Q[i] += self.B * Qsub / nsec


        # Power
        P = Q * Omega*pi/30.0  # RPM to rad/s

        # normalize if necessary
        if coefficient:
            q = 0.5 * self.rho * Uinf**2
            A = pi * self.rotorR**2
            CP = P / (q * A * Uinf)
            CT = T / (q * A)
            CQ = Q / (q * self.rotorR * A)

            if self.derivatives:

                # s = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, Uinf, Omega, pitch]

                dR_ds = np.array([-self.Rtip*sin(self.precone)*pi/180.0 + self.precurveTip*cos(self.precone)*pi/180.0,
                    0.0, 0.0, 0.0, cos(self.precone), sin(self.precone), 0.0, 0.0, 0.0, 0.0, 0.0])
                dR_ds = np.dot(np.ones((npts, 1)), np.array([dR_ds]))  # same for each operating condition

                dA_ds = 2*pi*self.rotorR*dR_ds

                dU_ds = np.zeros((npts, 11))
                dU_ds[:, 8] = 1.0

                dOmega_ds = np.zeros((npts, 11))
                dOmega_ds[:, 9] = 1.0

                dq_ds = (self.rho*Uinf*dU_ds.T).T

                dCT_ds = (CT * (dT_ds.T/T - dA_ds.T/A - dq_ds.T/q)).T
                dCT_dv = (dT_dv.T / (q*A)).T

                dCQ_ds = (CQ * (dQ_ds.T/Q - dA_ds.T/A - dq_ds.T/q - dR_ds.T/self.rotorR)).T
                dCQ_dv = (dQ_dv.T / (q*self.rotorR*A)).T

                dCP_ds = (CP * (dQ_ds.T/Q + dOmega_ds.T/Omega - dA_ds.T/A - dq_ds.T/q - dU_ds.T/Uinf)).T
                dCP_dv = (dQ_dv.T * CP/Q).T

                if self.freeform:
                    dCT_dafp = (dT_dafp.T / (q*A)).T
                    dCQ_dafp = (dQ_dafp.T / (q*self.rotorR*A)).T
                    dCP_dafp = (dQ_dafp.T * CP/Q).T
                    dCT, dCQ, dCP = self.__thrustTorqueDictionary(dCT_ds, dCQ_ds, dCP_ds, dCT_dv, dCQ_dv, dCP_dv, npts, dCT_dafp, dCQ_dafp, dCP_dafp)
                # pack derivatives into dictionary
                else:
                    dCT, dCQ, dCP = self.__thrustTorqueDictionary(dCT_ds, dCQ_ds, dCP_ds, dCT_dv, dCQ_dv, dCP_dv, npts)


                return CP, CT, CQ, dCP, dCT, dCQ

            else:
                return CP, CT, CQ


        if self.derivatives:
            # scalars = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, Uinf, Omega, pitch]
            # vectors = [r, chord, theta, precurve, presweep]

            dP_ds = (dQ_ds.T * Omega*pi/30.0).T
            dP_ds[:, 9] += Q*pi/30.0
            dP_dv = (dQ_dv.T * Omega*pi/30.0).T

            # pack derivatives into dictionary
            if self.freeform:
                dP_dafp = (dQ_dafp.T * Omega*pi/30.0).T
                dT, dQ, dP = self.__thrustTorqueDictionary(dT_ds, dQ_ds, dP_ds, dT_dv, dQ_dv, dP_dv, npts, dT_dafp, dQ_dafp, dP_dafp)
            else:
                dT, dQ, dP = self.__thrustTorqueDictionary(dT_ds, dQ_ds, dP_ds, dT_dv, dQ_dv, dP_dv, npts)
            return P, T, Q, dP, dT, dQ

        else:
            return P, T, Q


    def __thrustTorqueDeriv(self, Np, Tp, dNp_dX, dTp_dX, dNp_dprecurve, dTp_dprecurve,
            r, precurve, presweep, precone, Rhub, Rtip, precurveTip, presweepTip, dNp_dafp=None, dTp_dafp=None):
        """derivatives of thrust and torque"""

        Tb = np.array([1, 0])
        Qb = np.array([0, 1])
        Npb, Tpb, rb, precurveb, presweepb, preconeb, Rhubb, Rtipb, precurvetipb, presweeptipb = \
            _bem.thrusttorque_bv(Np, Tp, r, precurve, presweep, precone, Rhub, Rtip, precurveTip, presweepTip, Tb, Qb)


        # X = [r, chord, theta, Rhub, Rtip, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega, pitch]
        dT_dNp = Npb[0, :]
        dQ_dNp = Npb[1, :]
        dT_dTp = Tpb[0, :]
        dQ_dTp = Tpb[1, :]

        # chain rule
        dT_dX = dT_dNp*dNp_dX + dT_dTp*dTp_dX
        dQ_dX = dQ_dNp*dNp_dX + dQ_dTp*dTp_dX

        dT_dr = dT_dX[0, :] + rb[0, :]
        dQ_dr = dQ_dX[0, :] + rb[1, :]
        dT_dchord = dT_dX[1, :]
        dQ_dchord = dQ_dX[1, :]
        dT_dtheta = dT_dX[2, :]
        dQ_dtheta = dQ_dX[2, :]
        dT_dRhub = np.sum(dT_dX[3, :]) + Rhubb[0]
        dQ_dRhub = np.sum(dQ_dX[3, :]) + Rhubb[1]
        dT_dRtip = np.sum(dT_dX[4, :]) + Rtipb[0]
        dQ_dRtip = np.sum(dQ_dX[4, :]) + Rtipb[1]
        dT_dpresweep = dT_dX[5, :] + presweepb[0, :]
        dQ_dpresweep = dQ_dX[5, :] + presweepb[1, :]
        dT_dprecone = np.sum(dT_dX[6, :]) + preconeb[0]*pi/180.0
        dQ_dprecone = np.sum(dQ_dX[6, :]) + preconeb[1]*pi/180.0
        dT_dtilt = np.sum(dT_dX[7, :])
        dQ_dtilt = np.sum(dQ_dX[7, :])
        dT_dhubht = np.sum(dT_dX[8, :])
        dQ_dhubht = np.sum(dQ_dX[8, :])
        dT_dprecurvetip = precurvetipb[0]
        dQ_dprecurvetip = precurvetipb[1]
        dT_dpresweeptip = presweeptipb[0]
        dQ_dpresweeptip = presweeptipb[1]
        dT_dprecurve = np.sum(dT_dNp*dNp_dprecurve + dT_dTp*dTp_dprecurve, axis=1) + precurveb[0, :]
        dQ_dprecurve = np.sum(dQ_dNp*dNp_dprecurve + dQ_dTp*dTp_dprecurve, axis=1) + precurveb[1, :]
        dT_dyaw = np.sum(dT_dX[9, :])
        dQ_dyaw = np.sum(dQ_dX[9, :])
        dT_dUinf = np.sum(dT_dX[11, :])
        dQ_dUinf = np.sum(dQ_dX[11, :])
        dT_dOmega = np.sum(dT_dX[12, :])
        dQ_dOmega = np.sum(dQ_dX[12, :])
        dT_dpitch = np.sum(dT_dX[13, :])
        dQ_dpitch = np.sum(dQ_dX[13, :])


        # scalars = [precone, tilt, hubHt, Rhub, Rtip, precurvetip, presweeptip, yaw, Uinf, Omega, pitch]
        dT_ds = np.array([dT_dprecone, dT_dtilt, dT_dhubht, dT_dRhub, dT_dRtip,
            dT_dprecurvetip, dT_dpresweeptip, dT_dyaw, dT_dUinf, dT_dOmega, dT_dpitch])
        dQ_ds = np.array([dQ_dprecone, dQ_dtilt, dQ_dhubht, dQ_dRhub, dQ_dRtip,
            dQ_dprecurvetip, dQ_dpresweeptip, dQ_dyaw, dQ_dUinf, dQ_dOmega, dQ_dpitch])


        # vectors = [r, chord, theta, precurve, presweep]
        dT_dv = np.vstack((dT_dr, dT_dchord, dT_dtheta, dT_dprecurve, dT_dpresweep))
        dQ_dv = np.vstack((dQ_dr, dQ_dchord, dQ_dtheta, dQ_dprecurve, dQ_dpresweep))

        if self.freeform:
            dT_dafp = np.dot(dT_dNp.reshape(1, 17), dNp_dafp) + np.dot(dT_dTp.reshape(1, 17), dTp_dafp)
            dQ_dafp = np.dot(dQ_dNp.reshape(1, 17), dNp_dafp) + np.dot(dQ_dTp.reshape(1, 17), dTp_dafp)
            return dT_ds, dQ_ds, dT_dv, dQ_dv, dT_dafp, dQ_dafp

        return dT_ds, dQ_ds, dT_dv, dQ_dv



    def __thrustTorqueDictionary(self, dT_ds, dQ_ds, dP_ds, dT_dv, dQ_dv, dP_dv, npts, dT_dafp=None, dQ_dafp=None, dP_dafp=None):


        # pack derivatives into dictionary
        dT = {}
        dQ = {}
        dP = {}

        # npts x 1
        dT['dprecone'] = dT_ds[:, 0].reshape(npts, 1)
        dQ['dprecone'] = dQ_ds[:, 0].reshape(npts, 1)
        dP['dprecone'] = dP_ds[:, 0].reshape(npts, 1)
        dT['dtilt'] = dT_ds[:, 1].reshape(npts, 1)
        dQ['dtilt'] = dQ_ds[:, 1].reshape(npts, 1)
        dP['dtilt'] = dP_ds[:, 1].reshape(npts, 1)
        dT['dhubHt'] = dT_ds[:, 2].reshape(npts, 1)
        dQ['dhubHt'] = dQ_ds[:, 2].reshape(npts, 1)
        dP['dhubHt'] = dP_ds[:, 2].reshape(npts, 1)
        dT['dRhub'] = dT_ds[:, 3].reshape(npts, 1)
        dQ['dRhub'] = dQ_ds[:, 3].reshape(npts, 1)
        dP['dRhub'] = dP_ds[:, 3].reshape(npts, 1)
        dT['dRtip'] = dT_ds[:, 4].reshape(npts, 1)
        dQ['dRtip'] = dQ_ds[:, 4].reshape(npts, 1)
        dP['dRtip'] = dP_ds[:, 4].reshape(npts, 1)
        dT['dprecurveTip'] = dT_ds[:, 5].reshape(npts, 1)
        dQ['dprecurveTip'] = dQ_ds[:, 5].reshape(npts, 1)
        dP['dprecurveTip'] = dP_ds[:, 5].reshape(npts, 1)
        dT['dpresweepTip'] = dT_ds[:, 6].reshape(npts, 1)
        dQ['dpresweepTip'] = dQ_ds[:, 6].reshape(npts, 1)
        dP['dpresweepTip'] = dP_ds[:, 6].reshape(npts, 1)
        dT['dyaw'] = dT_ds[:, 7].reshape(npts, 1)
        dQ['dyaw'] = dQ_ds[:, 7].reshape(npts, 1)
        dP['dyaw'] = dP_ds[:, 7].reshape(npts, 1)


        # npts x npts (diagonal)
        dT['dUinf'] = np.diag(dT_ds[:, 8])
        dQ['dUinf'] = np.diag(dQ_ds[:, 8])
        dP['dUinf'] = np.diag(dP_ds[:, 8])
        dT['dOmega'] = np.diag(dT_ds[:, 9])
        dQ['dOmega'] = np.diag(dQ_ds[:, 9])
        dP['dOmega'] = np.diag(dP_ds[:, 9])
        dT['dpitch'] = np.diag(dT_ds[:, 10])
        dQ['dpitch'] = np.diag(dQ_ds[:, 10])
        dP['dpitch'] = np.diag(dP_ds[:, 10])


        # npts x n
        dT['dr'] = dT_dv[:, 0, :]
        dQ['dr'] = dQ_dv[:, 0, :]
        dP['dr'] = dP_dv[:, 0, :]
        dT['dchord'] = dT_dv[:, 1, :]
        dQ['dchord'] = dQ_dv[:, 1, :]
        dP['dchord'] = dP_dv[:, 1, :]
        dT['dtheta'] = dT_dv[:, 2, :]
        dQ['dtheta'] = dQ_dv[:, 2, :]
        dP['dtheta'] = dP_dv[:, 2, :]
        dT['dprecurve'] = dT_dv[:, 3, :]
        dQ['dprecurve'] = dQ_dv[:, 3, :]
        dP['dprecurve'] = dP_dv[:, 3, :]
        dT['dpresweep'] = dT_dv[:, 4, :]
        dQ['dpresweep'] = dQ_dv[:, 4, :]
        dP['dpresweep'] = dP_dv[:, 4, :]

        if self.freeform:
            dT['dafp'] = dT_dafp
            dQ['dafp'] = dQ_dafp
            dP['dafp'] = dP_dafp

        return dT, dQ, dP

import os, sys, csv, subprocess
import cmath
import mpmath
from math import factorial
def cst_to_y_coordinates_given_x_Complexx(wl, wu, N, dz, xl, xu):

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1
    yl = ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates
    return yl, yu
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
        konfig = deepcopy(config)
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
            konfig = deepcopy(config)
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

def cfdAirfoilsSolveParallel(alphas, Res, afps, airfoil_analysis_options):
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

        config.CONSOLE = 'QUIET'

        cl = np.zeros(len(alphas))
        cd = np.zeros(len(alphas))
        alphas = np.degrees(alphas)
        procTotal = []
        konfigTotal = []
        konfigDirectTotal = []
        ztateTotal = []
        Re = airfoil_analysis_options['Re']
        for i in range(len(alphas)):
            meshFileName = basepath + os.path.sep + 'mesh_airfoil'+str(i+1)+'.su2'
            # Create airfoil coordinate file for SU2
            [x, y] = cst_to_coordinates_full(afps[i])
            airfoilFile = basepath + os.path.sep + 'airfoil'+str(i+1)+'_coordinates.dat'
            coord_file = open(airfoilFile, 'w')
            print >> coord_file, 'Airfoil Parallel'
            for j in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[j], y[j])
            coord_file.close()

            konfig = deepcopy(config)
            ztate = deepcopy(state)
            konfig.MESH_OUT_FILENAME = meshFileName
            konfig.DV_KIND = 'AIRFOIL'
            tempname = basepath + os.path.sep + 'config_DEF_direct.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = 'SU2_DEF ' + tempname
            the_Command = base_Command % the_Command
            sys.stdout.flush()
            proc = subprocess.Popen( the_Command, shell=True    ,
                             stdout=open(basepath + os.path.sep + 'mesh_deformation_airfoil'+str(i+1)+'.txt', 'w')      ,
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

            konfig.MESH_FILENAME = meshFileName
            ztate.FILES.MESH = config.MESH_FILENAME
            Uinf = 10.0
            Ma = Uinf / 340.29  # Speed of sound at sea level
            konfig.MACH_NUMBER = Ma
            konfig.REYNOLDS_NUMBER = Re

            if restart:
                konfig.RESTART_SOL = 'YES'
                konfig.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil'+str(i+1)+'.dat'
                konfig.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil'+str(i+1)+'_SOLVED.dat'
            else:
                konfig.RESTART_SOL = 'NO'
                konfig.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil'+str(i+1)+'.dat'
                konfig.SOLUTION_ADJ_FILENAME = basepath + os.path.sep + 'solution_adj_airfoil'+str(i+1)+'.dat'
                konfig.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'restart_flow_airfoil'+str(i+1)+'.dat'
                konfig.RESTART_ADJ_FILENAME = basepath + os.path.sep + 'restart_adj_airfoil'+str(i+1)+'.dat'
                konfig.SURFACE_ADJ_FILENAME = basepath + os.path.sep + 'surface_adjoint_airfoil' + str(i+1)
                konfig.SURFACE_FLOW_FILENAME = basepath + os.path.sep + 'surface_flow_airfoil' + str(i+1)


            x_vel = Uinf * cos(np.radians(alphas[i]))
            y_vel = Uinf * sin(np.radians(alphas[i]))
            konfig.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
            konfig.AoA = alphas[i]
            konfig.CONV_FILENAME = basepath + os.path.sep + 'history_airfoil'+str(i+1)
            #state = SU2.io.State(state)

            konfig_direct = deepcopy(konfig)
            # setup direct problem
            konfig_direct['MATH_PROBLEM']  = 'DIRECT'
            konfig_direct['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'

            # Run Solution
            tempname = basepath + os.path.sep + 'config_CFD_airfoil'+str(i+1)+'.cfg'
            konfig_direct.dump(tempname)
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
            cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil'+str(i+1)+'.txt', 'w')
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=cfd_output      ,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.close()
            procTotal.append(deepcopy(proc))
            konfigDirectTotal.append(deepcopy(konfig_direct))
            konfigTotal.append(deepcopy(konfig))
            ztateTotal.append(deepcopy(state))

        for i in range(len(alphas)):
            while procTotal[i].poll() is None:
                pass
            konfig = konfigDirectTotal[i]
            konfig['SOLUTION_FLOW_FILENAME'] = konfig['RESTART_FLOW_FILENAME']
            #oldstdout = sys.stdout
            #sys.stdout = oldstdout
            plot_format      = konfig['OUTPUT_FORMAT']
            plot_extension   = SU2.io.get_extension(plot_format)
            history_filename = konfig['CONV_FILENAME'] + plot_extension
            special_cases    = SU2.io.get_specialCases(konfig)

            final_avg = config.get('ITER_AVERAGE_OBJ',0)
            aerodynamics = SU2.io.read_aerodynamics( history_filename , special_cases, final_avg )
            config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM']  })
            info = SU2.io.State()
            info.FUNCTIONS.update( aerodynamics )
            ztateTotal[i].update(info)

            cl[i], cd[i] = info.FUNCTIONS['LIFT'], info.FUNCTIONS['DRAG']

        if airfoil_analysis_options['ComputeGradient']:
            print "computeGradients", airfoil_analysis_options['ComputeGradient']
            dcl_dx = []
            dcd_dx = []
            dcl_dafp = []
            dcd_dafp = []
            dcl_dalpha = []
            dcd_dalpha = []
            dcl_dRe = []
            dcd_dRe = []
            dx_dafpTotal = []
            procTotal = []
            konfigDragTotal = []
            konfigLiftTotal = []
            for i in range(len(alphas)):
                konfig = deepcopy(konfigTotal[i])
                ztate = ztateTotal[i]
                konfig.RESTART_SOL = 'NO'
                mesh_data = SU2.mesh.tools.read(konfig.MESH_FILENAME)
                points_sorted, loop_sorted = SU2.mesh.tools.sort_airfoil(mesh_data, marker_name='airfoil')

                SU2.io.restart2solution(konfig, ztate)
                # RUN FOR DRAG GRADIENTS
                konfig.OBJECTIVE_FUNCTION = 'DRAG'

                # setup problem
                konfig['MATH_PROBLEM']  = 'ADJOINT'
                konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_adjoint'

                # Run Solution
                #oldstdout = sys.stdout
                #sys.stdout = open(basepath + os.path.sep + 'output_cfd_adjoint_airfoil_drag'+str(i+1)+'.txt', 'w')
                tempname = basepath + os.path.sep + 'config_CFD_airfoil'+str(i+1)+'_drag.cfg'
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
                cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil'+str(i+1)+'_drag.txt', 'w')
                proc = subprocess.Popen( the_Command, shell=True    ,
                             stdout=cfd_output      ,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
                proc.stderr.close()
                proc.stdin.close()

                # merge

                procTotal.append(deepcopy(proc))
                konfigDragTotal.append(deepcopy(konfig))
                # ztateTotal.append(deepcopy(state))

            for i in range(len(alphas)):
                while procTotal[i].poll() is None:
                    pass
                konfig = konfigDragTotal[i]
                konfig['SOLUTION_ADJ_FILENAME'] = konfig['RESTART_ADJ_FILENAME']

                # filenames
                plot_format      = konfig['OUTPUT_FORMAT']
                plot_extension   = SU2.io.get_extension(plot_format)
                history_filename = konfig['CONV_FILENAME'] + plot_extension
                special_cases    = SU2.io.get_specialCases(konfig)

                # get history
                history = SU2.io.read_history( history_filename )

                # update super config
                config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM'] ,
                                'OBJECTIVE_FUNCTION'  : konfig['OBJECTIVE_FUNCTION']   })

                # files out
                objective    = konfig['OBJECTIVE_FUNCTION']
                adj_title    = 'ADJOINT_' + objective
                suffix       = SU2.io.get_adjointSuffix(objective)
                restart_name = konfig['RESTART_FLOW_FILENAME']
                restart_name = SU2.io.add_suffix(restart_name,suffix)

                # info out
                info = SU2.io.State()
                info.FILES[adj_title] = restart_name
                info.HISTORY[adj_title] = history

                #info = SU2.run.adjoint(konfig)
                ztate.update(info)
                dcd_dx1, xl, xu = su2Gradient(loop_sorted, konfig.SURFACE_ADJ_FILENAME + '.csv')
                dcd_dx.append(dcd_dx1)
                dcd_dalpha1 = ztate.HISTORY.ADJOINT_DRAG.Sens_AoA[-1]
                dcd_dalpha.append(dcd_dalpha1)
                n = 8
                m = 200
                dx_dafp = np.zeros((n, m))
                wl_original, wu_original, N, dz = CST_to_kulfan(afps[i])
                step_size = airfoil_analysis_options['cs_step']
                cs_step = complex(0, step_size)

                for k in range(0, n):
                    wl_new = deepcopy(wl_original.astype(complex))
                    wu_new = deepcopy(wu_original.astype(complex))
                    if k < n/2:
                        wl_new[k] += cs_step
                    else:
                        wu_new[k-4] += cs_step
                    yl_new, yu_new = cst_to_y_coordinates_given_x_Complexx(wl_new, wu_new, N, dz, xl, xu)
                    for j in range(m):
                        if k < n/2:
                            if j < len(yl_new):
                                dx_dafp[k][j] = (np.imag(yl_new[j])) / step_size
                            else:
                                dx_dafp[k][j] = 0.0
                        else:
                            if j > m - len(yu_new):
                                dx_dafp[k][j] = np.imag(yu_new[j- (m-len(yu_new))]) / step_size
                            else:
                                dx_dafp[k][j] = 0.0
                dx_dafpTotal.append(dx_dafp)
            procTotal = []
            for i in range(len(alphas)):
                konfig = deepcopy(konfigTotal[i])
                ztate = ztateTotal[i]

                konfig.OBJECTIVE_FUNCTION = 'LIFT'

                # setup problem
                konfig['MATH_PROBLEM']  = 'ADJOINT'
                konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_adjoint'

                # Run Solution
                #oldstdout = sys.stdout
                #sys.stdout = open('output_cfd_adjoint_airfoil_lift'+str(i+1)+'.txt', 'w')
                tempname = basepath + os.path.sep + 'config_CFD_airfoil_lift'+str(i+1)+'.cfg'
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
                cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil'+str(i+1)+'_lift.txt', 'w')
                proc = subprocess.Popen( the_Command, shell=True    ,
                             stdout=cfd_output      ,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
                proc.stderr.close()
                proc.stdin.close()

                # merge

                procTotal.append(deepcopy(proc))
                konfigLiftTotal.append(deepcopy(konfig))
                # ztateTotal.append(deepcopy(state))

            for i in range(len(alphas)):
                while procTotal[i].poll() is None:
                    pass
                konfig = konfigLiftTotal[i]
                konfig['SOLUTION_ADJ_FILENAME'] = konfig['RESTART_ADJ_FILENAME']

                # filenames
                plot_format      = konfig['OUTPUT_FORMAT']
                plot_extension   = SU2.io.get_extension(plot_format)
                history_filename = konfig['CONV_FILENAME'] + plot_extension
                special_cases    = SU2.io.get_specialCases(konfig)

                # get history
                history = SU2.io.read_history( history_filename )

                # update super config
                config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM'] ,
                                'OBJECTIVE_FUNCTION'  : konfig['OBJECTIVE_FUNCTION']   })

                # files out
                objective    = konfig['OBJECTIVE_FUNCTION']
                adj_title    = 'ADJOINT_' + objective
                suffix       = SU2.io.get_adjointSuffix(objective)
                restart_name = konfig['RESTART_FLOW_FILENAME']
                restart_name = SU2.io.add_suffix(restart_name,suffix)

                # info out
                info = SU2.io.State()
                info.FILES[adj_title] = restart_name
                info.HISTORY[adj_title] = history
                surface_adjoint = konfig.SURFACE_ADJ_FILENAME + '.csv'
                #info = SU2.run.adjoint(konfig)
                ztate.update(info)
                dcl_dx1, xl, xu = su2Gradient(loop_sorted, surface_adjoint)
                dcl_dx.append(dcl_dx1)
                dcl_dalpha1 = ztate.HISTORY.ADJOINT_LIFT.Sens_AoA[-1]
                dcl_dalpha.append(dcl_dalpha1)
                # info = SU2.run.adjoint(konfig)
                # ztate.update(info)


                dafp_dx_ = np.matrix(dx_dafpTotal[i])
                dcl_dx_ = np.matrix(dcl_dx[i])
                dcd_dx_ = np.matrix(dcd_dx[i])

                dcl_dafp.append(np.asarray(dafp_dx_ * dcl_dx_.T).reshape(8))
                dcd_dafp.append(np.asarray(dafp_dx_ * dcd_dx_.T).reshape(8))
                #sys.stdout = oldstdout

                dcl_dRe.append(0.0)
                dcd_dRe.append(0.0)
            return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp #cl, cd, dcl_dafp, dcd_dafp, dcl_dalpha, dcd_dalpha

        return cl, cd

def su2Gradient(loop_sorted, surface_adjoint):
        data = np.zeros([500, 8])
        with open(surface_adjoint, 'rb') as f1:
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

def evaluate_direct_parallel(alphas, Res, afs, computeAlphaGradient=False, computeAFPGradient=False):
        indices_to_compute = []
        n = len(alphas)
        airfoil_analysis_options = afs[-1].airfoil_analysis_options
        cl = np.zeros(n)
        cd = np.zeros(n)
        dcl_dalpha = [0]*n
        dcd_dalpha = [0]*n
        dcl_dafp = [0]*n
        dcd_dafp = [0]*n
        dcl_dRe = [0]*n
        dcd_dRe = [0]*n

        for i in range(len(alphas)):
            alpha = alphas[i]
            Re = Res[i]
            af = afs[i]
            if af.afp is not None and abs(np.degrees(alpha)) < af.airfoil_analysis_options['maxDirectAoA']:
                if alpha in af.alpha_storage and alpha in af.dalpha_storage:
                    index = af.alpha_storage.index(alpha)
                    cl[i] = af.cl_storage[index]
                    cd[i] = af.cd_storage[index]
                    if computeAlphaGradient:
                        index = af.dalpha_storage.index(alpha)
                        dcl_dalpha[i] = af.dcl_storage[index]
                        dcd_dalpha[i] = af.dcd_storage[index]
                    if computeAFPGradient and alpha in af.dalpha_dafp_storage:
                        index = af.dalpha_dafp_storage.index(alpha)
                        dcl_dafp[i] = af.dcl_dafp_storage[index]
                        dcd_dafp[i] = af.dcd_dafp_storage[index]
                    dcl_dRe[i] = 0.0
                    dcd_dRe[i] = 0.0
                else:
                    indices_to_compute.append(i)
            else:
                cl[i] = af.cl_spline.ev(alpha, Re)
                cd[i] = af.cd_spline.ev(alpha, Re)
                tck_cl = af.cl_spline.tck[:3] + af.cl_spline.degrees  # concatenate lists
                tck_cd = af.cd_spline.tck[:3] + af.cd_spline.degrees

                dcl_dalpha[i] = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
                dcd_dalpha[i] = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

                if af.one_Re:
                    dcl_dRe[i] = 0.0
                    dcd_dRe[i] = 0.0
                else:
                    dcl_dRe[i] = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
                    dcd_dRe[i] = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
                if computeAFPGradient and af.afp is not None:
                    dcl_dafp[i], dcd_dafp[i] = af.splineFreeFormGrad(alpha, Re)
                else:
                    dcl_dafp[i], dcd_dafp[i] = np.zeros(8), np.zeros(8)
        if indices_to_compute is not None:
            alphas_to_compute = [alphas[i] for i in indices_to_compute]
            Res_to_compute = [Res[i] for i in indices_to_compute]
            afps_to_compute = [afs[i].afp for i in indices_to_compute]
            if airfoil_analysis_options['ComputeGradient']:
                cls, cds, dcls_dalpha, dcls_dRe, dcds_dalpha, dcds_dRe, dcls_dafp, dcds_dafp = cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, afps_to_compute, airfoil_analysis_options)
                for j in range(len(indices_to_compute)):
                    dcl_dalpha[indices_to_compute[j]] = dcls_dalpha[j]
                    dcl_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcd_dalpha[indices_to_compute[j]] = dcds_dalpha[j]
                    dcd_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcl_dafp[indices_to_compute[j]] = dcls_dafp[j]
                    dcd_dafp[indices_to_compute[j]] = dcds_dafp[j]

            else:
                cls, cds = cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, afps_to_compute, airfoil_analysis_options)
            for j in range(len(indices_to_compute)):
                cl[indices_to_compute[j]] = cls[j]
                cd[indices_to_compute[j]] = cds[j]

        if computeAFPGradient:
            try:
                return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp
            except:
                raise
        elif computeAlphaGradient:
            return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe
        else:
            return cl, cd


if __name__ == '__main__':

    # geometry
    Rhub = 1.5
    Rtip = 63.0

    r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
    chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    B = 3  # number of blades

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5

    # Airfoil specifications
    airfoil_analysis_options = dict(AnalysisMethod='CFD', AirfoilParameterization='CST',
                                CFDiterations=10000, CFDprocessors=32, FreeFormDesign=True, BEMSpline=True,
                                alphas=np.linspace(-15, 15, 30), Re=5e5, ComputeGradient=True)
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
    if airfoil_analysis_options['AnalysisMethod'] == 'Files':
        import os
        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
        basepath = '5MW_AFFiles' + os.path.sep

        # load all airfoils
        airfoil_types = [0]*8
        airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
        airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
        airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
        airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
        airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
        airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
        airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
        airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        af = [0]*len(r)
        for i in range(len(r)):
            af[i] = airfoil_types[af_idx[i]]

    else:
        # Specify airfoil parameters
        airfoil_parameterization = np.asarray([[-0.49209940079930325, -0.72861624849999296, -0.38147646962813714, 0.13679205926397994, 0.50396496117640877, 0.54798355691567613, 0.37642896917099616, 0.37017796580840234],
                                               [-0.38027535114760153, -0.75920832612723133, -0.21834261746205941, 0.086359012110824224, 0.38364567865371835, 0.48445264573011815, 0.26999944648962521, 0.34675843509167931],
                                               [-0.29817561716727448, -0.67909473119918973, -0.15737231648880162, 0.12798260780188203, 0.2842322211249545, 0.46026650967959087, 0.21705062978922526, 0.33758303223369945],
                                               [-0.27413320446357803, -0.40701949670950271, -0.29237424992338562, 0.27867844397438357, 0.23582783854698663, 0.43718573158380936, 0.25389099250498309, 0.31090780344061775],
                                               [-0.19600050454371795, -0.28861738331958697, -0.20594891135118523, 0.19143138186871009, 0.22876347660120994, 0.39940768357615447, 0.28896745336793572, 0.29519782561050112],
                                               [-0.17200255338600826, -0.13744743777735921, -0.24288986290945222, 0.15085289615063024, 0.20650016452789369, 0.35540642522188848, 0.32797634888819488, 0.2592276816645861]])

        af_input_init = CCAirfoil.initFromInput
        if airfoil_analysis_options['AirfoilParameterization'] == 'CST':
            af_freeform_init = CCAirfoil.initFromCST
        elif airfoil_analysis_options['AirfoilParameterization'] == 'NACA':
            af_freeform_init = CCAirfoil.initFromNACA
        else:
            af_freeform_init = CCAirfoil.initFromInput

        # load all airfoils
        non_airfoils_idx = 2
        airfoil_types = [0]*8
        non_airfoils_alphas = [-180.0, 0.0, 180.0]
        non_airfoils_cls = [0.0, 0.0, 0.0]
        non_airfoils_cds = [[0.5, 0.5, 0.5],[0.35, 0.35, 0.35]]
        print "Generating airfoil data..."
        for i in range(len(airfoil_types)):
            if i < non_airfoils_idx:
                airfoil_types[i] = af_input_init(non_airfoils_alphas, airfoil_analysis_options['Re'], non_airfoils_cls, non_airfoils_cds[i], non_airfoils_cls)
            else:
                time0 = time.time()
                airfoil_types[i] = af_freeform_init(airfoil_parameterization[i-2], airfoil_analysis_options)
                print "Airfoil ", str(i+1-2), " data generation complete in ", time.time() - time0, " seconds."
        print "Finished generating airfoil data"

        af = [0]*len(r)
        for i in range(len(af)):
            af[i] = airfoil_types[af_idx[i]]

    tilt = -5.0
    precone = 2.5
    yaw = 0.0
    shearExp = 0.2
    hubHt = 80.0
    nSector = 8

    aeroanalysis = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                           precone, tilt, yaw, shearExp, hubHt, nSector, airfoil_parameterization=airfoil_parameterization, airfoil_options=airfoil_analysis_options, derivatives=False)

    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 90

    ### LOADS
    Np, Tp = aeroanalysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)

    import matplotlib.pyplot as plt
    # rstar = (rload - rload[0]) / (rload[-1] - rload[0])
    plt.plot(r, Tp/1e3, 'k', label='lead-lag')
    plt.plot(r, Np/1e3, 'r', label='flapwise')
    plt.xlabel('blade fraction')
    plt.ylabel('distributed aerodynamic loads (kN)')
    plt.legend(loc='upper left')

    tsr = np.linspace(2, 14, 20)
    Omega = 10.0 * np.ones_like(tsr)
    Uinf = Omega*pi/30.0 * Rtip/tsr
    pitch = np.zeros_like(tsr)

    # COEFFICIENTS
    CP, CT, CQ = aeroanalysis.evaluate(Uinf, Omega, pitch, coefficient=True)
    print CP, CT, CQ

    wind_tunnel_CP_origin = [ 0.02344119,  0.0653068,   0.12733272,  0.19768979,  0.275223,    0.35764107,
                            0.41604225,  0.44387852,  0.45630932,  0.45969981,  0.45627368,  0.44741262,
                            0.43461535,  0.4190967,   0.40101026,  0.38017748,  0.35642367,  0.32954743,
                            0.29939923,  0.26601073]

    plt.figure()
    plt.plot(tsr, CP, 'xk-', label='CFD')
    plt.plot(tsr, wind_tunnel_CP_origin, '^r-', label='WT')
    plt.legend(loc='best')
    plt.xlabel('$\lambda$')
    plt.ylabel('$c_p$')
    plt.show()
