import numpy as np
import cmath
import mpmath
from math import factorial, pi, cos, sin
#from naca_generator import naca4, naca5
import sys, os, csv
import subprocess
from copy import deepcopy
import pyXLIGHT

class AirfoilAnalysis:
    """A helper class to store and evaluate airfoil coordinates"""
    def __init__(self, afp, airfoil_analysis_options, numCoordinates=200):
        self.afp = afp
        self.parameterization_method = airfoil_analysis_options['AirfoilParameterization']
        self.analysis_method = airfoil_analysis_options['AnalysisMethod']
        self.airfoil_analysis_options = airfoil_analysis_options
        self.numCoordinates = numCoordinates
        self.x, self.y, self.xl, self.xu, self.yl, self.yu = self.__setCoordinates()
        self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c = self.__setCoordinatesComplex()

    def getCoordinates(self, type='full'):
        if type=='full':
            return self.x, self.y
        elif type=='split':
            return self.xl, self.xu, self.yl, self.yu
        else:
            return self.x, self.y, self.xl, self.xu, self.yl, self.yu
    def getCoordinatesComplex(self, type='full'):
        if type=='full':
            return self.x_c, self.y_c
        elif type=='split':
            return self.xl_c, self.xu_c, self.yl_c, self.yu_c
        else:
            return self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c

    def getYCoordinatesGivenX(self, xl, xu):
        if self.parameterization_method == 'CST':
            yl, yu = self.__cstYgivenX(self.wl, self.wu, self.numCoordinates, self.dz, xl, xu)
        return yl, yu

    def getCoordinateDerivatives(self, xl=None, xu=None):
        if xl is None or xu is None:
            dy_dafp = self.__cstYDerivatives(self.wl, self.wu, self.numCoordinates, self.dz, self.xl, self.xu)
        else:
            dy_dafp = self.__cstYDerivatives(self.wl, self.wu, self.numCoordinates, self.dz, xl, xu)
        return dy_dafp

    def saveToFile(self, airfoilFile):
        coord_file = open(airfoilFile, 'w')
        print >> coord_file, 'Airfoil'
        for i in range(len(self.x)):
            print >> coord_file, '{:<10f}\t{:<10f}'.format(self.x[i], self.y[i])
        coord_file.close()

    def setNumCoordinates(self, numCoordinates):
        self.numCoordinates = numCoordinates

    def __setCoordinates(self):
        if self.parameterization_method == 'CST':
             x, y, xl, xu, yl, yu = self.__cstCoordinates()
        return x, y, xl, xu, yl, yu

    def __setCoordinatesComplex(self):
        if self.parameterization_method == 'CST':
             x, y, xl, xu, yl, yu = self.__cstCoordinatesComplex()
        return x, y, xl, xu, yl, yu

    def __cstToKulfan(self, CST):
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
        return wl, wu


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

    def __cstYgivenX(self, wl, wu, N, dz, xl, xu):

        # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
        N1 = 0.5
        N2 = 1
        yl = self.__ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates
        return yl, yu

    def __cstYgivenXComplex(self, wl, wu, N, dz, xl, xu):

        # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
        N1 = 0.5
        N2 = 1
        yl = self.__ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates
        return yl, yu

    def __cstYDerivatives(self, wl, wu, N, dz, xl, xu):

        # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
        N1 = 0.5
        N2 = 1
        dyl = self.__ClassShapeDerivative(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        dyu = self.__ClassShapeDerivative(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates
        dyl_dzeros = np.zeros((len(wl), N-len(xl)))
        dyu_dzeros = np.zeros((len(wu), N-len(xu)))
        dyl_dw = np.hstack((dyl, dyl_dzeros))
        dyu_dw = np.hstack((dyu_dzeros, dyu))
        dy_dafp = np.vstack((dyl_dw, dyu_dw))

        return dy_dafp

    def __cstCoordinates(self):
        self.wl, self.wu = self.__cstToKulfan(self.afp)
        self.dz = 0.0
        wl, wu, N, dz = self.wl, self.wu, self.numCoordinates, self.dz
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

        yl = self.__ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x))
        for k in range(len(x)):
            x1[k] = x[k][0]
        x = x1
        return x, y, xl, xu, yl, yu

    def __cstCoordinatesReal(self, wl, wu, N, dz):

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

        yl = self.__ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x))
        for k in range(len(x)):
            x1[k] = x[k][0]
        x = x1
        return x, y


    def __cstCoordinatesComplex2(self, wl, wu, N, dz):
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

        yl = self.__ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x), dtype=complex)
        for k in range(len(x)):
            x1[k] = x[k][0]
        x = x1
        return x, y

    def __cstCoordinatesComplex(self):
        wl, wu, N, dz = self.wl, self.wu, self.numCoordinates, self.dz
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

        yl = self.__ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x), dtype=complex)
        for k in range(len(x)):
            x1[k] = x[k][0]
        x = x1
        return x, y, xl, xu, yl, yu


    def __ClassShapeDerivative(self, w, x, N1, N2, dz):
        n = len(w) - 1
        dy_dw = np.zeros((n+1, len(x)))
        for i in range(len(x)):
            for j in range(0, n+1):
                dy_dw[j][i] = x[i]**N1*((1-x[i])**N2) * factorial(n)/(factorial(j)*(factorial((n)-(j)))) * x[i]**(j) * ((1-x[i])**(n-(j)))

        y = self.__ClassShape(w, x, N1, N2, dz)

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

                dnormx3 = y[i+1] - y[i]
                dnormy3 = -(x[i+1] - x[i])


                # normal vector of backward line with adjacent point
                dx2 = x[i] - x[i-1]
                dy2 = y[i] - y[i-1]
                dnormy2 = dx2 - -dx2
                dnormx2 = -dy2 - dy2

                dnormx4 = y[i] - y[i-1]
                dnormy4 = -(x[i] - x[i-1])

                # dnormx = dnormx1 + dnormx2
                # dnormy = dnormy1 + dnormy2

                dnormx = dnormx3 + dnormx4
                dnormy = dnormy3 + dnormy4


                norm_y = dnormy / np.sqrt(dnormy**2 + dnormx**2)

            for j in range(0, n+1):
                dy_total[j][i] = dy_dw[j][i] * norm_y
        return dy_total

    def computeSpline(self):
        if self.airfoil_analysis_options['BEMSpline'] == 'CFD':
            cl, cd, cm, alphas, failure  = self.__cfdSpline()
        else:
            cl, cd, cm, alphas, failure  = self.__xfoilSpline()
        return cl, cd, cm, alphas, failure

    def __xfoilSpline(self):
        self.airfoilShapeFile = 'airfoil_shape.dat'
        self.saveToFile(self.airfoilShapeFile)
        alphas, Re = self.airfoil_analysis_options['alphas'], self.airfoil_analysis_options['Re']
        airfoil = pyXLIGHT.xfoilAnalysis(self.airfoilShapeFile, x=self.x, y=self.y)
        airfoil.re, airfoil.mach, airfoil.iter = Re, 0.0, 100
        cl, cd, cm, to_delete = np.zeros(len(alphas)), np.zeros(len(alphas)), np.zeros(len(alphas)), np.zeros(0)
        failure = False
        for j in range(len(alphas)):
            cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(alphas[j])
            if lexitflag:
                cl[j], cd[j] = -10.0, 0.0

        # Make sure none of the values are too far outliers
        cl_diff = np.diff(np.asarray(cl))
        cd_diff = np.diff(np.asarray(cd))
        for zz in range(len(cl_diff)):
            if abs(cd_diff[zz]) > 0.02 or abs(cl_diff[zz]) > 0.5:
                to_delete = np.append(to_delete, zz)

        # error handling in case of XFOIL failure
        for k in range(len(cl)):
            if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 1.0 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
                to_delete = np.append(to_delete, k)

        cl, cd, cm = np.delete(cl, to_delete), np.delete(cd, to_delete), np.delete(cm, to_delete)

        if not cl.size or len(cl) < 3 or max(cl) < 0.0:
            print "XFOIL Failure! Using default backup airfoil." # for CST = [-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25]
            cl = [-1.11249573, -1.10745928, -1.10242437, -1.10521061, -1.03248528, -0.9272929, -0.81920516, -0.70843745, -0.58962942, -0.45297636, -0.34881162, -0.26194, -0.17375163, -0.09322158, -0.01072867,  0.07232111,
                  0.15326737,  0.22932199, 0.29657574,  0.36818004,  0.45169576,  0.55640456 , 0.68532189,  0.81592085, 0.93355555,  1.04754944,  1.06513144,  1.07821432 , 1.09664777,  1.11425611]
            cd = [ 0.03966997,  0.03289554,  0.02783541,  0.02418726,  0.02120267,  0.01849611,  0.01623273,  0.01424686,  0.0124225 ,  0.01083306,  0.00973778,  0.00908278, 0.00867001,  0.00838171,  0.00823596,  0.00820681,
                   0.00828496 , 0.00842328,  0.00867177,  0.00921659,  0.01004469,  0.01129231,  0.01306175 , 0.01509252, 0.01731396,  0.01986422,  0.02234169 , 0.02555122,  0.02999641 , 0.03574208]
            cm = np.zeros(len(cl))
            alphas = np.linspace(-15, 15, len(cl))
            failure = True
        else:
            alphas = np.delete(alphas, to_delete)
        return cl, cd, cm, alphas, failure

    def computeDirect(self, alpha, Re):
        if self.analysis_method == 'CFD':
            cl, cd, dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp, lexitflag = self.__cfdDirect(alpha, Re, GenerateMESH=True)
        else:
            cl, cd, dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp, lexitflag = self.__xfoilDirect(alpha, Re)
        dcl_dRe, dcd_dRe = 0.0, 0.0
        return cl, cd, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dafp, dcd_dafp, lexitflag

    def __cfdSpline(self):
        alphas = self.airfoil_analysis_options['alphas']
        cl, cd, cm, failure = np.zeros(len(alphas)), np.zeros(len(alphas)), np.zeros(len(alphas)), False
        if self.airfoil_analysis_options['ParallelAirfoils']:
            cl, cd = self.__cfdParallelSpline(np.radians(alphas), self.airfoil_analysis_options['Re'], self.airfoil_analysis_options)
        else:
            for j in range(len(alphas)):
                if j == 0:
                    mesh = True
                else:
                    mesh = False
                cl[j], cd[j] = self.__cfdDirect(np.radians(alphas[j]), self.airfoil_analysis_options['Re'], GenerateMESH=mesh)
        return cl, cd, cm, alphas, failure

    def __xfoilDirect(self, alpha, Re):
        airfoil_shape_file = None
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=self.x, y=self.y)
        airfoil.re = self.airfoil_analysis_options['Re']
        airfoil.mach = 0.00
        airfoil.iter = 100
        dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp = 0.0, 0.0, np.zeros(8), np.zeros(8)
        if self.airfoil_analysis_options['ComputeGradient']:
            cs_step = 1e-20
            angle = complex(np.degrees(alpha), cs_step)
            cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
            dcl_dalpha, dcd_dalpha = 180.0/np.pi*np.imag(deepcopy(cl))/ cs_step, 180.0/np.pi*np.imag(deepcopy(cd)) / cs_step
            cl, cd = np.real(np.asscalar(cl)), np.real(np.asscalar(cd))
            if abs(dcl_dalpha) > 10.0 or abs(dcd_dalpha) > 10.0:
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
            if self.airfoil_analysis_options['FreeFormDesign']:
                dcl_dafp, dcd_dafp = self.xfoilGradients(alpha, Re)
                if np.any(abs(dcl_dafp)) > 100.0 or np.any(abs(dcd_dafp) > 100.0):
                    print "Error in complex step splines"
        else:
            cl, cd, cm, lexitflag = airfoil.solveAlpha(np.degrees(alpha))
            cl, cd = np.asscalar(cl), np.asscalar(cd)
        return cl, cd, dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp, lexitflag

    def xfoilGradients(self, alpha, Re):
        alpha = np.degrees(alpha)
        wl, wu, N, dz = self.wl, self.wu, self.numCoordinates, self.dz
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
                from akima import Akima
                from airfoilprep_free import Airfoil
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

        x, y = self.__cstCoordinatesComplex2(wl_complex, wu_complex, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.airfoil_analysis_options['Re']
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl_complex, cd_complex, cm_complex, lexitflag = airfoil.solveAlphaComplex(alpha)
        return deepcopy(cl_complex), deepcopy(cd_complex), deepcopy(lexitflag)

    def xfoilFlowReal(self, alpha, wl, wu, N, dz):
        airfoil_shape_file = None
        x, y = self.__cstCoordinatesReal(wl, wu, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.airfoil_analysis_options['Re']
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl, cd, cm, lexitflag = airfoil.solveAlpha(alpha)
        return np.asscalar(cl), np.asscalar(cd), deepcopy(lexitflag)

    def __cfdDirect(self, alpha, Re, GenerateMESH=True, airfoilNum=0):
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2
        airfoil_analysis_options = self.airfoil_analysis_options
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CFDFiles')
        config_filename = basepath + os.path.sep + airfoil_analysis_options['cfdConfigFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = airfoil_analysis_options['CFDprocessors']
        config.EXT_ITER    = airfoil_analysis_options['CFDiterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = basepath + os.path.sep + 'mesh_AIRFOIL_serial.su2'

        if GenerateMESH:
            return_code = self.__generateMesh(meshFileName, config, state, basepath)
            restart = False
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
        config.REYNOLDS_NUMBER = airfoil_analysis_options['Re']

        if restart:
            config.RESTART_SOL = 'YES'
            config.RESTART_FLOW_FILENAME = 'solution_flow_AIRFOIL' + str(airfoilNum+1) +'.dat'
            config.SOLUTION_FLOW_FILENAME = 'solution_flow_SOLVED_AIRFOIL' + str(airfoilNum+1) + '.dat'
        else:
            config.RESTART_SOL = 'NO'
            config.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil.dat'
            config.SOLUTION_ADJ_FILENAME = basepath + os.path.sep + 'solution_adj_airfoil.dat'
            config.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'restart_flow_airfoil.dat'
            config.RESTART_ADJ_FILENAME = basepath + os.path.sep + 'restart_adj_airfoil.dat'
            config.SURFACE_ADJ_FILENAME = basepath + os.path.sep + 'surface_adjoint_airfoil'
            config.SURFACE_FLOW_FILENAME = basepath + os.path.sep + 'surface_flow_airfoil'

        konfig = deepcopy(config)
        konfig['MATH_PROBLEM']  = 'DIRECT'
        konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'
        tempname = basepath + os.path.sep + 'config_CFD_airfoil.cfg'
        konfig.dump(tempname)
        SU2_RUN = os.environ['SU2_RUN']
        sys.path.append( SU2_RUN )

        processes = konfig['NUMBER_PART']
        the_Command = 'SU2_CFD ' + tempname
        base_Command = os.path.join(SU2_RUN,'%s')
        the_Command = base_Command % the_Command
        if konfig['NUMBER_PART'] > 0:
            mpi_Command = 'mpirun -n %i %s'
            the_Command = mpi_Command % (processes,the_Command)
        else:
            mpi_Command = ''
        if processes > 0:
            if not mpi_Command:
                raise RuntimeError , 'could not find an mpi interface'
        cfd_direct_output = open(basepath + os.path.sep + 'cfd_direct_output.txt', 'w')

        sys.stdout.flush()
        proc = subprocess.Popen( the_Command, shell=True    ,
                     stdout=cfd_direct_output,
                     stderr=subprocess.PIPE,
                     stdin=subprocess.PIPE)
        proc.stderr.close()
        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise ValueError('Error in mesh deformation. Error code: %c' % (return_code))

        konfig['SOLUTION_FLOW_FILENAME'] = konfig['RESTART_FLOW_FILENAME']
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

        cl, cd = info.FUNCTIONS['LIFT'], info.FUNCTIONS['DRAG']

        if self.airfoil_analysis_options['ComputeGradient']:
            konfig = deepcopy(config)
            ztate = deepcopy(state)

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
            tempname = basepath + os.path.sep + 'config_CFD_airfoil_drag.cfg'
            konfig.dump(tempname)

            SU2_RUN = os.environ['SU2_RUN']
            sys.path.append( SU2_RUN )
            processes = konfig['NUMBER_PART']
            the_Command = 'SU2_CFD ' + tempname
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = base_Command % the_Command
            if konfig['NUMBER_PART'] > 0:
                mpi_Command = 'mpirun -n %i %s'
                the_Command = the_Command = mpi_Command % (processes,the_Command)
            else:
                mpi_Command = ''
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError , 'could not find an mpi interface'

            sys.stdout.flush()
            cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil_drag.txt', 'w')
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=cfd_output      ,
                         stderr=cfd_output,
                         stdin=subprocess.PIPE)
            # proc.stderr.close()
            proc.stdin.close()
            return_code = proc.wait()
            if return_code != 0:
                raise ValueError('Error in CFD Direct. Error code: %c' % (return_code))

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
            dcd_dx, xl, xu = su2Gradient(loop_sorted, surface_adjoint)
            dcd_dalpha = ztate.HISTORY.ADJOINT_DRAG.Sens_AoA[-1]
            dy_dafp = self.__cstYDerivatives(self.wl, self.wu, self.numCoordinates, self.dz, xl, xu)
            # info = SU2.run.adjoint(konfig)
            # ztate.update(info)
            konfig = deepcopy(config)
            ztate = deepcopy(state)

            konfig.RESTART_SOL = 'NO'
            mesh_data = SU2.mesh.tools.read(konfig.MESH_FILENAME)
            points_sorted, loop_sorted = SU2.mesh.tools.sort_airfoil(mesh_data, marker_name='airfoil')

            SU2.io.restart2solution(konfig, ztate)
            # RUN FOR DRAG GRADIENTS

            konfig.OBJECTIVE_FUNCTION = 'LIFT'

            # setup problem
            konfig['MATH_PROBLEM']  = 'ADJOINT'
            konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_adjoint'

            # Run Solution
            tempname = basepath + os.path.sep + 'config_CFD_airfoil_lift.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            sys.path.append( SU2_RUN )
            processes = konfig['NUMBER_PART']
            the_Command = 'SU2_CFD ' + tempname
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = base_Command % the_Command
            if konfig['NUMBER_PART'] > 0:
                mpi_Command = 'mpirun -n %i %s'
                the_Command = the_Command = mpi_Command % (processes,the_Command)
            else:
                mpi_Command = ''
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError , 'could not find an mpi interface'

            sys.stdout.flush()
            cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil_lift.txt', 'w')
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=cfd_output      ,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.close()
            return_code = proc.wait()
            if return_code != 0:
                raise ValueError('Error in CFD Drag Adjoint. Error code: %c' % (return_code))
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
            dcl_dx, xl, xu = su2Gradient(loop_sorted, surface_adjoint)
            dcl_dalpha = ztate.HISTORY.ADJOINT_LIFT.Sens_AoA[-1]

            dafp_dx_ = np.matrix(dy_dafp)
            dcl_dx_ = np.matrix(dcl_dx)
            dcd_dx_ = np.matrix(dcd_dx)

            dcl_dafp = np.asarray(dafp_dx_ * dcl_dx_.T).reshape(8)
            dcd_dafp = np.asarray(dafp_dx_ * dcd_dx_.T).reshape(8)
            lexitflag = False
        else:
            dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp, lexitflag = 0.0, 0.0, np.zeros(8), np.zeros(8), False
        return cl, cd, dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp, lexitflag

    def __cfdParallelSpline(self, alphas, Re, airfoil_analysis_options):
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
        return_code = self.__generateMesh(meshFileName, config, state, basepath)

        config.MESH_FILENAME = meshFileName
        state.FILES.MESH = config.MESH_FILENAME
        Uinf = 10.0
        Ma = Uinf / 340.29  # Speed of sound at sea level
        config.MACH_NUMBER = Ma
        config.REYNOLDS_NUMBER = Re
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
            konfig['MATH_PROBLEM']  = 'DIRECT'
            konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'
            tempname = basepath + os.path.sep + 'config_CFD'+str(int(alphas[i]))+'.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            sys.path.append( SU2_RUN )
            mpi_Command = 'mpirun -n %i %s'
            processes = konfig['NUMBER_PART']
            the_Command = 'SU2_CFD ' + tempname
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = base_Command % the_Command
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError , 'could not find an mpi interface'
            the_Command = mpi_Command % (processes,the_Command)
            sys.stdout.flush()
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=sys.stdout, #open(basepath + os.path.sep + 'cfd_output'+str(i+1)+'.txt', 'w'),
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
            plot_format      = konfig['OUTPUT_FORMAT']
            plot_extension   = SU2.io.get_extension(plot_format)
            history_filename = konfig['CONV_FILENAME'] + plot_extension
            special_cases    = SU2.io.get_specialCases(konfig)

            final_avg = config.get('ITER_AVERAGE_OBJ',0)
            aerodynamics = SU2.io.read_aerodynamics( history_filename , special_cases, final_avg)
            config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM']  })
            info = SU2.io.State()
            info.FUNCTIONS.update( aerodynamics )
            state.update(info)

            cl[i], cd[i] = info.FUNCTIONS['LIFT'], info.FUNCTIONS['DRAG']
        return cl, cd

    def __generateMesh(self, meshFileName, config, state, basepath):
        airfoilFile = basepath + os.path.sep + 'airfoil_shape_parallel.dat'
        self.saveToFile(airfoilFile)
        konfig = deepcopy(config)
        konfig.VISUALIZE_DEFORMATION = 'NO'
        konfig.MESH_OUT_FILENAME = meshFileName
        konfig.DV_KIND = 'AIRFOIL'
        tempname = basepath + os.path.sep + 'config_DEF.cfg'
        konfig.dump(tempname)
        SU2_RUN = os.environ['SU2_RUN']
        base_Command = os.path.join(SU2_RUN,'%s')
        the_Command = 'SU2_DEF ' + tempname
        the_Command = base_Command % the_Command
        sys.stdout.flush()
        cfd_mesh_output = open(basepath + os.path.sep + 'mesh_deformation_direct.txt', 'w')
        proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout= cfd_mesh_output    ,
                         stderr= cfd_mesh_output,
                         stdin=subprocess.PIPE)
        #proc.stderr.close()
        proc.stdin.write(airfoilFile+'\n')
        proc.stdin.write('Selig\n')
        proc.stdin.write('1.0\n')
        proc.stdin.write('No\n')
        proc.stdin.write('clockwise\n')
        proc.stdin.close()
        return_code = proc.wait()
        return return_code


def cfdSolveBladeParallel(self, alphas, Res, afps, airfoil_analysis_options):
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

        airfoilFile = basepath + os.path.sep + 'airfoil'+str(i+1)+'_coordinates.dat'
        coord_file = open(airfoilFile, 'w')
        print >> coord_file, 'Airfoil Parallel'
        for j in range(len(self.x)):
            print >> coord_file, '{:<10f}\t{:<10f}'.format(self.x[j], self.y[j])
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
            dcd_dx1, xl, xu = self.su2Gradient(loop_sorted, konfig.SURFACE_ADJ_FILENAME + '.csv')
            dcd_dx.append(dcd_dx1)
            dcd_dalpha1 = ztate.HISTORY.ADJOINT_DRAG.Sens_AoA[-1]
            dcd_dalpha.append(dcd_dalpha1)
            afcoor = AirfoilAnalysis(afps[i], 'CST')
            dx_dafp = afcoor.getCoordinateDerivatives(xl, xu)
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


            dcl_dRe.append(0.0)
            dcd_dRe.append(0.0)
        return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp

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

