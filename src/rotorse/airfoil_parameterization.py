import numpy as np
import cmath
import mpmath
from math import factorial, pi, cos, sin
#from naca_generator import naca4, naca5
import sys, os, csv
import subprocess
from copy import deepcopy
import pyXLIGHT
from scipy.interpolate import RectBivariateSpline, bisplev

class AirfoilAnalysis:
    """A helper class to store and evaluate airfoils"""
    def __init__(self, afp, airfoilOptions, numCoordinates=200, computeModel=True):
        self.afp = afp
        self.parameterization_method = airfoilOptions['AirfoilParameterization']
        self.analysis_method = airfoilOptions['AnalysisMethod']
        self.airfoilOptions = airfoilOptions
        self.numCoordinates = numCoordinates

        # Create folder for doing calculations
        analysisFolder = 'AirfoilAnalysisFiles'
        if not os.path.exists(analysisFolder):
            os.makedirs(analysisFolder)
        self.basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), analysisFolder)

        # Check if methods installed
        if self.parameterization_method == 'XFOIL':
            try:
                import pyXLIGHT
            except:
                raise ValueError('XFOIL not installed corrrectly.')
        elif self.parameterization_method == 'CFD':
            try:
                import SU2
            except:
                raise ValueError('SU2 CFD not installed corrrectly.')

        # Generate coordinates or precomputational model
        if self.parameterization_method != 'Precomputational':
            self.x, self.y, self.xl, self.xu, self.yl, self.yu, self.Wl, self.Wu = self.__setCoordinates()
            self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c = self.__setCoordinatesComplex()
        else:
            self.n, self.thick_max, self.thick_min = airfoilOptions['PrecomputationalOptions']['numAirfoilsToCompute'], airfoilOptions['PrecomputationalOptions']['tcMax'], airfoilOptions['PrecomputationalOptions']['tcMin']
            self.precomp_param = airfoilOptions['PrecomputationalOptions']['AirfoilParameterization']
            if computeModel:
                self.__generatePreCompModel()


    ### COORDINATE METHODS ###
    def getCoordinates(self, type='full'):
        if type == 'full':
            return self.x, self.y, self.Wl, self.Wu, self.xl, self.xu
        elif type == 'split':
            return self.xl, self.xu, self.yl, self.yu
        else:
            return self.x, self.y, self.xl, self.xu, self.yl, self.yu

    def getCoordinatesComplex(self, type='full'):
        if type == 'full':
            return self.x_c, self.y_c
        elif type == 'split':
            return self.xl_c, self.xu_c, self.yl_c, self.yu_c
        else:
            return self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c

    def getYCoordinatesGivenX(self, xl, xu):
        if self.parameterization_method != 'CST':
            raise ValueError('Not currently working for non-CST parameterization')
        else:
            yl, yu = self.__cstYgivenX(self.wl, self.wu, self.numCoordinates, self.dz, xl, xu)
        return yl, yu

    def getCoordinateDerivatives(self, xl=None, xu=None):
        if self.parameterization_method != 'CST':
            raise ValueError('Not currently working for non-CST parameterization')
        else:
            if xl is None or xu is None:
                dy_dafp = self.__cstYDerivatives(self.wl, self.wu, self.numCoordinates, self.dz, self.xl, self.xu)
            else:
                dy_dafp = self.__cstYDerivatives(self.wl, self.wu, self.numCoordinates, self.dz, xl, xu)
        return dy_dafp

    def saveCoordinateFile(self, airfoilFile):
        coord_file = open(airfoilFile, 'w')
        print >> coord_file, 'Airfoil'
        for i in range(len(self.x)):
            print >> coord_file, '{:<20f}\t{:<20f}'.format(self.x[i], self.y[i])
        coord_file.close()

    def saveCoordinateFileFromCoordinates(self, airfoilFile, x, y):
        coord_file = open(airfoilFile, 'w')
        print >> coord_file, 'Airfoil'
        for i in range(len(x)):
            print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])
        coord_file.close()

    def readCoordinateFile(self, airfoilFile):
        coord_file = open(airfoilFile, 'r')
        x, y = [], []
        for row in coord_file:
            try:
                row = row.split()
                x.append(float(row[0]))
                y.append(float(row[1]))
            except:
                pass
        coord_file.close()
        x = np.asarray(x)
        y = np.asarray(y)
        self.x = x
        self.y = y
        return x, y

    def getPreCompCoordinates(self, t_c, type='full'):
        self.xx0, self.yy0, self.thicknesses0 = self.__generatePreCompCoordinates(0)
        self.xx1, self.yy1, self.thicknesses1 = self.__generatePreCompCoordinates(1)
        tc = t_c[0]
        weight = t_c[1]

        for i in range(len(self.thicknesses0)):
            if tc >= self.thicknesses0[i] and tc < self.thicknesses0[i+1]:
                x0 = self.xx0[i]
                y0 = self.yy0[i]
        yy0 = self.__convertTCCoordinates(tc, y0)
        for i in range(len(self.thicknesses1)):
            if tc >= self.thicknesses1[i] and tc < self.thicknesses1[i+1]:
                x1 = self.xx1[i]
                y1 = self.yy1[i]
        yy1 = self.__convertTCCoordinates(tc, y1)

        xx = np.zeros(len(x1))
        yy = np.zeros(len(x1))
        if len(x1) == len(x0):
            for i in range(len(x0)):
                xx[i] = x1[i]
                yy[i] = yy0[i] + weight*(yy1[i] - yy0[i])
        else:
            print "Uneven blended airfoils"

        try:
            zerind = np.where(xx == 0)  # Used to separate upper and lower surfaces
            zerind = zerind[0][0]
        except:
            zerind = len(xx)/2

        xl = np.zeros(zerind)
        xu = np.zeros(len(xx)-zerind)
        yl = np.zeros(zerind)
        yu = np.zeros(len(xx)-zerind)

        for z in range(len(xl)):
            xu[z] = xx[z]        # Lower surface x-coordinates
            yu[z] = yy[z]
        for z in range(len(xu)):
            xl[z] = xx[z + zerind]   # Upper surface x-coordinates
            yl[z] = yy[z + zerind]

        # Get in ascending order if not already
        if xl[int(len(xl)/4)] > 0.5:
            xl = xl[::-1]
            yl = yl[::-1]
        if xu[int(len(xu)/4)] > 0.5:
            xu = xu[::-1]
            yu = yu[::-1]

        if xu[0] != 0.0:
            xu[0] = 0.0
        if xl[0] != 0.0:
            xl[0] = 0.0
        if xu[-1] != 1.0:
            xu[-1] = 1.0
        if xl[-1] != 1.0:
            xl[-1] = 1.0

        if yu[0] != 0.0:
            yu[0] = 0.0
        if yl[0] != 0.0:
            yl[0] = 0.0
        if yu[-1] != 0.0:
            yu[-1] = 0.0
        if yl[-1] != 0.0:
            yl[-1] = 0.0

        # Get in right order for precomp
        xl = xl[::-1]
        yl = yl[::-1]


        if type == 'full':
            return xl, xu, yl, yu
        else:
            return xx, yy

    def __setCoordinates(self):
        if self.parameterization_method == 'CST':
             x, y, xl, xu, yl, yu, Wl, Wu = self.__cstCoordinates()
        else:
            x, y, xl, xu, yl, yu = self.__tcCoordinates()
        return x, y, xl, xu, yl, yu, Wl, Wu

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
        W = np.zeros((len(x), 4))
        for i in range(len(x)):
            S[i] = 0
            for j in range(0, n+1):
                S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))
                W[i][j] = (w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))) * C[i]

        # Calculate y output
        y = np.zeros(len(x))
        for i in range(len(y)):
            y[i] = C[i] * S[i] + x[i] * dz

        return y, W

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

        yl, Wl = self.__ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu, Wu = self.__ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x))
        for k in range(len(x)):
            x1[k] = x[k][0]
        x = x1
        return x, y, xl, xu, yl, yu, Wl, Wu

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

    def setNumCoordinates(self, numCoordinates):
        self.numCoordinates = numCoordinates

    ### PRECOMPUTATIONAL MODEL METHODS ###
    def __generatePreCompModel(self):
        self.cl_total_spline0, self.cd_total_spline0, self.xx0, self.yy0, self.thicknesses0 = self.__generatePreCompSpline(0)
        self.cl_total_spline1, self.cd_total_spline1, self.xx1, self.yy1, self.thicknesses1 = self.__generatePreCompSpline(1)

    def __generatePreCompCoordinates(self, splineNum):
        n = self.n
        airfoilsSpecified = self.airfoilOptions['PrecomputationalOptions']['BaseAirfoilsCoordinates'+str(splineNum)]

        xs, ys, airfoil_thicknesses = [], [], []

        for i in range(len(airfoilsSpecified)):
            x, y = self.readCoordinateFile(airfoilsSpecified[i])
            airfoil_thickness = max(y) - min(y)
            xs.append(x), ys.append(y), airfoil_thicknesses.append(airfoil_thickness)
        self.airfoilsSpecified = deepcopy(airfoil_thicknesses)
        yx = zip(airfoil_thicknesses,xs)
        yx.sort()
        x_sorted = [x for y, x in yx]
        yx = zip(airfoil_thicknesses,ys)
        yx.sort()
        y_sorted = [x for y, x in yx]
        airfoil_thicknesses.sort()
        # Calculate thicknesses just past min and max because gradient at edges are zero
        thicknesses = [self.thick_min-1e-3] + airfoil_thicknesses + [self.thick_max+1e-3]
        yy = [self.__convertTCCoordinates(self.thick_min-1e-3, y_sorted[0])] + y_sorted + [self.__convertTCCoordinates(self.thick_max+1e-3, y_sorted[-1])]
        xx = [x_sorted[0]] + x_sorted + [x_sorted[-1]]
        airfoils_to_add = n - len(thicknesses)
        if airfoils_to_add > 0:
            to_insert = np.linspace(self.thick_min, self.thick_max, 2+airfoils_to_add)
            for j in range(len(to_insert)-2):
                alreadyFound = False
                for k in range(len(thicknesses)):
                    if to_insert[j+1] >= thicknesses[k] and to_insert[j+1] <= thicknesses[k+1] and not alreadyFound:
                        thicknesses.insert(k+1, to_insert[j+1])
                        yy.insert(k+1, self.__convertTCCoordinates(to_insert[j+1], yy[k]))
                        xx.insert(k+1, xx[k])
                        alreadyFound = True
        return xx, yy, thicknesses

    def __generatePreCompSpline(self, splineNum):

        xx, yy, thicknesses = self.__generatePreCompCoordinates(splineNum)
        cls, cds, cms, alphass, failures = [], [], [], [], []
        from airfoilprep import Airfoil, Polar
        from akima import Akima
        alphas_set = np.linspace(np.radians(-180), np.radians(180), 360)
        clGrid = np.zeros((len(alphas_set), len(thicknesses)))
        cdGrid = np.zeros((len(alphas_set), len(thicknesses)))
        if self.airfoilOptions['AnalysisMethod'] == 'Files' and self.n > len(self.airfoilsSpecified):
            computeCorrection = True
            compute = True
        elif self.airfoilOptions['AnalysisMethod'] != 'Files':
            computeCorrection = False
            compute = True
        else:
            computeCorrection = False
            compute = False

        if compute:
            for i in range(len(thicknesses)):
                self.x, self.y = xx[i], yy[i]
                cl, cd, cm, alphas, failure = self.__computeSplinePreComp()
                p1 = Polar(self.airfoilOptions['SplineOptions']['Re'], alphas, cl, cd, cm)
                af = Airfoil([p1])
                if self.airfoilOptions['SplineOptions']['correction3D']:
                    af = af.correction3D(self.airfoilOptions['SplineOptions']['r_over_R'], self.airfoilOptions['SplineOptions']['chord_over_r'], self.airfoilOptions['SplineOptions']['tsr'])
                af_extrap = af.extrapolate(self.airfoilOptions['SplineOptions']['cd_max'])
                alpha_ext, Re_ext, cl_ext, cd_ext, cm_ext = af_extrap.createDataGrid()
                if not all(np.diff(alpha_ext)):
                    to_delete = np.zeros(0)
                    diff = np.diff(alpha_ext)
                    for z in range(len(alpha_ext)-1):
                        if not diff[z] > 0.0:
                            to_delete = np.append(to_delete, z)
                    alpha_ext = np.delete(alpha_ext, to_delete)
                    cl_ext = np.delete(cl_ext, to_delete)
                    cd_ext = np.delete(cd_ext, to_delete)
                cls.append(cl_ext), cds.append(cd_ext), cms.append(cm_ext), alphass.append(alpha_ext), failures.append(failure)

        # Do XFOIL correction on file inputs if there is not enough data
        cl_correction = np.zeros(len(alphas_set))
        cd_correction = np.zeros(len(alphas_set))
        cls_files, cds_files, cms_files, alphass_files, failures_files = [], [], [], [], []
        if computeCorrection:
            for i in range(len(self.airfoilsSpecified)):
                aerodynFile = self.airfoilOptions['PrecomputationalOptions']['BaseAirfoilsData'+str(splineNum)][i]
                af = Airfoil.initFromAerodynFile(aerodynFile)
                alpha_ext, Re_ext, cl_ext, cd_ext, cm_ext = af.createDataGrid()
                failure = False
                index = thicknesses.index(self.airfoilsSpecified[i])
                if not all(np.diff(alpha_ext)):
                    to_delete = np.zeros(0)
                    diff = np.diff(alpha_ext)
                    for z in range(len(alpha_ext)-1):
                        if not diff[z] > 0.0:
                            to_delete = np.append(to_delete, z)
                    alpha_ext = np.delete(alpha_ext, to_delete)
                    cl_ext = np.delete(cl_ext, to_delete)
                    cd_ext = np.delete(cd_ext, to_delete)
                cls_files.append(cl_ext), cds_files.append(cd_ext), cms_files.append(cm_ext), alphass_files.append(alpha_ext), failures_files.append(failure)

                cl_spline_xfoil = Akima(np.radians(alphass[index]), cls[index], delta_x=0)
                cl_set_xfoil, _ = cl_spline_xfoil.interp(alphas_set)
                cd_spline_xfoil = Akima(np.radians(alphass[index]), cds[index], delta_x=0)
                cd_set_xfoil, _, = cd_spline_xfoil.interp(alphas_set)
                cl_spline_files = Akima(np.radians(alpha_ext), cl_ext, delta_x=0)
                cl_set_files, _, = cl_spline_files.interp(alphas_set)
                cd_spline_files = Akima(np.radians(alpha_ext), cd_ext, delta_x=0)
                cd_set_files, _, = cd_spline_files.interp(alphas_set)
                for k in range(len(alphas_set)):
                    cl_correction[k] += cl_set_files[k] - cl_set_xfoil[k]
                    cd_correction[k] += cd_set_files[k] - cd_set_xfoil[k]
            cl_correction /= float(len(self.airfoilsSpecified))
            cd_correction /= float(len(self.airfoilsSpecified))

        for i in range(len(thicknesses)):
            if not all(np.diff(alphass[i])):
                to_delete = np.zeros(0)
                diff = np.diff(alphass[i])
                for z in range(len(alphass[i])-1):
                    if not diff[z] > 0.0:
                        to_delete = np.append(to_delete, z)
                alphass[i] = np.delete(alphass[i], to_delete)
                cls[i] = np.delete(cls[i], to_delete)
                cds[i] = np.delete(cds[i], to_delete)
            cl_spline = Akima(np.radians(alphass[i]), cls[i])
            cd_spline = Akima(np.radians(alphass[i]), cds[i])

            cl_set, _, _, _ = cl_spline.interp(alphas_set)
            cd_set, _, _, _ = cd_spline.interp(alphas_set)
            if computeCorrection:
                for w in range(len(cl_set)):
                    cl_set[w] += cl_correction[w]
                    cd_set[w] += cd_correction[w]
            if thicknesses[i] in self.airfoilsSpecified and self.airfoilOptions['AnalysisMethod'] == 'Files':
                index = self.airfoilsSpecified.index(thicknesses[i])
                cl_spline = Akima(np.radians(alphass_files[index]), cls_files[index])
                cd_spline = Akima(np.radians(alphass_files[index]), cds_files[index])

                cl_set, _, _, _ = cl_spline.interp(alphas_set)
                cd_set, _, _, _ = cd_spline.interp(alphas_set)

            for j in range(len(alphas_set)):
                clGrid[j][i] = cl_set[j]
                cdGrid[j][i] = cd_set[j]
        kx = min(len(alphas_set)-1, 3)
        ky = min(len(thicknesses)-1, 3)
        cl_total_spline = RectBivariateSpline(alphas_set, thicknesses, clGrid, kx=kx, ky=ky, s=0.001)
        cd_total_spline = RectBivariateSpline(alphas_set, thicknesses, cdGrid, kx=kx, ky=ky, s=0.0005)
        return cl_total_spline, cd_total_spline, xx, yy, thicknesses

    def __convertTCCoordinates(self, tc, y):
        yy = np.zeros(len(y))
        base_tc = max(y) - min(y)
        for i in range(len(y)):
            yy[i] = y[i] * tc / base_tc
        return yy

    def evaluatePreCompModel(self, alpha, afp):
        tc = afp[0]
        bf = afp[1]

        cl0 = self.cl_total_spline0.ev(alpha, tc)
        cd0 = self.cd_total_spline0.ev(alpha, tc)
        cl1 = self.cl_total_spline1.ev(alpha, tc)
        cd1 = self.cd_total_spline1.ev(alpha, tc)
        cl = cl0 + bf*(cl1-cl0)
        cd = cd0 + bf*(cd1-cd0)
        self.cl1, self.cl0, self.cd1, self.cd0 = cl1, cl0, cd1, cd0

        return cl, cd

    def derivativesPreCompModel(self, alpha, afp):

        tc = afp[0]
        bf = afp[1]

        # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
        tck_cl0 = self.cl_total_spline0.tck[:3] + self.cl_total_spline0.degrees  # concatenate lists
        tck_cd0 = self.cd_total_spline0.tck[:3] + self.cd_total_spline0.degrees

        dcl_dalpha0 = bisplev(alpha, tc, tck_cl0, dx=1, dy=0)
        dcd_dalpha0 = bisplev(alpha, tc, tck_cd0, dx=1, dy=0)

        dcl_dt_c0 = bisplev(alpha, tc, tck_cl0, dx=0, dy=1)
        dcd_dt_c0 = bisplev(alpha, tc, tck_cd0, dx=0, dy=1)

        tck_cl1 = self.cl_total_spline1.tck[:3] + self.cl_total_spline1.degrees  # concatenate lists
        tck_cd1 = self.cd_total_spline1.tck[:3] + self.cd_total_spline1.degrees

        dcl_dalpha1 = bisplev(alpha, tc, tck_cl1, dx=1, dy=0)
        dcd_dalpha1 = bisplev(alpha, tc, tck_cd1, dx=1, dy=0)

        dcl_dt_c1 = bisplev(alpha, tc, tck_cl1, dx=0, dy=1)
        dcd_dt_c1 = bisplev(alpha, tc, tck_cd1, dx=0, dy=1)

        dcl_dalpha = dcl_dalpha0 + bf*(dcl_dalpha1-dcl_dalpha0)
        dcd_dalpha = dcd_dalpha0 + bf*(dcd_dalpha1-dcd_dalpha0)

        dcl_dtc = dcl_dt_c0 + bf*(dcl_dt_c1-dcl_dt_c0)
        dcd_dtc = dcd_dt_c0 + bf*(dcd_dt_c1-dcd_dt_c0)

        dcl_dweight = self.cl1-self.cl0
        dcd_dweight = self.cd1-self.cd0
        if self.airfoilOptions['PrecomputationalOptions']['AirfoilParameterization'] == 'Blended':
            dcl_dafp = np.asarray([dcl_dtc, dcl_dweight])
            dcd_dafp = np.asarray([dcd_dtc, dcd_dweight])
        else:
            dcl_dafp = np.asarray([dcl_dtc])
            dcd_dafp = np.asarray([dcd_dtc])

        return dcl_dalpha, dcl_dafp, dcd_dalpha, dcd_dafp

    def plotPreCompModel(self):
        import matplotlib.pylab as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        n = 200
        thick = np.linspace(self.thick_min, self.thick_max, n)
        alpha = np.linspace(-np.pi, np.pi, n)
        CL = np.zeros((n, n))
        CD = np.zeros((n, n))
        [X, Y] = np.meshgrid(alpha, thick)
        for i in range(n):
            for j in range(n):
                CL[i, j] = self.cl_total_spline.ev(X[i, j], Y[i, j])
                CD[i, j] = self.cd_total_spline.ev(X[i, j], Y[i, j])

        font_size = 14
        fig4 = plt.figure()
        ax4 = fig4.gca(projection='3d')
        surf = ax4.plot_surface(np.degrees(X), Y, CD, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        plt.xlim(xmin=-180, xmax=180)
        plt.xticks(np.arange(-180, 180+1, 60.0))
        plt.yticks(np.arange(0.15, 0.46, 0.10))
        ax4.set_zlabel(r'$c_d$')
        # plt.title('C_D Surface')
        plt.xlabel(r'$\alpha$ (deg)')
        plt.ylabel('t/c (\%)')
        fig4.colorbar(surf) #, shrink=0.5, aspect=5)
        # plt.rcParams['font.size'] = font_size
        # plt.savefig('cd_fin_surface.pdf')
        # plt.savefig('cd_fin_surface.png')

        fig5 = plt.figure()
        ax5 = fig5.gca(projection='3d')
        surf2 = ax5.plot_surface(np.degrees(X), Y, CL, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        fig5.colorbar(surf2) #, shrink=0.5, aspect=5)
        plt.xlim(xmin=-180, xmax=180)
        plt.xticks(np.arange(-180, 180+1, 60.0))
        plt.yticks(np.arange(0.15, 0.46, 0.10))
        # plt.title('C_L Surface')
        ax5.set_zlabel(r'$c_l$')
        plt.xlabel(r'$\alpha$ (deg)')
        plt.ylabel('t/c (\%)')
        # plt.rcParams['font.size'] = font_size
        plt.savefig('/Users/ryanbarr/Desktop/cl_fin_surface.pdf')
        plt.savefig('/Users/ryanbarr/Desktop/cl_fin_surface.png')
        # ax4.zaxis.set_major_locator(LinearLocator(10))
        # ax4.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.show()

    def computeDirect(self, alpha, Re):
        if self.analysis_method == 'CFD':
            cl, cd, dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp, lexitflag = self.__cfdDirect(alpha, Re, GenerateMESH=True)
        else:
            cl, cd, dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp, lexitflag = self.__xfoilDirect(alpha, Re)
        dcl_dRe, dcd_dRe = 0.0, 0.0

        if self.airfoilOptions['GradientOptions']['ComputeGradient']:
            return cl, cd, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dafp, dcd_dafp, lexitflag
        else:
            return cl, cd

    def computeSpline(self):
        if self.airfoilOptions['SplineOptions']['AnalysisMethod'] == 'CFD':
            cl, cd, cm, alphas, failure  = self.__cfdSpline()
        else:
            cl, cd, cm, alphas, failure  = self.__xfoilSpline()
        return cl, cd, cm, alphas, failure

    def __computeSplinePreComp(self):
        if self.analysis_method == 'CFD':
            cl, cd, cm, alphas, failure = self.__cfdSpline()
        else:
            cl, cd, cm, alphas, failure = self.__xfoilSpline()
        return cl, cd, cm, alphas, failure

    def __xfoilSpline(self):
        airfoilShapeFile = self.basepath + os.sep + 'airfoil_shape.dat'
        self.saveCoordinateFile(airfoilShapeFile)
        alphas, Re = self.airfoilOptions['SplineOptions']['alphas'], self.airfoilOptions['SplineOptions']['Re']
        airfoil = pyXLIGHT.xfoilAnalysis(airfoilShapeFile, x=self.x, y=self.y)
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



    def __cfdSpline(self):
        alphas = self.airfoilOptions['SplineOptions']['alphas']
        Re = self.airfoilOptions['SplineOptions']['Re']
        cl, cd, cm, failure = np.zeros(len(alphas)), np.zeros(len(alphas)), np.zeros(len(alphas)), False
        if self.airfoilOptions['CFDOptions']['computeAirfoilsInParallel']:
            cl, cd = self.__cfdParallelSpline(np.radians(alphas), Re, self.airfoilOptions)
        else:
            for j in range(len(alphas)):
                if j == 0:
                    mesh = True
                else:
                    mesh = False
                cl[j], cd[j] = self.__cfdDirect(np.radians(alphas[j]), Re, GenerateMESH=mesh)
        return cl, cd, cm, alphas, failure

    def __xfoilDirect(self, alpha, Re):
        airfoil_shape_file = None
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=self.x, y=self.y)
        airfoil.re = self.airfoilOptions['SplineOptions']['Re']
        airfoil.mach = 0.00
        airfoil.iter = 100
        dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp = 0.0, 0.0, np.zeros(8), np.zeros(8)
        if self.airfoilOptions['GradientOptions']['ComputeGradient']:
            cs_step = 1e-20
            angle = complex(np.degrees(alpha), cs_step)
            cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
            dcl_dalpha, dcd_dalpha = 180.0/np.pi*np.imag(deepcopy(cl))/ cs_step, 180.0/np.pi*np.imag(deepcopy(cd)) / cs_step
            cl, cd = np.real(np.asscalar(cl)), np.real(np.asscalar(cd))
            if abs(dcl_dalpha) > 10.0 or abs(dcd_dalpha) > 10.0:
                fd_step = 1.e-6
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
            if self.airfoilOptions['GradientOptions']['ComputeAirfoilGradients']:
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
                fd_step = 1.e-6 #self.airfoilOptions['GradientOptions']['fd_step']
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
                from airfoilprep import Airfoil
                af1 = Airfoil.initFromCST(self.afp, self.airfoilOptions)
                af_extrap11 = af1.extrapolate(1.5)
                alphas_cur, Re_cur, cl_cur, cd_cur, cm_cur = af_extrap11.createDataGrid()
                cl_spline_cur = Akima(alphas_cur, cl_cur)
                cd_spline_cur = Akima(alphas_cur, cd_cur)
                cl_fd_cur, _, _, _ = cl_spline_cur.interp(alpha)
                cd_fd_cur, _, _, _ = cd_spline_cur.interp(alpha)
                afp_new = deepcopy(self.afp)
                afp_new[i] += fd_step
                af = Airfoil.initFromCST(afp_new, self.airfoilOptions)
                af_extrap1 = af.extrapolate(1.5)
                alphas_new, Re_new, cl_new, cd_new, cm_new = af_extrap1.createDataGrid()
                cl_spline = Akima(alphas_new, cl_new)
                cd_spline = Akima(alphas_new, cd_new)
                cl_fd_new, _, _, _ = cl_spline.interp(alpha)
                cd_fd_new, _, _, _ = cd_spline.interp(alpha)
                dcl_dafp[i] = (cl_fd_new - cl_fd_cur) / fd_step
                dcd_dafp[i] = (cd_fd_new - cd_fd_cur) / fd_step
        return dcl_dafp, dcd_dafp

    def xfoilSolveComplex(self, alpha, wl_complex, wu_complex, N, dz):
        airfoil_shape_file = None

        x, y = self.__cstCoordinatesComplex2(wl_complex, wu_complex, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.airfoilOptions['SplineOptions']['Re']
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl_complex, cd_complex, cm_complex, lexitflag = airfoil.solveAlphaComplex(alpha)
        return deepcopy(cl_complex), deepcopy(cd_complex), deepcopy(lexitflag)

    def xfoilSolveReal(self, alpha, wl, wu, N, dz):
        airfoil_shape_file = None
        x, y = self.__cstCoordinatesReal(wl, wu, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.airfoilOptions['SplineOptions']['Re']
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl, cd, cm, lexitflag = airfoil.solveAlpha(alpha)
        return np.asscalar(cl), np.asscalar(cd), deepcopy(lexitflag)

    def __cfdDirect(self, alpha, Re, GenerateMESH=True, airfoilNum=0):
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2
        airfoilOptions = self.airfoilOptions
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')
        config_filename = basepath + os.path.sep + airfoilOptions['CFDOptions']['configFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = airfoilOptions['CFDOptions']['processors']
        config.EXT_ITER    = airfoilOptions['CFDOptions']['iterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = basepath + os.path.sep + 'mesh_AIRFOIL_serial.su2'
        config.MESH_FILENAME = basepath + os.path.sep + config.MESH_FILENAME

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
        config.MACH_NUMBER = 0.2 #TODO Change back
        config.REYNOLDS_NUMBER = airfoilOptions['SplineOptions']['Re']

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
        print the_Command
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

        if self.airfoilOptions['GradientOptions']['ComputeGradient']:
            konfig = deepcopy(config)
            ztate = deepcopy(state)

            konfig.RESTART_SOL = 'NO'
            mesh_data = SU2.mesh.tools.read(konfig.MESH_FILENAME)
            points_sorted, loop_sorted = SU2.mesh.tools.sort_airfoil(mesh_data, marker_name='airfoil')

            SU2.io.restart2solution(konfig, ztate)
            # RUN FOR DRAG GRADIENTS
            konfig.OBJECTIVE_FUNCTION = 'DRAG'

            # setup problem
            konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
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
            print the_Command
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
            dcd_dx, xl, xu = self.su2Gradient(loop_sorted, surface_adjoint)
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
            konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
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
            dcl_dx, xl, xu = self.su2Gradient(loop_sorted, surface_adjoint)
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

    def __cfdParallelSpline(self, alphas, Re, airfoilOptions):
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')
        config_filename = basepath + os.path.sep + airfoilOptions['CFDOptions']['configFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = int(airfoilOptions['CFDOptions']['processors']/ float(len(alphas)))
        remainder =  airfoilOptions['CFDOptions']['processors'] % len(alphas) - 1
        if remainder <= 0 or config.NUMBER_PART == 0:
            remainder = 0
        config.MESH_FILENAME = basepath + os.path.sep + config.MESH_FILENAME
        config.EXT_ITER    = airfoilOptions['CFDOptions']['iterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = basepath + os.path.sep + 'mesh_AIRFOIL_spline_parallel.su2'
        config.CONSOLE = 'QUIET'

        return_code = self.__generateMesh(meshFileName, config, state, basepath)
        if return_code != 0:
            print "Error in mesh deformation."

        config.MESH_FILENAME = meshFileName
        state.FILES.MESH = config.MESH_FILENAME
        Uinf = 10.0
        Ma = Uinf / 340.29  # Speed of sound at sea level
        config.MACH_NUMBER = 0.2 #Ma
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
            if i <= remainder:
                processes = konfig['NUMBER_PART'] + 1
            else:
                processes = konfig['NUMBER_PART']
            the_Command = 'SU2_CFD ' + tempname
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = base_Command % the_Command
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError , 'could not find an mpi interface'
            the_Command = mpi_Command % (processes,the_Command)
            print the_Command
            sys.stdout.flush()
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=open(basepath + os.path.sep + 'cfd_spline'+str(i+1)+'.txt', 'w'),
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
        airfoilFile = basepath + os.path.sep + 'airfoil_shape.dat'
        self.saveCoordinateFile(airfoilFile)
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


    def cfdSolveBladeParallel(self, alphas, Res, afps, airfoilOptions):
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')

        config_filename = basepath + os.path.sep + airfoilOptions['CFDOptions']['configFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = airfoilOptions['CFDOptions']['processors']
        config.EXT_ITER    = airfoilOptions['CFDOptions']['iterations']
        config.WRT_CSV_SOL = 'YES'
        config.CONSOLE = 'QUIET'

        cl = np.zeros(len(alphas))
        cd = np.zeros(len(alphas))
        alphas = np.degrees(alphas)
        procTotal = []
        konfigTotal = []
        konfigDirectTotal = []
        ztateTotal = []
        Re = airfoilOptions['SplineOptions']['Re']
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
            konfig.MACH_NUMBER = 0.2#  Ma
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

        if airfoilOptions['GradientOptions']['ComputeGradient']:
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
                konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
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
                konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
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
                dcl_dx1, xl, xu = self.su2Gradient(loop_sorted, surface_adjoint)
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


    def su2Gradient(self, loop_sorted, surface_adjoint):
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

    def cfdDirectSolveParallel(self, alphas, Re, afp, airfoilOptions):
            # Import SU2
            sys.path.append(os.environ['SU2_RUN'])
            import SU2

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')

            config_filename = basepath + os.path.sep + airfoilOptions['CFDOptions']['configFile']
            config = SU2.io.Config(config_filename)
            state  = SU2.io.State()
            config.NUMBER_PART = airfoilOptions['CFDOptions']['processors']
            config.EXT_ITER    = airfoilOptions['CFDOptions']['iterations']
            config.WRT_CSV_SOL = 'YES'
            meshFileName = basepath + os.path.sep + 'mesh_AIRFOIL_parallel.su2'
            config.CONSOLE = 'QUIET'

            # Create airfoil coordinate file for SU2
            afanalysis = AirfoilAnalysis(afp, airfoilOptions)
            x, y = afanalysis.x, afanalysis.y
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
            config.MACH_NUMBER = 0.2 #Ma
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

                if i >= len(alphas) - remainder:
                    processes = konfig['NUMBER_PART'] + 1
                else:
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

    def cfdAirfoilsSolveParallel(self, alphas, Res, afps, airfoilOptions):
            # Import SU2
            sys.path.append(os.environ['SU2_RUN'])
            import SU2

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')

            config_filename = basepath + os.path.sep + airfoilOptions['CFDOptions']['configFile']
            config = SU2.io.Config(config_filename)
            state  = SU2.io.State()
            config.NUMBER_PART = int(airfoilOptions['CFDOptions']['processors'] / float(len(alphas)))
            remainder = airfoilOptions['CFDOptions']['processors'] % len(alphas)
            config.EXT_ITER    = airfoilOptions['CFDOptions']['iterations']
            config.WRT_CSV_SOL = 'YES'

            config.CONSOLE = 'QUIET'

            cl = np.zeros(len(alphas))
            cd = np.zeros(len(alphas))
            alphas = np.degrees(alphas)
            procTotal = []
            konfigTotal = []
            konfigDirectTotal = []
            ztateTotal = []
            Re = airfoilOptions['SplineOptions']['Re']
            for i in range(len(alphas)):
                meshFileName = basepath + os.path.sep + 'mesh_airfoil'+str(i+1)+'.su2'
                # Create airfoil coordinate file for SU2
                afanalysis = AirfoilAnalysis(afps[i], airfoilOptions)
                x, y = afanalysis.x, afanalysis.y
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
                konfig.MACH_NUMBER = 0.2 # Ma
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
                if i >= len(alphas) - remainder:
                    processes = konfig['NUMBER_PART'] + 1
                else:
                    processes = konfig['NUMBER_PART']

                the_Command = 'SU2_CFD ' + tempname
                the_Command = base_Command % the_Command
                if processes > 0:
                    if not mpi_Command:
                        raise RuntimeError , 'could not find an mpi interface'
                the_Command = mpi_Command % (processes,the_Command)
                sys.stdout.flush()
                cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil'+str(i+1)+'.txt', 'w')
                print the_Command
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

            if airfoilOptions['GradientOptions']['ComputeGradient']:
                print "computeGradients", airfoilOptions['GradientOptions']['ComputeGradient']
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
                    konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
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
                    print the_Command
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
                    n = 8
                    m = 200
                    afanalysis = AirfoilAnalysis(afps[i], airfoilOptions)
                    dy_dafp = afanalysis.getCoordinateDerivatives(xl, xu)
                    dx_dafpTotal.append(dy_dafp)
                procTotal = []
                for i in range(len(alphas)):
                    konfig = deepcopy(konfigTotal[i])
                    ztate = ztateTotal[i]

                    konfig.OBJECTIVE_FUNCTION = 'LIFT'

                    # setup problem
                    konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
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
                    print the_Command
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
                    dcl_dx1, xl, xu = self.su2Gradient(loop_sorted, surface_adjoint)
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

    def cfdDirectSolveParallel(self, alphas, Re, afp, airfoilOptions):
            # Import SU2
            sys.path.append(os.environ['SU2_RUN'])
            import SU2

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')

            config_filename = basepath + os.path.sep + airfoilOptions['cfdConfigFile']
            config = SU2.io.Config(config_filename)
            state  = SU2.io.State()
            config.NUMBER_PART = airfoilOptions['CFDprocessors']
            config.EXT_ITER    = airfoilOptions['CFDiterations']
            config.WRT_CSV_SOL = 'YES'
            meshFileName = basepath + os.path.sep + 'mesh_AIRFOIL_parallel.su2'
            config.CONSOLE = 'QUIET'

            # Create airfoil coordinate file for SU2
            afanalysis = AirfoilAnalysis(afp, airfoilOptions)
            x, y = afanalysis.x, afanalysis.y
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
            config.MACH_NUMBER = 0.2 #Ma
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

                if i >= len(alphas) - remainder:
                    processes = konfig['NUMBER_PART'] + 1
                else:
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


    from scipy.interpolate import RectBivariateSpline, bisplev
    def evaluate_direct_parallel(self, alphas, Res, afs, computeAlphaGradient=False, computeAFPGradient=False):
            indices_to_compute = []
            n = len(alphas)
            airfoilOptions = afs[-1].airfoilOptions
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
                if af.afp is not None and abs(np.degrees(alpha)) < af.airfoilOptions['SplineOptions']['maxDirectAoA']:
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
            if indices_to_compute:
                alphas_to_compute = [alphas[i] for i in indices_to_compute]
                Res_to_compute = [Res[i] for i in indices_to_compute]
                afps_to_compute = [afs[i].afp for i in indices_to_compute]
                if airfoilOptions['GradientOptions']['ComputeGradient']:
                    cls, cds, dcls_dalpha, dcls_dRe, dcds_dalpha, dcds_dRe, dcls_dafp, dcds_dafp = self.cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, afps_to_compute, airfoilOptions)
                    for j in range(len(indices_to_compute)):
                        dcl_dalpha[indices_to_compute[j]] = dcls_dalpha[j]
                        dcl_dRe[indices_to_compute[j]] = dcls_dRe[j]
                        dcd_dalpha[indices_to_compute[j]] = dcds_dalpha[j]
                        dcd_dRe[indices_to_compute[j]] = dcls_dRe[j]
                        dcl_dafp[indices_to_compute[j]] = dcls_dafp[j]
                        dcd_dafp[indices_to_compute[j]] = dcds_dafp[j]
                else:
                    cls, cds = self.cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, afps_to_compute, airfoilOptions)
                for j in range(len(indices_to_compute)):
                    cl[indices_to_compute[j]] = cls[j]
                    cd[indices_to_compute[j]] = cds[j]

            print cl, cd, dcl_dalpha, dcd_dalpha, dcl_dafp, dcd_dafp
            if computeAFPGradient:
                try:
                    return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dafp, dcd_dafp
                except:
                    raise
            elif computeAlphaGradient:
                return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe
            else:
                return cl, cd
