#!/usr/bin/env python
# encoding: utf-8
"""
test_environment_gradients.py

Created by Andrew Ning on 2013-12-18.
Copyright (c) NREL. All rights reserved.
"""


import unittest
import numpy as np
from commonse.utilities import check_gradient
from commonse.environment import PowerWind, LogWind, LinearWaves, TowerSoil


class TestPowerWind(unittest.TestCase):


    def test1(self):

        pw = PowerWind()
        pw.Uref = 10.0
        pw.zref = 100.0
        pw.z0 = 0.0
        pw.z = np.linspace(0.0, 100.0, 20)
        pw.shearExp = 0.2
        pw.betaWind = 0.0

        names, errors = check_gradient(pw)

        tol = 1e-6
        for name, err in zip(names, errors):

            if name == 'd_U[0] / d_z[0]':
                continue  # the derivative at z==0 is a discontinuity.  this node must not move in the optimization

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



    def test2(self):

        pw = PowerWind()
        pw.Uref = 10.0
        pw.zref = 100.0
        pw.z0 = 0.0
        pw.z = np.linspace(-10.0, 110.0, 20)
        pw.shearExp = 0.2
        pw.betaWind = 5.0


        names, errors = check_gradient(pw)

        tol = 1e-6
        for name, err in zip(names, errors):

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e


    def test3(self):

        pw = PowerWind()
        pw.Uref = 10.0
        pw.zref = 100.0
        pw.z0 = 0.0
        pw.z = np.linspace(-10.0, 90.0, 20)
        pw.shearExp = 0.2
        pw.betaWind = 5.0


        names, errors = check_gradient(pw)

        tol = 1e-6
        for name, err in zip(names, errors):

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e


class TestLogWind(unittest.TestCase):


    def test1(self):

        lw = LogWind()
        lw.Uref = 12.0
        lw.zref = 100.0
        lw.z0 = 5.0
        lw.z = np.linspace(4.9, 5.3, 20)
        lw.z_roughness = 10.0
        lw.betaWind = 5.0

        names, errors = check_gradient(lw)

        tol = 1e-6
        for name, err in zip(names, errors):

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



    def test2(self):

        lw = LogWind()
        lw.Uref = 12.0
        lw.zref = 100.0
        lw.z0 = 5.0
        lw.z = np.linspace(5.0, 100.0, 20)
        lw.z_roughness = 10.0
        lw.betaWind = 5.0

        names, errors = check_gradient(lw)

        tol = 1e-6
        for name, err in zip(names, errors):

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



class TestLinearWave(unittest.TestCase):


    def test1(self):

        lw = LinearWaves()
        lw.Uc = 7.0
        lw.z_surface = 20.0
        lw.hs = 10.0
        lw.T = 2.0
        lw.z_floor = 0.0
        lw.betaWave = 3.0
        lw.z = np.linspace(0.0, 20.0, 20)

        names, errors = check_gradient(lw)

        tol = 1e-4
        for name, err in zip(names, errors):

            if name in ('d_U[0] / d_z[0]', 'd_U[19] / d_z[19]', 'd_A[0] / d_z[0]', 'd_A[19] / d_z[19]'):
                continue  # the boundaries are not differentiable across bounds. these nodes must not move

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e


    def test2(self):

        lw = LinearWaves()
        lw.Uc = 5.0
        lw.z_surface = 20.0
        lw.hs = 2.0
        lw.T = 10.0
        lw.z_floor = 0.0
        lw.betaWave = 3.0
        lw.z = np.linspace(-5.0, 50.0, 20)

        names, errors = check_gradient(lw)

        tol = 1e-6
        for name, err in zip(names, errors):

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



class TestSoil(unittest.TestCase):


    def test1(self):

        soil = TowerSoil()
        soil.r0 = 10.0
        soil.depth = 30.0
        soil.G = 140e6
        soil.nu = 0.4
        soil.rigid = [False, False, False, False, False, False]

        names, errors = check_gradient(soil)

        tol = 1e-6
        for name, err in zip(names, errors):

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



if __name__ == '__main__':
    unittest.main()
