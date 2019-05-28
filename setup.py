#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_packages
import platform


setup(
    name='RotorSE',
    version='0.1.1',
    description='Rotor Systems Engineering Model',
    author='NREL WISDEM Team',
    author_email='systems.engineering@nrel.gov',
    install_requires=['commonse', 'ccblade', 'pbeam'],
    package_dir={'': 'src'},
    packages=['rotorse','rotorse.test','rotorse.5MW_AFFiles','rotorse.5MW_PreCompFiles','rotorse.10MW_AFFiles','rotorse.10MW_PreCompFiles','rotorse.3_35MW_AFFiles','rotorse.3_35MW_PreCompFiles','rotorse.BAR_00_AFFiles','rotorse.BAR_00_PreCompFiles'],
    license='Apache License, Version 2.0',
    dependency_links=['https://github.com/WISDEM/CCBlade/tarball/master#egg=ccblade',
        'https://github.com/WISDEM/pBEAM/tarball/master#egg=pbeam',
        'https://github.com/WISDEM/CommonSE/tarball/master#egg=commonse'],
    zip_safe=False,
    package_data={'':['*.inp']},
    include_package_data = True
)


from numpy.distutils.core import setup, Extension
import os

if platform.system() == 'Windows':
    # Note: must use mingw compiler on windows or a Visual C++ compiler version that supports std=c++11
    arglist = ['-O3','-std=gnu++11','-fPIC']
else:
    arglist = ['-O3','-std=c++11','-fPIC']

setup(
    name='precomp',
    package_dir={'': 'src/rotorse'},
    ext_modules=[Extension('_precomp', sources=[os.path.join('src','rotorse','PreCompPy.cpp')], extra_compile_args=arglist, include_dirs=[os.path.join('src','include')])]
)
