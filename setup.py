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
    packages=['rotorse','rotorse.test','rotorse.turbine_inputs', 'rotorse.geometry_tools'],
    package_data={'':['*.inp']},
    include_package_data=True,
    license='Apache License, Version 2.0',
    dependency_links=['https://github.com/WISDEM/CCBlade/tarball/master#egg=ccblade',
        'https://github.com/WISDEM/pBEAM/tarball/master#egg=pbeam',
        'https://github.com/WISDEM/CommonSE/tarball/master#egg=commonse'],
    zip_safe=False
)


from numpy.distutils.core import setup, Extension
setup(
    name='precomp',
    package_dir={'': 'src/rotorse'},
    ext_modules=[Extension('_precomp', ['src/rotorse/PreCompPy.f90'], extra_compile_args=['-O2','-fPIC','-shared'], extra_link_args=['-shared'])],
    # ext_modules=[Extension('_precomp', ['src/rotorse/PreCompPy.f90'], extra_compile_args=['-O2'])],
)
