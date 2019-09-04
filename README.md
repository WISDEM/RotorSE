# DEPRECATED
------------

**THIS REPOSITORY IS DEPRECATED AND WILL BE ARCHIVED (READ-ONLY) IN NOVEMBER 2019.**

WISDEM has moved to a single, integrated repository at https://github.com/wisdem/wisdem

---------------
# RotorSE

RotorSE is a Systems engineering model for wind turbines rotors implemented as an OpenMDAO assembly.

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov) 

## Documentation

See local documentation in the `docs`-directory or access the online version at <http://wisdem.github.io/RotorSE/>

## Prerequisites

RotorSE requires C++ and Fortran compilers

## Installation

For detailed installation instructions of WISDEM modules see <https://github.com/WISDEM/WISDEM> or to install RotorSE by itself do:

    $ python setup.py install

## Run Unit Tests

To check if installation was successful try to import the package:

    $ python
    > import rotorse.rotor

You may also run the example

    $ python src/rotorse/rotor.py

For software issues please use <https://github.com/WISDEM/RotorSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).


