RotorSE is a Systems engineering model for wind turbines rotors implemented as an OpenMDAO assembly.

Author: [S. Andrew Ning](mailto:andrew.ning@nrel.gov)

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/RotorSE/>

## Prerequisites

Fortran compiler, C compiler, NumPy, SciPy, OpenMDAO, CommonSE

## Installation

Install RotorSE with the following command.

    $ python setup.py install

## Run Unit Tests

To check if installation was successful try to import the module

    $ python
    > import rotorse.rotor

You may also run the unit tests.

    $ python src/rotorse/test/test_rotoraero_gradients.py
    $ python src/rotorse/test/test_rotor_gradients.py

For software issues please use <https://github.com/WISDEM/RotorSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).


