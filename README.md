RotorSE is a Systems engineering model for wind turbines rotors implemented as an OpenMDAO assembly.

Author: [S. Andrew Ning](mailto:nrel.wisdem+rotorse@gmail.com)

## Version

This software is a beta version 0.1.1.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/RotorSE/>

## Prerequisites

General: C compiler, Fortran compiler, NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

## Dependencies

Wind Plant Framework: [FUSED-Wind](http://fusedwind.org) (Framework for Unified Systems Engineering and Design of Wind Plants)

Sub-Models: CommonSE, CCBlade, Akima

Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

## Installation (Windows)

First, clone the [repository](https://github.com/WISDEM/RotorSE)
or download the releases and uncompress/unpack (RotorSE.py-|release|.tar.gz or RotorSE.py-|release|.zip) from the website link at the bottom the [RotorSE site](http://nwtc.nrel.gov/RotorSE).

These instructions assume you are using MinGW and have already installed gcc and g++.
Also you should already have successfully installed Python (for a [single user only](http://bugs.python.org/issue5459#msg101098)), NumPy, and setuptools.
The example directories may need to be modified depending on where you installed things.  See this [Windows guideline set](https://nwtc.nrel.gov/system/files/Windows%20OpenMDAO%20Install%20Tips.pdf) for additional support on installing python.

1) Edit (or create) a distutils config 'distutils.cfg' file in your Python Lib directory or in your openmdao Lib directory if working from an activated openmdao environment.

    C:\Python27\Lib\distutils\distutils.cfg

or

    "Path to openmdao"\Lib\distutils.cfg

and put the following in it:

    [build]
    compiler=mingw32


2) Download [Boost](http://www.boost.org) (v 1.55 as of this writing) and setup bjam

At the command prompt:

    > cd boost_1_55_0\tools\build\v2\engine
    > build.bat mingw

This should create a folder called: bin.ntx86.  For convenience in the next step you can add this folder to your PATH so bjam is accessible.  Otherwise, use the whole path when calling bjam.

    C:\boost_1_55_0\tools\build\v2\engine\bin.ntx86

3) Compile Boost.Python

In the boost root directory (must be in the root directory) type the following at the command prompt:

    > bjam toolset=gcc --with-python link=shared

the libraries should be built in stage/lib and will be needed in steps 5 and 6.


4) Install LAPACK and BLAS.  I just used [prebuilt libraries](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries).  Make sure to grab all three libraries - BLAS, LAPACK and LAPACKE and make sure they are the 32-bit versions.
Remember the location for steps 5 and 6.

5) Make sure the following are on your system PATH.  The dynamic libraries are needed in order to actually run pBEAM.

    C:\Python27  (for Python)
    C:\Python27\Scripts  (for easy_install)
    C:\MinGW\bin  (for g++, gcc, etc.)
    C:\lapack  (LAPACK dynamic libraries)
    C:\boost_1_55_0\stage\lib  (Boost Python dynamic libraries)

For the remainder of the setup, use the below directions for *nix systems.  If you have issues with installation of pBEAM and RotorSE, then do this additional step:

6) Modify the 'setup.py' script in RotorSE's main directory.  Unlike GCC on *nix systems, Windows does not have typical locations to store headers and libraries (e.g., /usr/local/include) and so you will need manually specify them.  Add the header locations for Boost in the include_dirs.  Add the library locations for Boost and LAPACK.  You may also need to rename the boost_python library.  Use the example below, modifying as needed based on where you installed things.  Note that setup.py expects unix style slashes (forward), and that you do not need to include 'lib' at the front of the library names (i.e., 'lapack' corresponds to 'liblapack.dll' or 'liblapack.a').  Note: make sure your boost version matches the boost version installed (i.e. mgw48, mgw46, etc).

    include_dirs=[join(path, 'pBEAM'), 'C:/boost_1_55_0'],
    library_dirs=['C:/boost_1_55_0/stage/lib', 'C:/lapack'],
    libraries=['boost_python-mgw48-mt-1_55', 'lapack']


## Installation (OS X, Linux)

Install pBEAM with the following command:

    $ python setup.py install

or from within an activated OpenMDAO environment:

	$ plugin install


## Run Unit Tests

To check if installation was successful try to import the module from within an activated OpenMDAO environment:

    $ python
    > import rotorse.rotor

You may also run the unit tests.

    $ python src/rotorse/test/test_rotoraero_gradients.py
    $ python src/rotorse/test/test_rotor_gradients.py

For software issues please use <https://github.com/WISDEM/RotorSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).


