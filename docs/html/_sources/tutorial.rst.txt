.. _tutorial-label:

.. currentmodule:: rotorse.rotoraero

Tutorial
--------

The module :mod:`rotorse.rotor_aeropower` contains methods for generating power curves and computing annual energy production (AEP) with any aerodynamic tool, any wind speed distribution (implementing :mod:`PDFBase` and :mod:`PDFBase`), any drivetrain efficiency function (implementing :mod:`DrivetrainLossesBase`), and any machine type amongst the four combinations of variable/fixed speed and variable/fixed pitch.

The module :mod:`rotorse.rotor_geometry` provides specific implementations of reference rotor designs (implementing :mod:`ReferenceBlade`).  `CCBlade <https://github.com/WISDEM/CCBlade>`_ is used for the aerodynamic analysis and `CommonSE <https://github.com/WISDEM/CommonSE>`_ provides Weibull and Rayleigh wind speed distribution.

The module :mod:`rotorse.rotor_structure` provides structural analyses including methods for managing the composite secion analysis, computing deflections, computing mass properties, etc.  The module :mod:`rotorse.rotor` provides the coupling between rotor_aeropower and rotor_sturucture for combined analysis.  

Two examples are included in this tutorial section: aerodynamic simulation and optimization of a rotor and aero/structural analysis of a rotor.



Rotor Aerodynamics
==================

.. currentmodule:: rotorse.rotor_aeropower

This example is available _____ or can be viewed as an interactive Jupyter notebook _____.  The first step is to import the relevant files.

.. literalinclude:: /examples/rotorse_example1.py
    :language: python
    :start-after: # --- Import Modules
    :end-before: # ---

When setting up our Problem, a rotor design that is an implimentation of :mod:`ReferenceBlade`, is used to initialize the Group.  Two reference turbine designs are included as examples in :mod:`rotorse.rotor_geometry`, the :mod:`NREL5MW` and the :mod:`DTU10MW`.  For this tutorial, we will be working with the DTU 10 MW.

.. literalinclude:: /examples/rotorse_example1.py
    :language: python
    :start-after: # --- Init Problem
    :end-before: # ---

A number of input variablers covering the blade geometry, atmospheric conditions, and controls system must be set by the use.  While the reference blade design provides this information, it must be again set at the Problem level.  This provides flexibility for modifications by the user or by an optimizer (discussed more in _______).  The user can choose to use the default values

.. literalinclude:: /examples/rotorse_example1.py
    :language: python
    :start-after: # --- default inputs
    :end-before: # ---

Or set their own values.  First, the geometry is defined.  Spanwise blade variables such as chord and twist are definied using control points, which :class:`BladeGeometry` uses to generate the spanwise distribution using Akima splines according to :num:`Figures #chord-param-fig` and :num:`#twist-param-fig`.

.. literalinclude:: /examples/rotorse_example1_b.py
    :language: python
    :start-after: # === blade grid ===
    :end-before: # ---

.. _chord-param-fig:

.. figure:: /images/chord_dtu10mw.*
    :height: 4in
    :align: left

    Chord parameterization

.. _twist-param-fig:

.. figure:: /images/theta_dtu10mw.*
    :height: 4in
    :align: center

    Twist parameterization

Atmospheric properties are defined.  The wind speed distribution parameters are determined based on the wind turbine class.

.. literalinclude:: /examples/rotorse_example1_b.py
    :language: python
    :start-after: # === atmosphere ===
    :end-before: # ---

The relevant control parameters are set

.. literalinclude:: /examples/rotorse_example1_b.py
    :language: python
    :start-after: # === control ===
    :end-before: # ---

Finally, a few configuation parameters are set.  The the following drivetrain types are supported: 'geared', 'single_stage', 'multi_drive', or 'pm_direct_drive'.

.. literalinclude:: /examples/rotorse_example1_b.py
    :language: python
    :start-after: # === aero and structural analysis options ===
    :end-before: # ---

We can now run the analysis, print the outputs, and plot the power curve.

.. literalinclude:: /examples/rotorse_example1.py
    :language: python
    :start-after: # === run and outputs ===
    :end-before: # ---


>>> AEP = 46811339.16312428
>>> diameter = 197.51768195144518
>>> ratedConditions.V = 11.674033110109226
>>> ratedConditions.Omega = 8.887659696962098
>>> ratedConditions.pitch = 0.0
>>> ratedConditions.T = 1514792.8710181064
>>> ratedConditions.Q = 10744444.444444444

.. figure:: /images/power_curve_dtu10mw.*
    :height: 4in
    :align: center

    Power curve


Rotor Aerodynamics Optimization
===============================

This section describes a simple optimization continuing off of the same setup as the previous section.  First, we import relevant modules and initialize the problem.

.. literalinclude:: /examples/rotorse_example2.py
    :language: python
    :start-after: # --- Import Modules
    :end-before: # ---

The optimizer must be selected and configured, in this example I choose SLSQP.

.. literalinclude:: /examples/rotorse_example2.py
    :language: python
    :start-after: # --- Optimizer
    :end-before: # ---

We now set the objective, and in this example it is normalized by the starting AEP for better convergence behavior.

.. literalinclude:: /examples/rotorse_example2.py
    :language: python
    :start-after: # --- Objective
    :end-before: # ---

The rotor chord, twist, and tip-speed ratio in Region 2 are added as design variables.

.. literalinclude:: /examples/rotorse_example2.py
    :language: python
    :start-after: # --- Design Variables
    :end-before: # ---

A recorder is added to display each iteration to the screen.

.. literalinclude:: /examples/rotorse_example2.py
    :language: python
    :start-after: # --- Recorder
    :end-before: # ---

Input variables must be set, see previous example.

.. literalinclude:: /examples/rotorse_example2.py
    :language: python
    :start-after: # --- Setup
    :end-before: # ---

Running the optimization (may take several minutes) yields a new design with a 1.35% percent increase in AEP.

.. literalinclude:: /examples/rotorse_example2.py
    :language: python
    :start-after: # --- run and outputs
    :end-before: # ---

>>> rotor.chord_sub = [0.4, 5.3, 3.77661242, 2.64450736]
>>> rotor.r_max_chord = 0.189540229541
>>> rotor.theta_sub: [13.46656381, 6.45837507, 4.59557023, 2.85016239]
>>> rotor.control.tsr = 7.79327456713
>>> Objective_0 = -1.01353182592


Rotor Aero/Structures
=====================

This examples includes both aerodynamic and structural analysis.  In this case, they are not fully coupled.  The aerodynamic loads feed into the structural analysis, but there is no feedback from the structural deflections.  `TurbineSE <http://wisdem.github.io/TurbineSE>`_ provides an example with fully coupled aero/structural solutions.  We first import the modules we will use and instantiate the objects.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === import and instantiate
    :end-before: # ---

Initial grids are set.  From these definitions only changes to the aerodynamic grid needs to be specified (through ``r_aero`` in the next section) and the locations along the aerodynamic and structural grids will be kept in sync.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === blade grid
    :end-before: # ---

Next, geometric parameters are defined.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === blade geometry
    :end-before: # ---

Airfoil data is loaded in the same way as in the aero only case.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === airfoil files
    :end-before: # ---

The atmospheric data also includes defining the IEC turbine and turbulence class, which are used to compute the average wind speed for the site and the survival wind speed.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === atmosphere
    :end-before: # ---

Parameters are defined for the steady-state control conditions.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === control
    :end-before: # ---

Various optional parameters for the analysis can be defined.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === aero and structural analysis options
    :end-before: # ---

Like the airfoil data, the material and composite layup data is most conveniently defined from files.  This implementation uses `PreComp <http://wind.nrel.gov/designcodes/preprocessors/precomp/>`_ files.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === materials and composite layup
    :end-before: # ---

A simplistic fatigue analysis can be done if damage equivalent moments are supplied.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === fatigue
    :end-before: # ---

Finally, we run the assembly and print/plot some of the outputs.  :num:`Figures #strain-spar-fig` and :num:`#strain-te-fig` show the strian distributions for the suction and pressure surface, as well as the critical strain load for buckling, in both the spar caps and trailing-edge panels.

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: # === run and outputs
    :end-before: # ---



>>> AEP = 23488621.0394
>>> diameter = 125.955004536
>>> ratedConditions.V = 11.7373200354
>>> ratedConditions.Omega = 12.0
>>> ratedConditions.pitch = 0.0
>>> ratedConditions.T = 716473.1724
>>> ratedConditions.Q = 3978873.5773
>>> mass_one_blade = 18224.9319804
>>> mass_all_blades = 54674.7959412
>>> I_all_blades = [35571709.72265241, 17105524.21138387, 14058743.19199818, 0., 0., 0.]
>>> freq = [0.98179088, 1.16157427, 2.93911298, 4.31122765, 6.52668043]
>>> tip_deflection = 5.76697788958
>>> root_bending_moment = 14036694.0247



.. _strain-spar-fig:

.. figure:: /images/strain_spar.*
    :height: 4in
    :align: left

    Strain in spar cap

.. _strain-te-fig:

.. figure:: /images/strain_te.*
    :height: 4in
    :align: center

    Strain in trailing-edge panels




