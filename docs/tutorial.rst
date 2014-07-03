.. _tutorial-label:

.. currentmodule:: rotorse.rotoraero

Tutorial
--------

The module :mod:`rotorse.rotoraero` contains methods for generating power curves and computing annual energy production (AEP) with any aerodynamic tool (implementing :class:`AeroBase` and :class:`GeomtrySetupBase`), any wind speed distribution (implementing :class:`PDFBase` and :class:`CDFBase`), any drivetrain efficiency function (implementing :class:`DrivetrainLossesBase`), and any machine type amongst the four combinations of variable/fixed speed and variable/fixed pitch.

The module :mod:`rotorse.rotoraerodefaults` provides specific implementations for use with :mod:`rotorse.rotoraero`.  It uses `CCBlade <https://github.com/WISDEM/CCBlade>`_ for the aerodynamic analysis, provides Weibull and Rayleigh wind speed distribution, and a drivetrain efficiency function from `NREL's Cost and Scaling Model <http://www.nrel.gov/docs/fy07osti/40566.pdf>`_.

The module :mod:`rotorse.rotor` builds upon rotoraero and provides structural analyses.  These include methods for coupling the aerodynamic and structural grids, managing the composite section analyses, transferring loads, computing deflections, computing mass properties, etc.

Two examples are included in this tutorial section: aerodynamic simulation and optimization of a rotor and aero/structural analysis of a rotor.



Rotor Aerodynamics
==================

.. currentmodule:: rotorse.rotoraerodefaults

We instantiate a variable-speed variable-pitch rotor preconfigured to use CCBlade (:class:`RotorAeroVSVPWithCCBlade`).  Similar methods exist for the other types of fixed/variable speed, fixed/variable pitch machines.  At instantiation, the cumulative distribution function can be selected (``cdf_type``).  Valid options are 'rayleigh' and 'weibull'.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- instantiate rotor
    :end-before: # ---

Next, the geometry is defined.  These parameters are specific to CCBlade and a spline component :class:`GeometrySpline` which uses Akima splines for the chord and twist distribution according to :num:`Figures #chord-param-fig` and :num:`#twist-param-fig`.  The power-curve is essentially the same as the previous example.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- rotor geometry
    :end-before: # ---

.. _chord-param-fig:

.. figure:: /images/chord-param2.*
    :height: 4in
    :align: left

    Chord parameterization

.. _twist-param-fig:

.. figure:: /images/theta-param2.*
    :height: 4in
    :align: center

    Twist parameterization



Next, the airfoils are loaded, again using CCBlade parameters.  The locations of the airfoil on a nondimensional blade are defined, as well as the location where the cylinder section ends.  This defines where the twist distribution begins.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- airfoils
    :end-before: # ---

Atmospheric properties and wind speed distribution are defined.  The parameter ``weibull_shape_factor`` is only relevant if the ``weibull`` type CDF is selected.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- site characteristics
    :end-before: # ---

.. currentmodule:: rotorse.rotoraero

The relevant control parameters will vary depending on whether a variable speed or fixed speed machine is chosen (see :class:`VarSpeedMachine` and :class:`FixedSpeedMachine`).

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- control settings
    :end-before: # ---

The Cost and Scaling model defines drivetrain efficiency functions for the following drivetrain types: 'geared', 'single_stage', 'multi_drive', or 'pm_direct_drive'.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- drivetrain model for efficiency
    :end-before: # ---

Finally, various optional configuration parameters are set, many of which are specific to CCBlade.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- analysis options
    :end-before: # ---


We can now run the analysis, print the AEP, and plot the power curve.


.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- run
    :end-before: # ---


>>> AEP0 = 9716744.29201

.. figure:: /images/powercurve.*
    :height: 4in
    :align: center

    Power curve


Rotor Aerodynamics Optimization
===============================

This section describes a simple optimization continuing off of the same setup as the previous section.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- optimizer imports
    :end-before: # ---

The optimizer must first be selected and configured, in this example I choose SNOPT.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- Setup Pptimizer
    :end-before: # ---

We now set the objective, and in this example it is normalized by the starting AEP for better convergence behavior.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- Objective
    :end-before: # ---

The rotor chord, twist, and tip-speed ratio in Region 2 are added as design variables.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- Design Variables
    :end-before: # ---

A recorder is added to display each iteration to the screen.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- recorder
    :end-before: # ---

There are no constraints for this problem, but because of a bug in OpenMDAO we need to add a dummy one.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- Constraints
    :end-before: # ---

Running the optimization (takes a little less than a minute) yields a new design with a 1.35% percent increase in AEP.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :language: python
    :start-after: # --- run opt
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




