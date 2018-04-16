About
=====

A library for fitting a continuous piecewise linear function f(x) to
data. Just specify the number of line segments you desire and your data
set.

Check out the
`examples <https://github.com/cjekel/piecewise_linear_fit_py/tree/master/examples>`__!

Read the `blog
post <http://jekel.me/2017/Fit-a-piecewise-linear-function-to-data/>`__.

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/examplePiecewiseFit.png
   :alt: Example of a continuous piecewise linear fit to a data set.

   Example of a continuous piecewise linear fit to a data set.

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/sinWaveFit.png
   :alt: Example of a continuous piecewise linear fit to a sin wave

   Example of a continuous piecewise linear fit to a sin wave

Features
========

For a specified number of line segments, you can determine (and predict
from) the optimal continuous piecewise linear function f(x). See `this
example <https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitForSpecifiedNumberOfLineSegments.py>`__.

You can fit and predict a continuous piecewise linear function f(x) if
you know the specific x locations where the line segments terminate. See
`this
example <https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitWithKnownLineSegmentLocations.py>`__.

If you want to pass different keywords for the SciPy differential
evolution algorithm see `this
example <https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitForSpecifiedNumberOfLineSegments_passDiffEvoKeywords.py>`__.

You can use a different optimization algorithm to find the optimal
location for line segments by using the objective function that
minimizes the sum of square of residuals. See `this
example <https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/useCustomOptimizationRoutine.py>`__.

Instead of using differential evolution, you can now use a multi-start
gradient optimization with fitfast() function. You can specify the
number of starting points to use. The default is 50. This means that a
latin hyper cube sampling of 50 is used to run 50 L-BFGS-B
optimizations. See `this
example <https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/sineWave_time_compare.py>`__
which runs fit() function, then runs the fitfast() to compare the
runtime differences!

Installation
============

You can now install with pip.

::

    sudo pip install pwlf

Or clone the repo

::

    git clone https://github.com/cjekel/piecewise_linear_fit_py.git

then install with pip

::

    sudo pip install piecewise_linear_fit_py/

or easy\_install

::

    sudo easy_install piecewise_linear_fit_py/

or using setup.py

::

    cd piecewise_linear_fit_py/
    sudo python setup.py install

How it works
============

This is based on a formulation of a piecewise linear least squares fit,
where the user must specify the location of break points. See `this
post <http://jekel.me/2018/Continous-piecewise-linear-regression/>`__
which goes through the derivation of a least squares regression problem
if the break point locations are known. Alternatively check out
`Golovchenko
(2004) <http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf>`__.

Global optimization is used to find the best location for the user
defined number of line segments. I specifically use the `differential
evolution <https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html>`__
algorithm in SciPy. I default the differential evolution algorithm to be
aggressive, and it is probably overkill for your problem. So feel free
to pass your own differential evolution keywords to the library. See
`this
example <https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitForSpecifiedNumberOfLineSegments_passDiffEvoKeywords.py>`__.

Why
===

All other methods require the user to specify the specific location of
break points, but in most cases the best location for these break points
is unknown. It makes more sense to rather have the user specify the
desired number of line segments, and then to quantitatively choose the
best location for the ends of these line segments.

Changelog
=========

-  2018/04/15 Now you can find piecewise linear fits that go through
   specified data points! Read `this
   post <http://jekel.me/2018/Force-piecwise-linear-fit-through-data/>`__
   for the details.
-  2018/04/09 Intelligently converts your x, y, or breaks to be numpy
   array.
-  2018/04/06 Speed! pwlf just got better and faster! A vast majority of
   this library has been entirely rewritten! New naming convention. The
   class piecewise\_lin\_fit() is being depreciated, now use the class
   PiecewiseLinFit(). See `this
   post <http://jekel.me/2018/Continous-piecewise-linear-regression/>`__
   for details on the new formulation. New test function that tests
   predict().
-  2018/03/25 Default now hides optimization results. Use disp\_res=True
   when initializing piecewise\_lin\_fit to change. The multi-start
   fitfast() function now defaults to the minimum population of 2.
-  2018/03/11 Added try/except behavior for fitWithBreaks function such
   that the function could be used in an optimization routine. In
   general when you have a singular matrix, the function will now return
   np.inf.
-  2018/02/16 Added new fitfast() function which uses multi-start
   gradient optimization instead of Differential Evolution. It may be
   substantially faster for your application. Also it would be a good
   candidate if you don't need the best solution, but just a reasonable
   fit. Fixed bug in tests function where assert was checking bound, not
   SSr. New requirement, pyDOE library. New 0.1.0 Version.
-  2017/11/03 add setup.py, new tests folder and test scripts, new
   version tracking, initialize break0 breakN in the beginning
-  2017/10/31 bug fix related to the case where break points exactly
   equal to x data points ( as per issue
   https://github.com/cjekel/piecewise\_linear\_fit\_py/issues/1 ) and
   added attributes .sep\_data\_x, .sep\_data\_y, .sep\_predict\_data\_x
   for troubleshooting issues related to the separation of data points
   to their respective regions
-  2017/10/20 remove determinant calculation and use try-except instead,
   this will offer a larger performance boost for big problems. Change
   library name to something more Pythonic. Add version attribute.
-  2017/08/03 gradients (slopes of the line segments) now stored as
   piecewise\_lin\_fit.slopes (or myPWLF.slopes) after they have been
   calculated by performing a fit or predicting
-  2017/04/01 initial release

Requirements
============

Python 2.7+ (Python 2.7 and Python 3.4 have been tested)

NumPy (Tested on version >= 1.14.0)

SciPy (Tested on version >= 0.19.0)

pyDOE (Tested on version >= 0.3.8)

License
=======

MIT License
