How it works
============

This
`paper <https://github.com/cjekel/piecewise_linear_fit_py/raw/master/paper/pwlf_Jekel_Venter_v2.pdf>`__
explains how this library works in detail.

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
