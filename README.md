# About
A library for fitting a continuous piecewise linear function f(x) to data. Just specify the number of line segments you desire and your data set.

Check out the [examples](https://github.com/cjekel/piecewise_linear_fit_py/tree/master/examples)!

Read the [blog post](http://jekel.me/2017/Fit-a-piecewise-linear-function-to-data/).

![Example of a continuous piecewise linear fit to a data set.](https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/examplePiecewiseFit.png)

![Example of a continuous piecewise linear fit to a sin wave](https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/sinWaveFit.png)

# Features
For a specified number of line segments, you can determine (and predict from) the optimal continuous piecewise linear function f(x). See [this example](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitForSpecifiedNumberOfLineSegments.py).

You can fit and predict a continuous piecewise linear function f(x) if you know the specific x locations where the line segments terminate. See [this example](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitWithKnownLineSegmentLocations.py).

If you want to pass different keywords for the SciPy differential evolution algorithm see [this example](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitForSpecifiedNumberOfLineSegments_passDiffEvoKeywords.py).

You can use a different optimization algorithm to find the optimal location for line segments by using the objective function that minimizes the sum of square of residuals. See [this example](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/useCustomOptimizationRoutine.py).

Instead of using differential evolution, you can now use a multi-start gradient optimization with fitfast() function. You can specify the number of starting points to use. The default is 2. This means that a latin hyper cube sampling (space filling DOE) of 2 is used to run 2 L-BFGS-B optimizations. See [this example](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/sineWave_time_compare.py) which runs fit() function, then runs the fitfast() to compare the runtime differences!

# Installation

You can now install with pip.
```
[sudo] pip install pwlf
```

Or clone the repo
```
git clone https://github.com/cjekel/piecewise_linear_fit_py.git
```

then install with pip
```
[sudo] pip install ./piecewise_linear_fit_py
```

# How it works
This is based on a formulation of a piecewise linear least squares fit, where the user must specify the location of break points. See [this post](http://jekel.me/2018/Continous-piecewise-linear-regression/) which goes through the derivation of a least squares regression problem if the break point locations are known. Alternatively check out [Golovchenko (2004)](http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf).

Global optimization is used to find the best location for the user defined number of line segments. I specifically use the [differential evolution](https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html) algorithm in SciPy. I default the differential evolution algorithm to be aggressive, and it is probably overkill for your problem. So feel free to pass your own differential evolution keywords to the library. See [this example](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitForSpecifiedNumberOfLineSegments_passDiffEvoKeywords.py).

# Why
All other methods require the user to specify the specific location of break points, but in most cases the best location for these break points is unknown. It makes more sense to rather have the user specify the desired number of line segments, and then to quantitatively choose the best location for the ends of these line segments.

# Changelog
All changes now stored in [CHANGELOG.md](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/CHANGELOG.md)

New r_squared() function, new p_values() function,  all docstrings follows numpydoc style... 

# Requirements
Python 2.7+ (Python 2.7 and Python 3.4 have been tested)

NumPy (Tested on version >= 1.14.0)

SciPy (Tested on version >= 0.19.0)

pyDOE (Tested on version >= 0.3.8)

# License
MIT License
