# About
A library for fitting continuous piecewise linear functions to data. Just specify the number of line segments you desire and provide the data.

![Downloads a month](https://img.shields.io/pypi/dm/pwlf.svg) [![Build Status](https://travis-ci.org/cjekel/piecewise_linear_fit_py.svg?branch=master)](https://travis-ci.org/cjekel/piecewise_linear_fit_py)  [![Coverage Status](https://coveralls.io/repos/github/cjekel/piecewise_linear_fit_py/badge.svg?branch=master)](https://coveralls.io/github/cjekel/piecewise_linear_fit_py?branch=master)![PyPI version](https://img.shields.io/pypi/v/pwlf)

Check out the [documentation](https://jekel.me/piecewise_linear_fit_py)!

Read the [blog post](http://jekel.me/2017/Fit-a-piecewise-linear-function-to-data/).

![Example of a continuous piecewise linear fit to data.](https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/examplePiecewiseFit.png)

![Example of a continuous piecewise linear fit to a sine wave.](https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/sinWaveFit.png)

Now you can perform segmented constant fitting and piecewise polynomials!
![Example of multiple degree fits to a sine wave.](https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/figs/multi_degree.png)

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
This [paper](https://github.com/cjekel/piecewise_linear_fit_py/raw/master/paper/pwlf_Jekel_Venter_v2.pdf) explains how this library works in detail.

This is based on a formulation of a piecewise linear least squares fit, where the user must specify the location of break points. See [this post](http://jekel.me/2018/Continous-piecewise-linear-regression/) which goes through the derivation of a least squares regression problem if the break point locations are known. Alternatively check out [Golovchenko (2004)](http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf).

Global optimization is used to find the best location for the user defined number of line segments. I specifically use the [differential evolution](https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html) algorithm in SciPy. I default the differential evolution algorithm to be aggressive, and it is probably overkill for your problem. So feel free to pass your own differential evolution keywords to the library. See [this example](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/fitForSpecifiedNumberOfLineSegments_passDiffEvoKeywords.py).

# Changelog
All changes now stored in [CHANGELOG.md](https://github.com/cjekel/piecewise_linear_fit_py/blob/master/CHANGELOG.md)

New ```weights=``` keyword allows you to perform weighted pwlf fits! Removed TensorFlow code which can now be found [here](https://github.com/cjekel/piecewise_linear_fit_py_tf). 

# Requirements
Python 2.7+

NumPy >= 1.14.0

SciPy >= 1.2.0

pyDOE >= 0.3.8

setuptools >= 38.6.0

# License
MIT License

# Citation

```bibtex
@Manual{pwlf,
	author = {Jekel, Charles F. and Venter, Gerhard},
	title = {{pwlf:} A Python Library for Fitting 1D Continuous Piecewise Linear Functions},
	year = {2019},
	url = {https://github.com/cjekel/piecewise_linear_fit_py}
}
```
