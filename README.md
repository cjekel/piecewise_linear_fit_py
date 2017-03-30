# About
fit piecewise linear data for a specified number of breakpoints

Simply provide your data and the number of piecewise lines to use, and the routine will return the best piecewise linear fit to your data.

# How it works
This is based on a formulation of a piecewise linear least squares fit, where the user must specify the location of break points. A simple derivation of this fit has been done by [Golovchenko (2004)](http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf). The routine for fitting the piecewise linear function is based on Golovchenko's MATLAB code (which I can't seem to find on the internet), which I ported to Python. Alternatively you can view [this code](https://www.mathworks.com/matlabcentral/fileexchange/40913-piecewise-linear-least-square-fit).

Global optimization is used to find the best location for the user defined number of break points. I specifically use the [differential evolution](https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html) algorithm in SciPy.

# Why
All other methods require the user to specify the specific location of break points, but in most cases the best location for these break points is unknown. It makes more sense to rather have the user specify the number of break points to use, and then to quantitatively choose the best location for the break points.

# Requirements
Python 2.7+
NumPy
Matplotlib
SciPy

# License
MIT License
