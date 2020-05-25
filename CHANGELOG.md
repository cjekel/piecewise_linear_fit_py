# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2020-05-25
### Changed
- Fixed an encoding bug that would not let pwlf install on windows. Thanks to h-vetinari for the [PR](https://github.com/cjekel/piecewise_linear_fit_py/pull/71)!

## [2.0.1] - 2020-05-24
### Changed
- Removed setuptools for importlib single source versioning
### Added
- Requirement for importlib-metadata if Python version is less than 3.8
### Removed
- Requriement for setuptools

## [2.0.0] - 2020-04-02
### Added
- Added supports for pwlf to fit to weighted data sets! Check out [this example](https://github.com/cjekel/piecewise_linear_fit_py/tree/master/examples#weighted-least-squares-fit).
### Changed
- Setup.py now grabs markdown file for long description
### Removed
- Tensorflow support has been removed. It hasn't been updated in a long time. If you still require this object, check out [pwlftf](https://github.com/cjekel/piecewise_linear_fit_py_tf)


## [1.1.7] - 2020-02-05
### Changed
- Minimum SciPy version is now 1.2.0 because of issues with MacOS and the old SciPy versions. See issue https://github.com/cjekel/piecewise_linear_fit_py/issues/40 and thanks to bezineb5 !

## [1.1.6] - 2020-01-22
### Changed
- Single source version now found in setup.py instead of pwlf/VERSION see issue https://github.com/cjekel/piecewise_linear_fit_py/issues/53
- New setuptools requirement to handle new version file
- Fix bug where forcing pwlf through points didn't work with higher degrees. See issue https://github.com/cjekel/piecewise_linear_fit_py/issues/54

## [1.1.5] - 2019-11-21
### Changed
- Fix minor typo in docstring of ```calc_slopes```
- Initialized all attributes in the ```__init__``` funciton
- All attributes are now documented in the ```__init__``` function. To view this docstring, use ```pwlf.PiecewiseLinFit?```.

## [1.1.4] - 2019-10-24
### Changed
- TensorFlow 2.0.0 is not (and most probably will not) be supported. DepreciationWarning is displayed when using the ```PiecewiseLinFitTF``` object. Setup.py checks for this optional requirement. Tests are run on Tensorflow<2.0.0.
- TravisCi now checks Python version 3.7 in addition to 3.6, 3.5, 2.7.
- TravisCi tests should now be run daily.

## [1.1.3] - 2019-09-14
### Changed
- Make .ssr stored with fit_with_break* functions

## [1.1.2] - 2019-08-19
### Changed 
- Bug fix in non-linear standard error, predict was calling y instead of x. https://github.com/cjekel/piecewise_linear_fit_py/pull/46 Thanks to @tcanders

## [1.1.1] - 2019-08-18
### Changed
- Raise the correct AttributeError when a fit has not yet been performed

## [1.1.0] - 2019-06-16
### Added
- Now you can calculate standard errors for non-linear regression using the Delta method! Check out this [example](https://github.com/cjekel/piecewise_linear_fit_py/tree/master/examples#non-linear-standard-errors-and-p-values). 

## [1.0.1] - 2019-06-15
### Added
- Now you can fit constants and continuous polynomials with pwlf! Just specify the keyword ```degree=``` when initializing the ```PiecewiseLinFit``` object. Note that ```degree=0``` for constants, ```degree==1``` for linear (default), ```degree==2``` for quadratics, etc.
- You can manually specify the optimization bounds for each breakpoint when calling the ```fit``` functions by using the ```bounds=``` keyword. Check out the related example.
### Changed
- n_parameters is now calculated based on the shape of the regression matrix
- assembly of the regression matrix now considers which degree polynomial
- n_segments calculated from break points...
- Greatly reduce teststf.py run time

## [1.0.0] - 2019-05-16
### Changed
- Numpy matrix assembly is now ~100x times faster, which will translate to much faster fits! See this [comment](https://github.com/cjekel/piecewise_linear_fit_py/issues/20#issuecomment-492860953) about the speed up. There should no longer be any performance benefits with using the ```PiecewiseLinFitTF``` (TensorFlow) object, so the only reason to use ```PiecewiseLinFitTF``` is if you want access to TensorFlow's optimizers.
### Removed
- There are no sort or order optional parameters in ```PiecewiseLinFit```. The new matrix assembly method doesn't need sorted data. This may break backwards compatibility with your code. 

## [0.5.1] - 2019-05-05
### Changed
- Fixed ```PiecewiseLinFitTF``` for Python 2.

## [0.5.0] - 2019-04-15
### Added
- New ```PiecewiseLinFitTF``` class which uses TensorFlow to accelerate pwlf. This class is nearly identical to ```PiecewiseLinFit```, with the exception of the removed ```sorted_data``` options. If you have TensorFlow installed you'll be able to use ```PiecewiseLinFitTF```. If you do not have TensorFlow installed, importing pwlf will issue a warning that ```PiecewiseLinFitTF``` is not available. See this blog [post](https://jekel.me/2019/Adding-tensorflow-to-pwlf/) for more information and benchmarks. The new class includes an option to use float32 or float64 data types. 
- ```lapack_driver``` option to choose between the least squares backend. For more see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html and http://www.netlib.org/lapack/lug/node27.html
### Changed
- Now use scipy.linalg instead of numpy.linalg because scipy is always compiled with lapack
- least squares fit now defaults to scipy instead of numpy
### Removed
- ```rcond``` optional parameter; was not necessary with scipy.linalg

## [0.4.3] - 2019-04-02
### Changed
- You can now manually specify ```rcond``` for the numpy least squares solver. For more see https://github.com/cjekel/piecewise_linear_fit_py/issues/21 . 

## [0.4.2] - 2019-03-22
### Changed
- ```assemble_regression_matrix()``` now checks if breaks is a numpy array

## [0.4.1] - 2019-03-18
### Changed
- p_values() now uses the correct degrees of freedom (Thanks to Tong Qiu for pointing this out!)
- p_values() now returns a value that can be compared to alpha; previous values would have been compared to alpha/2

## [0.4.0] - 2019-03-14
### Added
- new ```assemble_regression_matrix()``` function that returns the linear regression matrix ```A``` which can allow you to do some more complicated [fits](https://jekel.me/2019/detect-number-of-line-segments-in-pwlf/)
- test function for the linear regression matrix
- new ```fit_guess()``` function to perform a fit when you have an estimate of the breakpoint locations
### Changed
- consolidated the assembly of the linear regression matrix to a single function (and removed the duplicate code)

## [0.3.5] - 2019-02-25
### Changed
- minor correction to r_squared docstring example

## [0.3.4] - 2019-02-06
### Added
- Uploaded paper and citation information
- Added example of what happens when you have more unknowns than data
### Changed
- Examples now include figures

## [0.3.3] - 2019-01-32
### Added
- Documentation and link to documentation
### Changed
- Minor changes to docstrings (spelling, formatting, attribute fixes)

## [0.3.2] - 2019-01-24
### Added
- y-intercepts are now calculated when running the calc_slopes() function. The Y-intercept for each line is stored in self.intercepts. 

## [0.3.1] - 2018-12-09
### Added
- p_values() function to calculate the p-value of each beta parameter (WARNING! you may only want to use this if you specify the break point locations, as this does not account for uncertainty in break point locations)

### Changed
- Now changes stored in CHANGELOG.md

## [0.3.0] - 2018-12-05
### Added
- r_squared() function to calculate the coefficent of determination after a fit has been performed

### Changed
- complete docstring overhaul to match the numpydoc style
- fix issues where sum-of-squares of residuals returned array_like, now should always return float

### Removed
- legacy piecewise_lin_fit object has been removed

## [0.2.10] - 2018-12-04
### Changed
- Minor docstring changes
- Fix spelling mistakes throughout
- Fix README.rst format for PyPI

## [0.0.X - 0.2.9] - from 2017-04-01 to 2018-10-03
- 2018/10/03 Add example of bare minimum model persistance to predict for new data (see examples/model_persistence_prediction.py). Bug fix in predict function for custom parameters. Add new test function to check that predict works with custom parameters.
- 2018/08/11 New function which calculates the predication variance for given array of x locations. The predication variance is the squared version of the standard error (not to be confused with the standard errors of the previous change). New example prediction_variance.py shows how to use the new function.
- 2018/06/16 New function which calculates the standard error for each of the model parameters (Remember model parameters are stored as my_pwlf.beta). Standard errors are calculated by calling se = my_pwlf.standard_errors() after you have performed a fit. For more information about standard errors see [this](https://en.wikipedia.org/wiki/Standard_error). Fix docstrings for all functions.
- 2018/05/11 New sorted_data key which can be used to avoided sorting already ordered data. If your data is already ordered as x[0] < x[1] < ... < x[n-1], you may consider using sorted_data=True for a slight performance increase. Additionally the predict function can take the sorted_data key if the data you want to predict at is already sorted. Thanks to [V-Kh](https://github.com/V-Kh) for the idea and PR. 
- 2018/04/15 Now you can find piecewise linear fits that go through specified data points! Read [this post](http://jekel.me/2018/Force-piecwise-linear-fit-through-data/) for the details.
- 2018/04/09 Intelligently converts your x, y, or breaks to be numpy array.
- 2018/04/06 Speed! pwlf just got better and faster! A vast majority of this library has been entirely rewritten! New naming convention. The class piecewise_lin_fit() is being depreciated, now use the class PiecewiseLinFit(). See [this post](http://jekel.me/2018/Continous-piecewise-linear-regression/) for details on the new formulation. New test function that tests predict().
- 2018/03/25 Default now hides optimization results. Use disp_res=True when initializing piecewise_lin_fit to change. The multi-start fitfast() function now defaults to the minimum population of 2.
- 2018/03/11 Added try/except behavior for fitWithBreaks function such that the function could be used in an optimization routine. In general when you have a singular matrix, the function will now return np.inf.
- 2018/02/16 Added new fitfast() function which uses multi-start gradient optimization instead of Differential Evolution. It may be substantially faster for your application. Also it would be a good candidate if you don't need the best solution, but just a reasonable fit. Fixed bug in tests function where assert was checking bound, not SSr. New requirement, pyDOE library. New 0.1.0 Version.
- 2017/11/03 add setup.py, new tests folder and test scripts, new version tracking, initialize break0 breakN in the beginning
- 2017/10/31 bug fix related to the case where break points exactly equal to x data points ( as per issue https://github.com/cjekel/piecewise_linear_fit_py/issues/1 ) and added attributes .sep_data_x, .sep_data_y, .sep_predict_data_x for troubleshooting issues related to the separation of data points to their respective regions
- 2017/10/20 remove determinant calculation and use try-except instead, this will offer a larger performance boost for big problems. Change library name to something more Pythonic. Add version attribute.
- 2017/08/03 gradients (slopes of the line segments) now stored as piecewise_lin_fit.slopes (or myPWLF.slopes) after they have been calculated by performing a fit or predicting
- 2017/04/01 initial release
