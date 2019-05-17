Examples
========

All of these examples will use the following data and imports.

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pwlf

    # your data
    y = np.array([0.00000000e+00, 9.69801700e-03, 2.94350340e-02,
                  4.39052750e-02, 5.45343950e-02, 6.74104940e-02,
                  8.34831790e-02, 1.02580042e-01, 1.22767939e-01,
                  1.42172312e-01, 0.00000000e+00, 8.58600000e-06,
                  8.31543400e-03, 2.34184100e-02, 3.39709150e-02,
                  4.03581990e-02, 4.53545600e-02, 5.02345260e-02,
                  5.55253360e-02, 6.14750770e-02, 6.82125120e-02,
                  7.55892510e-02, 8.38356810e-02, 9.26413070e-02,
                  1.02039790e-01, 1.11688258e-01, 1.21390666e-01,
                  1.31196948e-01, 0.00000000e+00, 1.56706510e-02,
                  3.54628780e-02, 4.63739040e-02, 5.61442590e-02,
                  6.78542550e-02, 8.16388310e-02, 9.77756110e-02,
                  1.16531753e-01, 1.37038283e-01, 0.00000000e+00,
                  1.16951050e-02, 3.12089850e-02, 4.41776550e-02,
                  5.42877590e-02, 6.63321350e-02, 8.07655920e-02,
                  9.70363280e-02, 1.15706975e-01, 1.36687642e-01,
                  0.00000000e+00, 1.50144640e-02, 3.44519970e-02,
                  4.55907760e-02, 5.59556700e-02, 6.88450940e-02,
                  8.41374060e-02, 1.01254006e-01, 1.20605073e-01,
                  1.41881288e-01, 1.62618058e-01])
    x = np.array([0.00000000e+00, 8.82678000e-03, 3.25615100e-02,
                  5.66106800e-02, 7.95549800e-02, 1.00936330e-01,
                  1.20351520e-01, 1.37442010e-01, 1.51858250e-01,
                  1.64433570e-01, 0.00000000e+00, -2.12600000e-05,
                  7.03872000e-03, 1.85494500e-02, 3.00926700e-02,
                  4.17617000e-02, 5.37279600e-02, 6.54941000e-02,
                  7.68092100e-02, 8.76596300e-02, 9.80525800e-02,
                  1.07961810e-01, 1.17305210e-01, 1.26063930e-01,
                  1.34180360e-01, 1.41725010e-01, 1.48629710e-01,
                  1.55374770e-01, 0.00000000e+00, 1.65610200e-02,
                  3.91016100e-02, 6.18679400e-02, 8.30997400e-02,
                  1.02132890e-01, 1.19011260e-01, 1.34620080e-01,
                  1.49429370e-01, 1.63539960e-01, -0.00000000e+00,
                  1.01980300e-02, 3.28642800e-02, 5.59461900e-02,
                  7.81388400e-02, 9.84458400e-02, 1.16270210e-01,
                  1.31279040e-01, 1.45437090e-01, 1.59627540e-01,
                  0.00000000e+00, 1.63404300e-02, 4.00086000e-02,
                  6.34390200e-02, 8.51085900e-02, 1.04787860e-01,
                  1.22120350e-01, 1.36931660e-01, 1.50958760e-01,
                  1.65299640e-01, 1.79942720e-01])

1.  `fit with known breakpoint
    locations <#fit-with-known-breakpoint-locations>`__
2.  `fit for specified number of line
    segments <#fit-for-specified-number-of-line-segments>`__
3.  `fitfast for specified number of line
    segments <#fitfast-for-specified-number-of-line-segments>`__
4.  `force a fit through data
    points <#force-a-fit-through-data-points>`__
5.  `use custom optimization
    routine <#use-custom-optimization-routine>`__
6.  `pass differential evolution
    keywords <#pass-differential-evolution-keywords>`__
7.  `find the best number of line
    segments <#find-the-best-number-of-line-segments>`__
8.  `model persistence <#model-persistence>`__
9.  `bad fits when you have more unknowns than
    data <#bad-fits-when-you-have-more-unknowns-than-data>`__
10. `fit with a breakpoint guess <#fit-with-a-breakpoint-guess>`__
11. `get the linear regression
    matrix <#get-the-linear-regression-matrix>`__
12. `use of TensorFlow <#use-of-TensorFlow>`__

fit with known breakpoint locations
-----------------------------------

You can perform a least squares fit if you know the breakpoint
locations.

.. code:: python

    # your desired line segment end locations
    x0 = np.array([min(x), 0.039, 0.10, max(x)])

    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # fit the data with the specified break points
    # (ie the x locations of where the line segments
    # will terminate)
    my_pwlf.fit_with_breaks(x0)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/figs/fit_breaks.png
   :alt: fit with known breakpoint locations

   fit with known breakpoint locations

fit for specified number of line segments
-----------------------------------------

Use a global optimization to find the breakpoint locations that minimize
the sum of squares error. This uses `Differential
Evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`__
from scipy.

.. code:: python

    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # fit the data for four line segments
    res = my_pwlf.fit(4)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/figs/numberoflines.png
   :alt: fit for specified number of line segments

   fit for specified number of line segments

fitfast for specified number of line segments
---------------------------------------------

This performs a fit for a specified number of line segments with a
multi-start gradient based optimization. This should be faster than
`Differential
Evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`__
for a small number of starting points.

.. code:: python

    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # fit the data for four line segments
    # this performs 3 multi-start optimizations
    res = my_pwlf.fitfast(4, pop=3)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/figs/fitfast.png
   :alt: fitfast for specified number of line segments

   fitfast for specified number of line segments

force a fit through data points
-------------------------------

Sometimes it's necessary to force the piecewise continuous model through
a particular data point, or a set of data points. The following example
finds the best 4 line segments that go through two data points.

.. code:: python

    # initialize piecewise linear fit with your x and y data
    myPWLF = pwlf.PiecewiseLinFit(x, y)

    # fit the function with four line segments
    # force the function to go through the data points
    # (0.0, 0.0) and (0.19, 0.16) 
    # where the data points are of the form (x, y)
    x_c = [0.0, 0.19]
    y_c = [0.0, 0.2]
    res = myPWLF.fit(4, x_c, y_c)

    # predict for the determined points
    xHat = np.linspace(min(x), 0.19, num=10000)
    yHat = myPWLF.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/figs/force.png
   :alt: force a fit through data points

   force a fit through data points

use custom optimization routine
-------------------------------

You can use your favorite optimization routine to find the breakpoint
locations. The following example uses scipy's
`minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__
function.

.. code:: python

    from scipy.optimize import minimize
    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # initialize custom optimization
    number_of_line_segments = 3
    my_pwlf.use_custom_opt(number_of_line_segments)

    # i have number_of_line_segments - 1 number of variables
    # let's guess the correct location of the two unknown variables
    # (the program defaults to have end segments at x0= min(x)
    # and xn=max(x)
    xGuess = np.zeros(number_of_line_segments - 1)
    xGuess[0] = 0.02
    xGuess[1] = 0.10

    res = minimize(my_pwlf.fit_with_breaks_opt, xGuess)

    # set up the break point locations
    x0 = np.zeros(number_of_line_segments + 1)
    x0[0] = np.min(x)
    x0[-1] = np.max(x)
    x0[1:-1] = res.x

    # calculate the parameters based on the optimal break point locations
    my_pwlf.fit_with_breaks(x0)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

pass differential evolution keywords
------------------------------------

You can pass keyword arguments from the ``fit`` function into scipy's
`Differential
Evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`__.

.. code:: python

    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # fit the data for four line segments
    # this sets DE to have an absolute tolerance of 0.1
    res = my_pwlf.fit(4, atol=0.1)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

find the best number of line segments
-------------------------------------

This example uses EGO (bayesian optimization) and a penalty function to
find the best number of line segments. This will require careful use of
the penalty parameter ``l``. Use this template to automatically find the
best number of line segments.

.. code:: python

    from GPyOpt.methods import BayesianOptimization
    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # define your objective function


    def my_obj(x):
        # define some penalty parameter l
        # you'll have to arbitrarily pick this
        # it depends upon the noise in your data,
        # and the value of your sum of square of residuals
        l = y.mean()*0.001
        f = np.zeros(x.shape[0])
        for i, j in enumerate(x):
            my_pwlf.fit(j[0])
            f[i] = my_pwlf.ssr + (l*j[0])
        return f


    # define the lower and upper bound for the number of line segments
    bounds = [{'name': 'var_1', 'type': 'discrete',
               'domain': np.arange(2, 40)}]

    np.random.seed(12121)

    myBopt = BayesianOptimization(my_obj, domain=bounds, model_type='GP',
                                  initial_design_numdata=10,
                                  initial_design_type='latin',
                                  exact_feval=True, verbosity=True,
                                  verbosity_model=False)
    max_iter = 30

    # perform the bayesian optimization to find the optimum number
    # of line segments
    myBopt.run_optimization(max_iter=max_iter, verbosity=True)

    print('\n \n Opt found \n')
    print('Optimum number of line segments:', myBopt.x_opt)
    print('Function value:', myBopt.fx_opt)
    myBopt.plot_acquisition()
    myBopt.plot_convergence()

    # perform the fit for the optimum
    my_pwlf.fit(myBopt.x_opt)
    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

model persistence
-----------------

You can save fitted models with pickle. Alternatively see
`joblib <https://joblib.readthedocs.io/en/latest/>`__.

.. code:: python

    # if you use Python 2.x you should import cPickle
    # import cPickle as pickle
    # if you use Python 3.x you can just use pickle
    import pickle

    # your desired line segment end locations
    x0 = np.array([min(x), 0.039, 0.10, max(x)])

    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # fit the data with the specified break points
    my_pwlf.fit_with_breaks(x0)

    # save the fitted model
    with open('my_fit.pkl', 'wb') as f:
        pickle.dump(my_pwlf, f, pickle.HIGHEST_PROTOCOL)

    # load the fitted model
    with open('my_fit.pkl', 'rb') as f:
        my_pwlf = pickle.load(f)

bad fits when you have more unknowns than data
----------------------------------------------

You can get very bad fits with pwlf when you have more unknowns than
data points. The following example will fit 99 line segments to the 59
data points. While this will result in an error of zero, the model will
have very weird predictions within the data. You should not fit more
unknowns than you have data with pwlf!

.. code:: python

    break_locations = np.linspace(min(x), max(x), num=100)
    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    my_pwlf.fit_with_breaks(break_locations)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/figs/badfit.png
   :alt: bad fits when you have more unknowns than data

   bad fits when you have more unknowns than data

fit with a breakpoint guess
---------------------------

In this example we see two distinct linear regions, and we believe a
breakpoint occurs at 6.0. We'll use the fit\_guess() function to find
the best breakpoint location starting with this guess. These fits should
be much faster than the ``fit`` or ``fitfast`` function when you have a
reasonable idea where the breakpoints occur.

.. code:: python

    import numpy as np
    import pwlf
    x = np.array([4., 5., 6., 7., 8.])
    y = np.array([11., 13., 16., 28.92, 42.81])
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit_guess([6.0])

Note specifying one breakpoint will result in two line segments. If we
wanted three line segments, we'll have to specify two breakpoints.

.. code:: python

    breaks = my_pwlf.fit_guess([5.5, 6.0])

get the linear regression matrix
--------------------------------

In some cases it may be desirable to work with the linear regression
matrix directly. The following example grabs the linear regression
matrix ``A`` for a specific set of breakpoints. In this case we assume
that the breakpoints occur at each of the data points. Please see the
`paper <https://github.com/cjekel/piecewise_linear_fit_py/tree/master/paper>`__
for details about the regression matrix ``A``.

.. code:: python

    import numpy as np
    import pwlf
    # select random seed for reproducibility
    np.random.seed(123)
    # generate sin wave data
    x = np.linspace(0, 10, num=100)
    y = np.sin(x * np.pi / 2)
    ytrue = y.copy()
    # add noise to the data
    y = np.random.normal(0, 0.05, 100) + ytrue

    my_pwlf_en = pwlf.PiecewiseLinFit(x, y)
    # copy the x data to use as break points
    breaks = my_pwlf_en.x_data.copy()
    # create the linear regression matrix A 
    A = my_pwlf_en.assemble_regression_matrix(breaks, my_pwlf_en.x_data)

We can perform fits that are more complicated than a least squares fit
when we have the regression matrix. The following uses the Elastic Net
regularizer to perform an interesting fit with the regression matrix.

.. code:: python

    from sklearn.linear_model import ElasticNetCV
    # set up the elastic net
    en_model = ElasticNetCV(cv=5,
                            l1_ratio=[.1, .5, .7, .9,
                                      .95, .99, 1],
                            fit_intercept=False,
                            max_iter=1000000, n_jobs=-1)
    # fit the model using the elastic net
    en_model.fit(A, my_pwlf_en.y_data)

    # predict from the elastic net parameters
    xhat = np.linspace(x.min(), x.max(), 1000)
    yhat_en = my_pwlf_en.predict(xhat, breaks=breaks,
                                 beta=en_model.coef_)

.. figure:: https://raw.githubusercontent.com/cjekel/piecewise_linear_fit_py/master/examples/figs/sin_en_net_fit.png
   :alt: interesting elastic net fit

   interesting elastic net fit

use of TensorFlow
-----------------

You'll be able to use the ``PiecewiseLinFitTF`` class if you have
TensorFlow installed, which may offer performance improvements for
larger data sets over the original ``PiecewiseLinFit`` class. For
performance benchmarks see this blog
`post <https://jekel.me/2019/Adding-tensorflow-to-pwlf/>`__.

The use of the TF class is nearly identical to the original class,
however note the following exceptions. ``PiecewiseLinFitTF`` does:

-- not have a ``lapack_driver`` option
 -  have an optional parameter ``dtype``, so you can choose between the
    float64 and float32 data types
 -  have an optional parameter ``fast`` to switch between Cholesky
    decomposition (default ``fast=True``), and orthogonal decomposition
    (``fast=False``)

.. code:: python

    # your desired line segment end locations
    x0 = np.array([min(x), 0.039, 0.10, max(x)])

    # initialize TF piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFitTF(x, y, dtype='float32)

    # fit the data with the specified break points
    # (ie the x locations of where the line segments
    # will terminate)
    my_pwlf.fit_with_breaks(x0)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)
