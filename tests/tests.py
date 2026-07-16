import numpy as np
import unittest
import pwlf


class TestEverything(unittest.TestCase):

    x_small = np.array((0.0, 1.0, 1.5, 2.0))
    y_small = np.array((0.0, 1.0, 1.1, 1.5))
    x_sin = np.linspace(0, 10, num=100)
    y_sin = np.sin(x_sin * np.pi / 2)

    def test_break_point_spot_on(self):
        # check that I can fit when break points spot on a
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        ssr = my_fit1.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr, 0.0))

    def test_break_point_spot_on_list(self):
        # check that I can fit when break points spot on a
        my_fit1 = pwlf.PiecewiseLinFit(list(self.x_small), list(self.y_small))
        x0 = self.x_small.copy()
        ssr = my_fit1.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr, 0.0))

    def test_degree_not_supported(self):
        try:
            _ = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                     degree=100)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_x_c_not_supplied(self):
        try:
            mf = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                      degree=1)
            mf.fit(2, x_c=[1.0])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_y_c_not_supplied(self):
        try:
            mf = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                      degree=1)
            mf.fit(2, y_c=[1.0])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_lapack_driver(self):
        # check that I can fit when break points spot on a
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                       lapack_driver='gelsy')
        x0 = self.x_small.copy()
        ssr = my_fit1.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr, 0.0))

    def test_assembly(self):
        # check that I can fit when break points spot on a
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        A = my_fit.assemble_regression_matrix(x0, my_fit.x_data)
        Asb = np.array([[1.,  0.,  0.,  0.],
                        [1.,  x0[1]-x0[0],  0.,  0.],
                        [1.,  x0[2]-x0[0], x0[2]-x0[1], 0.],
                        [1.,  x0[3]-x0[0], x0[3]-x0[1], x0[3]-x0[2]]])
        self.assertTrue(np.allclose(A, Asb))

    def test_assembly_list(self):
        # check that I can fit when break points spot on a
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        A = my_fit.assemble_regression_matrix(x0, list(my_fit.x_data))
        Asb = np.array([[1.,  0.,  0.,  0.],
                        [1.,  x0[1]-x0[0],  0.,  0.],
                        [1.,  x0[2]-x0[0], x0[2]-x0[1], 0.],
                        [1.,  x0[3]-x0[0], x0[3]-x0[1], x0[3]-x0[2]]])
        self.assertTrue(np.allclose(A, Asb))

    def test_break_point_spot_on_r2(self):
        # test r squared value with known solution
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        my_fit1.fit_with_breaks(x0)
        rsq = my_fit1.r_squared()
        self.assertTrue(np.isclose(rsq, 1.0))

    def test_break_point_diff_x0_0(self):
        # check diff loc
        my_fit2 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        x0[1] = 1.00001
        x0[2] = 1.50001
        ssr2 = my_fit2.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr2, 0.0))

    def test_break_point_diff_x0_1(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit3 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0[1] = 0.9
        ssr3 = my_fit3.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr3, 0.0))

    def test_break_point_diff_x0_2(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit4 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0[1] = 1.1
        ssr4 = my_fit4.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr4, 0.0))

    def test_break_point_diff_x0_3(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit5 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0[2] = 1.6
        ssr5 = my_fit5.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr5, 0.0))

    def test_break_point_diff_x0_4(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit6 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0[2] = 1.4
        ssr6 = my_fit6.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr6, 0.0))

    def test_diff_evo(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        my_pwlf.fit(4, disp=False)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))

    def test_custom_bounds1(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        n_segments = 3
        bounds = np.zeros((n_segments-1, 2))
        bounds[0, 0] = self.x_small[0]
        bounds[0, 1] = self.x_small[2]
        bounds[1, 0] = self.x_small[1]
        bounds[1, 1] = self.x_small[-1]
        my_pwlf.fit(n_segments, bounds=bounds)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))

    def test_custom_bounds2(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        n_segments = 3
        bounds = np.zeros((n_segments-1, 2))
        bounds[0, 0] = self.x_small[0]
        bounds[0, 1] = self.x_small[2]
        bounds[1, 0] = self.x_small[1]
        bounds[1, 1] = self.x_small[-1]
        my_pwlf.fitfast(n_segments, pop=20, bounds=bounds)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))

    def test_predict(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        xmax = np.max(self.x_small)
        xmin = np.min(self.x_small)
        my_pwlf.fit_with_breaks((xmin, xmax))
        x = np.linspace(xmin, xmax, 10)
        yHat = my_pwlf.predict(x)
        self.assertTrue(np.isclose(np.sum(yHat), 8.085714285714287))

    def test_custom_opt(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        my_pwlf.use_custom_opt(3)
        x_guess = np.array((0.9, 1.1))
        from scipy.optimize import minimize
        res = minimize(my_pwlf.fit_with_breaks_opt, x_guess)
        self.assertTrue(np.isclose(res['fun'], 0.0))

    def test_custom_opt_with_con(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        my_pwlf.use_custom_opt(3, x_c=[0.], y_c=[0.])
        x_guess = np.array((0.9, 1.1))
        from scipy.optimize import minimize
        _ = minimize(my_pwlf.fit_with_breaks_opt, x_guess)
        self.assertTrue(True)

    def test_single_force_break_point1(self):
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x_c = [-0.5]
        y_c = [-0.5]
        my_fit.fit_with_breaks_force_points([0.2, 0.7], x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_single_force_break_point2(self):
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x_c = [2.0]
        y_c = [1.5]
        my_fit.fit_with_breaks_force_points([0.2, 0.7], x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_single_force_break_point_degree(self):
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small, degree=2)
        x_c = [2.0]
        y_c = [1.5]
        my_fit.fit_with_breaks_force_points([0.2, 0.7], x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_single_force_break_point_degree_zero(self):
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small, degree=0)
        x_c = [2.0]
        y_c = [1.5]
        my_fit.fit_with_breaks_force_points([0.2, 0.7], x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_opt_fit_with_force_points(self):
        # I need more data points to test this function because of
        # ill conditioning in the least squares problem...
        x = np.linspace(0.0, 1.0, num=100)
        y = np.sin(6.0*x)
        my_fit = pwlf.PiecewiseLinFit(x, y, disp_res=True)
        x_c = [0.0]
        y_c = [0.0]
        my_fit.fit(3, x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_opt_fit_single_segment(self):
        # Test fit for a single segment without force points
        x = np.linspace(0.0, 1.0, num=100)
        y = x + 1
        my_fit = pwlf.PiecewiseLinFit(x, y, disp_res=True)
        my_fit.fit(1)
        xhat = 0
        yhat = my_fit.predict(xhat)
        self.assertTrue(np.isclose(xhat + 1, yhat))

    def test_opt_fit_with_force_points_single_segment(self):
        # Test fit for a single segment (same as above)
        # but with a force point
        x = np.linspace(0.0, 1.0, num=100)
        y = x + 1
        my_fit = pwlf.PiecewiseLinFit(x, y, disp_res=True)
        x_c = [0.0]
        y_c = [0.0]
        my_fit.fit(1, x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_se(self):
        # check to see if it will let me calculate standard errors
        my_pwlf = pwlf.PiecewiseLinFit(np.random.random(20),
                                       np.random.random(20))
        my_pwlf.fitfast(2)
        my_pwlf.standard_errors()

    def test_p(self):
        # check to see if it will let me calculate p-values
        my_pwlf = pwlf.PiecewiseLinFit(np.random.random(20),
                                       np.random.random(20))
        my_pwlf.fitfast(2)
        my_pwlf.p_values()

    def test_nonlinear_p_and_se(self):
        # generate a true piecewise linear data
        np.random.seed(1)
        n_data = 20
        x = np.linspace(0, 1, num=n_data)
        y = np.random.random(n_data)
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        true_beta = np.array((1.0, 0.2, 0.2))
        true_breaks = np.array((0.0, 0.5, 1.0))
        y = my_pwlf.predict(x, beta=true_beta, breaks=true_breaks)
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        my_pwlf.fitfast(2)
        # calculate p-values
        p = my_pwlf.p_values(method='non-linear', step_size=1e-4)
        self.assertTrue(p.max() <= 0.05)

    def test_pv(self):
        # check to see if it will let me calculate prediction variance for
        # random data
        my_pwlf = pwlf.PiecewiseLinFit(np.random.random(20),
                                       np.random.random(20))
        my_pwlf.fitfast(2)
        my_pwlf.prediction_variance(np.random.random(20))

    def test_predict_with_custom_param(self):
        # check to see if predict runs with custom parameters
        x = np.random.random(20)
        my_pwlf = pwlf.PiecewiseLinFit(x, np.random.random(20))
        my_pwlf.predict(x, beta=np.array((1e-4, 1e-2, 1e-3)),
                        breaks=np.array((0.0, 0.5, 1.0)))

    def test_fit_guess(self):
        x = np.array([4., 5., 6., 7., 8.])
        y = np.array([11., 13., 16., 28.92, 42.81])
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit_guess([6.0])
        self.assertTrue(np.isclose(breaks[1], 6.0705297))

    def test_fit_guess_kwrds(self):
        x = np.array([4., 5., 6., 7., 8.])
        y = np.array([11., 13., 16., 28.92, 42.81])
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit_guess([6.0], m=10,
                                   factr=1e2, pgtol=1e-05,
                                   epsilon=1e-6, iprint=-1,
                                   maxfun=1500000, maxiter=150000,
                                   disp=None)
        self.assertTrue(np.isclose(breaks[1], 6.0705297))

    def test_multi_start_fitfast(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        my_pwlf.fitfast(4, 50)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))

    # =================================================
    # Start of degree tests
    def test_n_parameters_correct(self):
        my_pwlf_0 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=0)
        my_pwlf_1 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=1)
        my_pwlf_2 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=2)
        breaks = np.array([0.,  1.03913513,  3.04676334,  4.18647526, 10.])

        A0 = my_pwlf_0.assemble_regression_matrix(breaks, self.x_sin)
        A1 = my_pwlf_1.assemble_regression_matrix(breaks, self.x_sin)
        A2 = my_pwlf_2.assemble_regression_matrix(breaks, self.x_sin)
        self.assertTrue(A0.shape[1] == my_pwlf_0.n_parameters)
        self.assertTrue(A1.shape[1] == my_pwlf_1.n_parameters)
        self.assertTrue(A2.shape[1] == my_pwlf_2.n_parameters)
        # Also check n_segments correct
        self.assertTrue(4 == my_pwlf_0.n_segments)
        self.assertTrue(4 == my_pwlf_1.n_segments)
        self.assertTrue(4 == my_pwlf_2.n_segments)

    def test_force_fits_through_points_other_degrees(self):
        # generate sin wave data
        x = np.linspace(0, 10, num=100)
        y = np.sin(x * np.pi / 2)
        # add noise to the data
        y = np.random.normal(0, 0.15, 100) + y

        # linear fit
        my_pwlf_1 = pwlf.PiecewiseLinFit(x, y, degree=1)
        my_pwlf_1.fit(n_segments=6, x_c=[0], y_c=[0])
        y_predict_1 = my_pwlf_1.predict(x)

        # quadratic fit
        my_pwlf_2 = pwlf.PiecewiseLinFit(x, y, degree=2)
        my_pwlf_2.fit(n_segments=5, x_c=[0], y_c=[0])
        y_predict_2 = my_pwlf_2.predict(x)
        self.assertTrue(np.isclose(0, y_predict_1[0]))
        self.assertTrue(np.isclose(0, y_predict_2[0]))

    def test_fitfast(self):
        my_pwlf_0 = pwlf.PiecewiseLinFit(
            self.x_sin, self.y_sin, degree=0, seed=123
        )
        my_pwlf_1 = pwlf.PiecewiseLinFit(
            self.x_sin, self.y_sin, degree=1, seed=123,
        )
        my_pwlf_2 = pwlf.PiecewiseLinFit(
            self.x_sin, self.y_sin, degree=2, seed=123,
        )

        # fit the data for four line segments
        my_pwlf_0.fitfast(4, pop=10)
        my_pwlf_1.fitfast(4, pop=10)
        my_pwlf_2.fitfast(4, pop=10)

        self.assertTrue(my_pwlf_0.ssr <= 35.)
        self.assertTrue(my_pwlf_1.ssr <= 15.)
        self.assertTrue(my_pwlf_2.ssr <= 2.0)

    def test_fit(self):
        my_pwlf_0 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=0)
        my_pwlf_1 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=1)
        my_pwlf_2 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=2)

        # fit the data for four line segments
        np.random.seed(123123)
        my_pwlf_0.fit(5)
        my_pwlf_1.fit(5)
        my_pwlf_2.fit(5)

        self.assertTrue(my_pwlf_0.ssr <= 10.)
        self.assertTrue(my_pwlf_1.ssr <= 7.0)
        self.assertTrue(my_pwlf_2.ssr <= 0.5)

    def test_se_no_fit(self):
        my_pwlf_0 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin)
        try:
            my_pwlf_0.standard_errors()
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

    def test_se_no_method(self):
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        _ = my_fit1.fit_with_breaks(x0)
        try:
            my_fit1.standard_errors(method='blah')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_pv_list(self):
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        _ = my_fit1.fit_with_breaks(x0)
        my_fit1.prediction_variance(list(self.x_small))

    def test_pv_no_fit(self):
        my_pwlf_0 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin)
        try:
            my_pwlf_0.prediction_variance(self.x_sin)
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

    def test_r2_no_fit(self):
        my_pwlf_0 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin)
        try:
            my_pwlf_0.r_squared()
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

    def test_pvalue_no_fit(self):
        my_pwlf_0 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin)
        try:
            my_pwlf_0.p_values()
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

    def test_pvalues_wrong_method(self):
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        _ = my_fit1.fit_with_breaks(x0)
        try:
            my_fit1.p_values(method='blah')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_all_stats(self):
        np.random.seed(121)
        my_pwlf_0 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=0)
        my_pwlf_0.fitfast(3)
        my_pwlf_0.standard_errors()
        my_pwlf_0.prediction_variance(np.random.random(20))
        my_pwlf_0.p_values()
        my_pwlf_0.r_squared()
        my_pwlf_0.calc_slopes()

        my_pwlf_2 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=2)
        my_pwlf_2.fitfast(3)
        my_pwlf_2.standard_errors()
        my_pwlf_2.prediction_variance(np.random.random(20))
        my_pwlf_2.p_values()
        my_pwlf_2.r_squared()
        my_pwlf_2.calc_slopes()

        my_pwlf_3 = pwlf.PiecewiseLinFit(self.x_sin, self.y_sin, degree=3)
        my_pwlf_3.fitfast(3)
        my_pwlf_3.standard_errors()
        my_pwlf_3.prediction_variance(np.random.random(20))
        my_pwlf_3.p_values()
        my_pwlf_3.r_squared()
        my_pwlf_3.calc_slopes()
    # End of degree tests
    # =================================================

    # =================================================
    # Start weighted least squares tests
    def test_weighted_same_as_ols(self):
        # test that weighted least squares is same as OLS
        # when the weight is equal to 1.0
        n_segments = 2
        my = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x = np.random.random()
        breaks = my.fit_guess([x])
        my_w = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                    weights=np.ones_like(self.x_small))
        breaks_w = my_w.fit_guess([x])

        self.assertTrue(np.isclose(my.ssr, my_w.ssr))
        for i in range(n_segments+1):
            self.assertTrue(np.isclose(breaks[i], breaks_w[i]))

    def test_heteroscedastic_data(self):
        n_segments = 3
        weights = self.y_small.copy()
        weights[0] = 0.01
        weights = 1.0 / weights
        my_w = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                    weights=weights)
        _ = my_w.fit(n_segments)
        _ = my_w.standard_errors()

    def test_not_supported_fit(self):
        x = np.linspace(0.0, 1.0, num=100)
        y = np.sin(6.0*x)
        w = np.random.random(size=100)
        my_fit = pwlf.PiecewiseLinFit(x, y, disp_res=True, weights=w)
        x_c = [0.0]
        y_c = [0.0]
        try:
            my_fit.fit(3, x_c, y_c)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_not_supported_fit_with_breaks_force_points(self):
        x = np.linspace(0.0, 1.0, num=100)
        y = np.sin(6.0*x)
        w = list(np.random.random(size=100))
        my_fit = pwlf.PiecewiseLinFit(x, y, disp_res=True, weights=w)
        x_c = [0.0]
        y_c = [0.0]
        try:
            my_fit.fit_with_breaks_force_points([0.1, 0.2, 0.3], x_c, y_c)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_custom_opt_not_supported(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                       weights=self.y_small)
        try:
            my_pwlf.use_custom_opt(3, x_c=[0], y_c=[0])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_random_seed_fit(self):
        np.random.seed(1)
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                       seed=123)
        fit1 = my_pwlf.fit(2)
        np.random.seed(2)
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                       seed=123)
        fit2 = my_pwlf.fit(2)
        same_breaks = np.isclose(fit1, fit2)
        self.assertTrue(same_breaks.sum() == same_breaks.size)

    def test_random_seed_fitfast(self):
        # specifically test for seed = 0
        np.random.seed(1)
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                       seed=0)
        fit1 = my_pwlf.fitfast(2)
        np.random.seed(2)
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small,
                                       seed=0)
        fit2 = my_pwlf.fitfast(2)
        same_breaks = np.isclose(fit1, fit2)
        self.assertTrue(same_breaks.sum() == same_breaks.size)

    def test_one_segment_fits(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        fit1 = my_pwlf.fitfast(1)
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        fit2 = my_pwlf.fit(1)
        same_breaks = np.isclose(fit1, fit2)
        self.assertTrue(same_breaks[0])
        self.assertTrue(same_breaks[1])

    def test_float32(self):
        my_pwlf = pwlf.PiecewiseLinFit(
            np.linspace(0, 10, 3, dtype=np.float32),
            np.random.random(3).astype(np.float32),
        )
        _ = my_pwlf.fitfast(2)
        self.assertTrue(True)

    def test_lfloat128(self):
        try:
            x = np.linspace(0, 10, 3, dtype=np.float128)
            y = np.random.random(3).astype(np.float128)
            my_pwlf = pwlf.PiecewiseLinFit(x, y)
            _ = my_pwlf.fitfast(2)
            self.assertTrue(True)
        except AttributeError:
            # this means that float128 is not supported
            self.assertTrue(True)

    def test_mixed_degree1(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 2, 3, 4, 4.25, 3.75, 4, 5, 6, 7]

        degree_list = [1, 0, 1]
        my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree_list)
        _ = my_pwlf.fit(3)

        # generate predictions
        x_hat = np.linspace(min(x), max(x), 1000)
        _ = my_pwlf.predict(x_hat)

    def test_mixed_degree2(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 2, 3, 4, 4.25, 3.75, 4, 5, 6, 7]

        degree_list = [1, 1, 1]
        my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree_list)
        _ = my_pwlf.fit(3)

        # generate predictions
        x_hat = np.linspace(min(x), max(x), 1000)
        _ = my_pwlf.predict(x_hat)

    def test_mixed_degree3(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 2, 3, 4, 4.25, 3.75, 4, 5, 6, 7]

        degree_list = [1, 3]
        try:
            my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree_list)
            _ = my_pwlf.fit(3)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_mixed_degree_wrong_list(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 2, 3, 4, 4.25, 3.75, 4, 5, 6, 7]
        degree_list = [1, 1]
        my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree_list)
        try:
            _ = my_pwlf.fit(3)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        try:
            _ = my_pwlf.fitfast(3)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        try:
            _ = my_pwlf.fit_with_breaks([0, 3, 4, 5])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_mixed_degree_force_point(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 2, 3, 4, 4.25, 3.75, 4, 5, 6, 7]
        degree_list = [1, 1]
        my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree_list)
        try:
            _ = my_pwlf.fit(2, x_c=[0,], y_c=[0,])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
