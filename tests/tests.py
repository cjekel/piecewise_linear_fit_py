import numpy as np
import unittest
import pwlf


class TestEverything(unittest.TestCase):
    # let's just test all of my use cases...
    # def __init__(self):
    x_small = np.array((0.0, 1.0, 1.5, 2.0))
    y_small = np.array((0.0, 1.0, 1.1, 1.5))

    # def test_is_five_prime(self):
    #     """Is five successfully determined to be prime?"""
    #     self.assertTrue(is_prime(5))

    def test_break_point_spot_on(self):
        # check that I can fit when break points spot on a
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
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

    def test_break_point_spot_on_r2(self):
        # test r squared value with known solution
        my_fit1 = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        ssr = my_fit1.fit_with_breaks(x0)
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
        res = my_pwlf.fit(4, disp=False)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))

    def test_predict(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        xmax = np.max(self.x_small)
        xmin = np.min(self.x_small)
        res = my_pwlf.fit_with_breaks((xmin, xmax))
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

    def test_single_force_break_point1(self):
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x_c = [-0.5]
        y_c = [-0.5]
        ssr = my_fit.fit_with_breaks_force_points([0.2, 0.7], x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_single_force_break_point2(self):
        my_fit = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        x_c = [2.0]
        y_c = [1.5]
        ssr = my_fit.fit_with_breaks_force_points([0.2, 0.7], x_c, y_c)
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
        res = my_fit.fit(3, x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_multi_start_fitfast(self):
        print('Last test! - multi start (fitfast) test')
        my_pwlf = pwlf.PiecewiseLinFit(self.x_small, self.y_small)
        res = my_pwlf.fitfast(4, 50)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))

    def test_se(self):
        # check to see if it will let me calculate standard errors
        my_pwlf = pwlf.PiecewiseLinFit(np.random.random(20),
                                       np.random.random(20))
        ssr = my_pwlf.fitfast(2)
        se = my_pwlf.standard_errors()

    def test_p(self):
        # check to see if it will let me calculate p-values
        my_pwlf = pwlf.PiecewiseLinFit(np.random.random(20),
                                       np.random.random(20))
        ssr = my_pwlf.fitfast(2)
        p = my_pwlf.p_values()

    def test_pv(self):
        # check to see if it will let me calculate prediction variance for
        # random data
        my_pwlf = pwlf.PiecewiseLinFit(np.random.random(20),
                                       np.random.random(20))
        ssr = my_pwlf.fitfast(2)
        pv = my_pwlf.prediction_variance(np.random.random(20))

    def test_predict_with_custom_param(self):
        # check to see if predict runs with custom parameters
        x = np.random.random(20)
        my_pwlf = pwlf.PiecewiseLinFit(x, np.random.random(20))
        my_pwlf.predict(x, beta=np.array((1e-4, 1e-2, 1e-3)),
                        breaks=np.array((0.0, 0.5, 1.0)))


if __name__ == '__main__':
    unittest.main()
