import numpy as np
import unittest
import pwlf
import tensorflow as tf
import os


class TestEverything(unittest.TestCase):
    # let's just test all of my use cases...
    # def __init__(self):
    x_small = np.array((0.0, 1.0, 1.5, 2.0))
    y_small = np.array((0.0, 1.0, 1.1, 1.5))
    xk = np.linspace(0, 1, num=10)
    yk = 2.0*xk + -.5
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36,
                  112.25, 126.14, 140.03])

    def test_break_point_spot_on(self):
        # check that I can fit when break points spot on a
        my_fit1 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        ssr = my_fit1.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr, 0.0))

    def test_fast_false(self):
        # check that I can fit when break points spot on a
        my_fit1 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small,
                                         fast=False)
        x0 = self.x_small.copy()
        ssr = my_fit1.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr, 0.0))

    def test_single_precision(self):
        # check that I can fit when break points spot on a
        my_fit1 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small,
                                         dtype='float32')
        x0 = self.x_small.copy()
        ssr = my_fit1.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr, 0.0))

    def test_assembly(self):
        # check that I can fit when break points spot on a
        my_fit = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        A = my_fit.assemble_regression_matrix(x0, my_fit.x_data)
        Asb = np.array([[1.,  0.,  0.,  0.],
                        [1.,  x0[1]-x0[0],  0.,  0.],
                        [1.,  x0[2]-x0[0], x0[2]-x0[1], 0.],
                        [1.,  x0[3]-x0[0], x0[3]-x0[1], x0[3]-x0[2]]])
        with tf.Session():
            A = A.eval()
        self.assertTrue(np.allclose(A, Asb))

    def test_break_point_spot_on_r2(self):
        # test r squared value with known solution
        my_fit1 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        my_fit1.fit_with_breaks(x0)
        rsq = my_fit1.r_squared()
        self.assertTrue(np.isclose(rsq, 1.0))

    def test_break_point_diff_x0_0(self):
        # check diff loc
        my_fit2 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0 = self.x_small.copy()
        x0[1] = 1.00001
        x0[2] = 1.50001
        ssr2 = my_fit2.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr2, 0.0))

    def test_break_point_diff_x0_1(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit3 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0[1] = 0.9
        ssr3 = my_fit3.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr3, 0.0))

    def test_break_point_diff_x0_2(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit4 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0[1] = 1.1
        ssr4 = my_fit4.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr4, 0.0))

    def test_break_point_diff_x0_3(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit5 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0[2] = 1.6
        ssr5 = my_fit5.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr5, 0.0))

    def test_break_point_diff_x0_4(self):
        # check if my duplicate is in a different location
        x0 = self.x_small.copy()
        my_fit6 = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x0[2] = 1.4
        ssr6 = my_fit6.fit_with_breaks(x0)
        self.assertTrue(np.isclose(ssr6, 0.0))

    def test_diff_evo(self):
        my_pwlf = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        np.random.seed(12)
        my_pwlf.fit(3)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))

    def test_predict(self):
        my_pwlf = pwlf.PiecewiseLinFitTF(self.xk, self.yk)
        my_pwlf.fit_with_breaks([0.5])
        x = np.linspace(self.xk.min(), self.xk.max(), 10)
        yHat = my_pwlf.predict(x)
        e = np.sum(np.abs(yHat - self.yk))
        self.assertTrue(np.isclose(e, 0.0))

    def test_custom_opt(self):
        my_pwlf = pwlf.PiecewiseLinFitTF(self.xk, self.yk)
        my_pwlf.use_custom_opt(3)
        x_guess = np.array((0.7, 0.9))
        from scipy.optimize import minimize
        with tf.Session():
            res = minimize(my_pwlf.fit_with_breaks_opt, x_guess)
        self.assertTrue(np.isclose(res['fun'], 0.0))

    def test_single_force_break_point1(self):
        my_fit = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x_c = [-0.5]
        y_c = [-0.5]
        my_fit.fit_with_breaks_force_points([0.2, 1.0], x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_single_force_break_point2(self):
        my_fit = pwlf.PiecewiseLinFitTF(self.x_small, self.y_small)
        x_c = [2.0]
        y_c = [1.5]
        my_fit.fit_with_breaks_force_points([0.2, 1.0], x_c, y_c)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_opt_fit_with_force_points(self):
        # I need more data points to test this function because of
        # ill conditioning in the least squares problem...
        x = np.linspace(0.0, 1.0, num=100)
        y = np.sin(6.0*x)
        my_fit = pwlf.PiecewiseLinFitTF(x, y, disp_res=True)
        np.random.seed(1231)
        x_c = [0.0]
        y_c = [0.0]
        my_fit.fit(3, x_c, y_c, popsize=2, maxiter=2, disp=False)
        yhat = my_fit.predict(x_c)
        self.assertTrue(np.isclose(y_c, yhat))

    def test_se(self):
        # check to see if it will let me calculate standard errors
        my_pwlf = pwlf.PiecewiseLinFitTF(np.linspace(0, 100, num=100),
                                         np.linspace(0, 100, num=100))
        my_pwlf.fitfast(2)
        my_pwlf.standard_errors()
        self.assertTrue(True)

    def test_p(self):
        # check to see if it will let me calculate p-values
        my_pwlf = pwlf.PiecewiseLinFitTF(np.linspace(0, 100, num=100),
                                         np.linspace(0, 100, num=100))
        my_pwlf.fitfast(2)
        my_pwlf.p_values()
        self.assertTrue(True)

    def test_pv(self):
        # check to see if it will let me calculate prediction variance for
        # random data
        my_pwlf = pwlf.PiecewiseLinFitTF(np.linspace(0, 100, num=100),
                                         np.linspace(0, 100, num=100))
        my_pwlf.fitfast(2)
        my_pwlf.prediction_variance(np.random.random(20))
        self.assertTrue(True)

    def test_predict_with_custom_param(self):
        # check to see if predict runs with custom parameters
        x = np.random.random(20)
        my_pwlf = pwlf.PiecewiseLinFitTF(x, np.random.random(20))
        my_pwlf.predict(x, beta=np.array((1e-4, 1e-2, 1e-3)),
                        breaks=np.array((0.0, 0.5, 1.0)))
        self.assertTrue(True)

    def test_fit_guess(self):
        # x = np.array([4., 5., 6., 7., 8.])
        # y = np.array([11., 13., 16., 28.92, 42.81])
        my_pwlf = pwlf.PiecewiseLinFitTF(self.x, self.y)
        breaks = my_pwlf.fit_guess([6.0])
        self.assertTrue(np.isclose(breaks[1],  5.99819559))

    def test_fit_guess_kwrds(self):
        my_pwlf = pwlf.PiecewiseLinFitTF(self.x, self.y)
        breaks = my_pwlf.fit_guess([6.0], m=10,
                                   factr=1e2, pgtol=1e-05,
                                   epsilon=1e-6, iprint=-1,
                                   maxfun=1500000, maxiter=150000,
                                   disp=None)
        self.assertTrue(np.isclose(breaks[1],  5.99819559))

    def test_multi_start_fitfast(self):
        my_pwlf = pwlf.PiecewiseLinFitTF(self.xk, self.yk)
        my_pwlf.fitfast(2)
        self.assertTrue(np.isclose(my_pwlf.ssr, 0.0))


if __name__ == '__main__':
    # force TF to use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    unittest.main()
