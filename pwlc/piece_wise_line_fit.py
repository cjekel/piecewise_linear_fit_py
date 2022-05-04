import numpy as np
import matplotlib.pyplot as plt
import pwlf
import pickle
import float_hex
import sigmoid
from float2bi import decimal2hex
# from GPyOpt.methods import BayesianOptimization
# from sigmoid import sigmoid_11lines
# from sigmoid import sigmoid
from scipy import optimize

sample_num = 1601
# x = np.arange(-8, 8.1, 0.1)
x = np.linspace(-8, 8, num=sample_num, endpoint=True)
y = 1/(1+np.exp(-x))

# 根据指定断点拟合函数
# x0 = np.array([min(x), -4, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 4, max(x)])
# segnum = 15
# x0 = np.array([min(x), -4.5, -3, -2.5, -2, -1.5, -1, 1, 1.5, 2, 2.5, 3, 4.5, max(x)])
# segnum = 13
# x0 = np.array([min(x), -4.5, -3, -2.5, -2, -1, 1, 2, 2.5, 3, 4.5, max(x)])
# segnum = 11
x0 = np.array([min(x), -4.5, -3, -2, -1, 1, 2, 3, 4.5, max(x)])
segnum = 9
# x0 = np.array([min(x), -4, -3, -2, -1, 1, 2, 3, 4, max(x)])
# segnum = 9
# x0 = np.array([min(x), -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, max(x)])
# segnum = 9
# x0 = np.array([min(x), -4.5, -3, -1.5, -1, 1, 1.5, 3, 4.5, max(x)])
# segnum = 9
# x0 = np.array([min(x), -4.5, -2.5, -1, -0.5, 0.5, 1, 2.5, 4.5, max(x)])
# segnum = 9
# x0 = np.array([min(x), -4, -2, -1, 1, 2, 4, max(x)])
# segnum = 7
# x0 = np.array([min(x), -3, -1, 1, 3, max(x)])
# segnum = 5
# x0 = np.array([min(x), -2, 2, max(x)])
# segnum = 3
# x0 = np.array([min(x), -1.6, 1.6, max(x)])
# segnum = 3

# 根据自变量和因变量初始化分段函数
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# 根据指定断点拟合分段函数
my_pwlf.fit_with_breaks(x0)
# print(my_pwlf.beta)

# 根据分段点数量均匀地生成分段点
xHat = np.linspace(min(x), max(x), num=sample_num)

# 读取np格式，字符型数组不能直接转数字数组
# yHat = np.loadtxt("sigmoidout.txt", dtype=str)
# for n, ele in enumerate(yHat):
#     yHat[n] = np.array(float(float_hex.hex_to_float(str(ele))))
# yHat = np.array(list(map(float, yHat)))  
# print(yHat)

# 读取list格式
# with open('sigmoidout.txt', 'r') as f:
#     yHat = f.readlines()  # txt中所有字符串读入data，list类型
# n = 0
# for line in yHat:
#     yHat[n] = (float_hex.hex_to_float(str(line)))
#     n += 1
# yHat = list(map(float, yHat))  # 转化为浮点数

# xHat = xHat.astype(np.float32)
# yHat = sigmoid.sigmoid_11hardlines(xHat)
yHat = my_pwlf.predict(xHat)
# print(xHat.dtype)
# print(yHat.dtype)

# 按照区间分段从左到右打印斜率、截距和端点横坐标
for index in range(0, segnum):
    # funcSlope = my_pwlf.slopes[index]
    # yIntercept = my_pwlf.slopes[index]*(-my_pwlf.fit_breaks[index]) + my_pwlf.predict(my_pwlf.fit_breaks[index])
    print(float_hex.float_to_hex(my_pwlf.slopes[index]), '; //', my_pwlf.slopes[index], '\t',
          # float_hex.float_to_hex(yIntercept[0]), '; //', yIntercept[0], '\t',
          float_hex.float_to_hex(my_pwlf.intercepts[index]), '; //', my_pwlf.intercepts[index], '\t',
          float_hex.float_to_hex(my_pwlf.fit_breaks[index]), '; //', my_pwlf.fit_breaks[index], '\t')

# 打印python分段函数代码
# for index in range(segnum-1,-1,-1):
#     print('elif z[index0][index1] > '+str(my_pwlf.fit_breaks[index])+':')
#     print('\tout[index0][index1] =  ' + str(my_pwlf.slopes[index]) + ' * z[index0][index1] + ' + str(my_pwlf.intercepts[index]))

# # 打印verilog分段函数斜率截距值
# for index in range(0, segnum):
#     print('parameter signed k'+str(index+1)+' = 16\'h'+(decimal2hex(my_pwlf.slopes[index])) + '; //' + str(my_pwlf.slopes[index]))
#     print('parameter signed t'+str(index+1)+' = 16\'h'+(decimal2hex(my_pwlf.intercepts[index])) + '; //' + str(my_pwlf.intercepts[index]))

# # 打印分段函数latex表达式
# for index in range(0,segnum):
#     print(str(float('%.5f' % my_pwlf.slopes[index])) + ' * x + ' + str(float('%.5f' % my_pwlf.intercepts[index]))+', & '+
#           str('{:g}'.format(my_pwlf.fit_breaks[index])) + '<x\leq ' + str('{:g}'.format(my_pwlf.fit_breaks[index+1]))+' \\'+'\\ ')

# # # 打印beta数值
# for index in range(0, segnum):
#     # print(float('%.5f' % my_pwlf.slopes[index]), str('* x +'), float('%.5f' % my_pwlf.intercepts[index]))
#     print(my_pwlf.beta[index])

# 打印分段点的估计值
# for index in range(0, segnum):
#     print(float('%.5f' % my_pwlf.predict(my_pwlf.fit_breaks[index])), float('%.5f' % my_pwlf.predict(my_pwlf.fit_breaks[index+1])))


# 打印绝对误差的最大值和平均值
def ssr(x):
    return abs(1 / (1 + np.exp(-x))-my_pwlf.predict(x)[0])
maxp = optimize.fminbound(lambda x: -ssr(x), 0, 8, xtol=1e-10, maxfun=5000)
print(maxp, np.argmax(abs(yHat-y))/100-8, ssr(-1))
print(ssr(maxp))
print(max(abs(yHat-y)))
print(np.average(abs(yHat-y)))
# print(my_pwlf.predict(my_pwlf.fit_breaks))

# 打印拟合图像
plt.figure()
plt.plot(x, y, '--', linewidth=2.4)
plt.plot(xHat, yHat, '-', markersize=1)
for index in range(0, segnum+1):
    plt.plot(my_pwlf.fit_breaks[index], my_pwlf.predict(my_pwlf.fit_breaks[index]), '^', color='green', markersize=6)
plt.legend((u'sigmoid(x)', u'piece-wise-line-fit(x)'),
        loc='upper left')

# 保存图片到本目录中
# plt.savefig('sigmoid.pdf', format='pdf', bbox_inches='tight', pad_inches=0.0)
# plt.savefig('sigmoid.png', format='png', bbox_inches='tight', pad_inches=0.0, dpi=800)
# plt.show()


# 保存模型参数
# my_pwlf.fit_with_breaks(x)
# save the fitted model
# with open('sigmoid'+str(segnum)+'segments.pkl', 'wb') as f:
#     pickle.dump(my_pwlf, f, pickle.HIGHEST_PROTOCOL)
# # load the fitted model
# with open('sigmoid'+str(segnum)+'segments.pkl', 'rb') as f:
#     my_pwlf = pickle.load(f)

# for index in range(0, segnum):
#     print(float_hex.float_to_hex(my_pwlf.slopes[index]), '; //', my_pwlf.slopes[index])
#
# for index in range(0, segnum):
#     print(float_hex.float_to_hex(my_pwlf.intercepts[index]), '; //', my_pwlf.intercepts[index])
#
# print(my_pwlf.fit_breaks)

