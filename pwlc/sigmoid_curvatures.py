import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sympy import *
from scipy import optimize
import numpy.linalg as LA

# 获取sigmoid函数、导数、曲率的图像
sample_num = 1601
x = np.linspace(-8, 8, num=sample_num, endpoint=True)
y = 1/(1+np.exp(-x))
z = np.exp(-x)/pow(1+np.exp(-x), 2)
c = abs((np.exp(-2*x)-np.exp(-x))/(1+np.exp(-x))**3)/(1+np.exp(-x)**2/((1+np.exp(-x))**4))**1.5
x_major_locator=MultipleLocator(1)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.1)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
# 把y轴的主刻度设置为10的倍数
plt.xlim(-8.5, 8)
plt.ylim(0, 1)
plt.plot(x, y, '-', linewidth=1)
plt.plot(x, z, '--', markersize=1)
plt.plot(x, c, '-.', markersize=1)
plt.grid(axis='y', linestyle='--')
plt.legend((u'sigmoid(x)', u'derivative of sigmoid(x)', u'curvature of sigmoid(x)'),
           loc='upper left')
plt.rcParams.update({'font.size': 16})
# 保存sigmoid函数、导数、曲率的图像到本目录中
# plt.savefig('sigmoiddc.pdf', format='pdf', bbox_inches='tight', pad_inches=0.0)
# plt.savefig('sigmoiddc.png', format='png', bbox_inches='tight', pad_inches=0.0, dpi=800)
# plt.show()

# # 获取指定导数分度值对应的横坐标
# x = symbols('x')
# length = 6
# points = np.zeros(length*2-4)
# for slope in np.linspace(0, 0.25, num=length, endpoint=True):
#     if slope == 0 or slope == 0.25:
#         continue
#     points[int(slope/0.25*(length-1)-1)] = solve(exp(-x)/((1+exp(-x))**2)-slope, x)[0]
#     points[int(length*2-4-slope/0.25*(length-1))] = solve(exp(-x)/((1+exp(-x))**2)-slope, x)[1]
# print(np.round(points, 2).tolist())

# 获取sigmoid函数曲率的最大值及其坐标
def ka(x):
    return abs((np.exp(-2*x)-np.exp(-x))/(1+np.exp(-x))**3)/(1+np.exp(-x)**2/((1+np.exp(-x))**4))**1.5
maxp = optimize.fminbound(lambda x: -ka(x), 0, 8, xtol=1e-10, maxfun=5000)
print(maxp, ka(maxp))
# x = symbols('x')
# print(solve(((exp(-2*x)-exp(-x))/(1+exp(-x))**3)/(1+exp(-x)**2/((1+exp(-x))**4))**1.5-0.05, x))

# 获取sigmoid函数中各点对应的曲率
kaeg = []
samples = np.linspace(0, 6.0, num=12, endpoint=False)
for index in samples:
    kaeg.append(ka(index))
print("\t".join(str(i) for i in samples))
kaegf = [float('{:.3f}'.format(i)) for i in kaeg]
print("\t".join(str(i) for i in kaegf))
