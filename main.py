import math

import matplotlib.pyplot as plt
import numpy as np

print("Введите промежуток (x1, x2) через пробел")
x1, x2 = 0.0, 1.0
y1, y2 = 1.0, 2.0
n = 5
h = (x2 - x1) / n
D0, D1 = y1 + h, h

y = [[y1, D0], [0, D1]]


def p(x):   return 1


def q(x):   return 1


def f(x):   return 3 * (np.e ** x)


def get_c1():
    global n
    return (y2 - y[0][n]) / y[1][n]


def get_solv_y_i(i): return y[0][i] + get_c1() * y[1][i]


x = np.linspace(x1, x2, n + 1)


def div(a, b):
    return a / b


for i in range(1, n):
    y[0].append(
        div(
            (math.pow(h, 2) * f(x[i]) - (1.0 - (h / 2) * p(x[i])) * y[0][i - 1] - (h ** 2 * q(x[i]) - 2) * y[0][i]),
            1 + h / 2 * p(x[i])
        )
    )
    y[1].append(
        div(
            -(1 - h / 2 * p(x[i])) * y[1][i - 1] - (h ** 2 * q(x[i]) - 2) * y[1][i],
            1 + h / 2 * p(x[i])
        )
    )

plt.plot(x, [get_solv_y_i(i) for i in range(n + 1)])
plt.show()

for i in range(n):
    print(x[i], get_solv_y_i(i))
