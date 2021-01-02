import numpy as np


def f(x):
    return x * x


def pi(iterations):
    a = 1
    b = 1 / (np.sqrt(2))
    s = 0.25
    for i in range(0, iterations):
        A = (a + b) / 2
        B = np.sqrt(a * b)
        S = s - pow(2, i) * (a - A) * (a - A)
        a = A
        b = B
        s = S
    return (A * A) / S


def integralRECT(func, a, b):
    genauigkeit = int(1E+6)
    sum = 0
    delta_x = np.abs(b - a) / genauigkeit
    x = a
    for i in range(0, genauigkeit + 1):
        sum += delta_x * func(x)
        x += delta_x
        if i % (genauigkeit / 1000) == 0:
            print(i / (genauigkeit / 100), "%")
            for _ in range(1, 11):
                print(chr(27) + "[2J")
    return sum


def integralRECT2(func, a, b, l):
    genauigkeit = int(1E+3)
    sum = 0
    delta_x = np.abs(b - a) / genauigkeit
    x = a
    for i in range(0, genauigkeit + 1):
        sum += delta_x * func(x, l)
        x += delta_x
        if i % (genauigkeit / 1000) == 0:
            print(i / (genauigkeit / 100), "%")
            for _ in range(1, 11):
                print(chr(27) + "[2J")
    return sum


def integralSIMPSON(func, a, b):
    genauigkeit = int(1E+5)
    ret1 = (b - a) / genauigkeit
    ret2 = 0.5 * func(a)
    ret3 = 0
    ret4 = 0.5 * func(b)
    for k in range(1, genauigkeit):
        ret3 += func(a + k * ((b - a) / (genauigkeit)))
        if k % (genauigkeit / 1000) == 0:
            print(k / (genauigkeit / 100), "%")
            for _ in range(1, 11):
                print(chr(27) + "[2J")
    return ret1 * (ret2 + int(ret3) + ret4)


def derivative(func, x):
    genauigkeit = int(1E+6)
    h = 1 / genauigkeit
    div = (func(x + h) - f(x)) / h
    return div


def kapital(k, q, n, r):
    return (k * np.power(q, n)) + (r * q * ((np.power(q, n) - 1) / (q - 1)))


a = np.matrix([[1,5],[3,4]])
b = np.matrix([[2,4],[4,9]])
print(a)