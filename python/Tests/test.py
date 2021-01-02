"""from math import *



def isPrime(number):
    if number <= 1: return False
    if number == 2: return True
    if number % 2 == 0: return False
    for i in range(2, ceil(sqrt(number)) + 1):
        if number % i == 0: return False
    return True


def factorize(number):
    primes = [i for i in range(0, number + 1) if isPrime(i)]
    factors = [i for i in primes if number % i == 0]
    dummy = []
    return factors



for i in range(10000):
    print(factorize(i), i)
"""

import numpy as np


def main3():
    x = "Love"
    print("    x   x\nx x x x\nx   x    x".replace("x", x))
    for i in range(5):
        print(" " * i + x + " " * (9 - i * 2), x)
    print(" " * 6, x)
main3()