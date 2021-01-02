from matplotlib import pyplot as plt
import numpy as np
from random import *
def exsits(Obj):
    return Obj is not None

NichtDa = None

Schatz = NichtDa
print("\n\nSchatz:", Schatz)
if exsits(Schatz):
    print("Ich bin")
else :
    print("Ich bin nicht")

print("\nGib den Name deines Schatzes ein")


Schatz = input()
print("Schatz:", Schatz)

if exsits(Schatz):
    if Schatz == "Julia" or Schatz == "Moritz":
        print("Ich bin")
    elif Schatz != "Julia":
        print("Ich bin nicht")
else:
    print("Ich bin nicht")
