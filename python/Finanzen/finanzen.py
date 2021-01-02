import requests
from datetime import date
import time
from matplotlib import pyplot as plt
import numpy as np


class KURSNAME:
    dax = "index/dax"
    ftse_100 = "index/ftse_100"
    concentra = "fonds/concentra"
    industria = "fonds/industria"
    kapitalplus = "fonds/kapital_plus"


def getKURS(name):
    page = requests.get('https://www.finanzen.net/' + name)
    content = page.text
    b = content.find('<div class="col-xs-5 col-sm-4 text-sm-right text-nowrap">')
    kurs = []
    for i in range(b + 59, b + 70):
        kurs.append(content[i])
    if "." in kurs:
        kurs.remove('.')
    kurs = ''.join(kurs).replace(',', '.')
    kurs = kurs.split(".")[0]
    kommastellen = []
    try:
        for i in range(b + 61 + len(kurs), b + 63 + len(kurs)):
            kommastellen.append(content[i])
        kommastellen = ''.join(kommastellen)
        kommastellen = float(kommastellen)

    except ValueError:
        kommastellen = []
        for i in range(b + 60 + len(kurs), b + 62 + len(kurs)):
            kommastellen.append(content[i])
        kommastellen = ''.join(kommastellen)
        kommastellen = float(kommastellen)

    kurs = float(kurs) + kommastellen / 100
    return kurs


def load():
    try:
        f = open("anteile.txt", "r")
        content = f.read()
        content = content.split("\n")
        c = float(content[0])
        i = float(content[1])
        k = float(content[2])
        return c, i, k
    except FileNotFoundError:
        pass


def save():
    if getWeekday() != "Sonntag" and getWeekday() != "Samstag":
        if 8 <= getHour() <= 22:
            f = open("bilanz.txt", "a+")
            w = open("C:/xampp/htdocs/index.html", "w+")
            f.write("\n" + str(SummeB))
            w.write('<img src="plot.svg">')
            f.close()
            w.close()


def getWeekday():
    return ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")[date.today().weekday()]


def getHour():
    return int(time.strftime("%H"))


concentra = getKURS(KURSNAME.concentra)
industria = getKURS(KURSNAME.industria)
kapitalplus = getKURS(KURSNAME.kapitalplus)

concentraMenge = load()[0]
industriaMenge = load()[1]
kapitalplusMenge = load()[2]

concentraGeld = concentra * concentraMenge
industriaGeld = industria * industriaMenge
kapitalplusGeld = kapitalplus * kapitalplusMenge

SummeB = concentraGeld + industriaGeld + kapitalplusGeld
SummeB = format(SummeB, '.2f')

print("Werte nach Börse Stuttgart")
print("Concentra A:  ", concentra, "€ pro Stück x", concentraMenge, " Stück  = ", concentraGeld.__round__(2), "€")
print("Industria A:  ", industria, " € pro Stück x", industriaMenge, " Stück  = ", industriaGeld.__round__(2), "€")
print("KaptialPlus A:", kapitalplus, " € pro Stück x", kapitalplusMenge, "Stück  = ", kapitalplusGeld.__round__(2), "€")
print("\n")

print("Gesamtvermögen :", SummeB, "€")

try:
    import avg as Average
except ModuleNotFoundError:
    import Finanzen.avg as Average
f = open("bilanz.txt", "r")
content = f.read()
f.close()
content = content.split("\n")
amount = len(content)
money = []
for i in range(0, amount):
    money.append(float(content[i]))

Average.calcAvg()
a = open("avgs.txt", "r")
content2 = a.read()
a.close()
content2 = content2.split("\n")
amount2 = len(content2)
avgs = []
for i in range(0, amount2):
    avgs.append(float(content2[i]))

max, min, avg = [], [], []
maxmoney = np.amax(money)
minmoney = np.amin(money)
avgmoney = round((maxmoney + minmoney) / 2, 2)
aktuell = money[len(money)-1]
for i in range(0, amount):
    max.append(maxmoney)
    min.append(minmoney)
    avg.append(avgmoney)
plt.plot(money)
plt.ylabel("Kapital")
plt.plot(max)
plt.plot(min)
plt.plot(avg)
plt.plot(avgs)
plt.suptitle(SummeB)
plt.savefig("C:/xampp/htdocs/plot.svg")
save()