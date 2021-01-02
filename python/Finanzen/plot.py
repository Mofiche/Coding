from matplotlib import pyplot as plt
import numpy as np

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
print("Max : ", maxmoney, "\nMin : ", minmoney, "\nAvg : ", avgmoney,"\nAvg2: ",round(avgs[len(avgs)-1],2),  "\nAkt : ", aktuell)
plt.show()


