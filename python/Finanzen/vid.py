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
vid = []
for j in range(7563):
    vid.append(money[j])
    print(round(j/7563,2))
for j in range(7563, len(money)):
    vid.append(money[j])
    Average.calcAvg()
    a = open("avgs.txt", "r")
    content2 = a.read()
    a.close()
    content2 = content2.split("\n")
    amount2 = len(content2)
    avgs = []
    for i in range(0, j+1):
        avgs.append(float(content2[i]))
    max, min, avg = [], [], []
    maxmoney = np.amax(vid)
    minmoney = np.amin(vid)
    avgmoney = round((maxmoney + minmoney) / 2, 2)
    aktuell = money[len(vid) - 1]
    for i in range(0, j + 1):
        max.append(maxmoney)
        min.append(minmoney)
        avg.append(avgmoney)
    plt.plot(vid, color="blue")
    plt.ylabel("Kapital")
    plt.plot(max, color="orange")
    plt.plot(min, color="green")
    plt.plot(avg, color="red")
    plt.plot(avgs, color="violet")
    #print("Max : ", maxmoney, "\nMin : ", minmoney, "\nAvg : ", avgmoney, "\nAvg2: ", round(avgs[len(avgs) - 1], 2),"\nAkt : ", aktuell)
    if j < 10:
        plt.savefig("vids/000%s.png" % j)
    elif j < 100:
        plt.savefig("vids/00%s.png" % j)
    elif j < 1000:
        plt.savefig("vids/0%s.png" % j)
    elif j < 10000:
        plt.savefig("vids/%s.png" % j)
    plt.close()
    print(j)

