def sumarray(array):
    ret = 0
    for i in array:
        ret += i
    return ret


def calcAvg():
    f = open("bilanz.txt", "r")
    w = open("avgs.txt", "w")
    content = f.read()
    f.close()
    content = content.split("\n")
    amount = len(content)
    money = []
    avg = []
    for i in range(0, amount):
        money.append(float(content[i]))
        avg.append(sumarray(money) / (i + 1))

    a = True
    for i in avg:
        if a:
            w.write(str(i))
            a = False
        else:
            w.write("\n" + str(i))
    w.close()
