import socket
from numpy import *


def isPrime(number):
    if number > 1:
        if number == 2:
            return True
        if number % 2 == 0:
            return False
        for i in range(3, int(sqrt(number) + 1), 2):
            if number % i == 0:
                return False
        return True
    return False

poss_primes = []
for i in range(1, int(1E7)):
    poss_primes.append(i) if isPrime(i) else None
    if i % (1E7 /100) == 0: print(str(i/((1E7 /100))) + "%")


def getPrimeFactors(number):
    factors = []
    if isPrime(number):
        return number
    for i in poss_primes:
        factors.append(i) if number % i == 0 else None
    return factors


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.2.104', 1337))
print("Server ist gestartet")
server_socket.listen(1)
while True:
    (client_socket, addr) = server_socket.accept()
    msg = str(client_socket.recv(2 ** 10), 'utf8')
    antwort = bytes(str(getPrimeFactors(int(msg))), 'utf8')
    print(msg, str(antwort, 'utf8'))
    try:
        client_socket.send(antwort)
    except:
        pass
