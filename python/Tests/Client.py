import random
import socket

number = random.randint(int(1E5),int(1E6))

file = open('primfac.txt', "w+")
for number in range(1, int(1E4)):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.2.104', 1337))
    client_socket.send(bytes(str(number), 'utf8'))
    msg = str(client_socket.recv(2**10), 'utf8')
    client_socket.close()
    print(number, msg)
    file.write(str(number) + " - " + (str(msg) if msg != [] else number) + "\n")
file.close()
