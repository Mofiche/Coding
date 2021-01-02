import cv2
import numpy as np
import os
from os.path import isfile, join

def convert(PathIn, Pathout, fps):
    frames = []
    files = [f for f in os.listdir(PathIn) if isfile(join(PathIn, f))]

    size = (640,480)
    #files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename = PathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        frames.append(img)

    out = cv2.VideoWriter(Pathout,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def main():
    pathIn= "vids/"
    pathOut = "video480.avi"
    pathOut2 = "video60.avi"
    fps = 480
    fps2 = 60
    convert(pathIn,pathOut, fps)
    convert(pathIn, pathOut2, fps2)

if __name__ == '__main__':
    main()