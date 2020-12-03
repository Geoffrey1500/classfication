import numpy as np
import cv2
from math import cos,sin,radians
import matplotlib.pyplot as plt

target = cv2.imread("B_MU_pic.png")
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print(th2.shape)
