from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import random as rng


source_window = 'Image'
maxTrackbar = 25
rng.seed(12345)


def goodFeaturesToTrack_Demo(val):
    maxCorners = max(val, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.5


    circle = np.zeros(src_gray.shape, dtype="uint8")
    central_x, central_y = src_gray.shape
    mask_ = cv.circle(circle, (int(central_x/2), int(central_y/2)), int((central_x + central_y)/16), 255, -1)
    blockSize = int((central_x + central_y) / 16)
    # cv.imshow("Circle", circle)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Copy the source image
    copy = np.copy(src)
    print(copy.shape, "照片大小")
    # Apply corner detection
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance=None, mask=mask_, blockSize=blockSize)
    # Draw corners detected
    print('** Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (corners[i,0,0], corners[i,0,1]), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)
    # Show what you got
    cv.namedWindow(source_window)
    cv.imshow(source_window, copy)
    # Set the needed parameters to find the refined corners
    winSize = (3, 3)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 50, 0.01)
    # Calculate the refined corner locations
    corners = cv.cornerSubPix(src_gray, corners, winSize, zeroZone, criteria)
    # Write them down
    for i in range(corners.shape[0]):
        print(" -- Refined Corner [", i, "]  (", corners[i,0,0], ",", corners[i,0,1], ")")
# Load source image and convert it to gray


parser = argparse.ArgumentParser(description='Code for Shi-Tomasi corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='B_MU_pic.png')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))

# plt.figure()
# plt.imshow(src, cmap='gray')
# plt.show()

if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Create a window and a trackbar
cv.namedWindow(source_window)
maxCorners = 10 # initial threshold
cv.createTrackbar('Threshold: ', source_window, maxCorners, maxTrackbar, goodFeaturesToTrack_Demo)
cv.imshow(source_window, src)
goodFeaturesToTrack_Demo(maxCorners)
cv.waitKey()

