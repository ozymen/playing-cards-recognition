import base64
import time

import cv2
import numpy as np
import sys
import tkinter
from matplotlib import pyplot as plt
import pyzbar.pyzbar

import os
from os.path import join, dirname
from dotenv import load_dotenv

# Create .env file path.
dotenv_path = join(dirname(__file__), '.env')

# Load file from the path.
load_dotenv(dotenv_path)


suites = ["D", "S", "H", "C"]
numbers = ["7", "8", "Q", "K", "10", "A", "9", "J"]

cardnames = [""]

for s in suites:
    for n in numbers:
        cardnames.append(s + n)


# start video capture
cap = cv2.VideoCapture(os.getenv('STREAM_URL'))

# realtime plot
plt.draw()
plt.ion()
plt.show()


###############################################################################
# Utility code from
# http://git.io/vGi60A
# Thanks to author of the sudoku example for the wonderful blog posts!
###############################################################################

def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


print('1')
print(cap.isOpened())


while (cap.isOpened()):
    ret, frame = cap.read()
    # imx = cv2.resize(frame,(1000,600))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([0, 0, 0], dtype=np.uint8)
    upper_blue = np.array([255, 40, 255], dtype=np.uint8)


    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    frame2 = cv2.bitwise_and(frame, frame, mask=mask)


    lower_blue = np.array([150, 150, 160], dtype=np.uint8)
    upper_blue = np.array([255, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(frame, lower_blue, upper_blue)
    frame2 = cv2.bitwise_and(frame2, frame, mask=mask2)

    #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    image_area = gray.size  # this is area of the image

    thresh = gray

    # thresh = cv2.adaptiveThreshold(gray,255,1,1,5,2)

    #thresh = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY,65,30)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,65,20)
    # cv2.imshow('tresh', thresh)#,plt.draw(),
    # time.sleep(10)

    numcards = 10

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print hierarchy
    # contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]
    i = -1
    for card in contours:
        i = i + 1
        print(hierarchy[0][i])
        # only top level
        # if hierarchy[0][i][3] != -1:
        # print 'Contour not toplevel'
        # continue

        # print '2'
        if cv2.contourArea(card) > image_area / 2:
            continue

        if cv2.contourArea(card) < image_area / 300:
            continue

        peri = cv2.arcLength(card, True)

        if (len(cv2.approxPolyDP(card, 0.02 * peri, True)) != 4):
            continue

        approx = rectify(cv2.approxPolyDP(card, 0.02 * peri, True))

        box = np.int0(approx)


        # cv2.imshow('frane',frame)
        # time.sleep(15)
        # continue

        h = np.array([[0, 0], [249, 0], [249, 249], [0, 249]], np.float32)

        transform = cv2.getPerspectiveTransform(approx, h)
        warp = cv2.warpPerspective(frame, transform, (250, 250))



        cv2.drawContours(frame, [box], 0, (255, 255, 0), 6)

        #warpbw = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        #ret, warpbw = cv2.threshold(warpbw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        decoded = pyzbar.pyzbar.decode(warp)

        #cv2.imshow('matches2', warpbw)


        if len(decoded) > 0:
            if int(decoded[0].data) < 33:

                cv2.putText(frame, cardnames[int(decoded[0].data)], (
                int(box[0][0] + abs(round(box[2][0] - box[0][0]))), int(box[0][1] + abs(round(box[3][1] - box[0][1])))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

    #imx = cv2.resize(frame, (1000, 600))
    cv2.imshow('matches', frame)
    print('Next frame waiting..')



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
