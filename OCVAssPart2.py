## Cole Slugget, Kayla Wheeler
## Robot Vision 442
## OpenCV Assignment
## February 10, 2019


import numpy as np
import cv2

cap = cv2.VideoCapture(0)
status, img = cap.read()

#function to brighten the image
def brighten(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

avg = np.float32(img)

while True:

    status, img = cap.read()
    img2 = img
    #brightens the image
    bright = brighten(img)
    #blurs the image
    blur = cv2.blur(bright,(10,10))
    #takes running average of frame
    cv2.accumulateWeighted(blur, avg, .1)
    #swaos running average to same bits as frame
    res = cv2.convertScaleAbs(avg)
    #takes the difference of the blurred and the converted
    diff = cv2.absdiff(blur, res)
    #converts to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #thresholds grayscale with a low number
    ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    #blurs the grayscale
    blur2 = cv2.blur(thresh,(10,10))
    #thresholds grayscale again, but with a high number
    ret,thresh2 = cv2.threshold(blur2,175,255,cv2.THRESH_BINARY)
    
    #finds contours of the the threshold image
    con, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height, width, _ = img.shape
    xMin, yMin = width, height
    xMax = yMax = 0

    #draws the polygons of the blobs of movement
    for c, heir in zip(con, hierarchy):
        (x,y,w,h) = cv2.boundingRect(c)
        xMin, xMax = min(x, xMin), max(x+w, xMax)
        yMin, yMax = min(y, yMin), max(y+h, yMax)
        if w > 100 and h > 100:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)

    if xMax - xMin > 0 and yMax - yMin > 0:
        cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (255, 0, 0), 2)
    
    # shoes the original image and the image with motion detection 
    cv2.imshow("Orignal", img)
    cv2.imshow('Detection', thresh2)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
