## Cole Slugget, Kayla Wheeler
## Robot Vision 442
## OpenCV Assignment
## February 10, 2019


import numpy as np
import cv2
cap = cv2.VideoCapture(0)

def nothing(x):
  pass

#gets the x and y coordinates and hsv values of where the mouse clicked
def getHSV(event,x,y,flag,param):
  if(event==cv2.EVENT_LBUTTONDOWN):
    color = hsv[y,x]
    
    print("x =",x,"y =",y,"| hsv =",color)

kernel = np.ones((5,5),np.uint8)
cv2.namedWindow("Video")
cv2.namedWindow("HSV")
cv2.setMouseCallback("HSV", getHSV)

#creates trackbars for color chance
cv2.createTrackbar('minH',"Video",0,179,nothing)
cv2.createTrackbar('maxH',"Video",0,179,nothing)
cv2.createTrackbar('minS',"Video",0,255,nothing)
cv2.createTrackbar('maxS',"Video",0,255,nothing)
cv2.createTrackbar('minV',"Video",0,255,nothing)
cv2.createTrackbar('maxV',"Video",0,255,nothing)


while True:
    status, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.setMouseCallback("HSV", on_mouse)

    #gets the values the trackbars are set at
    hN = cv2.getTrackbarPos('minH', "Video")
    hX = cv2.getTrackbarPos('maxH', "Video")
    sN = cv2.getTrackbarPos('minS', "Video")
    sX = cv2.getTrackbarPos('maxS', "Video")
    vN = cv2.getTrackbarPos('minV', "Video")
    vX = cv2.getTrackbarPos('maxV', "Video")
    
    #puts hsv min values in an array
    minArray = np.array([hN,sN,vN])

    #puts hsv max values in an array
    maxArray = np.array([hX,sX,vX])
    
    #makes the image grayscale
    mask = cv2.inRange(hsv, minArray, maxArray)

    #Dilates,and erodes the grayscale image to get a better representation of tracked object
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    #shows the original, masked and hsv video feeds
    cv2.imshow("Mask", mask)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Video", img)
    
    #kill key
    k = cv2.waitKey(1)
    if k == 27:
        break
cv2.destroyAllWindows()
