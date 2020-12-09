import cv2
import numpy as np
# 0 means gray scale conversion
hand = cv2.imread('Capture.png',0)
#  min threshold=70, max threshold=255
ret, the = cv2.threshold(hand, 70, 255, cv2.THRESH_BINARY)
#contours,_=cv2.findContours(the.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#contours,hierarchy = cv2.findContours(the.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
###### contour is a technique tht is used to find out the outer area--->contour connectd pixels ko find out krta h
contours,_ = cv2.findContours(the.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

hull = [cv2.convexHull(c) for c in contours]
## result check krne ke liye hme contour ko draw krna pdega
final = cv2.drawContours(hand, hull, -1, (255,0,0))

cv2.imshow('Originals', hand)
cv2.imshow('Thresh',the)
cv2.imshow('Convex hull',final)

cv2.waitKey(0)
cv2.destroyAllWindows()