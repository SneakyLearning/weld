import cv2
from matplotlib import pyplot as plt
import numpy as np

kernel = np.ones((2,1),np.uint8)
img = cv2.imread('butt.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),0)
edge = cv2.Canny(blur,100,200)
ret,th = cv2.threshold(edge,0,255,cv2.THRESH_OTSU,0,)
erode = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)
lines = cv2.HoughLines(edge,0.1,np.pi/180,1)

for rho,theta in lines[0]:
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)

plt.imshow(img,cmap='gray')
plt.show()
