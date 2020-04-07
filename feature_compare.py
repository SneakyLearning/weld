import cv2
import numpy as np
from matplotlib import  pyplot as plt

img0 = cv2.imread(r'img0.jpg',0)
img1 = cv2.imread(r'img1.jpg',0)
img2 = cv2.imread(r'img2.jpg',0)

sift = cv2.xfeatures2d.SIFT_create()

#kp0,des0 = sift.detectAndCompute(img0,None)
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)
img1=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(1,2,1)
plt.imshow(img1,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img2,cmap='gray')

plt.show()
'''
bf = cv2.BFMatcher()
matcher1 = bf.knnMatch(des0,des1,k=2)
matcher2 = bf.knnMatch(des0,des2,k=2)
matcher = matcher1 and matcher2

good=[]
for m,n in matcher2:
    if m.distance<0.75*n.distance:
        good.append(m)


img3 = cv2.drawMatches(img0,kp0,img2,kp2,good,None,matchColor=(0,0,255),flags=2)
plt.imshow(img3)
plt.show()
'''