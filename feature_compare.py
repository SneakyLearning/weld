import cv2
import numpy as np
from matplotlib import  pyplot as plt

img0 = cv2.imread(r'img0.jpg',0)
img1 = cv2.imread(r'img1.jpg',0)
img2 = cv2.imread(r'img2.jpg',0)

def sift_feature(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    #kp0,des0 = sift.detectAndCompute(img0,None)
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)
    #img1=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img2=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp1,des1,kp2,des2

def orb_feature(img1,img2):
    orb = cv2.ORB_create()
    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2,None)
    return kp1,des1,kp2,des2

def bf_match(des1,des2):
    bf = cv2.BFMatcher(crossCheck=True)
    matchers = bf.match(des1,des2)
    return matchers

def bf_knn_match(des1,des2):
    bf = cv2.BFMatcher()
    matchers = bf.knnMatch(des1,des2,k=2)
    return matchers

def flann_match(des1,des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches_flann = flann.knnMatch(des1,des2,k=2)
    return matches_flann

def find_good(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def ransac(good,kp1,kp2,img1,img2):
    MIN_MATCH_COUNT=10
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,2,1)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2,1)
        M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
        matchmask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),cv2.LINE_AA)
        return mask
if __name__ == '__main__':
    #kp1, des1, kp2, des2 = orb_feature(img1,img2)
    kp1, des1, kp2, des2 = sift_feature(img0, img1)

    #matchers = bf_match(des1,des2)
    matches_flann = flann_match(des1, des2)
    good = find_good(matches_flann)

    mask = ransac(good, kp1, kp2, img0, img1)
    img_ransac = cv2.drawMatches(img0,kp1,img1,kp2,good,None)

    #matchers_knn = bf_knn_match(des1,des2)



    #img_knn = cv2.drawMatches(img0,kp1,img1,kp2,good,None)

    plt.imshow(img_ransac)
    plt.show()