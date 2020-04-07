import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,30,0.01)

objp = np.zeros((11*13,3),np.float32)
objp[:,:2] = np.mgrid[0:13,0:11].reshape(-1,2)

obj_points = []
img_points = []

imgs = ['mark1.jpg','mark2.jpg','mark3.jpg','mark4.jpg','mark5.jpg']

for fname in imgs:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, (13, 11), None)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        cv2.drawChessboardCorners(img,(13,11),corners,ret)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数