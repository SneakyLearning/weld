import cv2
import numpy as np
from matplotlib import pyplot as plt
point1 = [2832.0,1616.0]
point2 = [1775.0,1802.0]
u0,v0 = 2.02273721e+03,1.23130333e+03
f_x,f_y = 6.82758832e+03,4.63481183e+01
b = 30
x = b*(point1[0] - u0) / (point1[0] - point2[0])
y = b*f_x*(point1[1] - v0) / (f_x*(point1[0] - point2[0]));
z = b*f_x / (point1[0] - point2[0]);
print(x,y,z)