import cv2
import numpy as np
import matplotlib.pyplot as plt
img=[]
img.append( cv2.imread('inti.jpg'))
img.append(cv2.imread('IMG_1441.JPG'))
print(img)
print(type(img))
cv2.imshow('hello', img[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
