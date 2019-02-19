import cv2
import matplotlib.pyplot as plt
from math import log2

Gamma = 0.5
Dim = 3
rest_profundidad=6
img = cv2.imread("id.jpg")
X = cv2.resize(img, (512, 512))
x, y, z = X.shape
v = [2 ** i for i in range(int(log2(x)) - 1, rest_profundidad-1, -1)]
longitud=len(v)
X_data = []
L_data = []
X_data.append(X)
c = 0
plt.figure(1)
for i in v:
    au_x = cv2.resize(X_data[c], (i, i))
    au_x = cv2.GaussianBlur(au_x, (Dim, Dim), Gamma)
    X_data.append(au_x)
    au_x = cv2.resize(au_x, (i * 2, i * 2))
    L_data.append(cv2.absdiff(X_data[c], au_x))
    plt.subplot(2, 10, c + 1)
    plt.imshow(X_data[c])
    plt.axis('off')
    plt.subplot(2, 10, c + 11)
    plt.imshow(L_data[c])
    plt.axis('off')
    c += 1
# plt.show()
print(au_x.shape)
L_data = L_data[::-1]
X_data = X_data[::-1]

XX_data = []
x,y,z=X_data[0].shape
XX_data.append(cv2.resize(X_data[0], (x*2,x*2)))
plt.figure(2)
for i in range(c):
    au_x = cv2.add(XX_data[i], L_data[i])
    au_x = cv2.resize(au_x, (2 ** (i+rest_profundidad+2), 2 ** (i+rest_profundidad+2)))
    XX_data.append(au_x)
    plt.subplot(1,10,i+1)
    plt.imshow(XX_data[i])
    plt.axis("off")
plt.show()
cv2.imshow('yo',XX_data[c])
cv2.waitKey(0)
cv2.destroyAllWindows()