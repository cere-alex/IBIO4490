import cv2
import matplotlib.pyplot as plt
from math import log2

# Tamaño inicial de la imagen
a = 1024
# profundidad de  la imagen
n = 4
maximo = int(log2(a))
n = maximo - n
# parametros de la gausiana
Gamma = 0.5
Dim = 3
##########################
img = cv2.imread("id.jpg")  # lectura de la imagen
X = cv2.resize(img, (a, a))  # se dimensiona la imagen al tamanho necesario
# for simplificado, donde se obtiene mediante un logarirmo
# se le resta para que empieze de "dimension-1" hasta la posicion
# que se determine en la variable "n" y se genera un rango de max a min
v = [2 ** i for i in range(maximo - 1, maximo - n - 1, -1)]

X_data = []  # inicio de array multidimensional
L_data = []
X_data.append(X)  # se guardo la 1ra imagen
c = 0  # contador interno de la imagen
plt.figure(1)
for i in v:
    au_x = cv2.resize(X_data[c], (i, i))  # se dimesiona a una imagen mas pequenha
    au_x = cv2.GaussianBlur(au_x, (Dim, Dim), Gamma)  # uso de filtro gaussiano con imagen
    X_data.append(au_x)  # la imagen filtrada se guarda en esta variable
    au_x = cv2.resize(au_x, (i * 2, i * 2))  # se amplia la imagen filtrada
    L_data.append(cv2.absdiff(X_data[c], au_x))  # resta y guarda (imagen - imagen_filtrada)
    plt.subplot(2, 10, c + 1)
    plt.imshow(X_data[c])
    plt.axis('off')
    plt.subplot(2, 10, c + 11)
    plt.imshow(L_data[c])
    plt.axis('off')
    c += 1
plt.show()
print(au_x.shape)
L_data = L_data[::-1]
X_data = X_data[::-1]

XX_data = []
x, y, z = X_data[0].shape
XX_data.append(cv2.resize(X_data[0], (x * 2, x * 2)))
plt.figure(2)
for i in range(c):
    au_x = cv2.add(XX_data[i], L_data[i])
    au_x = cv2.resize(au_x, (2 ** (i + n + 2), 2 ** (i + n + 2)))
    XX_data.append(au_x)
    plt.subplot(1, 10, i + 1)
    plt.imshow(XX_data[i])
    plt.axis("off")
plt.show()
cv2.imshow('yo', XX_data[c])
cv2.waitKey(0)
cv2.destroyAllWindows()
