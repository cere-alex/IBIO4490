import cv2
import matplotlib.pyplot as plt
from math import log2


def gaussian_pyramid(img, a, n, Dim, Gamma, fig):
    maximo = int(log2(a))
    m = maximo - n
    # for simplificado, donde se obtiene mediante un logarirmo
    # se le resta para que empieze de "dimension-1" hasta la posicion
    # que se determine en la variable "n" y se genera un rango de max a min
    v = [2 ** i for i in range(maximo - 1, m - 1, -1)]
    L_data = []
    X = cv2.resize(img, (a, a))  # se dimensiona la imagen al tamanho necesario
    c = 0  # contador interno de la imagen
    plt.figure(fig)
    for i in v:
        au_x = cv2.resize(X, (i, i))  # se dimesiona a una imagen mas pequenha
        X_copy = X
        au_x = cv2.GaussianBlur(au_x, (Dim, Dim), Gamma)  # uso de filtro gaussiano con imagen
        X = au_x  # la imagen filtrada se guarda en esta variable sera reutilizada
        au_x = cv2.resize(au_x, (i * 2, i * 2))  # se amplia la imagen filtrada
        L_data.append(cv2.subtract(X_copy, au_x))  # resta y guarda (imagen - imagen_filtrada)
        plt.subplot(2, n, c + 1)
        plt.imshow(X_copy)
        plt.axis('off')
        plt.subplot(2, n, c + n + 1)
        plt.imshow(L_data[c])
        plt.axis('off')
        c += 1

    return X, L_data, plt, v


def laplacian_pyramid(X, L_data, v, n, fig):
    L_data = L_data[::-1]
    X_data = []
    v = list(map(lambda v: v * 2, v))
    c = 0
    plt.figure(fig)
    for i in reversed(v):
        X = cv2.resize(X, (i, i))
        X = cv2.add(X, L_data[c])
        X_data.append(X)
        plt.subplot(2, n, c + 1)
        plt.imshow(L_data[c])
        plt.axis('off')
        plt.subplot(2, n, c + n + 1)
        plt.imshow(X)
        plt.axis('off')
        c += 1
    return X, plt


# Tamaño inicial de la imagen
a = 512
# profundidad de  la imagen
n = 4
# parametros de la gausiana
Gamma = 15
Dim = 15
img = cv2.imread("dog1.jpg")  # lectura de la imagen
img2 = cv2.imread('dog2.jpg')
X, L_X, plt_X, v_X = gaussian_pyramid(img, a, n, Dim, Gamma, fig=1)
Y, L_Y, plt_Y, v_Y = gaussian_pyramid(img2, a, n, Dim, Gamma, fig=2)
div_v = 256
au_X = X[:, :int((len(X) / 2))]
au_Y = Y[:, int(len(Y) / 2):]
fusion = cv2.hconcat((au_X, au_Y))
au_L_X = [L_X[i][:, :int((len(L_X[i]) / 2))] for i in range(len(L_X))]
au_L_Y = [L_Y[i][:, int((len(L_X[i]) / 2)):] for i in range(len(L_Y))]
L_fusion = [cv2.hconcat((au_L_X[i], au_L_Y[i])) for i in range(len(au_L_X))]
X = laplacian_pyramid(X, L_X, v_X, n, fig=3)
Y = laplacian_pyramid(Y, L_Y, v_Y, n, fig=4)
fusion = laplacian_pyramid(fusion, L_fusion, v_Y, n, fig=5)

# plt_X.show()
# plt_Y.show()
final = cv2.hconcat((X[0], Y[0], fusion[0]))
cv2.imshow("I & dog", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
