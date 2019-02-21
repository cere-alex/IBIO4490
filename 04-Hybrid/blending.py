#!/usr/bin/python3
import os

import cv2
import matplotlib.pyplot as plt
from math import log2

busqueda = os.listdir("./")
existe = 'fotos' in busqueda
if existe:
    print("existe el archivo")
else:
    from google_drive_downloader import GoogleDriveDownloader as gdd

    archivo = '1jlS339zmwNur8kGi6vMk08eAgYPGlaqr'  # link del archivo
    gdd.download_file_from_google_drive(file_id=archivo, \
                                        dest_path='./fotos.zip', unzip=True, overwrite=True)
c = 0
dir_imagenes = []
# se usara walk que caminara en cada directorio
for root, dir, files in os.walk("./fotos/"):  # root = son las direcciones, dir= matriz, files = nombre de archivos
    for filename in files:  # busca las imagen
        c += 1
        dir_imagenes.append(root + filename)


# se crearon dos funciones
###############################################################################################
def gaussian_pyramid(img, a, n, Dim, sigma, fig):  # funcion que realiza la piramide gaussiana
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
        au_x = cv2.GaussianBlur(au_x, (Dim, Dim), sigmaX=int(i / 4),
                                sigmaY=int(i / 4))  # uso de filtro gaussiano con imagen
        X = au_x  # la imagen filtrada se guarda en esta variable sera reutilizada
        au_x = cv2.resize(au_x, (i * 2, i * 2))  # se amplia la imagen filtrada
        L_data.append(cv2.subtract(X_copy, au_x, cv2.NORM_L1))  # resta y guarda (imagen - imagen_filtrada)
        plt.subplot(2, n, c + 1)
        plt.imshow(X_copy)
        plt.axis('off')
        plt.subplot(2, n, c + n + 1)
        plt.imshow(L_data[c])
        plt.axis('off')
        c += 1

    return X, L_data, plt, v


################################################################################################
def laplacian_pyramid(X, L_data, v, Dim, n, fig):  # funsion que realiza la piramide laplaciana
    L_data = L_data[::-1]
    X_data = []
    v = list(map(lambda v: v * 2, v))
    c = 0
    plt.figure(fig)
    for i in reversed(v):
        X = cv2.resize(X, (i, i))
        X = cv2.add(X, L_data[c], cv2.NORM_L1)
        # en este segmento se esta agrego un filtro gaussiano por que al ir sumando las imagens no se generaba ningun cambio
        X = cv2.GaussianBlur(X, (Dim, Dim), sigmaX=int(i / 2), sigmaY=int(i / 2))
        X_data.append(X)
        plt.subplot(2, n, c + 1)
        plt.imshow(L_data[c])
        plt.axis('off')
        plt.subplot(2, n, c + n + 1)
        plt.imshow(X)
        plt.axis('off')
        c += 1
    return X, plt


###################################################################################
# Tamaño inicial de la imagen
a = 512
# profundidad de  la imagen
n = 4
# parametros de la gausiana
Dim = 3
sigma = 5
img = cv2.imread("./fotos/id.jpg")  # lectura de la imagen
img2 = cv2.imread('./fotos/id2.jpg')
X, L_X, plt_X, v_X = gaussian_pyramid(img, a, n, Dim, sigma, fig=1)  # se usa la funcion de piramide gaussiana
Y, L_Y, plt_Y, v_Y = gaussian_pyramid(img2, a, n, Dim, sigma, fig=2)
# se cortaron  por la mitad las dos imagenes tratadas
au_X = X[:, :int((len(X) / 2))]
au_Y = Y[:, int(len(Y) / 2):]
au_L_X = [L_X[i][:, :int((len(L_X[i]) / 2))] for i in range(len(L_X))]  #
au_L_Y = [L_Y[i][:, int((len(L_X[i]) / 2)):] for i in range(len(L_Y))]
# se junto la image "g" que servira de guia y las plantillas de L
fusion = cv2.hconcat((au_X, au_Y))
L_fusion = [cv2.hconcat((au_L_X[i], au_L_Y[i])) for i in range(len(au_L_X))]
# se empezo con la piramide laplaciana con las mitades juntadas
X = laplacian_pyramid(X, L_X, v_X, Dim, n, fig=3)  # muestra de la piramide laplaciana con imagenes no divididas
Y = laplacian_pyramid(Y, L_Y, v_Y, Dim, n, fig=4)  # muestra de la piramide laplaciana con imagenes no divididas

fusion = laplacian_pyramid(fusion, L_fusion, v_Y, Dim, n, fig=5) # piramide laplaciana de la imagen principal
# plt_X.show()
# plt_Y.show()
final = cv2.hconcat((X[0], Y[0], fusion[0]))# se concateno las imagenes necesarias
cv2.imshow("friends", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
