#!/usr/bin/python3
import cv2
import os

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

sigma_1 = 6
sigma_2 = 81
Dim = 81
img = cv2.imread("./fotos/dog2.jpg")
x = 512
y = 512
img_2 = cv2.resize(img, (x, y))
img_2_gaussian = cv2.GaussianBlur(img_2, (Dim, Dim), sigma_1)
img = cv2.imread("./fotos/dog1.jpg")
img_i = cv2.resize(img, (x, y))
img_i_gaussian = cv2.GaussianBlur(img_i, (Dim, Dim), sigma_2)
img_i_difference = cv2.absdiff(img_i, img_i_gaussian, cv2.NORM_L1)
img_sum = cv2.add(img_2_gaussian, img_i_difference, cv2.NORM_L1)

final = cv2.hconcat((img_2, img_i, img_sum))
cv2.imshow("amigos", final)
cv2.waitKey(0)
cv2.destroyAllWindows()

