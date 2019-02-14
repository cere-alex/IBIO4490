#!/usr/bin/python3
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
busqueda=os.listdir("./")
existe='Arte' in busqueda
if existe:
    pass
else:
    instalar = 'pip install googledrivedownloader'
    os.system(instalar)
    from google_drive_downloader import GoogleDriveDownloader as gdd
    archivo = '1ila5xlMApbvV0R6Ic_4my6AtDlH-aXYU'# link del archivo
    gdd.download_file_from_google_drive (file_id = archivo,\
    dest_path='./arte_nube.zip', unzip=True, overwrite=True)
os.system('rm -rf ./arte_new/')
c=0
# se usara walk que caminara en cada directorio
for root,dir,files in os.walk("./Arte/"):# root = son las direcciones, dir= matriz, files = nombre de archivos
    for filename in files:#busca las imagen
        c+=1
dir_imagenes=['']*(c)#se crea una matriz vacia del tamaño que se conto anteriormente
tipo_imagenes=['']*(c)
c=0
for root,dir, files in os.walk("./Arte/"):
    for(filename) in files:
        dir_imagenes[c]=root+"/"+filename # se realizo una matriz con la direccion necesaria
        tipo_imagenes[c]=root.split('/')#divide el string con relacion a un separador
        c+=1
a = list(range(c))# se realizo una lista del total de imagenes
k=9 # limite de imagenes a usar de todo los archivos
aleatorio=random.sample(a,k)#crea una matriz con numeros aleatorios sin repetir
img_muestras=['']*len(aleatorio)# se crea dos variable que almacenaran cadenas
img_tipo=['']*len(aleatorio)
c=0
for i in aleatorio:
    img_muestras[c]=dir_imagenes[i]
    img_tipo[c]=tipo_imagenes[i][2]#us solo la casilla 2 de la matriz
    c+=1
os.system('mkdir ./arte_new')#crea el directorio donde se guardaran las nuevas imagenes
plt.figure(1)
for i in range(k):#guardara las imagenes del rango de 9 images de k
    img=Image.open(img_muestras[i])#habre la imagen para ser usada
    img_new=img.resize((256,256)) # se recortara la imagen para
    texto =img_tipo[i]
    dibujar=ImageDraw.Draw(img_new)# Prepara la imagen para dibujarla
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)# usa el fondo de la letra para el texto
    (x, y) = dibujar.textsize(texto, font=fnt)#Toma las medidas necesarias para colocar el texto al medio
    color = 'red'# pinta el texto de color rojo
    dibujar.text(((256-x)/2, (256-y)/2), texto, fill=color,font=fnt)# sobrepone el texto sobre la imagen
    img_new.save('./arte_new/img_new_'+str(i+1)+"_"+texto+'.jpeg','jpeg')#Guarda la imagen en un archivo
    plt.subplot(3,3,(i+1)) # muetra la imagen en un conjuntno de subplots
    plt.axis('off')
    plt.imshow(img_new)
plt.show()
