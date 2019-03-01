import matplotlib.pyplot as plt
import cifar10 as cf
import numpy as np
from fbCreate import fbCreate
from fbRun import fbRun
from computeTextons import computeTextons
from assignTextons import assignTextons

# a = cf.load_cifar10(meta='cifar-10-batches-py', mode=1)  # carga las imagenes del modulo 2
b = cf.unpickle('cifar-10-batches-py/data_batch_2')  # carga de otra forma las imagnes del modelo 2
# convierte la imagen en escala de grises
bb = cf.get_data(b, sliced=1)  # se obtiene las matrices de las imagenes y sus labels
b1 = bb[0]  # se guarda los  matrices de imagenes
b2 = bb[1]  # se guarda los labels
aux = np.hstack(b1)  # matrizes mxnxo se convierte  nx(o*m)
fb = fbCreate(support=3, startSigma=2)  # funcion con 2 pero es mas lento
fb_respuesta = fbRun(fb, aux)
k = 10  # asignar clusters a tener

map, textons = computeTextons(fb_respuesta, k)  # creacion  de los textones

ubicacion = 0
b11 = b1[ubicacion]

im_uno = assignTextons(fbRun(fb, b11), textons.transpose())
ubicacion = 1000
b11 = b1[ubicacion]
im_dos = assignTextons(fbRun(fb, b11), textons.transpose())
ubicacion = 500
b11 = b1[ubicacion]
im_tres = assignTextons(fbRun(fb, b11), textons.transpose())


# b22 = b2[ubicacion]
# plt.imshow(b11)
# plt.title(b22)
# plt.show()

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X, bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i - 1] += 1
    return np.array(r)


D = np.linalg.norm(
    histc(im_uno.flatten(), np.arange(k)) / im_uno.size - histc(im_dos.flatten(), np.arange(k)) / im_dos.size)
D2 = np.linalg.norm(
    histc(im_uno.flatten(), np.arange(k)) / im_uno.size - histc(im_tres.flatten(), np.arange(k)) / im_tres.size)

print("D(moto1--moto2)= ", D, "; D2=(perro1--moto2)", D2)
