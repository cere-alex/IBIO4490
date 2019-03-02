from skimage import io, color
import matplotlib.pyplot as plt


def rgb_xy(im):
    from skimage import color
    aux = color.rgb2xyz(im)
    x = aux[:, :, 0]
    y = aux[:, :, 1]
    z = aux[:, :, 2]
    X = x/(x+y+z)
    Y = y/(x+y+z)
    xy = []
    xy.append(X)
    xy.append(Y)
    return xy


filename = "./BSDS_small/train/12003.jpg"
rgb = io.imread(filename)
lab = color.rgb2lab(rgb)
hsv = color.rgb2hsv(rgb)
xy = rgb_xy(rgb)
plt.figure(1)
plt.subplot(221)
plt.imshow(rgb)
plt.subplot(222)
plt.imshow(lab)
plt.subplot(223)
plt.imshow(hsv)
plt.subplot(224)
#plt.imshow(xy)
plt.show()
