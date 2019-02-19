from math import log2
from PIL import Image, ImageFilter, ImageChops
import matplotlib.pyplot as plt
import cv2

img = Image.open('inti2.jpg')
img = img.rotate(-90)
x = 512
y = 512
r = 20

img_dog = img.resize((x, y))
img_dog_gaussian = img_dog.filter(ImageFilter.GaussianBlur(radius=r / 1.5))
img = Image.open('id.jpg')
img = img.rotate(-90)
img_i = img.resize((x, y))
img_i_gaussian = img_i.filter(ImageFilter.GaussianBlur(radius=r * 1.5))
img_i_difference = ImageChops.difference(img_i, img_i_gaussian)
img_sum = ImageChops.add(img_dog_gaussian, img_i_difference)
plt.figure(1)
plt.subplot(221)
plt.imshow(img_dog)
plt.axis('off')
plt.subplot(222)
plt.imshow(img_dog_gaussian)
plt.axis('off')
plt.subplot(223)
plt.imshow(img_i)
plt.axis('off')
plt.subplot(224)
plt.imshow(img_i_difference)
plt.axis('off')
plt.figure(2)
plt.imshow(img_sum)
plt.axis("off")
plt.show()
print(log2(x))
