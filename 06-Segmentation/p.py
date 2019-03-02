import pdb
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio

img = 'BSDS_small/train/22090.jpg'
plt.imshow(imageio.imread(img))
plt.show()

# Load .mat
gt=sio.loadmat(img.replace('jpg', 'mat'))

#Load segmentation from sixth human
segm=gt['groundTruth'][0,5][0][0]['Segmentation']
plt.imshow(segm, cmap=plt.get_cmap('hot'))
plt.colorbar()
plt.show()

#Boundaries from third human
segm=gt['groundTruth'][0,2][0][0]['Boundaries']
plt.imshow(segm)
plt.colorbar()
plt.show()
