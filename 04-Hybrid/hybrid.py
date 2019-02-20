import cv2
Gamma_1 = 5
Gamma_2 = 45
Dim = 35
img = cv2.imread('id.jpg')
x = 512
y = 512
img_2 = cv2.resize(img,(x, y))
img_2_gaussian = cv2.GaussianBlur(img_2, (Dim, Dim), Gamma_1)
img = cv2.imread('id2.jpg')
img_i = cv2.resize(img,(x, y))
img_i_gaussian = cv2.GaussianBlur(img_i , (Dim, Dim), Gamma_2)
img_i_difference = cv2.subtract(img_i, img_i_gaussian)
img_sum = cv2.add(img_2_gaussian, img_i_difference)

final = cv2.hconcat((img_2_gaussian, img_i_difference,img_sum))
cv2.imshow("amigos",final)
cv2.waitKey(0)
cv2.destroyAllWindows()