import numpy as np
import cv2

img = cv2.imread('lung.jpg')

img = cv2.bitwise_not(img)

img2 = img.reshape((-1,3))
img2 = np.float32(img2)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# for i in range(12):
K = 4
# 2 - 14
attempts = 20

ret,label,center=cv2.kmeans(img2,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
print(f'center: {center}')

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

print(f'center: {center}, res: {res}')

name = 'segmented_lung_k4' + '.jpg'

cv2.imwrite(name, res2)