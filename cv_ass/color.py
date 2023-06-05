import cv2
import numpy as np

iname = 'hanser.jpg'
image = cv2.imread(iname)
print(image.shape)
grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(grayimg.shape)
print(grayimg.dtype)

# cv2.imshow('gray_img', grayimg)
# cv2.waitKey()
# cv2.destroyAllWindows()

# gray = grayimg.reshape(-1)
# print(gray.shape)
# cv2.imshow('gray_img', gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

mat1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
mat2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
avg = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# mat1 = np.uint8(mat1)
print(mat1.dtype)



res1 = cv2.filter2D(src=grayimg, kernel=mat1, ddepth=-1)
cv2.imwrite('vert.jpg', res1)
cv2.imshow('mask1', res1)
cv2.waitKey()

res2 = cv2.filter2D(src=grayimg, kernel=mat2, ddepth=-1)
cv2.imwrite('hor.jpg', res2)
cv2.imshow('mask2', res2)
cv2.waitKey()


res = (res1 + res2)
# res = cv2.filter2D(src=res, kernel=avg, ddepth=-1)
# res //= 9
cv2.imwrite('res.jpg', res)
cv2.imshow('mask', res)
cv2.waitKey()

cv2.destroyAllWindows()