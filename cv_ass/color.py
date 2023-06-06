import cv2
import numpy as np

iname = 'coins.jpg'
image = cv2.imread(iname, 2)
print(image.shape)

# image = image.reshape((-1,3))

# cv2.imshow('reshape', image)

############################ boundary extraction ##################################################

# ret, bw_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# kernel = np.ones((5,5), np.uint8)

# bw_img = cv2.bitwise_not(bw_img)

# closing = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)

# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# e = cv2.erode(closing, kernel2, iterations=2)

# bound = closing - e

# cv2.imshow('input', image)
# cv2.imshow('bw img', bw_img)
# cv2.imshow('close', closing)
# cv2.imshow('bound', bound)

# cv2.waitKey()
# cv2.destroyAllWindows()

######################### erosion on chip #####################################
## img name == 'circuit.jpg'

# erode = np.ones((5,1), np.uint8)
# erode2 = np.array([[1, 0, 0, 0, 1],
#                    [0, 1, 0, 1, 0],
#                    [0, 0, 1, 0, 0],
#                    [0, 1, 0, 1, 0],
#                    [1, 0, 0, 0, 1]])
# erode2 = np.uint8(erode2)

# ret, bw_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# print(type(bw_img))
# print(type(erode2))
# # converting to its binary form
# # bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


# cv2.imshow("Binary", bw_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# out = bw_img
# for i in range(50):
#     out = cv2.erode(out, erode2, iterations=1)
#     cv2.imshow(f'erode{i}', out)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

######################### feature extraction ##############################################################
## img name== 'hanser.jpg'


# grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(grayimg.shape)
# print(grayimg.dtype)

# mat1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
# mat2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
# dil = np.array([    [0, 1, 0],
#                     [1, 1, 1],
#                     [0, 1, 0]])

# dil2 = np.array([   [0, 0, 1, 0, 0],
#                     [0, 1, 1, 1, 0],
#                     [1, 1, 1, 1, 1],
#                     [0, 1, 1, 1, 0],
#                     [0, 0, 1, 0, 0]])

# dil = np.uint8(dil)
# dil2 = np.uint8(dil2)
# kernel = np.ones((3,3), np.uint8)
# # mat1 = np.uint8(mat1)
# print(mat1.dtype)

# res1 = cv2.filter2D(src=grayimg, kernel=mat1, ddepth=-1)
# cv2.imwrite('vert_dect.jpg', res1)
# # cv2.imshow('vert_dect', res1)
# # cv2.waitKey()

# res2 = cv2.filter2D(src=grayimg, kernel=mat2, ddepth=-1)
# cv2.imwrite('hor_dect.jpg', res2)
# # cv2.imshow('hor_dect', res2)
# # cv2.waitKey()


# res = (res1 + res2)
# # res = cv2.filter2D(src=res, kernel=avg, ddepth=-1)
# # res //= 9
# cv2.imwrite('res.jpg', res)
# # cv2.imshow('mask', res)
# # cv2.waitKey()

# gauss = cv2.GaussianBlur(res, (5,5), 0)
# cv2.imshow('Gauss', res)
# cv2.waitKey()


# cv2.destroyAllWindows()


# img_erosion = cv2.erode(res, kernel, iterations=1)
# img_dilation = cv2.dilate(res, kernel, iterations=1)

# try_e = cv2.erode(res, dil, iterations=1)
# try_d = cv2.dilate(res, dil, iterations=1)

# opening = cv2.dilate(cv2.erode(res, kernel, iterations=1), kernel, iterations=1)
# closing = cv2.erode(cv2.dilate(res, kernel, iterations=1), kernel, iterations=1)

# try_o = cv2.dilate(cv2.erode(res, dil, iterations=1), dil, iterations=1)
# try_c = cv2.erode(cv2.dilate(res, dil, iterations=1), dil, iterations=1)

# cv2.imwrite('erode.jpg', img_erosion)
# cv2.imwrite('dilate.jpg', img_dilation)
# cv2.imwrite('try_e.jpg', try_e)
# cv2.imwrite('try_d.jpg', try_d)
# cv2.imwrite('opening.jpg', opening)
# cv2.imwrite('closing.jpg', closing)
# cv2.imwrite('try_o.jpg', try_o)
# cv2.imwrite('try_c.jpg', try_c)

# # cv2.imshow('Erosion', img_erosion)
# # cv2.imshow('Dilation', img_dilation)
# # cv2.imshow('Dilation', img_dilation2)
# # cv2.waitKey(0)

# # cv2.destroyAllWindows()

#########################################################################################################