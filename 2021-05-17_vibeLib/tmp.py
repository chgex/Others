
import ctypes
from ctypes import *
import  cv2


my_image = cv2.imread("./img/001.jpg")  # reads colorfull image in python
dims = my_image.shape  # get image shape (h, w, c)
my_image = my_image.ravel()  # flattens 3d array into 1d
cppextenionmodule.np_to_mat(dims, my_image)

