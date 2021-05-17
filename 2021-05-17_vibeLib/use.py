


import time
import cv2
import numpy as np
import ctypes    
from ctypes import *  


import sys
# sys.path.append("..")
import darknet




ll = ctypes.cdll.LoadLibrary
libs = ll("./lib_vibe/vibe.so")    



# def image_to_Mat(image):
#     width, height,_ = image.shape
#     # print("image_size:" , width,height)
#     darknet_image = darknet.make_image(width, height, 3)
#     # print("darknet.copy_image_from_bytes")
#     libs.copy_image_from_bytes(darknet_image, image.tobytes())
#     return darknet_image



# firstFrame=cv2.imread("001.jpg")
############################# init()
image=cv2.imread("./001.jpg")

width, height,_ = image.shape

cross_image = darknet.make_image(width, height, 3)
libs.copy_image_from_bytes(cross_image, image.tobytes())
libs.init(cross_image)
darknet.free_image(cross_image)


################################ run
image=cv2.imread("./002.jpg")
cross_image = darknet.make_image(width, height, 3)
libs.copy_image_from_bytes(cross_image, image.tobytes())
libs.run(cross_image)
darknet.free_image(cross_image)


################################  run
image=cv2.imread("./003.jpg")
cross_image = darknet.make_image(width, height, 3)
libs.copy_image_from_bytes(cross_image, image.tobytes())
libs.run(cross_image)
darknet.free_image(cross_image)



def list_to_array(c_list,size):
    import numpy as np
    h,w,c=size
    print("convert list to array")
    image=np.zeros((h,w,c),dtype='float')
    count=0
    if c != 1:
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    image[i,j,k]=c_list[count]
                    count+=1
    else:
        for i in range(h):
            for j in range(w):
                    image[i,j]=c_list[count]
                    count+=1
    return image

# def getImg():
#     # libs.getImg.argtypes = c_int,
#     libs.getImg.restype = POINTER(c_float)
#     x=libs.getImg()
#     print("x[0]: ",x[0])
#     # C content to Python content
#     size=(416,416,3)
#     h,w,c=size
#     a = cast(x,POINTER(c_float *  (h * w * c) )).contents
#     a=list(a)
#     print("len(a): ",len(a))

#     img=list_to_array(a,size)
#     return img

# img=getImg()
# cv2.imwrite("getImg.jpg",img)

def getSeg():
    # libs.getImg.argtypes = c_int,
    libs.getSegList.restype = POINTER(c_float)
    x=libs.getSegList()
    # C content to Python content
    size = (416,416,1)
    h,w,c = size
    a = cast(x,POINTER(c_float *  (h * w * c) )).contents
    a=list(a)
    img=list_to_array(a,size)
    return img

# img=getSeg()
# cv2.imwrite("segImg.jpg",img)

img = getSeg()
cv2.imwrite("segImg.jpg",img)

