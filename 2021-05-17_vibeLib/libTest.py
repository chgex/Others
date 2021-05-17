


# import darknet
import time
import cv2
import numpy as np


import ctypes    
from ctypes import *  




ll = ctypes.cdll.LoadLibrary

# libs = ll("./vibe.so")    

libs = ll("./test.so")    



def image_to_Mat(image):
    width, height,_ = image.shape
    # print("image_size:" , width,height)
    darknet_image = darknet.make_image(width, height, 3)
    # print("darknet.copy_image_from_bytes")
    libs.copy_image_from_bytes(darknet_image, image.tobytes())
    return darknet_image



# firstFrame=cv2.imread("./img/001.jpg")
image=cv2.imread("./001.jpg")

    
# # darknet_image=image_to_Mat(image)
# width, height,_ = image.shape
# # print("image_size:" , width,height)
# darknet_image = darknet.make_image(width, height, 3)
# # print("darknet.copy_image_from_bytes")
# libs.copy_image_from_bytes(darknet_image, image.tobytes())
# # libs.init(width,height,darknet_image)
# libs.init(darknet_image)
# darknet.free_image(darknet_image)



# print("getImg=======")
# width, height = 416,416
# bright_image = darknet.make_image(width, height, 3)

# libs.getImg(bright_image)
# tmp=convert_to_array(bright_image)
# # cv2.imwrite("./getImg.jpg",tmp)


# darknet.free_image(bright_image)

def list_to_array(c_list,c=1):
    import numpy as np
    print("convert list to array")
    h,w=416,416
    image=np.zeros((h,w,c),dtype='float')
    count=0
    for i in range(h):
        for j in range(w):
            for k in range(c):
                image[i,j,k]=c_list[count]
                count+=1
    return image

def getImg():
    # libs.getImg.argtypes = c_int,
    libs.getImg.restype = POINTER(c_float)
    x=libs.getImg()
    print("x[0]: ",x[0])
    # C content to Python content
    h,w,c=416,416,3
    a = cast(x,POINTER(c_float *  (h * w * c) )).contents
    a=list(a)
    print("len(a): ",len(a))

    img=list_to_array(a,c)
    return img


img=getImg()
cv2.imwrite("getImg.jpg",img)




def getSeg():
    # libs.getImg.argtypes = c_int,
    libs.getImg.restype = POINTER(c_float)
    x=libs.getSeg()
    # C content to Python content
    h,w,c=416,416,1
    a = cast(x,POINTER(c_float *  (h * w * c) )).contents
    a=list(a)
    
    img=list_to_array(a,c)
    return img

img=getSeg()
cv2.imwrite("segImg.jpg",img)




"""
#  test list=========================
libs.giveList.argtypes = c_int,
libs.giveList.restype = POINTER(c_float)
x = libs.giveList(100)
print(x[0])

# 获得整个数组
a = cast(x,POINTER(c_float*100)).contents

a=list(a)
print(len(a))
#  test list=========================
"""

# print("init ==============")
# libs.init(firstFrame)
# print("init ==============")


# frame=cv2.imread("./img/002.jpg")
# libs.run(frame)

# SegModel = libs.getSeg()
# print("SegModel")
# UpdateModel = libs.getUpdate()
# print("UpdateModel")


