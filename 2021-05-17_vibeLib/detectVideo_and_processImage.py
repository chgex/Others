
import os
import glob
import random
import darknet

import cv2
import numpy as np
from queue import Queue


# 调用背景提取算法
import ctypes    
from ctypes import * 
ll = ctypes.cdll.LoadLibrary
libs = ll("./lib_vibe/vibe.so")    


""""
在darknet_images.py基础上，做了一些修改：
    去掉一些函数
    添加路径
    添加后优化函数

"""
######################################################################
def load_images(images_path):
    return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    print("image_detection:" , width,height)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    
    print("darknet.detect_image")
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    
    print("detections")
    print(detections)
    print("darknet.draw_boxes")
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

def load_network():
    # load network
    config_file="./dataspace/pp.cfg"
    data_file="./dataspace/soot.data"
    weights="./pp_6000.weights"
    batch_size = 1
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size
    )
    return network, class_names, class_colors

###########################################################################


def calcuFrameAbs(previousFrame,currentFrame,thresh):
    # 计算当前帧和前帧的不同 
    frameDelta = cv2.absdiff(previousFrame, currentFrame) 
    
    # 在frameDelta中找大于20（阈值）的像素点, 对应点设为255
    # 此处阈值可以帮助阴影消除, 图像二值化 
    frameDelta = cv2.threshold(frameDelta, thresh, 255, cv2.THRESH_BINARY)[1] 
    
    # 结构元素(内核矩阵)的尺寸
    g_nStructElementSize=3
    
    # 获取自定义核
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

    # 去除图像噪声: 腐蚀, 膨胀(形态学开运算) 
    
    # 腐蚀
    frameDelta=cv2.erode(frameDelta, kernel,3) 
    # 膨胀
    frameDelta = cv2.dilate(frameDelta, kernel,3) 
    
    return frameDelta

def bbox2points(bbox):
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def isHasMove(image,detections,frameDelta,thresh):
    # frameDelta归一化处理
    # assert len(frameDelta)  , "frameDelta must be single channel "
    frameDelta = frameDelta // 255
    # 计算bbox在frameDelta对应位置的动态信息数量
    for _, confidence, bbox in detections:
        x,y,w,h=bbox
        xmin,ymin,xmax,ymax = bbox2points(bbox)
        crop = frameDelta[xmin:xmax,ymin:ymax]
        # 计算动态信息占比
        sigma = np.sum(crop) / (w * h)
        """ 直接==
        # 不画边界框
        if sigma < thresh:
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            # cv2.putText(image, "suppression", (xmax, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)
            print("=====suppression this bounding box")
        else:
            # 画边界框
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        
        """
        # 画检测结果的边界框
        image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        
        if sigma < thresh:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            cv2.putText(image, "suppression", (xmax, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)
            print("=====suppression this bounding box")

    return image


# 加载图片数据
def load_data(frame_queue,image_path="dataspace/img"):
    # 加载数据
    # image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    image_names=load_images(image_path)
    image_names.sort()
    print("total image:",len(image_names))
    
    # images = [cv2.imread(image) for image in image_names]
    for image_name in image_names:
        image=cv2.imread(image_name)
        if image is not None:
            frame_queue.put(image)
    print("cv2.imread succ")

# 加载视频数据
def load_video_data(frame_queue,input_path):
    cap = cv2.VideoCapture(input_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    print("video imread succ")

# 内存不够，参考dataloader: 一次加载一个批量
def data_loader(cap,frame_queue,batch_size):
    frame_queue.queue.clear()
    cnt=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
        cnt+=1
        if cnt == batch_size:
            break
    print("batch_size frame of video imread succ")

#############################################################背景提取，C格式转换
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

def getSeg(width,height):
    # libs.getImg.argtypes = c_int,
    libs.getSegList.restype = POINTER(c_float)
    x=libs.getSegList()
    # C content to Python content
    size = (width,height,1)
    h,w,c = size
    a = cast(x,POINTER(c_float *  (h * w * c) )).contents
    a=list(a)
    img=list_to_array(a,size)
    return img

def getUpdate(width,height):
    # libs.getImg.argtypes = c_int,
    libs.getUpdateList.restype = POINTER(c_float)
    x=libs.getUpdateList()
    # C content to Python content
    size = (width,height,1)
    h,w,c = size
    a = cast(x,POINTER(c_float *  (h * w * c) )).contents
    a=list(a)
    img=list_to_array(a,size)
    return img



def make_memory_use():
    input_path="./dataspace/test.mp4"
    save_tmp="dataspace/saveTmp"

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,160)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120)
    index=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        index+=1
        save_frame_name = save_tmp + "/" + str(index).zfill(5) + ".jpg"
        cv2.imwrite( save_frame_name , frame)
    print("video to frame succ")
    cap.release()
    return index

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) 
    #scaleup为False的话表示图像只能缩小不能放大
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
 
    # Compute padding,计算padding的像素大小
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # return img, ratio, (dw, dh)
    return img

################################################################## main


"""
if __name__=='__main__':
    # 将处理之后的帧，保存在save_path文件夹
    save_path= "dataspace/saveImg"
    save_seg= "dataspace/saveSeg"
    save_update= "dataspace/saveUpdate"
    
    input_path="./dataspace/test.mp4"
    save_tmp="./dataspace/saveTmp"

    # 视频总帧数
    # frame_total = make_memory_use() 
    # print("frame_total: ",frame_total)   
    print("=========")
    image_list=os.listdir(save_tmp)
    image_list.sort()
    print("frame_total: ",len(image_list))   
    
    firstFrame=False
    print("loop ===============")    
    for index,image_name in enumerate(image_list):
        img = cv2.imread(save_tmp + "/" + image_name)
        print(save_tmp + "/" + image_name)
        print(img.shape)
        
        ######################## letterbox()
        image = letterbox(img,(416,416))
        width, height,_ = image.shape
        print("width,height",width,height)
        
        ######################## init()
        if firstFrame is False:
            # init() and run first frame
            cross_image = darknet.make_image(width, height, 3)
            libs.copy_image_from_bytes(cross_image, image.tobytes())
            libs.init(cross_image)
            firstFrame=True
            try:
                print("free =====")
                darknet.free_image(cross_image)    
            except:
                pass
            continue
        ########################  run()
        cross_image = darknet.make_image(width, height, 3)
        libs.copy_image_from_bytes(cross_image, image.tobytes())
        libs.run(cross_image)
        try:
            print("free=====")
            darknet.free_image(cross_image)
        except:
            pass
        ########################  getSeg()
        seg=getSeg(width,height)
        update=getUpdate(width,height)
        
        # 保存seg
        save_seg_name = save_seg + "/" + str(index).zfill(5) + ".jpg"
        cv2.imwrite( save_seg_name , seg)
        print("seg writed to ",save_seg_name)
        
        # 保存update
        save_update_name=save_update + "/" + str(index).zfill(5) + ".jpg"
        cv2.imwrite( save_update_name , update)
        print("updata writed to ",save_update_name)
        
"""
################################################################## main

# 转换坐标，与原图大小相对应
def detection_adjust(detections,frame):
    detections_adjusted=[]
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
    return detections_adjusted

def isMove(frame,detections,frameDelta,thresh,class_colors):
    print("detections ============= ")
    print(detections)
    print("detections ============= ")
    frameDelta = frameDelta // 255
    # 计算bbox在frameDelta对应位置的动态信息数量
    
    ################################  直接抑制 ####################
    remain_detections=[]
    for label, confidence, bbox in detections:
        x,y,w,h=bbox
        xmin,ymin,xmax,ymax = bbox2points(bbox)
        crop = frameDelta[xmin:xmax,ymin:ymax]
        # 计算动态信息占比
        sigma = np.sum(crop) / (w * h)
        if sigma > thresh:
            print("remain  this detection")
            remain_detections.append((str(label), confidence, bbox))
        else:
            print("==== suppression this bounding box ====")
    # 画检测结果的边界框
    image = darknet.draw_boxes(remain_detections, frame, class_colors)
    ################################  直接抑制 ####################
    
    ################################  标注 抑制 ########################
    """
    image = darknet.draw_boxes(remain_detections, frame, class_colors)
    for label, confidence, bbox in detections:
        x,y,w,h=bbox
        xmin,ymin,xmax,ymax = bbox2points(bbox)
        crop = frameDelta[xmin:xmax,ymin:ymax]
        # 计算动态信息占比
        sigma = np.sum(crop) / (w * h)
        if sigma < thresh:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            cv2.putText(image, "suppression", (xmax, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 1)
            print("===== suppression this bounding box ====")
    """
    ################################  标注 抑制 ##########################
    return image


################################################################## main
if __name__=='__main__':
    # 将处理之后的帧，保存在save_path文件夹
    save_path= "dataspace/saveImg"
    save_seg= "dataspace/saveSeg"
    save_update= "dataspace/saveUpdate"

    frame_queue = Queue()

    # load network
    network, class_names, class_colors = load_network()
    print("load network succ")

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    input_path="./dataspace/test.mp4"
    cap = cv2.VideoCapture(input_path)

    # 视频总帧数
    frame_total = int(cap.get(7)) 
    print("frame_total: ",frame_total)   
    # 一次加载batch_size张图片
    batch_size=25


    print("loop ====================== loop")
    for index in range(0, frame_total, batch_size):
        data_loader(cap,frame_queue,batch_size)
        print("queue size: ",frame_queue.qsize())
        print("loader ================  loader")

        firstFrame=False

        thresh=0.25
        while not frame_queue.empty():
            # frame保存原图
            frame=frame_queue.get()
            # img 用于提取背景
            img=frame.copy()

            #######################################  检测图片  ##################################
            print("***** image_detection *****")
            image, detections = image_detection(frame, network, class_names, class_colors,thresh)
            # 调整检测框size，与原图对应
            detections_adjust= detection_adjust(detections,frame)
            print("***** image_detection *****")
            #######################################  检测图片  ##################################


            ##################################### pyrdown and extract background  #################################
            # img = cv2.resize(img, (width, height),interpolation=cv2.INTER_LINEAR)
            # 下采样，使size缩小一半
            img=cv2.pyrDown(img)
            print("img_width,img_height: ",img.shape)
            img_width, img_height = img.shape[:2]
            ######################## init()
            if firstFrame is False:
                # init() and run first frame
                cross_image = darknet.make_image(img_width, img_height, 3)
                libs.copy_image_from_bytes(cross_image, img.tobytes())
                libs.init(cross_image)
                firstFrame=True
                try:
                    darknet.free_image(cross_image)
                except:
                    pass
                continue
            ########################  run()
            cross_image = darknet.make_image(img_width, img_height, 3)
            libs.copy_image_from_bytes(cross_image, img.tobytes())
            libs.run(cross_image)
            try:
                darknet.free_image(cross_image)
            except:
                pass
            ########################  getSeg(),getUpdate
            seg=getSeg(img_width,img_height)
            update=getUpdate(img_width,img_height)
            # 上采用，返回到原图size
            seg=cv2.pyrUp(seg)
            update=cv2.pyrUp(update)
            print("seg.shape: ",seg.shape)
            print("update.shape: ",update.shape)
            ##################################### pyrdown and extract background  #################################

            # 更新背景参考信息
            frameDelta=update
            
            ###########################################  逻辑判断 ###############################
            if len(detections_adjust) != 0 :
                # 有边界框，判断是否有运动信息，将结果修改到原图
                print("===========  detections  ============")
                image=isMove(frame,detections_adjust,frameDelta,0.001,class_colors)
                print("===========  detections  ============")
            else:
                print("=========== no  detections============")
                image=frame
            ###########################################  逻辑判断 ###############################
            
            # 显示
            # cv2.imshow('Inference', image)
            #######################################  保存jpg ###################################
            # 保存image
            save_image_name=save_path + "/" + str(index).zfill(5) + ".jpg"
            cv2.imwrite( save_image_name , image)
            print("image writed to ",save_image_name)
            
            # 保存seg
            save_seg_name = save_seg + "/" + str(index).zfill(5) + ".jpg"
            cv2.imwrite( save_seg_name , seg)
            print("seg writed to ",save_seg_name)
            # 保存update
            save_update_name=save_update + "/" + str(index).zfill(5) + ".jpg"
            cv2.imwrite( save_update_name , update)
            print("image writed to ",save_update_name)
            #######################################  保存jpg ###################################
            index+=1
    cap.release()


