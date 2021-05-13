
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
from queue import Queue


""""
在darknet_images.py基础上，做了一些修改：
    去掉一些函数
    添加路径
    添加后优化函数

"""


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
    config_file="./dataspace/wj3_v16.cfg"
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
        if sigma < thresh:
            # int(confidence) < 30.0
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            cv2.putText(image, "suppression", (xmax, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
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

if __name__=='__main__':
    # 将处理之后的帧，保存在save_path文件夹
    save_path= "dataspace/saveImg"
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


    print("loop ===============")
    for index in range(0, frame_total, batch_size):
        data_loader(cap,frame_queue,batch_size)
        print("queue size: ",frame_queue.qsize())
        print("loader ============================")
        ############################### 帧差
        frame_thresh=20
        previousFrame=None

        thresh=0.25
        while not frame_queue.empty():
            image=frame_queue.get()
            ################################## 帧差
            currentFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 如果第一帧是None，对其进行初始化 
            if previousFrame is None: 
                previousFrame = currentFrame
                continue
            # 得到帧差：单通道二值图
            frameDelta = calcuFrameAbs(previousFrame,currentFrame,0.001) 
        
            # 当前帧设置为下一帧的前帧 
            previousFrame = currentFrame.copy()
            ################################## 帧差

            ################################### 检测图片
            height,width,_ = image.shape
            print("width, height", width, height)
            
            # frame保存原图
            frame = image.copy()
            # detect image, image type is np.adarray()
            print("image_detection *****")
            
            image, detections = image_detection(image, network, class_names, class_colors,thresh)
            
            print("detections ============= ")
            print(detections)
            print("detections ============= ")
            # 转换坐标，与原图大小相对应
            detections_adjusted=[]
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            
            # 在原图上，画边界框
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            
            ################################### 逻辑判断
            if len(detections_adjusted) != 0 :
                # 有边界框，判断是否有运动信息，将结果修改到原图
                image=isHasMove(frame,detections_adjusted,frameDelta,0.01)
            # 显示
            # cv2.imshow('Inference', image)
            # 保存
            save_image_name=save_path + "/" + str(index).zfill(5) + ".jpg"
            # cv2.imwrite( save_image_name , image_resized)
            cv2.imwrite( save_image_name , image)
            print("image writed to ",save_image_name)
            
            index+=1
    cap.release()

    





# if __name__=='__main__':

#     save_path= "dataspace/saveImg"
#     frame_queue = Queue()

#     # # 加载数据：图片
#     # image_path="dataspace/img"
#     # load_data(frame_queue,image_path)
    
#     # 加载数据：视频
#     input_path="./dataspace/test.mp4"
#     load_video_data(frame_queue,input_path) 
    
#     print("queue size: ",frame_queue.qsize())

#     # load network
#     network, class_names, class_colors = load_network()
#     print("load network succ")

#     darknet_width = darknet.network_width(network)
#     darknet_height = darknet.network_height(network)





#     #####  帧差
#     frame_thresh=20
#     previousFrame=None
#     #####  帧差

#     thresh=.25
#     index = 1
#     while not frame_queue.empty():
#         image=frame_queue.get()
#         ################################## 帧差
#         currentFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # 如果第一帧是None，对其进行初始化 
#         if previousFrame is None: 
#             previousFrame = currentFrame
#             continue
#         # 得到帧差：单通道二值图
#         frameDelta = calcuFrameAbs(previousFrame,currentFrame,0.01) 
    
#         # 当前帧设置为下一帧的前帧 
#         previousFrame = currentFrame.copy()
#         ################################## 帧差

#         ################################### 检测图片
#         height,width,_ = image.shape
#         print("width, height", width, height)
        
#         # frame保存原图
#         frame = image.copy()
#         # detect image, image type is np.adarray()
#         print("image_detection =========")
#         try:
#             # free pointer报错，没找到原因，加个try🐕一会
#             image, detections = image_detection(image, network, class_names, class_colors,thresh)
#         except:
#             continue
#         print("detections ============= ", detections)

#         # 转换坐标，与原图大小相对应
#         detections_adjusted=[]
#         for label, confidence, bbox in detections:
#             bbox_adjusted = convert2original(frame, bbox)
#             detections_adjusted.append((str(label), confidence, bbox_adjusted))
#             image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        
#         # 在原图上，画边界框
#         image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        
#         ################################### 逻辑判断
#         if len(detections_adjusted) != 0 :
#             # 有边界框，判断是否有运动信息，将结果修改到原图
#             image=isHasMove(frame,detections_adjusted,frameDelta)
#         # 显示
#         # cv2.imshow('Inference', image)
#         # 保存
#         save_image_name=save_path + "/" + str(index).zfill(5) + ".jpg"
#         # cv2.imwrite( save_image_name , image_resized)
#         cv2.imwrite( save_image_name , image)
#         print("image writed to ",save_image_name)
        
#         index+=1

#     # if cv2.waitKey() & 0xFF == ord('q'):
#     #     break


