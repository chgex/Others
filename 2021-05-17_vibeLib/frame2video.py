'''
Author: liubai
Date: 2021-04-22 
LastEditTime: 2021-04-22 
'''
import cv2
import os 

# 图片重命名
def imageRename(image_path):
    image_list=os.listdir(image_path)
    total=len(image_list)
    # 第一张图片
    cnt=1
    for i in range(1,total+1):
        old_image_name=image_path + '/' + str(i) + '.jpg'
        new_image_name=image_path + '/' + str(i).zfill(5) + '.jpg'
        os.rename(old_image_name,new_image_name)
    print('rename success')

# 计算视频长度
def getTime(filename):
    total_time=0
    cap=cv2.VideoCapture(filename)
    if cap.isOpened():
    	rate = cap.get(5)    # 该函数返回 帧速率
    	fraNum=cap.get(7)    # 该函数返回 视频文件中的帧数
    	duration=fraNum/rate
    	total_time+=duration
    cap.release()
    return total_time
# 计算fps
def getFPS(filename='./test.mp4'):
    # filename='test.mp4'
    cap = cv2.VideoCapture(filename)
    total_frame = 0
    while(True):
        ret, frame = cap.read()
        if ret is False:
            break
        total_frame = total_frame + 1
    cap.release()
    # 视频长度：秒
    total_time=getTime(filename)
    # 计算fps
    fps=total_frame/total_time
    return int(fps)
    

# 将帧组合成视频
def frame2video(image_path,video_name):
    image_list=os.listdir(image_path)
    image_list.sort()
    # 第一张图片
    first_image = cv2.imread( image_path + '/' + image_list[0])
    fps = 20
    # fps=getFPS()
    print('fps: ',fps)
    # size
    size= (first_image.shape[1],first_image.shape[0])
    print(size)
    # 编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # MJPG
    # videowriter
    videoWrite = cv2.VideoWriter(video_name,fourcc,fps,size)
    for image in image_list:
        print(image)
        image=image_path + '/' + image
        img = cv2.imread(image,cv2.IMREAD_COLOR)
        # 调整大小 
        img = cv2.resize(img,size,interpolation=cv2.INTER_CUBIC) 
        # 写
        videoWrite.write(img)
    print('video write success')

if __name__=='__main__':
    
    image_path='./saveImg'
    video_name='out.mp4'
    frame2video(image_path,video_name)
    
    
    image_path='./saveSeg'
    video_name='seg.mp4'
    frame2video(image_path,video_name)
    
    image_path='./saveUpdate'
    video_name='update.mp4'
    frame2video(image_path,video_name)
    
    
    

