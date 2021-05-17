

#include <stdlib.h>
#include <sys/shm.h>

#include<iostream>
#include<string>
#include "opencv2/opencv.hpp"

#include "ViBePlus.h"

using namespace cv;
using namespace std;


typedef struct image {
    int w;
    int h;
    int c;
    unsigned char *data;
}imageBody;


extern "C" {

// =============================== 格式转换用到的函数
// =============================== 
void copy_image_from_bytes(imageBody im, char *pdata){
    int height=im.h;
    int width=im.w;
    int i, j, temp;
    //for(int k=0;k<3;k++)
    for(i = 0; i < height; i ++){
        for(j = 0; j < width; j ++){
            for(int k=0;k<3;k++){
                temp = (int)(pdata[ k*height*width + i * width + j]);
                im.data[ k * height * width +  i * width + j] = (unsigned char)temp;
            }
        }
    }
}

Mat convert(imageBody im){
    cout<<"convert"<<endl;
    Mat image = Mat(im.h,im.w,CV_8UC3);
    image.data = im.data;
    return image;    
}

float* giveArrayList(cv::Mat image){
    int height=image.cols;
    int width=image.rows;
    int channels = image.channels();
    cout << height << width << channels<<endl;
    int length= height * width * channels;

    float* list = (float*)malloc(sizeof(float)*length);
    int count=0;
    if(channels == 1){
        // 灰度图
        for(int i = 0; i < height; i ++)
            for(int j = 0; j < width; j ++)
                list[count ++ ]= (float) image.at<uchar>(i, j);
    }else{
        for(int i = 0; i < height; i ++)
            for(int j = 0; j < width; j ++)
                for(int k=0; k < channels ;k++) 
                    list[count ++ ]= (float) image.at<Vec3b>(i, j)[k]; 
    }
    return list;
}

// float* getSeg(){
//     Mat image=cv::imread("./001.jpg");
//     Mat Gray;
//     if(image.channels() == 3){
//         cvtColor(image, Gray, COLOR_BGR2GRAY);
//     }
//     else{
//         cvtColor(image, Gray, COLOR_BGR2GRAY);
//     }
//     //segModel = Mat::zeros(Gray.size(),CV_8UC1);
//     return giveArrayList(Gray);
// }


// =================================================== vibe 背景提取算法
// =================================================== vibe 背景提取算法

ViBePlus vibeplus;

void init(imageBody firstFrame){    
    cout<<"vibe::init()"<<endl;
    Mat frame = convert(firstFrame);

    // init and run 
    cout<<"framecapture"<<endl;
    vibeplus.FrameCapture(frame);
    cout<<"run"<<endl;
    vibeplus.Run();
}

void run(imageBody image){
    cout<<"vibe::run()" <<endl;
    Mat frame = convert(image);
    // run vibe+ algorithm
    vibeplus.FrameCapture(frame);
    vibeplus.Run();
}

Mat getSeg(){
    Mat SegModel;
    SegModel = vibeplus.getSegModel();
    return SegModel;
}

Mat getUpdate(){
    Mat updateModel;
    updateModel = vibeplus.getUpdateModel();
    return updateModel; 
}

float* getSegList(){
    //segModel = Mat::zeros(Gray.size(),CV_8UC1);
    Mat segModel = getSeg();
    return giveArrayList(segModel);
}

float* getUpdateList(){
    //updateModel = Mat::zeros(Gray.size(),CV_8UC1);
    Mat updateModel = getUpdate();
    return giveArrayList(updateModel);
}


}