


#include <stdlib.h>
#include <sys/shm.h>

#include<iostream>
#include<string>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


typedef struct image {
    int w;
    int h;
    int c;
    // float *data;
    unsigned char *data;
}imageBody;



// typedef struct getImage {
//     int w;
//     int h;
//     int c;
//     float *data;
//     //unsigned char *data;
// }getImage;


extern "C" {

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

    // Mat image = Mat(im.h,im.w,CV_8UC3,im.data); 
    Mat image = Mat(im.h,im.w,CV_8UC3);
    image.data = im.data;
/*
    unsigned char* pxvec =image.ptr<unsigned char>(0);
    int count = 0;
    for (int row = 0; row < im.h; row++){
      pxvec = image.ptr<unsigned char>(row);
      for(int col = 0; col < im.w; col++){
        for(int c = 0; c < 3; c++){
          pxvec[col*3+c] = im.data[count];
          count++;
        }
      }
    }
*/
    return image;    
}
imageBody make_empty_image(Mat image){
    imageBody out;
    out.data = 0;
    out.w = image.cols;
    out.h = image.rows;
    out.c = image.dims;
    return out;
}

// void mat_to_ndarray(getImage im ,cv::Mat image){
//     // imageBody im = make_empty_image(image);
//     //im.data = image.data
    
//     im.w = image.cols;
//     im.h = image.rows;
//     im.c = image.channels();


//     // im.data=(float)image.data;
    
//     int height=im.h;
//     int width=im.w;
//     int channels=im.c;
//     cout<<height << width << channels<<endl;
//     int i, j, temp;
//     //for(int k=0;k<3;k++)
//     int count=0;
//     for(i = 0; i < height; i ++){
//         for(j = 0; j < width; j ++){
//             for(int k=0; k<channels ;k++){
//                 im.data[ k* height*width + i * width + j ] = (float)(image.data[ count]);
//                 count++;
                
//                 // temp = (float)(image.data[ count++]);
//                 // im.data[ k*height*width + i * width + j ] = (float)temp;
//             }
//         }
//     }
//     cout<<"count: "<<count<<endl;
    
// }

void init(imageBody image){
    cout<<"test=========== init()"<<endl;
    // cv::Mat img(image.w,image.h,CV_8UC3);
    // img.data = image.data;

    cv::Mat img=convert(image);    
    cv::imwrite("./libTest.jpg",img);
}


float* giveArrayList(cv::Mat image)
{
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


float* getImg()  
{
    cout<<"test==========="<<endl;
    Mat image=cv::imread("./001.jpg");

    return giveArrayList(image);
    
}

float* getSeg(){
    Mat image=cv::imread("./001.jpg");
    Mat Gray;
    if(image.channels() == 3){
        cvtColor(image, Gray, COLOR_BGR2GRAY);
    }
    else{
        cvtColor(image, Gray, COLOR_BGR2GRAY);
    }
    //segModel = Mat::zeros(Gray.size(),CV_8UC1);
    return giveArrayList(Gray);
}



/*
// // test python list ========================
float* giveList(int length) 
{
    float* list2 = (float*)malloc(sizeof(float)*length);
    for (unsigned int i = 0; i < length; i++) 
        list2[i] = 0;
    return list2;
}
// // test python list ========================
*/


}
