/*
 * @Author: liubai
 * @Date: 2021-04-21 
 * @LastEditTime: 2021-04-21 
 */
/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<string>

using namespace cv;
using namespace std;

int cnt=0;
String path="./imgFd";

int main(int argc, char *argv[])
{
    //char name[]="./test.mp4";
    string name;
    cout<<"input video name: ";
    cin>>name;
    
    int thresh;
    cout<<"input thresh: ";
    cin>>thresh;
    
    VideoCapture capture;
    capture = VideoCapture(name);
    if(!capture.isOpened()){
        cout<<"ERROR: Did't find this video!"<<endl;
        return 0;
    }

    // 用于遍历capture中的帧，通道数为3，需要转化为单通道才可以处理
    Mat tmpFrame, tmpFrameF;
    // 当前帧，单通道，uchar / Float
    Mat currentFrame, currentFrameF;
    // 上一帧，单通道，uchar / Float
    Mat previousFrame, previousFrameF;

    int frameNum = 0;

    capture >> tmpFrame;
    while(!tmpFrame.empty())
    {
        capture >> tmpFrame;
        cnt++;
        //tmpFrame=cvQueryFrame(capture);
        frameNum++;
        if(frameNum == 1)
        {
            // 第一帧先初始化各个结构，为它们分配空间
            previousFrame.create(tmpFrame.size(), CV_8UC1);
            currentFrame.create(tmpFrame.size(), CV_8UC1);
            currentFrameF.create(tmpFrame.size(), CV_32FC1);
            previousFrameF.create(tmpFrame.size(), CV_32FC1);
            tmpFrameF.create(tmpFrame.size(), CV_32FC1);
        }

        if(frameNum >= 2)
        {
            // 转化为单通道灰度图，此时currentFrame已经存了tmpFrame的内容
            cvtColor(tmpFrame, currentFrame, COLOR_BGR2GRAY);
            currentFrame.convertTo(tmpFrameF, CV_32FC1);
            previousFrame.convertTo(previousFrameF, CV_32FC1);

            // 做差求绝对值
            absdiff(tmpFrameF, previousFrameF, currentFrameF);
            currentFrameF.convertTo(currentFrame, CV_8UC1);
            /*
            在currentFrameMat中找大于20（阈值）的像素点，把currentFrame中对应的点设为255
            此处阈值可以帮助阴影消除
            */
            //threshold(currentFrame, currentFrame, 30, 255, THRESH_BINARY);
            threshold(currentFrameF, currentFrame, thresh, 255.0, THRESH_BINARY);
            
            int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸
            // 获取自定义核
            Mat element = getStructuringElement(MORPH_RECT,
                                                Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
                                                Point( g_nStructElementSize, g_nStructElementSize ));
            // 膨胀
            dilate(currentFrame, currentFrame, element);
            // 腐蚀
            erode(currentFrame, currentFrame, element);
        }

        //把当前帧保存作为下一次处理的前一帧
        cvtColor(tmpFrame, previousFrame, COLOR_BGR2GRAY);

        // 显示图像
        imshow("Camera", tmpFrame);
        imshow("Moving Area", currentFrame);
        waitKey(25);
        //保存图像
        String imagePath = path + "/"  + to_string(cnt) + ".jpg"; 
        cout<<imagePath;
        imwrite(imagePath,currentFrame);
    }
}
