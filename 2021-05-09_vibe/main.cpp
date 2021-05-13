/*
 * @Author: liubai
 * @Date: 2021-04-21 
 * @LastEditTime: 2021-04-21 
 */
/*=================================================================
 * Extract Background & Foreground Model by ViBe+ Algorithm using OpenCV Library.
 *
 * Copyright (C) 2017 Chandler Geng. All rights reserved.
 *
 *     This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 *     You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
===================================================================
*/

#include "ViBePlus.h"
#include<iostream>
#include<string>
using namespace std;

int cnt=0;

String path1="./imgSeg";
String path2="./imgUpdate";

int main(int argc, char* argv[])
{
    //char name[]="./test.mp4";
    string name;
    cout<<"input video name: ";
    cin>>name;
    
    
    Mat frame, gray, SegModel, UpdateModel;
    VideoCapture capture;
    //capture = VideoCapture("./Video/Camera Road 01");
    capture = VideoCapture(name);
    if(!capture.isOpened())
    {
        cout<<"ERROR: Did't find this video!"<<endl;
        return 0;
    }
    capture.set(CAP_PROP_FRAME_WIDTH,160);
    capture.set(CAP_PROP_FRAME_HEIGHT,120);
    if (!capture.isOpened())
    {
        cout<<"No camera or video input!"<<endl;
        return -1;
    }

    // 程序运行时间统计变量
    // the Time Statistical Variable of Program Running Time
    double time;
    double start;

    ViBePlus vibeplus;
    bool count = true;

    while (1)
    {
        capture >> frame;
        cnt++;
        if (frame.empty())
            continue;
        // 捕获图像
        vibeplus.FrameCapture(frame);

        start = static_cast<double>(getTickCount());
        vibeplus.Run();
        time = ((double)getTickCount() - start) / getTickFrequency() * 1000;
        cout << "Time of Update ViBe+ Background: " << time << "ms"<<endl;

        SegModel = vibeplus.getSegModel();
        UpdateModel = vibeplus.getUpdateModel();
		morphologyEx(SegModel, SegModel, MORPH_OPEN, Mat());//
        
        imshow("SegModel", SegModel);
        imshow("UpdateModel", UpdateModel);
        imshow("input", frame);
        //vibeplus.drawbox();
        //保存图像
        String imagePath1 = path1 + "/"  + to_string(cnt) + ".jpg";
        String imagePath2 = path2 + "/"  + to_string(cnt) + ".jpg";  
        cout<<imagePath2<<endl;
        cv::imwrite(imagePath1,SegModel);
        //cv::imwrite(imagePath2,UpdateModel);
        if ( waitKey(25) == 27 )    break;
    }

    return 0;
}
