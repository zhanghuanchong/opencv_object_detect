#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

void colorReduce(const Mat &image, Mat &outImage, int div) {
    int nr = image.rows;
    int nc = image.cols;
    outImage.create(image.size(), image.type());
    if (image.isContinuous() && outImage.isContinuous()) {
        nr = 1;
        nc = nc * image.rows * image.channels();
    }
    for (int i = 0; i < nr; i++) {
        const uchar *inData = image.ptr<uchar>(i);
        uchar *outData = outImage.ptr<uchar>(i);
        for (int j = 0; j < nc; j++) {
            *outData++ = (uchar) (*inData++ / div * div + div / 2);
        }
    }
}

Mat getHistImg(const MatND &hist) {
    double maxVal = 0;
    double minVal = 0;

    //找到直方图中的最大值和最小值
    minMaxLoc(hist, &minVal, &maxVal, 0, 0);
    int histSize = hist.rows;
    Mat histImg(histSize, histSize, CV_8U, Scalar(255));
    // 设置最大峰值为图像高度的90%
    int hpt = static_cast<int>(0.9 * histSize);

    for (int h = 0; h < histSize; h++) {
        float binVal = hist.at<float>(h);
        int intensity = static_cast<int>(binVal * hpt / maxVal);
        line(histImg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
    }

    return histImg;
}

// 标记角点
void drawOnImage(const Mat& binary,Mat& image)
{
    for(int i=0;i<binary.rows;i++)
    {
        // 获取行指针
        const uchar* data=binary.ptr<uchar>(i);
        for(int j=0;j<binary.cols;j++)
        {
            if(data[j]) //角点图像上的白点
                circle(image,Point(j,i),8,Scalar(0,255,0));// 画圈
        }
    }
}

int main(int argc, char **argv) {
    Mat image=imread("../input/plan2.jpg");
   /* // 彩色转灰度
    cvtColor(image,image,CV_BGR2GRAY);
    Mat catEdge;
    morphologyEx(image,catEdge,MORPH_GRADIENT,Mat());

    // 阈值化
    threshold(catEdge,catEdge,40,255,THRESH_BINARY);*/

    // 定义结构元素
    Mat cross(5,5,CV_8U,Scalar(0));
    Mat diamond(5,5,CV_8U,Scalar(1));
    Mat square(5,5,CV_8U,Scalar(1));
    Mat x(5,5,CV_8U,Scalar(0));

    for(int i=0;i<5;i++)
    {
        cross.at<uchar>(2,i)=1;
        cross.at<uchar>(i,2)=1;

    }
    diamond.at<uchar>(0,0)=0;
    diamond.at<uchar>(0,1)=0;
    diamond.at<uchar>(1,0)=0;
    diamond.at<uchar>(4,4)=0;
    diamond.at<uchar>(3,4)=0;
    diamond.at<uchar>(4,3)=0;
    diamond.at<uchar>(4,0)=0;
    diamond.at<uchar>(4,1)=0;
    diamond.at<uchar>(3,0)=0;
    diamond.at<uchar>(0,4)=0;
    diamond.at<uchar>(0,3)=0;
    diamond.at<uchar>(1,4)=0;

    for(int i=0;i<5;i++){
        x.at<uchar>(i,i)=1;
        x.at<uchar>(4-i,i)=1;
    }

    Mat result;
    dilate(image,result,cross);
    erode(result,result,diamond);

    Mat result2;
    dilate(image,result2,x);
    erode(result2,result2,square);
    absdiff(result2,result,result);

    threshold(result,result,40,255,THRESH_BINARY);

    namedWindow("catEdge");imshow("catEdge",result);

    waitKey(0);

    return 0;
}