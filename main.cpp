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
    Mat I=imread("../input/plan2.jpg");
    cvtColor(I,I,CV_BGR2GRAY);

    Mat contours;
    Canny(I,contours,125,350);
    threshold(contours,contours,128,255,THRESH_BINARY);

    namedWindow("Canny");
    imshow("Canny",contours);
    waitKey();
    return 0;
}