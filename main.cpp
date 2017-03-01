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

int main(int argc, char **argv) {
    Mat image = imread("../input/plan2.jpg");
    colorReduce(image, image, 32);

    const int channels[3] = {0, 1, 2};
    const int histSize[3] = {256, 256, 256};
    float hranges[2] = {0, 255};
    const float *ranges[3] = {hranges, hranges, hranges};
    MatND hist;
    calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges);

    // 直方图归一化
    normalize(hist, hist, 1.0);

    // 直方图反向映射
    Mat result;
    calcBackProject(&image, 1, channels, hist, result, ranges, 255);
    // 将结果进行阈值化
    threshold(result, result, 255 * (0.05), 255, THRESH_BINARY);

    namedWindow("Window");
    imshow("Window", result);

    waitKey(0);

    return 0;
}