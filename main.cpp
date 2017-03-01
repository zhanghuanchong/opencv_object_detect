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
    Mat image;
    image = imread("../input/plan2.jpg", 0);

    const int channels[1] = {0};
    const int histSize[1] = {256};
    float hranges[2] = {0, 255};
    const float *ranges[1] = {hranges};
    MatND hist;
    calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
    Mat histImage = getHistImg(hist);
    namedWindow("Image hist");
    imshow("Image hist", histImage);

    Mat newImage;
    newImage.create(image.size(), image.type());
    colorReduce(image, newImage, 128);

    namedWindow("Color Reduced Image");
    imshow("Color Reduced Image", newImage);

//    cvtColor(newImage, newImage, CV_BGR2GRAY);


    /*CascadeClassifier cascade;
    string filename("../input/plan1.jpg");
    cascade.load(filename);

    Mat symbol;
    symbol = imread("../input/symbol.jpg");

    vector<Rect> objects;
    cascade.detectMultiScale(symbol, objects);

    for (int i = 0; i < objects.size(); i++) {
        Rect rect = objects[i];
        Point p1 = rect.tl();
        Point p2 = rect.br();
        cout << "Rect #" << i + 1 << p1.x << ", " << p1.y << ", " << p2.x << ", " << p2.y << "." <<endl;
    }*/


    waitKey(0);

    return 0;
}