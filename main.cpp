#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

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
void drawOnImage(const Mat &binary, Mat &image) {
    for (int i = 0; i < binary.rows; i++) {
        // 获取行指针
        const uchar *data = binary.ptr<uchar>(i);
        for (int j = 0; j < binary.cols; j++) {
            if (data[j]) //角点图像上的白点
                circle(image, Point(j, i), 8, Scalar(0, 255, 0));// 画圈
        }
    }
}

void drawDetectLines(Mat &image, const vector<Vec4i> &lines, Scalar &color) {
    // 将检测到的直线在图上画出来
    vector<Vec4i>::const_iterator it = lines.begin();
    while (it != lines.end()) {
        Point pt1((*it)[0], (*it)[1]);
        Point pt2((*it)[2], (*it)[3]);
        line(image, pt1, pt2, color, 2); //  线条宽度设置为2
        ++it;
    }
}

int main(int argc, char **argv) {
    Mat image1=imread("../input/plan2.jpg");
    Mat image2=imread("../input/plan1.jpg");
    // 检测surf特征点
    vector<KeyPoint> keypoints1,keypoints2;
    SurfFeatureDetector detector(400);
    detector.detect(image1, keypoints1);
    detector.detect(image2, keypoints2);
    // 描述surf特征点
    SurfDescriptorExtractor surfDesc;
    Mat descriptros1,descriptros2;
    surfDesc.compute(image1,keypoints1,descriptros1);
    surfDesc.compute(image2,keypoints2,descriptros2);

    // 计算匹配点数
    BruteForceMatcher<L2<float>>matcher;
    vector<DMatch> matches;
    matcher.match(descriptros1,descriptros2,matches);
    std::nth_element(matches.begin(),matches.begin()+24,matches.end());
    matches.erase(matches.begin()+25,matches.end());
    // 画出匹配图
    Mat imageMatches;
    drawMatches(image1,keypoints1,image2,keypoints2,matches,
                imageMatches,Scalar(255,0,0));

    namedWindow("image2");
    imshow("image2",image2);
    waitKey();
    return 0;
}