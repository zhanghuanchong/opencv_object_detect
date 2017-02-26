#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

void colorReduce(const Mat& image,Mat& outImage,int div)
{
    int nr=image.rows;
    int nc=image.cols;
    outImage.create(image.size(),image.type());
    if(image.isContinuous()&&outImage.isContinuous())
    {
        nr=1;
        nc=nc*image.rows*image.channels();
    }
    for(int i=0;i<nr;i++)
    {
        const uchar* inData=image.ptr<uchar>(i);
        uchar* outData=outImage.ptr<uchar>(i);
        for(int j=0;j<nc;j++)
        {
            *outData++=*inData++/div*div+div/2;
        }
    }
}

int main(int argc, char** argv) {
    Mat image;
    image = imread( argv[1], 0 );

    if( argc != 2 || !image.data )
    {
        printf( "No image data \n" );

        return -1;
    }

    namedWindow( "Display Image" );

    Mat newImage;
    newImage.create(image.size(), image.type());
    colorReduce(image, newImage, 128);

    // 计算颜色直方图
    const int channels[3]={0,1,2};
    const int histSize[3]={256,256,256};
    float hranges[2]={0,255};
    const float* ranges[3]={hranges,hranges,hranges};
    MatND hist;
    calcHist(&newImage,1,channels,Mat(),hist,3,histSize,ranges);

    // 直方图归一化
    normalize(hist,hist,1.0);

    // 直方图反向映射
    Mat result;
    calcBackProject(&newImage,1,channels,hist,result,ranges,255);
    // 将结果进行阈值化
    threshold(result,result,255*(0.05),255,THRESH_BINARY);

    imshow( "Display Image", newImage );


    waitKey(0);

    return 0;
}