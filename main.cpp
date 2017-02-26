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
    image = imread( "../input/plan1.jpg", 0 );

    namedWindow( "Display Image" );

    Mat newImage;
    newImage.create(image.size(), image.type());
    colorReduce(image, newImage, 128);

    imshow( "Display Image", newImage );


    waitKey(0);

    return 0;
}