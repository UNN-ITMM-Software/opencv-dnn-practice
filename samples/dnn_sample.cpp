#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace cv;

const String cmdlineKeys = "{model||}"
                        "{proto||}"
                        "{classes||}"
                        "{image||}";

void print_usage()
{
    cout << "-model=<[Caffe model file].caffemodel>"
            "-proto=<[prototype file].prototxt>"
            "-classes=<[human-readable classes file].txt>"
            "-image=<image-to-analyze>" << endl;
}


int main(int argc, char** argv)
{

    CommandLineParser parser(argc, argv, cmdlineKeys);
    const String ModelPath = parser.get<cv::String>("model");
    const String ProtoPath = parser.get<cv::String>("proto");
    const String ClassesPath = parser.get<cv::String>("classes");
    const String ImagePath = parser.get<cv::String>("image");
    
    if (!parser.check())
    {
        parser.printErrors();
        print_usage();
        return 1;
    }
    Point maxIdx;
    double maxVal;

    try
    {
        const Mat img = imread(ImagePath);

        dnn::Net net = dnn::readNetFromCaffe(ProtoPath, ModelPath);
        const Mat reversedRB = dnn::blobFromImage(img, 1.0, Size(224,224));

        const String imgName = "data";
        net.setInput(reversedRB, imgName);

        const Mat out = net.forward(String("prob"));

        minMaxLoc(out, NULL, &maxVal, NULL, &maxIdx);

//        cout << "INDEX = " << maxIdx.x << "." << maxIdx.y << ", LOC = " << maxVal << endl;
        
    }
    catch (Exception& ex)
    {
        cout << "caught exception: " << ex.what() << endl;
        return 1;
    }

    ifstream fs(ClassesPath);
    string outClass;
    for(unsigned i = 0; i < maxIdx.x ; ++i) {
        getline(fs,outClass);
    }
    cout << "RESULT: " << outClass << endl;

    return 0;
}
