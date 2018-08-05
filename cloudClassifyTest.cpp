#include <iostream>
#include <vector> 
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>


using namespace std;
using namespace cv;
using namespace ml;

String record = "E:\\testvs\\pdata\\srcImg\\lbpRename\\";
String loadLocation = "E:\\testvs\\pdata\\srcImg\\lbpRename\\cloud\\";
String loadLocationNot = "E:\\testvs\\pdata\\srcImg\\lbpRename\\other\\";
String fileType = ".jpg";
String svmFileName = "SVM_hog.xml";
int cloudAmount = 498;
int otherAmount = 251;
float histTemp[256] = {0};

int histogram(Mat lbp, Mat roi)
{
    int hcount[256] = {0};
    float histogram[256] = {0};
    int rows = lbp.rows;
    int cols = lbp.cols;
    int pixelCount = 0;
    for (int r = 0; r < rows; ++r)
    {
        const uchar *lbpdata = lbp.ptr<uchar>(r);
        const uchar *roidata = roi.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c)
        {
            if (!(roidata[3 * c] == 255 && roidata[3 * c + 1] == 0 && roidata[3 * c + 2] == 255))
            {
                hcount[lbpdata[c]]++;
                pixelCount++;
            }
            else
            {
                //cout << "MAGENTA" << endl;
            }
        }
    }
    lbp.release();
    roi.release();
    for (int z = 0; z < 256 && pixelCount != 0; z++)
    {
        //cout << "z = " << z << " hcount = ";
        //cout << hcount[z];
        histTemp[z] = (float)hcount[z] / pixelCount;
        //cout << " hist = " << histTemp[z] << " pixel = " << pixelCount << endl;
    }
    //cout << "Have " << pixelCount << "pixels." << endl;
    return 0;
}

HOGDescriptor *hog = new HOGDescriptor(Size(64, 64), Size(8, 8), Size(4, 4), Size(4, 4), 9, 1);
//vector<Mat> hogDatas;

void convert_to_ml(Mat &trainData, Mat hogDatas)
{
    //--Convert data
    // const int rows = (int)hogDatas.size();
    const int cols = (int)std::max(hogDatas.cols, hogDatas.rows);
    Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = Mat(1, cols, CV_32FC1);

    // for (size_t i = 0; i < hogDatas.size(); ++i)
    // {
        CV_Assert(hogDatas.cols == 1 || hogDatas.rows == 1);

        if (hogDatas.cols == 1)
        {
            transpose(hogDatas, tmp);
            tmp.copyTo(trainData.row(0));
        }
        else if (hogDatas.rows == 1)
        {
            hogDatas.copyTo(trainData.row(0));
        }
    // }
}

vector<float> hogCompute(Mat lbp)
{
    vector<float> hogDescriptors;
    resize(lbp, lbp, Size(64, 64), 0, 0, CV_INTER_AREA);
    hog->compute(lbp, hogDescriptors, Size(1, 1), Size(0, 0));
	cout << "hogDescriptors:  size->\n"<< hogDescriptors.size()<<endl;
	/*for (int i = 0; i < hogDescriptors.size(); i++)
	{
		cout << hogDescriptors[i]<<"  ";
	}
	cout << endl;*/
    return hogDescriptors;
    // hogDatas.push_back( Mat( hogDescriptors ).clone() );
}

bool classification(Mat src)
{
    Ptr<SVM> svm = SVM::create();
    svm = SVM::load(record + svmFileName);
    int response = svm->predict(src);
    return response;
}

int main()
{
    Mat lbp, roi;
    for (int i = 1; i <= 5; i++)
    {
        lbp = imread(loadLocation + to_string(i) + "_lbp" + fileType, 0);
        /*HISTOGRAM
        roi = imread(loadLocationNot + to_string(i) + "_roi" + fileType);
        //imshow("lbp", lbp);
        //imshow("roi", roi);
        histogram(lbp, roi);
        float testData[256] = { 0 };
        for (int i = 0; i < 256; i++) {
            testData[i] = histTemp[i];
            //cout << testData[i] << endl;
        }
        Mat src(1, 256, CV_32FC1, testData);
        */
        vector<float> src = hogCompute(lbp);
        Mat hogTrain_data;
        convert_to_ml(hogTrain_data, Mat(src));
        bool flag = classification(hogTrain_data);
        cout << "flag = " << flag << endl;
    }
	waitKey(0);
    system("pause");
    return 0;
}