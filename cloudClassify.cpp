#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector> 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace ml;

String record = "E:\\testvs\\pdata\\srcImg\\lbpRename\\";
String loadLocationC = "E:\\testvs\\pdata\\srcImg\\lbpRename\\cloud\\";
String loadLocationO = "E:\\testvs\\pdata\\srcImg\\lbpRename\\other\\";
String fileType = ".jpg";

int cloudAmount = 498 ;
int otherAmount = 251 ;
float histTemp[256] = { 0 };
float histogramCal[498 + 251][256] = { 0 };
int tag[498 + 251] = { 0 };


HOGDescriptor *hog = new HOGDescriptor(Size(64,64), Size(8,8), Size(4,4), Size(4,4), 9, 1 );

vector< Mat >  hogDatas;

int histogram(Mat lbp, Mat roi) {
	int hcount[256] = { 0 };
	float histogram[256] = { 0 };
	int rows = lbp.rows;
	int cols = lbp.cols;
	int pixelCount = 0;
	for (int r = 0; r < rows; ++r) {
		const uchar *lbpdata = lbp.ptr<uchar>(r);
		const uchar *roidata = roi.ptr<uchar>(r);
		for (int c = 0; c < cols; ++c)
		{
			//cout << roidata[3 * c] << "" << roidata[3 * c + 1] << "" << roidata[3 * c + 2] << endl;
			if (!(roidata[3 * c] == 255 && roidata[3 * c + 1] == 0 && roidata[3 * c + 2] == 255)) {
				hcount[lbpdata[c]]++;
				pixelCount++;
			}
			else {
				//cout << "MAGENTA" << endl;
			}
		}
	}
	lbp.release();
	roi.release();
	for (int z = 0; z<256; z++) {
		//cout << "z = " << z << " hcount = " << hcount[z];
		if (pixelCount == 0)
			histTemp[z] = 0;
		else
			histTemp[z] = (float)hcount[z] / pixelCount;
		//cout << " hist = " << histTemp[z] << " pixel = " << pixelCount << endl;
	}
	//cout << "Have " << pixelCount << "pixels." << endl;
	return 0;
}

void convert_to_ml( Mat& trainData )
{
    //--Convert data
    const int rows = (int)hogDatas.size();
    const int cols = (int)std::max( hogDatas[0].cols, hogDatas[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );

    for( size_t i = 0 ; i < hogDatas.size(); ++i )
    {
        CV_Assert( hogDatas[i].cols == 1 || hogDatas[i].rows == 1 );

        if( hogDatas[i].cols == 1 )
        {
            transpose( hogDatas[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( hogDatas[i].rows == 1 )
        {
            hogDatas[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void hogCompute(Mat lbp){
	vector<float>  hogDescriptors;
	resize(lbp, lbp, Size(64,64), 0, 0, CV_INTER_AREA);
	hog->compute(lbp, hogDescriptors,Size(1,1), Size(0,0)); 
	hogDatas.push_back( Mat( hogDescriptors ).clone() );
}

int main()
{
	Mat lbp, roi;
	for (int i = 1; i <= cloudAmount; i++) {
		//file << loadLocationC+to_string(i)+"_lbp"+fileType << "\n";
		cout << loadLocationC + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationC + to_string(i) + "_lbp" + fileType, 0);
		roi = imread(loadLocationC + to_string(i) + "_roi" + fileType);
		//imshow("lbp", lbp);
		//imshow("roi", roi);
		//histogram(lbp, roi);
		hogCompute(lbp);
		// for (int j = 0; j<256; j++) {
		// 	histogramCal[i - 1][j] = histTemp[j];
		// 	cout << histogramCal[i - 1][j] << endl;
		// }
	}
	for (int i = 1; i <= otherAmount; i++) {
		//file << loadLocationO+to_string(i)+"_lbp"+fileType << "\n";
		cout << loadLocationO + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationO + to_string(i) + "_lbp" + fileType, 0);
		roi = imread(loadLocationO + to_string(i) + "_roi" + fileType);
		//imshow("lbp", lbp);
		//imshow("roi", roi);
		//histogram(lbp, roi);
		hogCompute(lbp);
		// for (int j = 0; j<256; j++) {
		// 	histogramCal[i + cloudAmount - 1][j] = histTemp[j];
		// 	cout << histogramCal[i + cloudAmount - 1][j] << endl;
		// }
	}
	for (int i = 0; i<cloudAmount + otherAmount; i++) {
		if (i<cloudAmount)
			tag[i] = 1;
		else
			tag[i] = -1;
	}
	Mat hogTrain_data;
	convert_to_ml( hogTrain_data );

	const int num_data = cloudAmount + otherAmount; //資料數
	//const int num_column = 256; //欄位數

	//Mat trainingDataMat(num_data, num_column, CV_32FC1, histogramCal);
	Mat labelsMat(num_data, 1, CV_32SC1, tag);
	Ptr<TrainData> trainingData = TrainData::create(hogTrain_data, ROW_SAMPLE, labelsMat);
	
	SVM::ParamTypes params;
	SVM::KernelTypes kernel_type = SVM::RBF ;
	Ptr<SVM> svm = SVM::create();
	svm->setKernel(kernel_type);

	svm->trainAuto(trainingData);
	svm->save(record + "SVM_hog_8x8.xml");

	system("pause");
	return 0;
}