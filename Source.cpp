#include "opencv2/opencv.hpp"
#include <ctime>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat image;
	int whiteRGB = 120;
	Scalar bgColor = Scalar(255, 0, 255);
	String filePlace = "E:/testvs/pdata/0713/";
	String srcfileType = ".jpg";
	String srcfilePlace = "E:/testvs/pdata/srcImg/";
	int fIndex = 1;
	const time_t ctt = time(0);
	cout << asctime(localtime(&ctt)) << std::endl;
	while (fIndex < 3)
	{
		String imgName = to_string(fIndex);
		image = cv::imread(srcfilePlace + imgName + srcfileType);

		if (!image.data) // Check for invalid input
		{
			const time_t ctt = time(0);
			cout << asctime(localtime(&ctt)) << std::endl;
			cout << "Could not open or find the image" << std::endl;
			system("pause");
			return -1;
		}

		int allArea = image.cols * image.rows;

		//grabCut
		Mat result(image.size(), CV_8UC1, Scalar(GC_BGD)); // segmentation result (4 possible values) (second)
		// Mat resultTemp;									   // segmentation result (4 possible values) (fist)
		Mat bgModel, fgModel; // the models (internally used)
		Mat foregroundTemp(image.size(), CV_8UC3, bgColor);

		// 前後景MASK
		for (int y = 0; y < image.rows; y++)
		{
			uchar *ptr2 = result.ptr<uchar>(y);
			uchar *ptr1 = image.ptr<uchar>(y);
			for (int x = 0; x < image.cols; x++)
			{
				if (!((int)ptr1[3 * x] > whiteRGB && (int)ptr1[3 * x + 1] > whiteRGB && (int)ptr1[3 * x + 2] > whiteRGB))
				{
					ptr2[x] = GC_PR_BGD;
				}
				else if (((int)ptr1[3 * x] > whiteRGB && (int)ptr1[3 * x + 1] > whiteRGB && (int)ptr1[3 * x + 2] > whiteRGB))
				{
					ptr2[x] = GC_PR_FGD;
				}
			}
		}

		// compare(result, GC_PR_FGD, resultTemp, CMP_EQ);
		// image.copyTo(foregroundTemp, resultTemp); // bg pixels not copied

		grabCut(image,					// input image
				result,					// segmentation result
				cv::Rect(),				// rectangle containing foreground
				bgModel, fgModel,		// models
				2,						// number of iterations
				cv::GC_INIT_WITH_MASK); // use rectangle
										// Get the pixels marked as likely foreground

		compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
		Mat foreground(image.size(), CV_8UC3, bgColor);
		image.copyTo(foreground, result); //RGB cloud on backgroundColor

		Mat foregroundBinary(image.size(), CV_8UC1, cv::Scalar(0));
		Mat whiteImg(image.size(), CV_8UC1, cv::Scalar(255));
		whiteImg.copyTo(foregroundBinary, result); // White cloud on black background

		Mat getContours(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		Mat getHulls_b(image.size(), CV_8UC1, cv::Scalar(0));
		//Mat imgLBP;

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(foregroundBinary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		vector<vector<Point>> hull(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			double subArea = contourArea(contours[i], false);
			if (subArea / allArea > 0.0005)
			{
				convexHull(Mat(contours[i]), hull[i], false);
				drawContours(getHulls_b, hull, i, Scalar(255), CV_FILLED, 8, hierarchy);
			}
		}
		namedWindow("image" + to_string(fIndex));
		imshow("image" + to_string(fIndex), image);
		namedWindow("hull" + to_string(fIndex));
		imshow("hull" + to_string(fIndex), getHulls_b);

		fIndex++;
	}

	waitKey();
	return 0;
}