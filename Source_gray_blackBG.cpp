#include "opencv2/opencv.hpp"
#include <ctime>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat image;
	int whiteRGB = 150;
	Scalar bgColor = Scalar(255, 0, 255);
	String filePlace = "E:/testvs/pdata/1014/";
	String srcfileType = ".jpg";
	String srcfilePlace = "E:/testvs/pdata/srcImg/cutImg/";
	int fIndex = 1;
	const time_t ctt = time(0);
	cout << asctime(localtime(&ctt)) << std::endl;
	while (true)
	{
		String imgName = to_string(fIndex);
		image = cv::imread(srcfilePlace + imgName + srcfileType, CV_LOAD_IMAGE_GRAYSCALE);

		if (!image.data) // Check for invalid input
		{
			const time_t ctt = time(0);
			cout << asctime(localtime(&ctt)) << std::endl;
			cout << "Could not open or find the image" << std::endl;
			system("pause");
			return -1;
		}
		// namedWindow(imgName, WINDOW_AUTOSIZE);
		// imshow(imgName, image);

		int allArea = image.cols * image.rows;

		// ====================================================================================================//
		// ============================================ GRABCUT ===============================================//
		// ====================================================================================================//

		Mat result(image.size(), CV_8UC1, Scalar(GC_BGD)); // segmentation result (4 possible values) (second)
		Mat mask;										   // segmentation result (4 possible values) (fist)
		Mat bgModel, fgModel;							   // the models (internally used)
		Mat foregroundTemp(image.size(), CV_8UC1, Scalar(0));

		// ---------------------------------------- 前後景MASK ---------------------------------//
		for (int y = 0; y < image.rows; y++)
		{
			uchar *ptr2 = result.ptr<uchar>(y);
			uchar *ptr1 = image.ptr<uchar>(y);
			for (int x = 0; x < image.cols; x++)
			{
				if (!((int)ptr1[x] > whiteRGB))
				{
					ptr2[x] = GC_PR_BGD;
				}
				else if ((int)ptr1[x] > whiteRGB)
				{
					ptr2[x] = GC_PR_FGD;
				}
			}
		}
		compare(result, GC_PR_FGD, mask, CMP_EQ);
		image.copyTo(foregroundTemp, mask); // foregroundTemp = cloud with bgColor before grabcut

		// ----------------------------------- grabcut ---------------------------------//
		Mat rgbImage;
		cvtColor(image, rgbImage, CV_GRAY2BGR);
		grabCut(rgbImage,				// input image
				result,					// segmentation result
				cv::Rect(),				// rectangle containing foreground
				bgModel, fgModel,		// models
				2,						// number of iterations
				cv::GC_INIT_WITH_MASK); // use rectangle
										// Get the pixels marked as likely foreground

		// ====================================================================================================//
		// ==================================== FIND CLOUD ROI ================================================//
		// ====================================================================================================//

		// Mat foreground(image.size(), CV_8UC3, bgColor);
		Mat foregroundBinary(image.size(), CV_8UC1, Scalar(0));
		Mat foregroundBinary1(image.size(), CV_8UC1, Scalar(0));
		Mat whiteImg(image.size(), CV_8UC1, Scalar(255));
		Mat afterFilterArea(image.size(), CV_8UC1, Scalar(0));
		Mat getContours(image.size(), CV_8UC1, Scalar(0));
		// Mat getHulls_b(image.size(), CV_8UC1, Scalar(0));
		// Mat imgLBP;

		compare(result, GC_PR_FGD, result, cv::CMP_EQ); // result = mask after grabcut
		image.copyTo(foregroundBinary, result);			// foregroundBinary = one channel foreground
		whiteImg.copyTo(foregroundBinary1, result);		// foregroundBinary = one channel foreground

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(foregroundBinary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		cout << contours.size();
		for (int i = 0; i < contours.size(); i++)
		{
			double subArea = contourArea(contours[i], false);
			if (subArea / allArea > 0.0005)
			{
				drawContours(getContours, contours, i, Scalar(255), CV_FILLED, 8, hierarchy);
			}
		}

		compare(getContours, Scalar(255), getContours, CMP_EQ);
		foregroundBinary.copyTo(afterFilterArea, getContours);			// foregroundBinary = one channel foreground
		imwrite(filePlace + "/grabcut_area_filter/" + imgName + ".jpg", afterFilterArea);
		imwrite(filePlace + "/grabcut/" + imgName + ".jpg", foregroundBinary);
		imwrite(filePlace + "/grabcut_before/" + imgName + ".jpg", foregroundTemp);
		fIndex++;
	}

	waitKey();
	return 0;
}