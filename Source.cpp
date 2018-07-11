#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

Mat LBP(Mat src_image)
{
	Mat temp_image = src_image;
	Mat Image(temp_image.rows, temp_image.cols, CV_8UC1);
	Mat lbp(temp_image.rows, temp_image.cols, CV_8UC1);

	if (temp_image.channels() == 3)
		cvtColor(temp_image, Image, CV_BGR2GRAY);

	//imshow("src_image", Image);

	int center = 0;
	int center_lbp = 0;

	for (int row = 1; row < Image.rows - 1; row++)
	{
		for (int col = 1; col < Image.cols - 1; col++)
		{

			center = Image.at<uchar>(row, col);
			center_lbp = 0;

			if (center <= Image.at<uchar>(row - 1, col - 1))
				center_lbp += 1;

			if (center <= Image.at<uchar>(row - 1, col))
				center_lbp += 2;

			if (center <= Image.at<uchar>(row - 1, col + 1))
				center_lbp += 4;

			if (center <= Image.at<uchar>(row, col - 1))
				center_lbp += 8;

			if (center <= Image.at<uchar>(row, col + 1))
				center_lbp += 16;

			if (center <= Image.at<uchar>(row + 1, col - 1))
				center_lbp += 32;

			if (center <= Image.at<uchar>(row + 1, col))
				center_lbp += 64;

			if (center <= Image.at<uchar>(row + 1, col + 1))
				center_lbp += 128;

			//cout << "center lbp value: " << center_lbp << endl;
			lbp.at<uchar>(row, col) = center_lbp;
		}
	}

	//imshow("lbp_image", lbp);
	//waitKey(0);
	//destroyAllWindows();

	return lbp;
}

int main()
{
	Mat image;
	// "testData/Cloud_TestData.png"
	//testData/s/cloud1.jpg
	String imgName = "cloud2";
	String srcfileType = ".jpg";
	String srcfilePlace = "testData/s/";
	image = cv::imread(srcfilePlace+imgName+srcfileType);

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// define bounding rectangle
	int border = 20;
	int border2 = border + border;
	int whiteRGB = 120;
	int whiteV = 150;
	int leftX = image.cols, leftY = image.rows, rightX = 0, rightY = 0;
	int allArea = image.cols * image.rows;
	int ex = 0;
	String filePlace = "E:/testvs/pdata/0711/useimg";
	Scalar bgColor = Scalar(255, 0, 255);
	cout << leftX << " " << leftY << " " << rightX << " " << rightY << "\n";
	Mat imgBinary2;

	Mat imgHSV, imgGray;
	Mat imgBinary(image.size(), CV_8UC1, Scalar(0));
	Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(3, 3));

	//HSV
	cvtColor(image, imgHSV, CV_BGR2HSV);
	for (int y = 0; y < imgHSV.rows; y++)
	{
		uchar *ptr1 = imgHSV.ptr<uchar>(y);
		uchar *ptr2 = imgBinary.ptr<uchar>(y);
		for (int x = 0; x < imgHSV.cols; x++)
		{
			if ((int)ptr1[3 * x + 2] > 100 && (int)ptr1[3 * x + 1] < 50)
				ptr2[x] = 255;
		}
	}
	erode(imgBinary, imgBinary2, erodeStruct, Point(-1, -1), 2);
	dilate(imgBinary2, imgBinary2, Mat(), Point(-1, -1), 5);
	/*namedWindow("imgHSV");
	imshow("imgHSV", imgHSV);*/
	imwrite(filePlace + imgName + "-imgHSV.jpg", imgHSV);
	namedWindow("erode_dilate");
	imshow("erode_dilate", imgBinary2);
	imwrite(filePlace + imgName + "-erode_dilate.jpg", imgBinary2);
	// namedWindow("imgBinary");
	// imshow("imgBinary", imgBinary);
	imwrite(filePlace + imgName + "-imgBinary.jpg", imgBinary);
	Mat result(image.size(), CV_8UC1, Scalar(GC_BGD)); // segmentation result (4 possible values) (second)
	Mat resultTemp;									   // segmentation result (4 possible values) (fist)
	Mat bgModel, fgModel;							   // the models (internally used)
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
			/*if ((int)ptr1[x] == 0 )
			{
				ptr2[x] = GC_PR_BGD;
			}
			else if ((int)ptr1[x] == 255)
			{
				ptr2[x] = GC_PR_FGD;
			}*/
		}
	}

	compare(result, GC_PR_FGD, resultTemp, CMP_EQ);
	image.copyTo(foregroundTemp, resultTemp); // bg pixels not copied

	grabCut(image,					// input image
			result,					// segmentation result
			cv::Rect(),				// rectangle containing foreground
			bgModel, fgModel,		// models
			2,						// number of iterations
			cv::GC_INIT_WITH_MASK); // use rectangle
									// Get the pixels marked as likely foreground

	compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	Mat foreground(image.size(), CV_8UC3, bgColor);
	Mat foregroundBinary(image.size(), CV_8UC1, cv::Scalar(0));
	Mat whiteImg(image.size(), CV_8UC1, cv::Scalar(255));
	Mat getContours(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	Mat imgLBP;
	image.copyTo(foreground, result); // bg pixels not copied
	whiteImg.copyTo(foregroundBinary, result);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(foregroundBinary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++)
	{
		double subArea = contourArea(contours[i], false);
		if (subArea / allArea > 0.0005)
		{

			int contoursIndex = 0;

			Rect bounding_rect = boundingRect(contours[i]);

			drawContours(getContours, contours, i, bgColor, CV_FILLED, 8, hierarchy);
			cv::rectangle(getContours, bounding_rect, cv::Scalar(0, 0, 255), 2);

			Mat foregroundROI = foreground(bounding_rect);
			Mat foregroundROI_HSV;
			cvtColor(foregroundROI, foregroundROI_HSV, CV_BGR2HSV);
			Mat test(bounding_rect.size(), CV_8UC3, Scalar(0, 0, 0));
			int cloudPixels = 0;
			int cloudB = 0, cloudG = 0, cloudR = 0;
			int cloudH = 0, cloudS = 0, cloudV = 0;
			for (int h = 0; h < bounding_rect.height; h++)
			{
				uchar *ptr1 = foregroundROI.ptr<uchar>(h);
				uchar *ptr2 = foregroundROI_HSV.ptr<uchar>(h);
				uchar *ptr3 = test.ptr<uchar>(h);
				for (int w = 0; w < bounding_rect.width; w++)
				{
					if (Scalar((int)ptr1[3 * w], (int)ptr1[3 * w + 1], (int)ptr1[3 * w + 2]) != bgColor)
					{
						cloudPixels++;
						cloudB += ptr1[3 * w];
						cloudG += ptr1[3 * w + 1];
						cloudR += ptr1[3 * w + 2];
						cloudH += ptr2[3 * w];
						cloudS += ptr2[3 * w + 1];
						cloudV += ptr2[3 * w + 2];
						ptr3[3 * w] = ptr1[3 * w];
						ptr3[3 * w + 1] = ptr1[3 * w + 1];
						ptr3[3 * w + 2] = ptr1[3 * w + 2];
					}
				}
			}

			drawContours(image, contours, i, cv::Scalar(0, 0, 255), 2, 8, hierarchy);
			imgLBP = LBP(foregroundROI);
			String ii = to_string(i);
			namedWindow(ii);
			imshow(ii, imgLBP);
			imwrite(filePlace + imgName+"-"+ii+".jpg", imgLBP);
			cout<<"\n\ncontour-"<< i <<":\nArea: "<< subArea<<"   Rate: "<<subArea / allArea << "\n";
			cout << "ROI  x: " << bounding_rect.x << " y: " << bounding_rect.y << " width: " << bounding_rect.width << " height: " << bounding_rect.height << "\n";
			cout<<"average:\n";
			cout<<"R: "<<cloudR / cloudPixels<<"   G: "<<cloudG / cloudPixels<<"   B: "<<cloudB / cloudPixels<<"\n";
			cout<<"H: "<<cloudH / cloudPixels<<"   S: "<<cloudS / cloudPixels<<"   V: "<<cloudV / cloudPixels<<"\n";

			
		}
	}

	// draw rectangle on original image
	//cv::rectangle(image, rectangle, cv::Scalar(255, 255, 255), 1);
	namedWindow("Image");
	imshow("Image", image);
	imwrite(filePlace + imgName + "-Image.jpg", image);
	namedWindow("foregroundBinary");
	imshow("foregroundBinary", foregroundBinary);
	imwrite(filePlace + imgName + "-foregroundBinary.jpg", foregroundBinary);

	namedWindow("foregroundTemp");
	imshow("foregroundTemp", foregroundTemp);
	imwrite(filePlace + imgName + "-foregroundTemp.jpg", foregroundTemp);

	// display result
	namedWindow("foreground");
	imshow("foreground", foreground);
	imwrite(filePlace + imgName + "-foreground.jpg", foreground);

	/*namedWindow("getContours");
	imshow("getContours", getContours);
	imwrite(filePlace + imgName + "-getContours.jpg", getContours);
*/

	waitKey();
	return 0;
}