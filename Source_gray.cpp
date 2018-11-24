#include "opencv2/opencv.hpp"
#include <ctime>
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
	int whiteRGB = 180;
	Scalar bgColor = Scalar(255, 0, 255);
	String filePlace = "E:/testvs/pdata/1102/";
	String srcfileType = ".jpg";
	String srcfilePlace = "E:/testvs/pdata/srcImg/";
	int fIndex = 10;
	const time_t ctt = time(0);
	cout << asctime(localtime(&ctt)) << std::endl;
	while (true)
	{
		String imgName = to_string(fIndex);
		// image = cv::imread(srcfilePlace + imgName + srcfileType, CV_LOAD_IMAGE_GRAYSCALE);
		image = cv::imread(srcfilePlace + "Cloud_TestData.png", CV_LOAD_IMAGE_GRAYSCALE);
		
		if (!image.data) // Check for invalid input
		{
			const time_t ctt = time(0);
			cout << asctime(localtime(&ctt)) << std::endl;
			cout << "Could not open or find the image" << std::endl;
			system("pause");
			return -1;
		}
 		namedWindow(imgName, WINDOW_AUTOSIZE);  
    	imshow(imgName, image); 

		int allArea = image.cols * image.rows;

		// ====================================================================================================//
		// ============================================ GRABCUT ===============================================//
		// ====================================================================================================//

		Mat result(image.size(), CV_8UC1, Scalar(GC_BGD)); // segmentation result (4 possible values) (second)
		Mat mask;									   // segmentation result (4 possible values) (fist)
		Mat bgModel, fgModel;							   // the models (internally used)
		Mat foregroundTemp(image.size(), CV_8UC3, bgColor);

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
		grabCut(rgbImage,					// input image
				result,					// segmentation result
				cv::Rect(),				// rectangle containing foreground
				bgModel, fgModel,		// models
				2,						// number of iterations
				cv::GC_INIT_WITH_MASK); // use rectangle
										// Get the pixels marked as likely foreground


		// ====================================================================================================//
		// ==================================== FIND CLOUD ROI ================================================//
		// ====================================================================================================//

		Mat foreground(image.size(), CV_8UC3, bgColor);
		Mat toGetContoursWithBGColor(image.size(), CV_8UC3, bgColor);
		Mat foregroundWithBGColor(image.size(), CV_8UC3, bgColor);
		Mat foregroundBinary(image.size(), CV_8UC1, Scalar(0));
		Mat whiteImg(image.size(), CV_8UC1, Scalar(255));
		Mat getContours(image.size(), CV_8UC3, Scalar(0, 0, 0));
		Mat getHulls_b(image.size(), CV_8UC1, Scalar(0));
		Mat imgLBP;
		
		compare(result, GC_PR_FGD, result, cv::CMP_EQ);		// result = mask after grabcut
		rgbImage.copyTo(foreground, result); 				// foreground = foregroundTemp after grabcut
		whiteImg.copyTo(foregroundBinary, result);			// foregroundBinary = one channel foreground 
		// whiteImg.copyTo(toGetContoursWithBGColor, result);			// foregroundBinary = one channel foreground 

		// ----------------------------------- findContours ---------------------------------//
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(foregroundBinary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		vector<vector<Point>> hull(contours.size());

		// ----------------------------------- cloud rois ---------------------------------//
		for (int i = 0; i < contours.size(); i++)
		{
			double subArea = contourArea(contours[i], false);
			if (subArea / allArea > 0.0005)
			{
				int contoursIndex = 0;
				Rect bounding_rect = boundingRect(contours[i]);
				Mat foregroundROI = foreground(bounding_rect);
				imgLBP = LBP(foregroundROI);
				compare(foregroundROI, bgColor, mask, CMP_NE);
				foregroundROI.copyTo(foregroundTemp, mask);
				drawContours(getContours, contours, i, bgColor, CV_FILLED, 8, hierarchy);
				drawContours(toGetContoursWithBGColor, contours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy);
				drawContours(toGetContoursWithBGColor, contours, i, Scalar(93, 255, 0), 5, 8, hierarchy);
				cv::rectangle(getContours, bounding_rect, cv::Scalar(0, 0, 255), 2);
				cv::rectangle(toGetContoursWithBGColor, bounding_rect, cv::Scalar(255), 5);

				convexHull(Mat(contours[i]), hull[i], false);
				drawContours(getHulls_b, hull, i, Scalar(255), CV_FILLED, 8, hierarchy);
				drawContours(image, contours, i, cv::Scalar(0, 0, 255), 2, 8, hierarchy);
				imwrite(filePlace+"/lbp/" + imgName +"_" +to_string(i)+"_lbp.jpg", imgLBP);
				imwrite(filePlace + "/lbp/" + imgName + "_" + to_string(i) + "_roi.jpg", foregroundROI);
			}
		}

		imwrite(filePlace+"/img/" + imgName + "_a.jpg", image);
		imwrite(filePlace+"/img/" + imgName + "_b.jpg", foregroundBinary);
		imwrite(filePlace + "/img/" + imgName + "_c.jpg", getContours);
		imwrite(filePlace+"/img/" + imgName + "_d.jpg", toGetContoursWithBGColor);
		imwrite(filePlace+"/img/" + imgName + "_e.jpg", foreground);
		imwrite(filePlace+"/img/" + imgName + "_f.jpg", foregroundWithBGColor);
		imwrite(filePlace+"/hull/" + imgName + ".jpg", getHulls_b);

		fIndex++;
		break;
	}

	waitKey();
	return 0;
}