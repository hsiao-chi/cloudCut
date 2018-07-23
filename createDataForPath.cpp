#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat image;
	String filePlace = "E:/testvs/pdata/0723/hull64x32/";
	String srcfileType = ".png";
	String srcfilePlace = "E:/testvs/pdata/0723/hull/";
	int fIndex = 453;
	int ROISize_width = 64;
	int ROISize_height = 32;
	float ROIArea = ROISize_width * ROISize_height;
	float cloudRateThreshold = 0.1;
	float cloudDiffThreshold = 0.05;
	float noCloudRateThreshold = 0.1;
	int fullBlack = 0;
	int fullWhite = 0;
	int globalIndex = 1;
	int x, y, col, row, key;
	int leftCloud = 0, rightCloud = 0;
	Mat subImg;
	time_t ctt = time(0);
	cout << asctime(localtime(&ctt)) << std::endl;

// -------------- Start -------------- //
	while (true)
	{

	// --------------- read image -------------- //
		String imgName = to_string(fIndex);
		image = cv::imread(srcfilePlace + imgName + srcfileType, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data) // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			ctt = time(0);
			cout << asctime(localtime(&ctt)) << std::endl;
			system("pause");
			return -1;
		}
		resize(image,image,Size(image.cols/4,image.rows/4),0,0,CV_INTER_AREA);
	// ----------------- 算一張圖可以放下幾行幾列的小圖 ------------------//
		x = 0;
		y = 0;
		col = image.cols / ROISize_width;
		col = (image.cols % ROISize_width) > (ROISize_width / 2) ? (col + 1) : col;
		row = image.rows / ROISize_height;
		row = (image.rows % ROISize_height) > (ROISize_height / 2) ? (row + 1) : row;

	// ----------------- 切割開始 ------------------ //
		for (int i = 0; i < row; i++)
		{
			// ----------- (x, y)) = 目前小圖左上角 ------------ //
			y = i * ROISize_height;
			if (i == row - 1)
				y = image.rows - ROISize_height;
			for (int j = 0; j < col; j++)
			{
				x = j * ROISize_width;
				if (j == col - 1)
					x = image.cols - ROISize_width;
				// 小圖切出，存入subImg
				subImg = image(Rect(x, y, ROISize_width, ROISize_height)); 

			// ---------- 算image左右兩邊白色面積 ------------ //
				leftCloud = 0;
				rightCloud = 0;
				for (int k = 0; k < subImg.rows; k++)
				{
					uchar *ptr1 = subImg.ptr<uchar>(k);
					for (int z = 0; z < subImg.cols; z++)
					{
						if ((int)ptr1[z] == 255)
						{
							if (z < (subImg.cols / 2))
							{
								leftCloud++;
							}
							else
							{
								rightCloud++;
							}
						}
					}
				}
			// ----------- Target設定 (key = target) ------------- //
				if (leftCloud == 0 && rightCloud == 0)		//整張都黑的
				{
					key = 0;
					fullBlack++;
					if (fullBlack == 1000)
						cout << "FB=> " << globalIndex;
					if (fullBlack >= 1000)		// 超過1000張就不存
						continue;
				}
				else if (leftCloud == (int)(ROIArea / 2) && rightCloud == (int)(ROIArea / 2))		// 整張都白的
				{
					key = 0;
					fullWhite++;
					if (fullWhite == 1000)
						cout << "FW=> " << globalIndex;
					if (fullWhite >= 1000)		// 超過1000張就不存
						continue;
				}
				else if ((abs(leftCloud - rightCloud) / (ROIArea) < cloudDiffThreshold))		// 左右兩邊面積差異比例小於cloudDiffThreshold
				{
					key = 0;
				}
				else if (1-((leftCloud + rightCloud) / ROIArea) > noCloudRateThreshold)		// 總黑色面積比例大於noCloudRateThreshold
				{
					if (leftCloud > rightCloud && leftCloud / (ROIArea / 2) > cloudRateThreshold)	// 左邊多於右邊，且左邊白色面積比例大於cloudRateThreshold
					{
						key = 2;
					}
					else if (leftCloud <= rightCloud && rightCloud / (ROIArea / 2) > cloudRateThreshold)		//// 右邊多於左邊，且右邊白色面積比例大於cloudRateThreshold
					{
						key = 1;
					}
				}

				else
				{
					key = 0;
				}

				//namedWindow(imgName + "_" + to_string(i) + to_string(j));
				//imshow(imgName + "_" + to_string(i) + to_string(j), subImg);
				//key = waitKey(0);
				//cout << globalIndex << "  fIndex " << fIndex << " " << i << " " << j << " key = " << key << endl;
				float temp = (abs(leftCloud - rightCloud) / (ROIArea));
				//cout << "    L = " << leftCloud << " R = " << rightCloud << "  diff= " << temp << endl;
				// key num 1(left) = 49, num 2(top) = 50, num 3(right) = 51;
				imwrite(filePlace + to_string(globalIndex) + "_" + to_string(key) + ".png", subImg);
				globalIndex++;
			}
		}
		fIndex++;
	}
	ctt = time(0);
	cout << asctime(localtime(&ctt)) << std::endl;
	system("pause");
	return 0;
}