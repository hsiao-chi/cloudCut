#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat image;
	String filePlace = "E:/testvs/pdata/0717/hull128/";
	String srcfileType = ".jpg";
	String srcfilePlace = "E:/testvs/pdata/0713/hull/";
	int fIndex = 453;
	int ROISize = 128;
	float ROIArea = ROISize * ROISize;
	float cloudRateThreshold = 0.1;
	float cloudDiffThreshold = 0.05;
	int fullBlack = 0;
	int fullWhite = 0;
	int globalIndex = 1;
	int x, y, col, row, key;
	int leftCloud = 0, rightCloud = 0;
	Mat subImg;
	time_t ctt = time(0);
	cout << asctime(localtime(&ctt)) << std::endl;
	while (true)
	{
		

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
		x = 0;
		y = 0;
		col = image.cols / ROISize;
		col = (image.cols % ROISize) > (ROISize / 2) ? (col + 1) : col;
		row = image.rows / ROISize;
		row = (image.rows % ROISize) > (ROISize / 2) ? (row + 1) : row;

		for (int i = 0; i < row; i++)
		{
			y = i * ROISize;
			if (i == row - 1)
				y = image.rows - ROISize;
			for (int j = 0; j < col; j++)
			{
				x = j * ROISize;
				if (j == col - 1)
					x = image.cols - ROISize;
				subImg = image(Rect(x, y, ROISize, ROISize));
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

				if (leftCloud == 0 && rightCloud == 0)
				{
					key = 0;
					fullBlack++;
					if (fullBlack == 1000) cout << "FB=> "<< globalIndex;
					if (fullBlack >= 1000) continue;
				}
				else if (leftCloud == (int)(ROIArea / 2) && rightCloud == (int)(ROIArea / 2)) {
					key = 0;
					fullWhite++;
					if (fullWhite == 1000) cout << "FW=> " << globalIndex;
					if (fullWhite >= 1000) continue;
				}
				else if ((abs(leftCloud - rightCloud) / (ROIArea) < cloudDiffThreshold))
				{
					key = 0;
				}
				else if (leftCloud > rightCloud && leftCloud > cloudRateThreshold)
				{
					key = 2;
				}
				else if (leftCloud <= rightCloud && rightCloud > cloudRateThreshold)
				{
					key = 1;
				}
				else{
					key = 0;
				}

				//namedWindow(imgName + "_" + to_string(i) + to_string(j));
				//imshow(imgName + "_" + to_string(i) + to_string(j), subImg);
				//key = waitKey(0);
				//cout << globalIndex << "  fIndex " << fIndex << " " << i << " " << j << " key = " << key << endl;
				float temp = (abs(leftCloud - rightCloud) / (ROIArea));
				//cout << "    L = " << leftCloud << " R = " << rightCloud << "  diff= " << temp << endl;
				// key num 1(left) = 49, num 2(top) = 50, num 3(right) = 51;
				imwrite(filePlace + to_string(globalIndex) + "_" + to_string(key) + ".jpg", subImg);
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