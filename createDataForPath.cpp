#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat image;
	String filePlace = "E:/testvs/pdata/0713/hull128/";
	String srcfileType = ".jpg";
	String srcfilePlace = "E:/testvs/pdata/0713/hull/";
	int fIndex = 452;
	int ROISize = 128;
	int globalIndex = 157;
	int x, y, col, row, key;

	Mat subImg;
	while (true)
	{
		fIndex++;
		
		String imgName = to_string(fIndex);
		image = cv::imread(srcfilePlace + imgName + srcfileType);
		if (!image.data) // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
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
				namedWindow(imgName + "_" + to_string(i) + to_string(j));
				imshow(imgName + "_" + to_string(i) + to_string(j), subImg);
				cout << "fIndex " << fIndex <<" "<< i<<" "<<j<<" key = ";
				key = waitKey(0);
				cout << key << endl;
				// key num 1(left) = 49, num 2(top) = 50, num 3(right) = 51;
				imwrite(filePlace + to_string(globalIndex) + "_" + to_string(key - 50) + ".jpg", subImg);
				globalIndex++;
			}
		}
	}

	return 0;
}