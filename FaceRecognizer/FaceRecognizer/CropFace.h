#pragma once
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
using namespace cv;
using namespace std;
CascadeClassifier face_cascade;
Mat CropFace(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	//face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0, cv::Size(90, 90));
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	size_t ib = 0; // ib is index of biggest element
	int ab = 0; // ab is area of biggest element

	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		if (ac > ab)
		{
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		}

		crop = frame(roi_b);
		resize(crop, res, Size(200, 200), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		cvtColor(res, gray, CV_RGB2GRAY); // Convert cropped image to Grayscale

										   // Form a filename
		

		Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	// Show image
	//sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
	text = sstm.str();

	putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);


	if (!gray.empty())
	{

		return gray;

		//cv::waitKey();
	}
	else
	{
		destroyWindow("detected");
	}
} 
vector<Mat> Crop_SaveDB(vector<Mat> imgsDB, string face_cascade_path) {
	face_cascade.load(face_cascade_path);
	vector<Mat> CroppedImgs;
	string filename;
	int filenumber = 1;
	for (int i = 0; i < imgsDB.size(); i++) {
		filename = "cropImg";
		stringstream ssfn;
		ssfn << filenumber << ".jpg";
		filename = ssfn.str();
		filenumber++;
		Mat crpImg = CropFace(imgsDB[i]);
		imwrite("C:\\database\\" + filename, crpImg);
		CroppedImgs.push_back(crpImg);
	}
	return CroppedImgs;
}