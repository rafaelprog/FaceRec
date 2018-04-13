#pragma once

#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <fstream>
#include <sstream>

#include "CropFace.h"
#include "Database.h"
using namespace cv;
using namespace std;


string face_cascade_name = "haarcascades\\haarcascade_frontalface_alt.xml";
//CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;
vector<string> names;
int nlabels;
static Mat MatNorm(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}





void eigenFaceTrainer() {
	vector<Mat> images;
	vector<int> labels;

	try {
		string filename = "csv.ext";
		dbread_file(filename, images, labels);

		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	//create algorithm eigenface recognizer
	Ptr<face::FaceRecognizer>  model = face::createFisherFaceRecognizer();
	//train data
	model->train(images, labels);

	model->save("E:/FDB/yaml/eigenface.yml");

	cout << "Training finished...." << endl;
	////get eigenvalue of eigenface model
	//Mat eigenValue = model->getMat("eigenvalues");

	//////get eigenvectors display(eigenface)
	//Mat w = model->getMat("eigenvectors");

	//////get the sample mean from the training data
	//Mat mean = model->getMat("mean");

	//////save or display
	//imshow("mean", MatNorm(mean.reshape(1,images[0].rows)));
	////imwrite(format("%s/mean.png", output_folder.c_str()), MatNorm(mean.reshape(1, images[0].rows)));

	////display or save eigenfaces
	//for (int i = 0; i < min(10, w.cols); i++)
	//{
	//	string msg = format("Eigenvalue #%d = %.5f", i, eigenValue.at<double>(i));
	//	cout << msg << endl;

	//	//get the eigenvector #i
	//	Mat ev = w.col(i).clone();

	//	// Reshape to original size & normalize to [0...255] for imshow.
	//	Mat grayscale = MatNorm(ev.reshape(1, height));
	//	// Show the image & apply a Jet colormap for better sensing.
	//	Mat cgrayscale;
	//	applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
	//	//display or save
	//	imshow(format("eigenface_%d", i), cgrayscale);
	//	//imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), MatNorm(cgrayscale));
	//}

	////display or save image reconstruction
	//for (int num_components = min(w.cols, 10); num_components < min(w.cols, 300); num_components += 15)
	//{
	//	// slice the eigenvectors from the model
	//	Mat evs = Mat(w, Range::all(), Range(0, num_components));
	//	Mat projection = subspaceProject(evs, mean, images[0].reshape(1, 1));
	//	Mat reconstruction = subspaceReconstruct(evs, mean, projection);
	//	// Normalize the result:
	//	reconstruction = MatNorm(reconstruction.reshape(1, images[0].rows));
	//	// Display or save:
	//	imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
	//	//imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);	
	//		
	//}
	waitKey(10000);
}

void fisherFaceTrainer() {
	vector<Mat> images;
	vector<int> labels;
	
	try {
		string filename = "csv.ext";
		//dbread_file(filename, images, labels);
		dbread_cam(VideoCapture(0), images, labels,names);
		nlabels = labels.size();
		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
	}

	vector<Mat> CropedImgs = Crop_SaveDB(images, face_cascade_name);
	Ptr<face::FaceRecognizer> model = face::createFisherFaceRecognizer();
	model->train(CropedImgs, labels);

	int height = CropedImgs[0].rows;

	try {
		model->save("C:\\Users\\muril\\Desktop\\teste.yml");
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
	}


	cout << "Training finished...." << endl;

	waitKey(10000);
}

void LBPHFaceTrainer() {

	vector<Mat> images;
	vector<int> labels;

	try {
		string filename = "csv.ext";
		//dbread_file(filename, images, labels);
		dbread_cam(VideoCapture(0), images, labels, names);
		nlabels = labels.size();
		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
	}

	vector<Mat> CropedImgs = Crop_SaveDB(images, face_cascade_name);
	Ptr<face::FaceRecognizer> model = face::createLBPHFaceRecognizer();
	model->train(CropedImgs, labels);

	int height = CropedImgs[0].rows;

	try {
		model->save("C:\\Users\\muril\\Desktop\\testeLBPH.yml");
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
	}


	cout << "Training finished...." << endl;

	waitKey(10000);
}


void  FisherRecognition() {

	cout << "start recognizing..." << endl;

	Ptr<face::FaceRecognizer>  model = face::createFisherFaceRecognizer();
	model->load("C:\\Users\\muril\\Desktop\\teste.yml");

	Mat testSample = imread("C:\\database\\brenda (1).jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;

	//string classifier = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(face_cascade_name)) {
		cout << " Error loading file" << endl;
	}

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
	}


	namedWindow(window, 1);
	long count = 0;

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;

		count = count + 1;

		if (!frame.empty()) {

			original = frame.clone();

			cvtColor(original, graySacleFrame, COLOR_RGB2GRAY);

			equalizeHist(graySacleFrame, graySacleFrame);
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));
			//face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			string Pname = "";

			for (int i = 0; i < faces.size(); i++)
			{
				Rect face_i = faces[i];

				Mat face = graySacleFrame(face_i);

				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				int label = -1; double confidence = 0;

				model->predict(face_resized,label,confidence);

				cout << " confidence " << confidence << " label : " << label << endl;
				rectangle(original, face_i, CV_RGB(0, 0, 255), 1);
				//cout << label << endl;
				string text = "Detected";
				if (label < nlabels) {
					string text = format("Person is =%d", label);
					Pname = names[label];
				}
				else
					Pname = "unknown";
				/*if (label == 39) {
					string text = format("Person is  = %d", label);
					Pname = "Rafael";
				}
				else if (label == 42) {
					string text = format("Person is  = %d", label);
					Pname = "Brenda";
				}
				else {
					Pname = "unknown";
				}*/
				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);
			}

			putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);
			putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);

			cv::imshow(window, original);
			waitKey(1);
		}
		//if (waitKey(30) >= 0) break;
	}
}
void LBPHRecognition() {

	cout << "start recognizing..." << endl;

	Ptr<face::FaceRecognizer>  model = face::createLBPHFaceRecognizer();
	model->load("C:\\Users\\muril\\Desktop\\testeLBPH.yml");

	Mat testSample = imread("C:\\database\\brenda (1).jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;

	//string classifier = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(face_cascade_name)) {
		cout << " Error loading file" << endl;
	}

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
	}


	namedWindow(window, 1);
	long count = 0;

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;

		count = count + 1;

		if (!frame.empty()) {

			original = frame.clone();

			cvtColor(original, graySacleFrame, COLOR_RGB2GRAY);

			equalizeHist(graySacleFrame, graySacleFrame);
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));
			//face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			string Pname = "";

			for (int i = 0; i < faces.size(); i++)
			{
				Rect face_i = faces[i];

				Mat face = graySacleFrame(face_i);

				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				int label = -1; double confidence = 0;

				model->predict(face_resized,label,confidence);

				cout << " confidence " << confidence << " label : " << label << endl;
				rectangle(original, face_i, CV_RGB(0, 0, 255), 1);
				//cout << label << endl;
				string text = "Detected";
				if (label < nlabels) {
					string text = format("Person is =%d", label);
					Pname = names[label];
				}
				else
					Pname = "unknown";
				/*if (label == 39) {
					string text = format("Person is  = %d", label);
					Pname = "Rafael";
				}
				else if (label == 42) {
					string text = format("Person is  = %d", label);
					Pname = "Brenda";
				}
				else {
					Pname = "unknown";
				}*/
				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);
			}

			putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);
			putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);

			cv::imshow(window, original);
			waitKey(1);
		}
		//if (waitKey(30) >= 0) break;
	}
}
