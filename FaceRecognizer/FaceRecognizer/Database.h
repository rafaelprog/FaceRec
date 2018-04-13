#pragma once
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;
static void dbread_file(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {

	//string lugar = "C:\\database\\crop\\";
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			Mat aux = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
			Size size(200, 200);
			resize(aux, aux, size);
			images.push_back(aux);
			labels.push_back(atoi(classlabel.c_str()));
			//lugar.append("1");
			//imwrite( lugar + ".jpg", im);
		}
	}
}
void dbread_cam(VideoCapture cam, vector<Mat>& images, vector<int>& labels, vector<string>& names) {
	int nfaces;
	cout << "Digite o numero de faces que voce deseja treinar:";
	cin >> nfaces;
	for (int i = 0; i < nfaces; i++) {
		system("cls");
		cout << "Por favor olhe para a camera!\n";
		system("PAUSE");
		Mat frame;
		int a = 0;
		while (a < 50) {
			cam >> frame;
			images.push_back(frame);
			labels.push_back(i);
			a++;
		}
		string nome;
		cout << "Digite o nome do dono da face capturada:\n";
		cin >> nome;
		names.push_back(nome);
	}
	
}