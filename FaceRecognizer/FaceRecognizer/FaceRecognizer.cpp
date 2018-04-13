// FaceRecognizer.cpp : Define o ponto de entrada para a aplicação de console.
//

#include "stdafx.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include "Recognizer.h"

using namespace cv;
using namespace std;



int main() {

	LBPHFaceTrainer();
	LBPHRecognition();

	system("pause");
	return 0;
}