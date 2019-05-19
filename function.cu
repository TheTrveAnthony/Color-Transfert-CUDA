#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <cmath>
#include "function.h"
#include "matrix.h"

 
 
using namespace std::cout;
using namespace std::endl;
using namespace std::string;
using namespace cv;



void transfert(string nom1, string nom2){

	/////////// First of all let's load the images we'll use

	Mat target = imread(nom1, 1);
	Mat source = imread(nom2, 1);

	Mat result ;



}