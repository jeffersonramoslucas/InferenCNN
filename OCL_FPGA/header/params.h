#ifndef _PARAMS_H_
#define _PARAMS_H_


#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace std;

//################# START DEFINES #######################
#define ATF at<float>
#define ALIGNMENT 64
// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
// database
#define MNIST 0
#define CIFAR 1
#define FASHION 2
//################# END DEFINES #######################

class Params {

public:

	int NumConvLayers;
	int NumHiddenLayers;
	int NumHiddenNeurons;
	int nclasses;

	int samples_inchl;
	int samples_rows;
	int samples_cols;

	int nonlin;

	string conf_path;
	string dirfileMNIST_images;
	string dirfileCIFAR;

	int number_of_images;
	size_t batchSize;

	int database;

	Params();

	virtual ~Params(){};
};

#endif
