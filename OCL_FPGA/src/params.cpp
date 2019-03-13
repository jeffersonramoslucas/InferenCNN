#include "params.h"

Params::Params() {

	NumConvLayers		= 1;
	NumHiddenLayers		= 1;
	NumHiddenNeurons	= 100;
	nclasses			= 10;
	nonlin				= NL_RELU;
	database			= CIFAR;

	dirfileMNIST_images = "files/";//mnist images
	dirfileCIFAR 		= "files/test_batch.bin";//cifar10
	conf_path 			= "files/conf.txt";

	//ypath = "C:/Users/Lincs/Documents/CNN/single-layer-convnet-master//t10k-labels.idx1-ubyte";



	samples_inchl = 3;
	samples_rows = 128;
	samples_cols = 128;

	number_of_images = 1;
	batchSize = 1;
}
