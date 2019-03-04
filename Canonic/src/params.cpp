#include "params.h"

Params::Params() {

	nclasses			= 10;
	nonlin				= NL_RELU;
	database			= MNIST;

	dirFiles            = "files/";
	dirfilesUser        = dirFiles + "/user/";
	dirfileMNIST_images = dirFiles;
	dirfileCIFAR 		= dirFiles + "/test_batch.bin";
	conf_path 			= dirFiles + "/conf.txt";


	conf_path = dirFiles + "/conf.txt";

	samples_inchl = 1;// 3;//
	samples_rows = 128;// 224;//
	samples_cols = 128;// 224;//

	number_of_images = 5;// 20;
	batchSize = 5;// 2;
}
