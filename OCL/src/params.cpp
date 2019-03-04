#include "params.h"

Params::Params() {

	nclasses			= 10;
	nonlin				= NL_RELU;
	database			= USER;

	dirFiles            = "files/";
	dirfilesUser        = dirFiles + "/user/";
	dirfileMNIST_images = dirFiles;
	dirfileCIFAR 		= dirFiles + "/test_batch.bin";
	conf_path 			= dirFiles + "/conf.txt";


	samples_inchl = 3;// 3;//
	samples_rows = 128;// 224;//
	samples_cols = 128;// 224;//

	number_of_images = 6;// 20;
	batchSize = 6;// 2;
}
