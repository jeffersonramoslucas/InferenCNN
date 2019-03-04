#include "params.h"

Params::Params() {

	nclasses			= 10;
	nonlin				= NL_RELU;
	database			= CIFAR;
    
    
    dirFiles            = "/Users/jefferson.r.anjos/Documents/files/";
    dirfilesUser        = dirFiles + "/user/";
    dirfileMNIST_images = dirFiles;
	dirfileCIFAR 		= dirFiles + "/test_batch.bin";
	conf_path 			= dirFiles + "/conf.txt";


	samples_inchl = 3;
	samples_rows = 128;
	samples_cols = 128;

	number_of_images = 6;
	batchSize = 6;
}
