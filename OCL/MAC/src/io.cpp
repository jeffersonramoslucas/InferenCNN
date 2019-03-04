#include "io.h"

Io::Io(Params params_){

	this->params_.number_of_images	= params_.number_of_images;
	this->params_.batchSize			= params_.batchSize;

	read_conf_net(this->params_.conf_path, this->conf_net, this->conf_net_in);
}

int Io::ReverseInt(int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void Io::read_conf_net(string filename, vector<char*> &vec, vector<vector<char*>> &vec_in)
{
	printf("READ CONF\n\n");
	FILE *fp = fopen(filename.c_str(), "r");

	if (fp == NULL){
		printf("NO Open File Conf_Net");
		exit(-1);
	}



	while (1) 
	{
		int qtd_input;
		char *layer = (char*)malloc(sizeof(char) * 50);
		vector<char*> vec_input;

		if (fscanf(fp, "%s %d", layer, &qtd_input) == EOF) break;
		
		for (int i = 0; i < qtd_input; i++)
		{
			char *layer_in = (char*)malloc(sizeof(char) * 50);

			fscanf(fp, "%s", layer_in);

			vec_input.push_back(layer_in);
		}

		vec.push_back(layer);
		vec_in.push_back(vec_input);
	}

}

void Io::read_batch(){

	if (this->file->is_open())
	{
		int n_rows = 32;
		int n_cols = 32;

		this->samples = vector<Mat>();

		for (int i = 0; i < params_.batchSize; ++i)
		{
			unsigned char tplabelsuper = 0;
			unsigned char tplabel = 0;


			file->read((char*)&tplabel, sizeof(tplabel));
			cv::Mat matF;

			for (int ch = 0; ch < params_.samples_inchl; ++ch)
			{
				Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
				for (int r = 0; r < n_rows; ++r){
					for (int c = 0; c < n_cols; ++c){
						unsigned char temp = 0;
						file->read((char*)&temp, sizeof(temp));
						tp.at<uchar>(r, c) = (int)temp;
					}
				}
				resize(tp, tp, Size(params_.samples_rows, params_.samples_cols));//resize image

				matF.push_back(tp);
			}
        
			this->samples.push_back(matF);
			this->labels.at<float>(0, i) = (float)tplabel;
		}
	}
}

void Io::read_Mnist(){
    
    if (file->is_open()){
        
        if (this->samples.empty())
        {
            int magic_number = 0;
            int all_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file->read((char*)&magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            file->read((char*)&all_images, sizeof(all_images));
            all_images = ReverseInt(all_images);
            file->read((char*)&n_rows, sizeof(n_rows));
            n_rows = ReverseInt(n_rows);
            file->read((char*)&n_cols, sizeof(n_cols));
            n_cols = ReverseInt(n_cols);
        }
        
        int n_rows = 28;
        int n_cols = 28;
        
        this->samples = vector<Mat>();
        for (int i = 0; i < this->params_.batchSize; ++i){
            Mat tpmat = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for (int r = 0; r < n_rows; ++r){
                for (int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file->read((char*)&temp, sizeof(temp));
                    tpmat.at<uchar>(r, c) = (int)temp;
                }
            }
            resize(tpmat, tpmat, Size(params_.samples_rows, params_.samples_cols));//resize image
            
            this->samples.push_back(tpmat);
        }
    }
}


void Io::read_Mnist_Label()
{
	if (fileLabels->is_open()){

		if (this->labels.empty())
		{
			int magic_number = 0;
			int all_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			fileLabels->read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			fileLabels->read((char*)&all_images, sizeof(all_images));
			all_images = ReverseInt(all_images);
		}

		this->labels = Mat::zeros(1, (int)this->params_.batchSize, CV_32FC1);
		for (int i = 0; i < this->params_.batchSize; ++i){
			unsigned char temp = 0;
			fileLabels->read((char*)&temp, sizeof(temp));
			this->labels.at<float>(0, i) = (float)temp;
		}
	}
}

void Io::readDataSet(){
    
    this->labels = Mat::zeros(1, (int)this->params_.batchSize, CV_64FC1);
    
    vector<cv::String> fn;
    glob(this->params_.dirfilesUser + "/*", fn, false);
    
    
    this->samples = vector<Mat>();

    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++)
    {
        Mat tpmat = imread(fn[i]);
        resize(tpmat, tpmat, Size(params_.samples_rows, params_.samples_cols));
        
        Mat bgr[this->params_.samples_inchl];
        split(tpmat,bgr);
        
        cv::Mat matF;
        for(size_t j=0; j<this->params_.samples_inchl;j++)
            matF.push_back(bgr[j]);

        this->samples.push_back(matF);
    }
    
    for (int i = 0; i < this->samples.size(); i++){
        this->samples[i].convertTo(this->samples[i], CV_32FC1, 1.0/255, 0);
    }
}

void Io::readDataCIFAR(){

	printf("\n\nIO: this->params_.batchSize:%zu\n\n", this->params_.batchSize);
	this->labels = Mat::zeros(1, (int)this->params_.batchSize, CV_64FC1);

	
	read_batch();

	for (int i = 0; i < this->samples.size(); i++){
		this->samples[i].convertTo(this->samples[i], CV_32FC1, 1.0/255, 0);
	}
}

void Io::readDataMNIST(){
	
	//read MNIST iamge into OpenCV Mat vector
	read_Mnist();
	for (int i = 0; i < this->samples.size(); i++){
		this->samples[i].convertTo(this->samples[i], CV_32FC1, 1.0 / 255, 0);
	}

	//read MNIST label into float vector
	read_Mnist_Label();
}

void Io::readDataFASHION(){
    
    //read MNIST iamge into OpenCV Mat vector
    read_Mnist();
    for (int i = 0; i < this->samples.size(); i++){
        this->samples[i].convertTo(this->samples[i], CV_32FC1, 1.0 / 255, 0);
    }
    
    //read MNIST label into float vector
    read_Mnist_Label();
}

void Io::setbatchSize(size_t batchSize){
	this->params_.batchSize = batchSize;
}
