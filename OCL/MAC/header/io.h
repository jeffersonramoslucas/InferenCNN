#ifndef _IO_H_
#define _IO_H_

#include "params.h"

class Io {

	private:

		Params params_;
	public:

		vector<char*> conf_net;
		vector<vector<char*>> conf_net_in;

		vector<Mat> samples;
		Mat labels;

		ifstream *file			= NULL;
		ifstream *fileLabels	= NULL;

		Io(){}

		Io(Params params_);

		void readDataCIFAR();

		void readDataMNIST();
    
        void readDataFASHION();
    
        void readDataSet();

		void read_conf_net(string filename, vector<char*> &vec, vector<vector<char*>> &vec_in);

		void read_Mnist();

		void read_Mnist_Label();

		int	ReverseInt(int i);

		void read_batch();

		void setbatchSize(size_t batchSize);

		virtual ~Io(){}
};

#endif
