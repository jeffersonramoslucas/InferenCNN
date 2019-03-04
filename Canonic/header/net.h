#ifndef _NET_H_
#define _NET_H_


#include "params.h"
#include "io.h"

#include <time.h>
#include <algorithm>
#include <random>


#define DEBUG


//################# START STRUCTS #######################
typedef struct Img{
	Mat img;
	float label;
}Img;

typedef struct Kernel{
	vector< float* > W;
	float b;
}ConvK;

typedef struct ConvLayer{
	vector< ConvK* > layer;
	char *padding;
}Cvl;

typedef struct FullyConected{
	float* W;
	float* b;
}Fully;

//################# END STRUCTS #######################

class Net {

private:
	//################# START GLOBAL VARIABLES #######################
	Params params_;
	Io io;

	vector< short > KernelSize;
	vector< short > KernelAmount;
	vector< short > KernelDepth;

	vector< short > NormParamsSize;
	vector< short > ConvStride;

	vector< short > PoolingDim;
	vector< short > PoolStride;
	vector< char* > PoolingPad;
	
	vector< int > featuresDeep;
	vector< int > featuresSizesX;
	vector< int > featuresSizesY;
	vector< float* > featuresMaps;

	vector< short > PadTop;
	vector< short > PadBottom;
	vector< short > PadLeft;
	vector< short > PadRigth;

	vector< int > HiddenFeaturesX;
	vector< int > HiddenFeaturesY;

	vector< Cvl* > ConvLayers;
	vector< Fully > HiddenLayers;
	Fully *smr;

	//int features;

	//################# END GLOBAL VARIABLES #######################

public:

	Net(Params params_, Io io);

	void loadNetwork();

	void loadConvLayers();

	void loadConvLayersInception();

	void loadConvParam(FILE* pInW, FILE* pInB, ConvK* convk, int kernelSize, short kernelDepth);

	void load(FILE* pIn, float* M, int length);

	void loadHiddenLayers();

	void loadHiddenParam(FILE* pInW, FILE* pInB, Fully &ntw, int NumHiddenNeurons, int hiddenfeatures);

	void loadSmrLayers();

	void loadSmrParam(FILE* pInW, FILE* pInB, Fully *smr, int nclasses, int nfeatures);

	char* resultNetwork();

	char* resultNetworkInception();

	void convAndPooling(float* sample, float* &convOut, int rows, int cols);

	void convAndPoolingInception(float* &sample, int rows, int cols, int *layer);

	void Pooling(float* img, float *tmpOutPool, int poolingSizeX, int poolingSizeY, int strideX, int strideY, int rows, int cols);

	void convolution(float* img, float* kernel, float *tmpOutConv, int rows, int cols, int kernel_rows, int kernel_cols, int strideX, int strideY);

	float* padding(float* &input, int rows, int cols, int pad_top, int pad_bottom, int pad_left, int pad_rigth);

	void nonLinearity(float* M, int length);

	void Tanh(float* M, int length);

	void ReLU(float* M, int length);

	void sigmoid(float* M, int length);

	void MatrixMult(float* src, float* src2, float* dst, int rows, int cols);

	virtual ~Net();
};




int main(int argc, char** argv)
{

	long start, end;

	Params params_;

	Io io(params_);
	
#ifdef DEBUG
	for (int i = 0; i < io.conf_net.size(); i++)
	{
		printf("%s -> ", io.conf_net[i]);
		for (int j = 0; j < io.conf_net_in[i].size(); j++)
		{
			printf("%s ",io.conf_net_in[i][j]);
		}
		printf("\n");
	}
	printf("EXIT");
	getchar();
#endif // DEBUG

	Net net(params_, io);

	net.loadNetwork();

	
	start = clock();
		net.resultNetworkInception();
		//net.resultNetwork();
	end = clock();

	printf("Totally used time: %lf second\n", ((double)(end - start)) / CLOCKS_PER_SEC);
	//cout << "Totally used time: " << ((double)(end - start)) / CLOCKS_PER_SEC << " second" << endl;


	params_.~Params();
	io.~Io();
	net.~Net();

	/*
	vector<Mat> trainX;
	vector<Mat> testX;
	Mat trainY, testY;

	readDataPedestrian(trainX, trainY, "F:/ImagesBase/Pedestrian/Training/PedestrianTraining(", "F:/ImagesBase/NonPedestrian/Training/NonPedestrianTraining(");
	//readDataPedestrian(testX, testY, "F:/ImagesBase/Pedestrian/Test/PedestrianTest(", "F:/ImagesBase/NonPedestrian/Test/NonPedestrianTest(");


	for (int i = 0; i < trainX.size(); i++)
	{
	imshow("img1", trainX[i]);
	printf("%lf\n", trainY.ATD(0, i));
	waitKey(0);
	}*/

	printf("Press Exit \n");
	//cout << "Press Exit " << endl;
	waitKey(0);
	getchar();
	return 0;
}


#endif
