#ifndef _NET_H_
#define _NET_H_


#include "params.h"
#include "io.h"
#include "ocl.h"

#include <time.h>
#include <algorithm>
#include <random>

//#define DEBUG


//################# START STRUCTS #######################
typedef struct ConvLayer{
	cl_mem W;
	cl_mem b;
	char *padding;
}Cvl;

typedef struct FullyConected {
	cl_mem W;
	cl_mem b;
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
		//vector< float* > featuresMaps;
		
		vector< short > PadTop;
		vector< short > PadBottom;
		vector< short > PadLeft;
		vector< short > PadRigth;
		
		vector< int > HiddenFeaturesX;
		vector< int > HiddenFeaturesY;

		vector< Cvl* > ConvLayers;
		vector< Fully* > HiddenLayers;
		Fully *smr;

		//################# END GLOBAL VARIABLES #######################
	
	public:
		double start;
		double end;

		double ReadStart_DataTimer;
		double ReadEnd_DataTimer;

		double ConvStart_NDRangeKernelTimer;
		double ConvEnd_NDRangeKernelTimer;
		double PoolStart_NDRangeKernelTimer;
		double PoolEnd_NDRangeKernelTimer;
		double PadStart_NDRangeKernelTimer;
		double PadEnd_NDRangeKernelTimer;
		double ActStart_NDRangeKernelTimer;
		double ActEnd_NDRangeKernelTimer;
		double AddStart_NDRangeKernelTimer;
		double AddEnd_NDRangeKernelTimer;
		double AddBiasStart_NDRangeKernelTimer;
		double AddBiasEnd_NDRangeKernelTimer;
		double BatchNormStart_NDRangeKernelTimer;
		double BatchNormEnd_NDRangeKernelTimer;
		double DepthStart_NDRangeKernelTimer;
		double DepthEnd_NDRangeKernelTimer;


		double in;
		double ou;

		Ocl ocl;

		Net(Params params_, Io io, Ocl ocl);

		void loadNetwork();
		
		void loadConvLayersInception();

		void loadConvParam(FILE* pInW, FILE* pInB, cl_mem &convW, cl_mem &convB, int kernelAmount, int kernelSize, short kernelDepth);

		void load(FILE* pIn, float* M, int length);
		
		void loadHiddenLayers();

		void loadHiddenParam(FILE* pInW, FILE* pInB, Fully *ntw, int NumHiddenNeurons, int hiddenfeatures);

		void loadSmrLayers();

		void loadSmrParam(FILE* pInW, FILE* pInB, Fully *smr, int nclasses, int nfeatures);

		virtual ~Net();

		bool init_problem();

		bool init_problemInception();

		bool run_problemInception();

		void convAndPoolingInception(int rows, int cols, int* layer);

		bool run_problem();

		void convolutionOcl_fast(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void convolutionOcl2(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);
		
		void convolutionOcl6(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void convolutionOcl(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void convolutionOcl5(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void cnn_conv_global_local_workitens_kernel_local_layer(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void convolution_kernel_localOcl(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void convolution_Canonic_Global_Local_WorkItensOcl(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);
		//void convolutionOcl(vector< cl_mem > &input_conv, float* kernel_vector, float* bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void convolution_CanonicOcl(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);

		void addOcl(cl_mem &input_add_a, cl_mem &input_add_b, cl_mem &output_add, int rows, int cols, short deep);

		void poolingOcl(cl_mem &input_pool, cl_mem &output_pool, int rows, int cols, int poolingSizeX, int poolingSizeY, short kernel_amount, int featuresX, int featuresY, short strideX, short strideY);
		//void poolingOcl(vector< cl_mem > &input_pool, int rows, int cols, int poolingSizeX, int poolingSizeY, short kernel_amount, int featuresX, int featuresY, short strideX, short strideY);

		void paddingOcl(cl_mem &input_padd, cl_mem &output_padd, short kernel_depth, short rows, short cols, int pad_top, int pad_bottom, int pad_left, int pad_rigth);
		//void paddingOcl(vector< cl_mem > &input_padd, short kernel_depth, short rows, short cols, int pad_top, int pad_bottom, int pad_left, int pad_rigth);

		void reluOcl(cl_mem &input_relu, short kernel_amount, int featuresX, int featuresY);

		void eluOcl(vector< cl_mem > &input_elu, short kernel_amount, int featuresX, int featuresY);

		void sigmOcl(vector< cl_mem > &input_sigm, short kernel_amount, int featuresX, int featuresY);

		void tanhOcl(vector< cl_mem > &input_tanh, short kernel_amount, int featuresX, int featuresY);

		void hiddenOcl(cl_mem &input_hidden, cl_mem &output_hidden, cl_mem &hidden_weigth, cl_mem &hidden_bias, int hidden_rows, int hidden_cols);

		void batchNormOcl(cl_mem &input_padd, cl_mem &output_padd, cl_mem &kernel_vector, short kernel_depth, short rows, short cols);

		void convolutionDepthOcl(cl_mem &input_conv, cl_mem &output_conv, cl_mem &kernel_vector, int rows, int cols, int kernel_rows, int kernel_cols, short kernel_amount, short kernel_depth, int featuresX, int featuresY, short strideX, short strideY);
		//void hiddenOcl(vector< cl_mem > &input_hidden, float* hidden_weigth, float* hidden_bias, int hidden_rows, int hidden_cols);

		void add_Bias_Ocl(cl_mem &input_add_a, cl_mem &input_add_b, int rows, int cols, short deep);
};


int main(int argc, char** argv)
{	
	//long start, end;

	Params params_;

	
	Io io(params_);
	
#ifdef DEBUG
	for (int i = 0; i < io.conf_net.size(); i++)
	{
		printf("%s -> ", io.conf_net[i]);
		for (int j = 0; j < io.conf_net_in[i].size(); j++)
		{
			printf("%s ", io.conf_net_in[i][j]);
		}
		printf("\n");
	}
	printf("EXIT");
	getchar();
#endif // DEBUG

	Ocl ocl(params_);
	
	Net net(params_, io, ocl);
	
	net.ocl.init_opencl();
		
	net.loadNetwork();
	
	
	//net.start = clock();
	net.init_problemInception();
	//net.init_problem();
	//net.end = clock();

	printf("Totally used time: %lf second\n", ((double)(net.end - net.start)));
	//cout << "Totally used time: " << ((double)(end - start)) / CLOCKS_PER_SEC << " second" << endl;
	
	//params_.~Params();
	//io.~Io();
	//net.~Net();
	
	printf("Press Exit \n");
	//cout << "Press Exit " << endl;
	//waitKey(0);
	getchar();
	return 0;
}


#endif
