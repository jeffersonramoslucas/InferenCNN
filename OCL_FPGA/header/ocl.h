#ifndef _OCL_H_
#define _OCL_H_

#include <params.h>

#define __CL_ENABLE_EXCEPTIONS
#ifdef NV //NVIDIA
#include <CLUtil.h>
#include <CL/cl.hpp>
#else
#include <CL/opencl.h>
#endif

#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)
#define PARAM_K_BYTE 1024
#define DEBUG


#include <stdarg.h>
#include <time.h>

#ifdef _WIN32
	#include <windows.h>
#else
	#include <unistd.h>

#endif

#ifdef DEBUG
#define PRINT(A,...) printf(A,##__VA_ARGS__);
#else
#define PRINT(A,...) ;
#endif


class Ocl {

	public:

		Params params_;

		unsigned N = 1; // Problem size
		cl_uint num_devices		= 1;
		cl_int use_gpu			= 1;
		string pltaform_name	= "altera";

		cl_program prog_conv;
		cl_program prog_conv_depth;
		cl_program prog_relu;
		cl_program prog_padd;
		cl_program prog_multiplication;
		cl_program prog_bias;
		cl_program prog_batch_norm;



		//command_queue
		vector<cl_command_queue> queue; //cl_command_queue*	queue = NULL;
		//devices
		cl_device_id* devices = NULL;
		//context
		cl_context	context;
		//platform
		cl_platform_id platform;
		//kernel
		vector<cl_kernel> kernel_conv;
		vector<cl_kernel> kernel_depth;
		vector<cl_kernel> kernel_batch_norm;
		vector<cl_kernel> kernel_pool;
		vector<cl_kernel> kernel_padding;
		vector<cl_kernel> kernel_add;
		vector<cl_kernel> kernel_add_bias;
		vector<cl_kernel> kernel_relu;
		vector<cl_kernel> kernel_elu;
		vector<cl_kernel> kernel_sigm;
		vector<cl_kernel> kernel_tanh;
		vector<cl_kernel> kernel_matrix_mult;
		vector<cl_kernel> kernel_deep;

		vector<cl_mem> input_img_buf; // num_devices elements
		vector<cl_mem> input_kernel_buf; // num_devices elements
		vector<cl_mem> input_bias_buf; // num_devices elements
		vector<cl_mem> input_hidden_weigth_buf; // num_devices elements
		vector<cl_mem> input_hidden_bias_buf; // num_devices elements
		vector<cl_mem> output_conv_buf; // num_devices elements
		vector<cl_mem> output_padd_buf; // num_devices elements
		vector<cl_mem> output_pool_buf; // num_devices elements
		vector<cl_mem> output_hidden_buf; // num_devices elements
		vector<cl_mem> featuresMaps_buf;

		vector<unsigned> n_per_device; // num_devices elements

		// Launch the problem for each device.
		vector<cl_event> kernel_event;
		vector<cl_event> finish_event;

		size_t max_work_itens_size;
		

		Ocl(){}

		Ocl(Params params_);

		//Pega tempo de maior precisao
		double getCurrentTimestamp();

		void get_devices(cl_device_type device_type);

		cl_platform_id findPlatform(std::string platform_name_search);

		void info_plataform();

		cl_program creat_program();

		std::string getPlatformName(cl_platform_id pid);

		void display_device_info(cl_device_id device);

		void device_info_ulong(cl_device_id device, cl_device_info param, const char* name);

		void device_info_uint(cl_device_id device, cl_device_info param, const char* name);

		void device_info_bool(cl_device_id device, cl_device_info param, const char* name);

		void device_info_string(cl_device_id device, cl_device_info param, const char* name);

		void printError(cl_int error);

		void _checkError(cl_int line, const char* file, cl_int error, const char* msg, ...);

		bool init_opencl();

		std::string getDeviceName(cl_device_id did);

		bool fileExists(const char *file_name);

		std::string getBoardBinaryFile(const char *prefix, cl_device_id device);

		cl_program createProgramFromBinary(cl_context context, const char *binary_file_name, const cl_device_id *devices, unsigned num_devices);

		unsigned char *loadBinaryFile(const char *file_name, size_t *size);


};

#endif
