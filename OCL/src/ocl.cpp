#include "ocl.h"


Ocl::Ocl(Params params_){

	this->params_ = params_;
}

// High-resolution timer.
double Ocl::getCurrentTimestamp()
{
	#ifdef _WIN32 // Windows
	  // Use the high-resolution performance counter.

	  static LARGE_INTEGER ticks_per_second = {};
	  if(ticks_per_second.QuadPart == 0) {
	    // First call - get the frequency.
	    QueryPerformanceFrequency(&ticks_per_second);
	  }

	  LARGE_INTEGER counter;
	  QueryPerformanceCounter(&counter);

	  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
	  return seconds;
	#else         // Linux
	  timespec a;
	  clock_gettime(CLOCK_MONOTONIC, &a);
	  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
	#endif
}

void Ocl::get_devices(cl_device_type device_type)
{
	cl_int status;

	clGetDeviceIDs(this->platform, device_type, 0, NULL, &this->num_devices);
	
	printf("num_devices: %d\n\n", num_devices);

	this->devices = new cl_device_id[this->num_devices];
	//this->queue = new cl_command_queue[this->num_devices];

	status = clGetDeviceIDs(platform, device_type, this->num_devices, this->devices, NULL);
	checkError(status, "Error: clGetDeviceIDs");

	status = clGetDeviceIDs(platform, device_type, 0, NULL, &this->num_devices);
	checkError(status, "Query for number of devices failed");

	status = clGetDeviceIDs(platform, device_type, this->num_devices, this->devices, NULL);
	checkError(status, "Query for device ids");

	this->num_devices = 1;
#ifdef DEBUG // Display some device information.
	display_device_info((cl_device_id)this->devices[0]);
#endif
}

// Searches all platforms for the first platform whose name
// contains the search string (case-insensitive).
cl_platform_id Ocl::findPlatform(std::string platform_name_search) {
	
	cl_int status;
	cl_platform_id platform;


	std::transform(platform_name_search.begin(), platform_name_search.end(), platform_name_search.begin(), ::tolower);

	if (platform_name_search.find("amd") != std::string::npos)
		platform_name_search = "amd";
	else if (platform_name_search.find("intel") != std::string::npos)
		platform_name_search = "intel";
	else if (platform_name_search.find("nvidia") != std::string::npos)
		platform_name_search = "nvidia";
	//else
	//  platform_name_search = "altera";


	

	// Get number of platforms.
	cl_uint num_platforms;
	status = clGetPlatformIDs(0, NULL, &num_platforms);
	checkError(status, "Query for number of platforms failed");

	printf("\nnum_platforms: %d\n", num_platforms);
	
	// Get a list of all platform ids.
	cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms* sizeof(cl_platform_id));

	status = clGetPlatformIDs(num_platforms, platforms, NULL);
	checkError(status, "Query for all platform ids failed");

	// For each platform, get name and compare against the search string.
	for (cl_uint i = 0; i < num_platforms; ++i) {
		std::string name = getPlatformName(platforms[i]);

		printf("%s\n", name.c_str());
		// Convert to lower case.
		std::transform(name.begin(), name.end(), name.begin(), ::tolower);

		if (name.find(platform_name_search) != std::string::npos) {
			// Found!
			platform = platforms[i];
			free(platforms);
			return platform;
		}
	}

	free(platforms);
	// No platform found.
	return NULL;
}

void Ocl::info_plataform()
{
	char char_buffer[PARAM_K_BYTE];
	PRINT("Querying platform for info:\n");
	PRINT("==========================\n");
	clGetPlatformInfo(this->platform, CL_PLATFORM_NAME, PARAM_K_BYTE, char_buffer, NULL);
	PRINT("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
	clGetPlatformInfo(this->platform, CL_PLATFORM_VENDOR, PARAM_K_BYTE, char_buffer, NULL);
	PRINT("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
	clGetPlatformInfo(this->platform, CL_PLATFORM_VERSION, PARAM_K_BYTE, char_buffer, NULL);
	PRINT("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
}

cl_program Ocl::creat_program()
{
	cl_int status;
	cl_int sourcesize = PARAM_K_BYTE * PARAM_K_BYTE;

	char* source = (char*)calloc(sourcesize, sizeof(char));
	if (!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return NULL; }

	FILE*  fp;// read the kernel core source

	fp = fopen("ker/cnn.cl", "rb");

	if (!fp) { printf("ERROR: unable to open Cl\n"); return NULL; }

	fread(source + strlen(source), sourcesize, 1, fp);

	fclose(fp);

	// compile kernel
	const char*  slist[] = { source };
	size_t sourceSize[] = { strlen(source) };

	cl_program prog = clCreateProgramWithSource(this->context,
														1,
														slist,
														sourceSize,
														&status);
	checkError(status, "Failed to create program with binary");
	free(source);

	return prog;
}

// Returns the platform name.
std::string Ocl::getPlatformName(cl_platform_id pid) {
	cl_int status;

	size_t sz;
	status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, NULL, &sz);
	checkError(status, "Query for platform name size failed");

	char* name = (char*)malloc(sz* sizeof(char));
	//scoped_array<char> name(sz);
	status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, sz, name, NULL);
	checkError(status, "Query for platform name failed");

	return name;
}

// Query and display OpenCL information on device and runtime environment
void Ocl::display_device_info(cl_device_id device)
{

	printf("Querying device for info:\n");
	printf("========================\n");
	device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
	device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
	device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
	device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
	device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
	device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
	device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
	device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
	device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
	device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
	device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
	device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
	device_info_ulong(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");

	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &this->max_work_itens_size, NULL);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE            = %llu\n", this->max_work_itens_size);

	device_info_ulong(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
	device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");

	cl_ulong a[3];
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(cl_ulong)*3, a, NULL);
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES            = %llu  %llu  %llu\n", a[0], a[1], a[2]);

	device_info_uint(device, CL_DEVICE_MAX_SAMPLERS, "CL_DEVICE_MAX_SAMPLERS");
	device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
	device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

	{
		cl_command_queue_properties ccp;
		clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
		printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
		printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
	}
}

// Helper functions to display parameters returned by OpenCL queries
void Ocl::device_info_ulong(cl_device_id device, cl_device_info param, const char* name) {
	cl_ulong a;
	clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
	printf("%-40s = %llu\n", name, a);
}

void Ocl::device_info_uint(cl_device_id device, cl_device_info param, const char* name) {
	cl_uint a;
	clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
	printf("%-40s = %u\n", name, a);
}

void Ocl::device_info_bool(cl_device_id device, cl_device_info param, const char* name) {
	cl_bool a;
	clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
	printf("%-40s = %s\n", name, (a ? "true" : "false"));
}

void Ocl::device_info_string(cl_device_id device, cl_device_info param, const char* name) {
	char a[PARAM_K_BYTE];
	clGetDeviceInfo(device, param, PARAM_K_BYTE, &a, NULL);
	printf("%-40s = %s\n", name, a);
}

// Print the error associated with an error code
void Ocl::printError(cl_int error) {
	// Print error message
	switch (error)
	{
	case -1:
		printf("CL_DEVICE_NOT_FOUND ");
		break;
	case -2:
		printf("CL_DEVICE_NOT_AVAILABLE ");
		break;
	case -3:
		printf("CL_COMPILER_NOT_AVAILABLE ");
		break;
	case -4:
		printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
		break;
	case -5:
		printf("CL_OUT_OF_RESOURCES ");
		break;
	case -6:
		printf("CL_OUT_OF_HOST_MEMORY ");
		break;
	case -7:
		printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
		break;
	case -8:
		printf("CL_MEM_COPY_OVERLAP ");
		break;
	case -9:
		printf("CL_IMAGE_FORMAT_MISMATCH ");
		break;
	case -10:
		printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
		break;
	case -11:
		printf("CL_BUILD_PROGRAM_FAILURE ");
		break;
	case -12:
		printf("CL_MAP_FAILURE ");
		break;
	case -13:
		printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
		break;
	case -14:
		printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
		break;
	case -30:
		printf("CL_INVALID_VALUE ");
		break;
	case -31:
		printf("CL_INVALID_DEVICE_TYPE ");
		break;
	case -32:
		printf("CL_INVALID_PLATFORM ");
		break;
	case -33:
		printf("CL_INVALID_DEVICE ");
		break;
	case -34:
		printf("CL_INVALID_CONTEXT ");
		break;
	case -35:
		printf("CL_INVALID_QUEUE_PROPERTIES ");
		break;
	case -36:
		printf("CL_INVALID_COMMAND_QUEUE ");
		break;
	case -37:
		printf("CL_INVALID_HOST_PTR ");
		break;
	case -38:
		printf("CL_INVALID_MEM_OBJECT ");
		break;
	case -39:
		printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
		break;
	case -40:
		printf("CL_INVALID_IMAGE_SIZE ");
		break;
	case -41:
		printf("CL_INVALID_SAMPLER ");
		break;
	case -42:
		printf("CL_INVALID_BINARY ");
		break;
	case -43:
		printf("CL_INVALID_BUILD_OPTIONS ");
		break;
	case -44:
		printf("CL_INVALID_PROGRAM ");
		break;
	case -45:
		printf("CL_INVALID_PROGRAM_EXECUTABLE ");
		break;
	case -46:
		printf("CL_INVALID_KERNEL_NAME ");
		break;
	case -47:
		printf("CL_INVALID_KERNEL_DEFINITION ");
		break;
	case -48:
		printf("CL_INVALID_KERNEL ");
		break;
	case -49:
		printf("CL_INVALID_ARG_INDEX ");
		break;
	case -50:
		printf("CL_INVALID_ARG_VALUE ");
		break;
	case -51:
		printf("CL_INVALID_ARG_SIZE ");
		break;
	case -52:
		printf("CL_INVALID_KERNEL_ARGS ");
		break;
	case -53:
		printf("CL_INVALID_WORK_DIMENSION ");
		break;
	case -54:
		printf("CL_INVALID_WORK_GROUP_SIZE ");
		break;
	case -55:
		printf("CL_INVALID_WORK_ITEM_SIZE ");
		break;
	case -56:
		printf("CL_INVALID_GLOBAL_OFFSET ");
		break;
	case -57:
		printf("CL_INVALID_EVENT_WAIT_LIST ");
		break;
	case -58:
		printf("CL_INVALID_EVENT ");
		break;
	case -59:
		printf("CL_INVALID_OPERATION ");
		break;
	case -60:
		printf("CL_INVALID_GL_OBJECT ");
		break;
	case -61:
		printf("CL_INVALID_BUFFER_SIZE ");
		break;
	case -62:
		printf("CL_INVALID_MIP_LEVEL ");
		break;
	case -63:
		printf("CL_INVALID_GLOBAL_WORK_SIZE ");
		break;
	default:
		printf("UNRECOGNIZED ERROR CODE (%d)", error);
	}
}

// Print line, file name, and error code if there is an error. Exits the application upon error.
void Ocl::_checkError(cl_int line,
	const char* file,
	cl_int error,
	const char* msg,
	...)
{
	// If not successful
	if (error != CL_SUCCESS) {

		// Print line and file
		printf("ERROR: ");
		printError(error);
		printf("\nLocation: %s:%d\n", file, line);

		// Print custom message.
		va_list vl;
		va_start(vl, msg);
		vprintf(msg, vl);
		printf("\n");
		va_end(vl);

		exit(error);
	}
}


bool Ocl::init_opencl()
{
	cl_int status;

	this->platform = findPlatform(this->pltaform_name);
	if (platform == NULL) {
		printf("ERROR: Unable to find OpenCL platform specified in %s.\n\n\n", this->pltaform_name.c_str());
		return false;
	}

	
	//User-visible output - Platform information
	info_plataform();

	
	// Query the available OpenCL device.
	cl_device_type device_type = (this->use_gpu == 1) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
	get_devices(device_type);

	printf("Device number: %d\n", this->num_devices);

	// Create the context.
	context = clCreateContext(NULL, this->num_devices, this->devices, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	
	// Create program with source
	this->prog = creat_program();
	
	// Build program
	status = clBuildProgram(this->prog, num_devices, devices, NULL, NULL, NULL);

	FILE* log;
	char buildLog[16 * PARAM_K_BYTE];

	// Determine the reason for the error
	clGetProgramBuildInfo(	this->prog,
							devices[0],
							CL_PROGRAM_BUILD_LOG,
							sizeof(buildLog),
							buildLog,
							NULL);
	

	if (status != CL_SUCCESS){

	
		
#ifdef _WIN32
		fopen_s(&log, "buildLog.txt", "w");
#else
		log = fopen("buildLog.txt", "w");
#endif

		fprintf(log, "\nINFO: clBuildProgram() => %d\n buildLog: \n %s", status, buildLog);
		fclose(log);

		printf("ERROR: clBuildProgram() => %d\n buildLog: \n %s", status, buildLog); return false;

	}

	return 1;
}

