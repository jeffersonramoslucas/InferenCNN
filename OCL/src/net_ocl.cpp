#include "net_ocl.h"

void *_aligned_malloc(size_t required_bytes, size_t alignment) {
	void *p1;
	void **p2;
	int offset = alignment - 1 + sizeof(void*);
	p1 = malloc(required_bytes + offset);
	if (!p1) {
		return 0;
	}
	p2 = (void**) (((size_t) (p1) + offset) & ~(alignment - 1));
	if (!p2) {
		return 0;
	}
	p2[-1] = p1; //line 6
	return p2;
}

void _aligned_free(void *p) {
	free(((void**) p)[-1]);
}

Net::Net(Params params_, Io io, Ocl ocl) {

	this->params_ = params_;
	this->io = io;
	this->ocl = ocl;

	this->smr = new Fully();
}

void Net::loadNetwork() {

	printf("\nLoading ConvLayers ...\n");
	loadConvLayersInception();

	printf("\nLoading HiddenLayers ...\n");
	loadHiddenLayers();

	printf("\nLoading SMR ...\n");
	loadSmrLayers();
}

void Net::loadConvLayersInception() {

	cl_int status;
	string sW = "files/ConvLayersW.txt";
	string sB = "files/ConvLayersB.txt";

	FILE* pInW = fopen(sW.c_str(), "r");
	FILE* pInB = fopen(sB.c_str(), "r");

	fscanf(pInW, "%d", &params_.NumConvLayers);

	int input_featuresX = params_.samples_rows;
	int input_featuresY = params_.samples_cols;
	int input_deep = params_.samples_inchl;

	featuresSizesX.push_back(input_featuresX);
	featuresSizesY.push_back(input_featuresY);
	featuresDeep.push_back(input_deep);

	ocl.featuresMaps_buf.clear();

	cl_float initValue = 0.0;

	ocl.queue.clear();
	cl_command_queue cmq;
	ocl.queue.push_back(cmq);
	// Command queue.
	ocl.queue[0] = clCreateCommandQueue(ocl.context, ocl.devices[0], NULL,
			&status);
	//ocl.queue[i] = clCreateCommandQueueWithProperties(ocl.context, ocl.devices[i], NULL, &status);
	ocl.checkError(status, "Failed to create command queue");

	cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
			sizeof(float) * featuresSizesX[0] * featuresSizesY[0]
					* featuresDeep[0],
			NULL, &status);
	//Zerando Buffer de Saida
	clEnqueueFillBuffer(ocl.queue[0], features, &initValue, sizeof(float), 0,
			featuresDeep[0] * featuresSizesX[0] * featuresSizesY[0]
					* sizeof(cl_float), 0, NULL, NULL);
	clFinish(ocl.queue[0]);

	ocl.featuresMaps_buf.push_back(features);

	//float* feat = (float*)malloc(sizeof(float)*featuresSizesX[0] * featuresSizesY[0] * featuresDeep[0]);
	//this->featuresMaps.push_back(feat);//features da entrada

	int ker_amount_conc;
	int id_conf = 0;

	char* layer = io.conf_net[id_conf];

	int featuresX;
	int featuresY;
	int deep;

	while (strstr(io.conf_net_in[id_conf][0], "flatten") == NULL) {
		this->PadTop.push_back(0);
		this->PadBottom.push_back(0);
		this->PadLeft.push_back(0);
		this->PadRigth.push_back(0);

		this->ocl.featuresMaps_buf.push_back(NULL);	//features de todas as camadas
		this->ConvLayers.push_back(NULL);

		featuresSizesX.push_back(0);
		featuresSizesY.push_back(0);
		featuresDeep.push_back(0);	//

		KernelAmount.push_back(0);
		KernelDepth.push_back(0);
		KernelSize.push_back(0);

		NormParamsSize.push_back(0);
		ConvStride.push_back(0);

		PoolingDim.push_back(0);
		PoolStride.push_back(0);
		PoolingPad.push_back(0);

		for (int j = 0; j < io.conf_net.size(); j++) {
			if (strstr(io.conf_net_in[id_conf][0], "input")) {
				featuresX = input_featuresX;
				featuresY = input_featuresY;
				deep = input_deep;
			} else if (strstr(io.conf_net_in[id_conf + 1][0], io.conf_net[j])) {
				featuresX = featuresSizesX[j];
				featuresY = featuresSizesY[j];
				deep = featuresDeep[j];
			}
		}

		if (strstr(layer, "conv") || strstr(layer, "depth")) {
			Cvl* tpcvl = new Cvl();

			int tpKernelSizeX, tpKernelSizeY, tpKernelAmount, tpKernelDepth;
			int strideX = 1;
			int strideY = 1;

			char* padding = (char*) malloc(sizeof(char) * 20);

			fscanf(pInW, "%d %d %d %d %s %d %d", &tpKernelSizeX, &tpKernelSizeY,
					&tpKernelAmount, &tpKernelDepth, padding, &strideX,
					&strideY);

			KernelSize[id_conf] = ((tpKernelSizeX << 8) | tpKernelSizeY);
			KernelDepth[id_conf] = tpKernelDepth;
			KernelAmount[id_conf] = tpKernelAmount;
			ConvStride[id_conf] = ((strideX << 8) | strideY);
			tpcvl->padding = padding;

			if (strstr(layer, "conv"))
				deep = tpKernelAmount;
			else
				deep = tpKernelAmount * tpKernelDepth;

			if (strstr(tpcvl->padding, "valid")) {

#ifdef DEBUG
				printf("\n\nCONV - valid\n\n");
#endif // DEBUG

				featuresX = (int) ceil(
						(float) (featuresX - tpKernelSizeX + 1) / strideX);
				featuresY = (int) ceil(
						(float) (featuresY - tpKernelSizeY + 1) / strideY);
			} else {

				int output_heigth = (int) ceil((float) featuresX / strideX);
				int output_width = (int) ceil((float) featuresY / strideY);

				int pad_along_heigth = (output_heigth - 1) * strideX
						+ tpKernelSizeX - featuresX;
				int pad_along_width = (output_width - 1) * strideY
						+ tpKernelSizeY - featuresY;

				PadTop[id_conf] = (short) (floor((float) pad_along_heigth / 2));
				PadBottom[id_conf] = (pad_along_heigth - PadTop.back());
				PadLeft[id_conf] = (short) (floor((float) pad_along_width / 2));
				PadRigth[id_conf] = (pad_along_width - PadLeft.back());

				featuresX = output_heigth;// ceil((float)(featuresX + PadTop.back() + PadBottom.back() - tpKernelSizeX + 1) / strideX);
				featuresY = output_width;// ceil((float)(featuresY + PadLeft.back() + PadRigth.back() - tpKernelSizeY + 1) / strideY);

#ifdef DEBUG
				printf("\n\nCONV - same\n\n");
				printf(
						"\n\nCONV - featuresX: %d - featuresY: %d  (%d %d)  (%d, %d) (%d,%d)\n\n",
						featuresX, featuresY, tpKernelSizeX, tpKernelSizeY,
						PadTop.back(), PadBottom.back(), PadLeft.back(),
						PadRigth.back());
#endif // DEBUG
			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1] = deep;

#ifdef DEBUG
			printf("featuresX: %d\nfeaturesY: %d  -> %d\n\n", featuresX,
					featuresY, KernelAmount[id_conf]);
#endif // DEBUG

			cl_mem tmpConvW;
			cl_mem tmpConvB;
			loadConvParam(pInW, pInB, tmpConvW, tmpConvB, KernelAmount[id_conf],
					KernelSize[id_conf], KernelDepth[id_conf]);

			tpcvl->W = tmpConvW;
			tpcvl->b = tmpConvB;

			/*
			 for (int j = 0; j < KernelAmount[id_conf]; j++)
			 {
			 ConvK* tmpConvK = new ConvK();
			 loadConvParam(pInW, pInB, tmpConvK, KernelSize[id_conf], KernelDepth[id_conf]);

			 tpcvl->layer.push_back(tmpConvK);
			 }*/

			this->ConvLayers[id_conf] = (tpcvl);

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1,
					featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1],
					featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			cl_mem features = NULL;

			if (strstr(layer, "conv")) {
				features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
						sizeof(float) * featuresSizesX[id_conf + 1]
								* featuresSizesY[id_conf + 1]
								* featuresDeep[id_conf + 1],
						NULL, &status);

				//Zerando Buffer de Saida
				clEnqueueFillBuffer(ocl.queue[0], features, &initValue,
						sizeof(float), 0,
						featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
								* featuresSizesY[id_conf + 1]
								* sizeof(cl_float), 0, NULL, NULL);
				clFinish(ocl.queue[0]);
			} else {
				features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
						sizeof(float) * featuresSizesX[id_conf + 1]
								* featuresSizesY[id_conf + 1]
								* featuresDeep[id_conf + 1]
								* KernelAmount[id_conf],
						NULL, &status);

				//Zerando Buffer de Saida
				clEnqueueFillBuffer(ocl.queue[0], features, &initValue,
						sizeof(float), 0,
						featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
								* featuresSizesY[id_conf + 1]
								* KernelAmount[id_conf] * sizeof(cl_float), 0,
						NULL, NULL);
				clFinish(ocl.queue[0]);
			}

			ocl.featuresMaps_buf[id_conf + 1] = features;

			//float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			//this->featuresMaps[id_conf + 1] = feat;
		} else if (strstr(layer, "pool")) {
			char* padding = (char*) malloc(sizeof(char) * 20);

			int tpPoolSizeX, tpPoolSizeY;
			int strideX = 1;
			int strideY = 1;

			fscanf(pInW, "%d %d %s %d %d", &tpPoolSizeX, &tpPoolSizeY, padding,
					&strideX, &strideY);

			PoolingDim[id_conf] = ((tpPoolSizeX << 8) | tpPoolSizeY);
			PoolStride[id_conf] = ((strideX << 8) | strideY);
			PoolingPad[id_conf] = (padding);

			if (strstr(padding, "valid")) {

				int output_heigth = (int) ceil(
						((float) featuresX - tpPoolSizeX + 1) / strideX);
				int output_width = (int) ceil(
						((float) featuresY - tpPoolSizeY + 1) / strideY);

#ifdef DEBUG
				printf("\n\nPool - valid\n\n");
				printf(
						"\n\nPOOL - featuresX: %d - featuresY: %d  (%d %d) %d %d\n\n",
						featuresX, featuresY, tpPoolSizeX, tpPoolSizeY,
						output_heigth, output_width);
#endif // DEBUG

				featuresX = output_heigth;
				featuresY = output_width;
			} else {

				int output_heigth = (int) ceil((float) featuresX / strideX);
				int output_width = (int) ceil((float) featuresY / strideY);

				int pad_along_heigth = max(0,
						(output_heigth - 1) * strideX + tpPoolSizeX
								- featuresX);
				int pad_along_width = max(0,
						(output_width - 1) * strideY + tpPoolSizeY - featuresY);

				//printf("\n\npad_along_heigth: %d\n\n", pad_along_heigth);

				PadTop[id_conf] = (short) (floor((float) pad_along_heigth / 2));
				PadBottom[id_conf] = (pad_along_heigth - PadTop.back());
				PadLeft[id_conf] = (short) (floor((float) pad_along_width / 2));
				PadRigth[id_conf] = (pad_along_width - PadLeft.back());

				//AllocateVectorSizes.push_back((featuresX + PadTop.back() + PadBottom.back() << 8) | featuresY + PadLeft.back() + PadRigth.back());

				featuresX = output_heigth;// floor((float)(featuresX + PadTop.back() + PadBottom.back()) / tpPoolSizeX);
				featuresY = output_width;// floor((float)(featuresY + PadLeft.back() + PadRigth.back()) / tpPoolSizeY);

#ifdef DEBUG
				printf("\n\nPool - same\n\n");
				printf(
						"\n\nPOOL - featuresX: %d - featuresY: %d  (%d %d)  (%d, %d) (%d,%d) %d %d\n\n",
						featuresX, featuresY, tpPoolSizeX, tpPoolSizeY,
						PadTop.back(), PadBottom.back(), PadLeft.back(),
						PadRigth.back(), output_heigth, output_width);
#endif // DEBUG
			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1] = deep;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1,
					featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1],
					featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
					sizeof(float) * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1]
							* featuresDeep[id_conf + 1],
					NULL, &status);
			//Zerando Buffer de Saida
			clEnqueueFillBuffer(ocl.queue[0], features, &initValue,
					sizeof(float), 0,
					featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1] * sizeof(cl_float), 0,
					NULL, NULL);
			clFinish(ocl.queue[0]);

			ocl.featuresMaps_buf[id_conf + 1] = features;

			//float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			//this->featuresMaps[id_conf + 1] = feat;
		} else if (strstr(layer, "zero")) {
			int pad_top = 0;
			int pad_bottom = 0;
			int pad_left = 0;
			int pad_rigth = 0;

			fscanf(pInW, "%d %d %d %d", &pad_top, &pad_bottom, &pad_left,
					&pad_rigth);

			PadTop[id_conf] = pad_top;
			PadBottom[id_conf] = pad_bottom;
			PadLeft[id_conf] = pad_left;
			PadRigth[id_conf] = pad_rigth;

			featuresX = featuresX + pad_top + pad_bottom;
			featuresY = featuresY + pad_left + pad_rigth;

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1] = deep;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1,
					featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1],
					featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
					sizeof(float) * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1]
							* featuresDeep[id_conf + 1],
					NULL, &status);
			ocl.featuresMaps_buf[id_conf + 1] = features;
			//Zerando Buffer de Saida
			clEnqueueFillBuffer(ocl.queue[0], features, &initValue,
					sizeof(float), 0,
					featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1] * sizeof(cl_float), 0,
					NULL, NULL);
			clFinish(ocl.queue[0]);

			//float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			//this->featuresMaps[id_conf + 1] = feat;
		} else if (strstr(layer, "concatenate")) {
			ker_amount_conc = 0;

			for (int i = 0; i < io.conf_net_in[id_conf].size(); i++) {
				for (int j = 0; j < io.conf_net.size(); j++) {
					if (strstr(io.conf_net_in[id_conf][i], io.conf_net[j])) {
						//printf("%s  -  %s  -  %d - %d\n\n", io.conf_net_in[id_conf][i], io.conf_net[j],j,cont_flatten);
						//getchar();
						featuresX = featuresSizesX[j];
						featuresY = featuresSizesY[j];
						ker_amount_conc += KernelAmount[j];

						//printf("Concat: %d\n", j - cont_flatten);
						//printf("%d X %d X %d\n", featuresX, featuresY, KernelAmount[j - cont_flatten]);
						//getchar();
					}
				}
			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1] = ker_amount_conc;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1,
					featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1],
					featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
					sizeof(float) * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1]
							* featuresDeep[id_conf + 1],
					NULL, &status);
			//Zerando Buffer de Saida
			clEnqueueFillBuffer(ocl.queue[0], features, &initValue,
					sizeof(float), 0,
					featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1] * sizeof(cl_float), 0,
					NULL, NULL);
			clFinish(ocl.queue[0]);
			ocl.featuresMaps_buf[id_conf + 1] = features;

			//float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			//this->featuresMaps[id_conf + 1] = feat;
		} else if (strstr(layer, "flatten")) {
			if (!strstr(io.conf_net[id_conf + 1], "concatenate")) {
				for (int j = 0; j < io.conf_net.size(); j++) {
					if (strstr(io.conf_net_in[id_conf + 1][0], "input")) {
						featuresSizesX[id_conf + 1] = input_featuresX;
						featuresSizesY[id_conf + 1] = input_featuresY;
						featuresDeep[id_conf + 1] = input_deep;

						ocl.featuresMaps_buf[id_conf + 1] =
								ocl.featuresMaps_buf[0];
						//this->featuresMaps[id_conf + 1] = featuresMaps[0];
					} else if (strstr(io.conf_net_in[id_conf + 1][0],
							io.conf_net[j])) {
						//printf("%d %d %d\n", featuresSizesX[j + 1], featuresSizesY[j + 1], featuresDeep[j + 1]);
						//getchar();
						featuresSizesX[id_conf + 1] = featuresSizesX[j + 1];
						featuresSizesY[id_conf + 1] = featuresSizesY[j + 1];
						featuresDeep[id_conf + 1] = featuresDeep[j + 1];

						ocl.featuresMaps_buf[id_conf + 1] =
								ocl.featuresMaps_buf[j + 1];
						//this->featuresMaps[id_conf + 1] = featuresMaps[j + 1];
					}
				}
			}

#ifdef DEBUG
			//printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			//float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			//this->featuresMaps[id_conf + 1] = feat;
		} else if (strstr(layer, "add")) {
			for (int i = 0; i < io.conf_net_in[id_conf].size(); i++) {
				for (int j = 0; j < io.conf_net.size(); j++) {
					if (strstr(io.conf_net_in[id_conf][i], io.conf_net[j])) {

						featuresX = featuresSizesX[j + 1];
						featuresY = featuresSizesY[j + 1];
						deep = featuresDeep[j + 1];
					}
				}
			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1] = deep;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1,
					featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1],
					featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG
			cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
					sizeof(float) * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1]
							* featuresDeep[id_conf + 1],
					NULL, &status);
			//Zerando Buffer de Saida
			clEnqueueFillBuffer(ocl.queue[0], features, &initValue,
					sizeof(float), 0,
					featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1] * sizeof(cl_float), 0,
					NULL, NULL);
			clFinish(ocl.queue[0]);
			ocl.featuresMaps_buf[id_conf + 1] = features;
		} else if (strstr(layer, "batch_norm")) {
			printf("batch_norm %d\n\n", id_conf);

			int params_norm_size;
			fscanf(pInW, "%d ", &params_norm_size);

			NormParamsSize[id_conf] = params_norm_size;

			Cvl* tpcvl = new Cvl();

			float* W = (float*) _aligned_malloc(
					sizeof(float) * featuresSizesY[id_conf]
							* NormParamsSize[id_conf], ALIGNMENT);
			load(pInW, W, featuresSizesY[id_conf] * NormParamsSize[id_conf]);

			cl_mem Wnorm = clCreateBuffer(ocl.context,
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(float) * featuresSizesY[id_conf]
							* NormParamsSize[id_conf], W, &status);
			ocl.checkError(status, "Failed to create buffer for convW");

			_aligned_free(W);

			tpcvl->W = Wnorm;

			this->ConvLayers[id_conf] = (tpcvl);

			featuresSizesX[id_conf + 1] = featuresSizesX[id_conf];
			featuresSizesY[id_conf + 1] = featuresSizesY[id_conf];
			featuresDeep[id_conf + 1] = deep;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1,
					featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1],
					featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
					sizeof(float) * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1]
							* featuresDeep[id_conf + 1],
					NULL, &status);
			//Zerando Buffer de Saida
			clEnqueueFillBuffer(ocl.queue[0], features, &initValue,
					sizeof(float), 0,
					featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
							* featuresSizesY[id_conf + 1] * sizeof(cl_float), 0,
					NULL, NULL);
			clFinish(ocl.queue[0]);

			ocl.featuresMaps_buf[id_conf + 1] = features;
		}

		layer = io.conf_net[++id_conf];
	}

	PadTop.push_back(0);
	PadBottom.push_back(0);
	PadLeft.push_back(0);
	PadRigth.push_back(0);

	ocl.featuresMaps_buf.pop_back();
	//this->features = featuresSizesX[id_conf] * featuresSizesY[id_conf] * featuresDeep[id_conf];

#ifdef DEBUG
	printf("\n\n\n\n%d %d %d\n\n", featuresSizesX[id_conf],
			featuresSizesY[id_conf], featuresDeep[id_conf]);
	printf("End Conv %d \n\n",
			featuresSizesX[id_conf] * featuresSizesY[id_conf]
					* featuresDeep[id_conf]);
	//printf("End Conv %d \n\n", this->features);

	for (int i = 0; i < featuresSizesX.size(); i++) {
		printf("%d (%d %d %d)\n", i, featuresSizesX[i], featuresSizesY[i],
				featuresDeep[i]);
	}
	printf("\n\n");
	getchar();
#endif // DEBUG

	fclose(pInW);
	fclose(pInB);
}

void Net::load(FILE* pIn, float* M, int length) {
	for (int i = 0; i < length; i++)
		fscanf(pIn, "%f", &M[i]);
}

void Net::loadConvParam(FILE* pInW, FILE* pInB, cl_mem &convW, cl_mem &convB,
		int kernelAmount, int kernelSize, short kernelDepth) {
	cl_int status;

	int width = kernelSize >> 8;
	int heigth = kernelSize & 0xFF;

	float* W = (float*) _aligned_malloc(
			sizeof(float) * width * heigth * kernelDepth * kernelAmount,
			ALIGNMENT);
	if (W == NULL) {
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	float* b = (float*) _aligned_malloc(sizeof(float) * kernelAmount,
			ALIGNMENT);
	if (b == NULL) {
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	load(pInW, W, width * heigth * kernelDepth * kernelAmount);

	load(pInB, b, kernelAmount);

	convW = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(float) * kernelAmount * width * heigth * kernelDepth, W,
			&status);
	ocl.checkError(status, "Failed to create buffer for convW");

	convB = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(float) * kernelAmount, b, &status);
	ocl.checkError(status, "Failed to create buffer for convB");

	_aligned_free(W);
	_aligned_free(b);
}

void Net::loadHiddenLayers() {
	cl_int status;

	string sW = "files/HiddenLayersW.txt";
	string sB = "files/HiddenLayersB.txt";

	FILE* pInW = fopen(sW.c_str(), "r");
	FILE* pInB = fopen(sB.c_str(), "r");

	fscanf(pInW, "%d", &params_.NumHiddenLayers);

	printf("NumHiddenLayers: %d\n\n", params_.NumHiddenLayers);

	// Init Hidden layers
	for (int hl = 0; hl < params_.NumHiddenLayers; hl++) {

		int hiddenfeatures;

		fscanf(pInW, "%d %d", &params_.NumHiddenNeurons, &hiddenfeatures);

		printf("%d %d\n\n", params_.NumHiddenNeurons, hiddenfeatures);

		cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
				sizeof(float) * params_.NumHiddenNeurons,
				NULL, &status);
		ocl.featuresMaps_buf.push_back(features);

		//float* feat = (float*)malloc(sizeof(float) * params_.NumHiddenNeurons);
		//this->featuresMaps.push_back(feat);//features da entrada

		Fully *tpntw = new Fully();

		HiddenFeaturesX.push_back(params_.NumHiddenNeurons);
		HiddenFeaturesY.push_back(hiddenfeatures);

		loadHiddenParam(pInW, pInB, tpntw, params_.NumHiddenNeurons,
				hiddenfeatures);
		this->HiddenLayers.push_back(tpntw);
	}

	fclose(pInW);
	fclose(pInB);
}

void Net::loadHiddenParam(FILE* pInW, FILE* pInB, Fully *ntw,
		int NumHiddenNeurons, int hiddenfeatures) {
	cl_int status;

	float* W = (float*) _aligned_malloc(
			sizeof(float) * NumHiddenNeurons * hiddenfeatures, ALIGNMENT);
	if (W == NULL) {
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	float* b = (float*) _aligned_malloc(sizeof(float) * NumHiddenNeurons,
			ALIGNMENT);
	if (b == NULL) {
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	load(pInW, W, NumHiddenNeurons * hiddenfeatures);

	load(pInB, b, NumHiddenNeurons);

	ntw->W = clCreateBuffer(ocl.context,
			CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
			sizeof(float) * NumHiddenNeurons * hiddenfeatures, W, &status);
	ocl.checkError(status, "Failed to create buffer for FullyW");

	ntw->b = clCreateBuffer(ocl.context,
			CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
			sizeof(float) * NumHiddenNeurons, b, &status);
	ocl.checkError(status, "Failed to create buffer for Fullyb");

	_aligned_free(W);
	_aligned_free(b);
}

void Net::loadSmrLayers() {
	cl_int status;

	string sW = "files/SmrLayerW.txt";
	string sB = "files/SmrLayerB.txt";

	FILE* pInW = fopen(sW.c_str(), "r");
	FILE* pInB = fopen(sB.c_str(), "r");

	int nclasses, nfeatures;

	fscanf(pInW, "%d %d", &nclasses, &nfeatures);

	cl_mem features = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
			sizeof(float) * nclasses,
			NULL, &status);
	ocl.featuresMaps_buf.push_back(features);

	//float* feat = (float*)malloc(sizeof(float) * nclasses);
	//this->featuresMaps.push_back(feat);//features da entrada

	loadSmrParam(pInW, pInB, this->smr, nclasses, nfeatures);

	fclose(pInW);
	fclose(pInB);
}

void Net::loadSmrParam(FILE* pInW, FILE* pInB, Fully *smr, int nclasses,
		int nfeatures) {
	cl_int status;

	float* W = (float*) _aligned_malloc(sizeof(float) * nclasses * nfeatures,
			ALIGNMENT);
	if (W == NULL) {
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	float* b = (float*) _aligned_malloc(sizeof(float) * nclasses, ALIGNMENT);
	if (b == NULL) {
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	load(pInW, W, nclasses * nfeatures);

	load(pInB, b, nclasses);

	smr->W = clCreateBuffer(ocl.context,
			CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
			sizeof(float) * nclasses * nfeatures, W, &status);
	ocl.checkError(status, "Failed to create buffer for FullyW");

	smr->b = clCreateBuffer(ocl.context,
			CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
			sizeof(float) * nclasses, b, &status);
	ocl.checkError(status, "Failed to create buffer for Fullyb");

	_aligned_free(W);
	_aligned_free(b);
}

void visualizeMem(cl_mem input, size_t length, cl_command_queue cmq) {
	cl_int status;

	float * OutConv = (cl_float*) clEnqueueMapBuffer(cmq, // Corresponding command queue
			input, // Buffer to be mapped
			1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
			CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
			0, // offset
			length * sizeof(float), // number of bytes mapped
			0, // number of events in the wait list
			NULL, // event wait list
			NULL, // event
			&status); // error

	//ocl.checkError(status, "Failed to clEnqueueMapBuffer");

	for (unsigned int i = 0; i < length; i++) {
		printf("%lf\n", OutConv[i]);
	}
}

bool Net::init_problemInception() {
	cl_int status;

	if (ocl.num_devices == 0) {
		ocl.checkError(-1, "No devices");
	}

	/*
	 ocl.queue.clear();
	 cl_command_queue cmq;
	 ocl.queue.push_back(cmq);

	 // Command queue.
	 ocl.queue[0] = clCreateCommandQueueWithProperties(ocl.context, ocl.devices[0], NULL, &status);
	 ocl.checkError(status, "Failed to create command queue");

	 float * OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[0], // Corresponding command queue
	 smr->b, // Buffer to be mapped
	 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
	 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
	 0, // offset
	 10, // number of bytes mapped
	 0,// number of events in the wait list
	 NULL,// event wait list
	 NULL,// event
	 NULL);// error
	 ocl.checkError(status, "Failed to clEnqueueMapBuffer");

	 for(int i = 0; i < 10; i++)
	 {
	 printf("%lf\n", OutConv[i]);
	 }
	 */

	// Clear per-device objects.
	ocl.queue.clear();

	ocl.kernel_conv.clear();
	ocl.kernel_depth.clear();
	ocl.kernel_batch_norm.clear();
	ocl.kernel_add_bias.clear();
	ocl.kernel_pool.clear();
	ocl.kernel_add.clear();
	ocl.kernel_padding.clear();
	ocl.kernel_relu.clear();
	ocl.kernel_elu.clear();
	ocl.kernel_sigm.clear();
	ocl.kernel_tanh.clear();
	ocl.kernel_matrix_mult.clear();
	ocl.kernel_deep.clear();

	ocl.n_per_device.clear();

	// Create per-device objects.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		cl_command_queue cmq;

		cl_kernel ker_conv;
		cl_kernel ker_depth;
		cl_kernel ker_batch_norm;
		cl_kernel ker_pool;
		cl_kernel ker_add;
		cl_kernel ker_add_bias;
		cl_kernel ker_pad;
		cl_kernel ker_relu;
		cl_kernel ker_elu;
		cl_kernel ker_sigm;
		cl_kernel ker_tanh;
		cl_kernel ker_mat_mul;
		cl_kernel ker_deep;

		unsigned n_per;

		ocl.queue.push_back(cmq);

		ocl.kernel_conv.push_back(ker_conv);
		ocl.kernel_depth.push_back(ker_depth);
		ocl.kernel_batch_norm.push_back(ker_batch_norm);
		ocl.kernel_pool.push_back(ker_pool);
		ocl.kernel_add.push_back(ker_add);
		ocl.kernel_add_bias.push_back(ker_add_bias);
		ocl.kernel_padding.push_back(ker_pad);
		ocl.kernel_relu.push_back(ker_relu);
		ocl.kernel_elu.push_back(ker_elu);
		ocl.kernel_sigm.push_back(ker_sigm);
		ocl.kernel_tanh.push_back(ker_tanh);
		ocl.kernel_matrix_mult.push_back(ker_mat_mul);
		ocl.kernel_deep.push_back(ker_deep);

		ocl.n_per_device.push_back(n_per);
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {

		// Command queue.
		ocl.queue[i] = clCreateCommandQueue(ocl.context, ocl.devices[i], NULL,
				&status);
		//ocl.queue[i] = clCreateCommandQueueWithProperties(ocl.context, ocl.devices[i], NULL, &status);
		ocl.checkError(status, "Failed to create command queue");

		// Kernel conv.
		ocl.kernel_conv[i] = clCreateKernel(ocl.prog, "cnn_depth_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_conv_layer");

		// Kernel depth.
		ocl.kernel_depth[i] = clCreateKernel(ocl.prog, "cnn_depth_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_depth_layer");

		// Kernel batch_norm.
		ocl.kernel_batch_norm[i] = clCreateKernel(ocl.prog,
				"cnn_batch_norm_layer", &status);
		ocl.checkError(status, "Failed to create kernel cnn_batch_norm_layer");

		// Kernel Pool.
		ocl.kernel_pool[i] = clCreateKernel(ocl.prog, "cnn_pool_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_pool_layer");

		// Kernel Padding.
		ocl.kernel_padding[i] = clCreateKernel(ocl.prog, "cnn_padding_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_padding_layer");

		// Kernel Relu.
		ocl.kernel_relu[i] = clCreateKernel(ocl.prog, "cnn_relu_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_relu_layer");

		// Kernel Elu.
		ocl.kernel_elu[i] = clCreateKernel(ocl.prog, "cnn_elu_layer", &status);
		ocl.checkError(status, "Failed to create kernel cnn_elu_layer");

		// Kernel Sigmoide.
		ocl.kernel_sigm[i] = clCreateKernel(ocl.prog, "cnn_sigm_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_sigm_layer");

		// Kernel Tanh.
		ocl.kernel_tanh[i] = clCreateKernel(ocl.prog, "cnn_tanh_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_tanh_layer");

		// Kernel Matrix_Multi.
		ocl.kernel_matrix_mult[i] = clCreateKernel(ocl.prog,
				"cnn_matrixMul_layer", &status);
		ocl.checkError(status, "Failed to create kernel cnn_tanh_layer");

		// Kernel deep.
		ocl.kernel_deep[i] = clCreateKernel(ocl.prog, "cnn_conv_deep_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_deep_layer");

		// Kernel deep.
		ocl.kernel_add[i] = clCreateKernel(ocl.prog, "cnn_add_layer", &status);
		ocl.checkError(status, "Failed to create kernel cnn_add_layer");

		ocl.kernel_add_bias[i] = clCreateKernel(ocl.prog, "cnn_add_bias_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_add_bias_layer");

	}

	//printf("\n\nocl.featuresMaps_buf.size: %d", ocl.featuresMaps_buf.size());
	//getchar();

	run_problemInception();

	return true;
}

bool Net::init_problem() {
	cl_int status;

	if (ocl.num_devices == 0) {
		ocl.checkError(-1, "No devices");
	}

	int kernel_rows = this->KernelSize[0] >> 8;
	int kernel_cols = this->KernelSize[0] & 0xFF;

	printf("\n\nkernel_rows: %d \nkernel_rows:%d\n\n", kernel_rows,
			kernel_cols);

	// Clear per-device objects.
	ocl.queue.clear();
	ocl.kernel_conv.clear();
	ocl.kernel_pool.clear();
	ocl.kernel_padding.clear();
	ocl.kernel_relu.clear();
	ocl.kernel_elu.clear();
	ocl.kernel_sigm.clear();
	ocl.kernel_tanh.clear();
	ocl.kernel_matrix_mult.clear();

	ocl.n_per_device.clear();
	ocl.input_img_buf.clear();
	ocl.input_kernel_buf.clear();
	ocl.input_bias_buf.clear();
	ocl.input_hidden_weigth_buf.clear();
	ocl.input_hidden_bias_buf.clear();

	ocl.output_conv_buf.clear();
	ocl.output_pool_buf.clear();
	ocl.output_padd_buf.clear();
	ocl.output_hidden_buf.clear();

	// Create per-device objects.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		cl_command_queue cmq;

		cl_kernel ker_conv;
		cl_kernel ker_pool;
		cl_kernel ker_pad;
		cl_kernel ker_relu;
		cl_kernel ker_elu;
		cl_kernel ker_sigm;
		cl_kernel ker_tanh;
		cl_kernel ker_mat_mul;

		unsigned n_per;

		cl_mem in_img;
		cl_mem in_ker;
		cl_mem in_bias;
		cl_mem in_hidden_weight;
		cl_mem in_hidden_bias;
		cl_mem ou_cnv;
		cl_mem ou_pool;
		cl_mem ou_padd;
		cl_mem ou_hidden;

		ocl.queue.push_back(cmq);

		ocl.kernel_conv.push_back(ker_conv);
		ocl.kernel_pool.push_back(ker_pool);
		ocl.kernel_padding.push_back(ker_pad);
		ocl.kernel_relu.push_back(ker_relu);
		ocl.kernel_elu.push_back(ker_elu);
		ocl.kernel_sigm.push_back(ker_sigm);
		ocl.kernel_tanh.push_back(ker_tanh);
		ocl.kernel_matrix_mult.push_back(ker_mat_mul);

		ocl.n_per_device.push_back(n_per);

		ocl.input_img_buf.push_back(in_img);
		ocl.input_kernel_buf.push_back(in_ker);
		ocl.input_bias_buf.push_back(in_bias);
		ocl.input_bias_buf.push_back(in_bias);
		ocl.input_hidden_weigth_buf.push_back(in_hidden_weight);
		ocl.input_hidden_bias_buf.push_back(in_hidden_bias);

		ocl.output_conv_buf.push_back(ou_cnv);
		ocl.output_pool_buf.push_back(ou_pool);
		ocl.output_padd_buf.push_back(ou_padd);
		ocl.output_hidden_buf.push_back(ou_hidden);

	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {

		// Command queue.
		ocl.queue[i] = clCreateCommandQueue(ocl.context, ocl.devices[i], NULL,
				&status);
		//ocl.queue[i] = clCreateCommandQueueWithProperties(ocl.context, ocl.devices[i], NULL, &status);
		ocl.checkError(status, "Failed to create command queue");

		// Kernel conv.
		ocl.kernel_conv[i] = clCreateKernel(ocl.prog, "cnn_conv_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_conv_layer");

		// Kernel Pool.
		ocl.kernel_pool[i] = clCreateKernel(ocl.prog, "cnn_pool_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_pool_layer");

		// Kernel Padding.
		ocl.kernel_padding[i] = clCreateKernel(ocl.prog, "cnn_padding_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_padding_layer");

		// Kernel Relu.
		ocl.kernel_relu[i] = clCreateKernel(ocl.prog, "cnn_relu_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_relu_layer");

		// Kernel Elu.
		ocl.kernel_elu[i] = clCreateKernel(ocl.prog, "cnn_elu_layer", &status);
		ocl.checkError(status, "Failed to create kernel cnn_elu_layer");

		// Kernel Sigmoide.
		ocl.kernel_sigm[i] = clCreateKernel(ocl.prog, "cnn_sigm_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_sigm_layer");

		// Kernel Tanh.
		ocl.kernel_tanh[i] = clCreateKernel(ocl.prog, "cnn_tanh_layer",
				&status);
		ocl.checkError(status, "Failed to create kernel cnn_tanh_layer");

		// Kernel Matrix_Multi.
		ocl.kernel_matrix_mult[i] = clCreateKernel(ocl.prog,
				"cnn_matrixMul_layer", &status);
		ocl.checkError(status, "Failed to create kernel cnn_tanh_layer");

		// Determine the number of elements processed by this device.
		ocl.n_per_device[i] = ocl.N / ocl.num_devices; // number of elements handled by this device

		// Spread out the remainder of the elements over the first
		// N % num_devices.

		if (i < (ocl.N % ocl.num_devices)) {
			ocl.n_per_device[i]++;
		}
	}

	run_problem();

	return true;
}

void convolutionRowHost(float *h_Dst, float *h_Src, float *h_Kernel, int imageW,
		int imageH, int kernelR) {
	for (int y = 0; y < imageH; y++)
		for (int x = 0; x < imageW; x++) {
			double sum = 0;
			for (int k = -kernelR; k <= kernelR; k++) {
				int d = x + k;
				if (d >= 0 && d < imageW)
					sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
			}
			h_Dst[y * imageW + x] = (float) sum;
		}
}

void Net::convolutionOcl6(cl_mem &input_conv, cl_mem &output_conv,
		cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols,
		int kernel_rows, int kernel_cols, short kernel_amount,
		short kernel_depth, int featuresX, int featuresY, short strideX,
		short strideY) {

	//float *OutConv = NULL;
	cl_int status;
	size_t global_work_size;

	cl_mem teste = NULL;

	//printf("\n\n(%d %d)\n\n(%d %d)\n\n", kernel_amount, kernel_depth, kernel_rows, kernel_cols);
	//getchar();
	//FILE * file = fopen("FileConv.txt", "w");

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = kernel_amount * kernel_depth;

		teste = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
				sizeof(float) * kernel_amount * kernel_depth * featuresY
						* featuresX,
				NULL, &status);
		ocl.checkError(status, "Failed to create buffer for convW");

		cl_float initValue = 0.0;
		clEnqueueFillBuffer(ocl.queue[0], teste, &initValue, sizeof(float), 0,
				sizeof(float) * kernel_amount * kernel_depth * featuresY
						* featuresX, 0, NULL, NULL);
		clFinish(ocl.queue[0]);
	}
	in = ocl.getCurrentTimestamp();
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		const int BLOCK = 2;

		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&teste);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				BLOCK * kernel_rows * kernel_cols * sizeof(float), NULL);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		//status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, BLOCK*kernel_rows*kernel_cols*sizeof(float), NULL);
		//ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		const size_t dimension = 3;

		size_t vec_local[dimension];
		size_t vec_global[dimension];

		if (ocl.n_per_device[i] > ocl.max_work_itens_size) {
			vec_local[0] = 2;// (int)floor((float)ocl.max_work_itens_size / featuresY);
			vec_local[1] = 4;
			vec_local[2] = 64;		// featuresY;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = (int) ceil((float) featuresX / vec_local[1])
					* vec_local[1];
			;		// (int)ceil((float)featuresX / vec_local[1])*vec_local[1];
			vec_global[2] = (int) ceil((float) featuresY / vec_local[2])
					* vec_local[2];
			;
		} else {
			vec_local[0] = 2;// (int)floor((float)ocl.max_work_itens_size / featuresY);
			vec_local[1] = 4;
			vec_local[2] = 64;		// featuresY;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = (int) ceil((float) featuresX / vec_local[1])
					* vec_local[1];
			;		//
			vec_global[2] = (int) ceil((float) featuresY / vec_local[2])
					* vec_local[2];
		}

		//printf("%d -> (%d x %d x %d) - (%d x %d x %d)\n", ocl.n_per_device[i], vec_global[0], vec_global[1], vec_global[2], vec_local[0], vec_local[1], vec_local[2]);
		//getchar();

		ConvStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_conv[i],
				dimension, NULL, vec_global, vec_local, 0,
				NULL, NULL);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		ConvEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- ConvStart_NDRangeKernelTimer);
	}
	//printf("SAIU\n\n");
	//getchar();

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++, sizeof(cl_mem),
				&teste);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++, sizeof(cl_mem),
				&output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++, sizeof(cl_mem),
				&bias_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		ocl.n_per_device[i] = kernel_amount;

		const size_t dimension = 3;
		const int BLOCK = 8;

		size_t vec_local[dimension];
		size_t vec_global[dimension];

		if (ocl.n_per_device[i] > ocl.max_work_itens_size) {
			vec_local[0] = 1;
			vec_local[1] = BLOCK;
			vec_local[2] = 64;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = (int) ceil((float) featuresX / vec_local[1])
					* vec_local[1];
			vec_global[2] = (int) ceil((float) featuresY / vec_local[2])
					* vec_local[2];
		} else {
			vec_local[0] = 1;
			vec_local[1] = BLOCK;
			vec_local[2] = 64;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = (int) ceil((float) featuresX / vec_local[1])
					* vec_local[1];
			;
			vec_global[2] = (int) ceil((float) featuresY / vec_local[2])
					* vec_local[2];
		}

		//printf("%d -> (%d x %d x %d) - (%d x %d x %d)\n", ocl.n_per_device[i], vec_global[0], vec_global[1], vec_global[2], vec_local[0], vec_local[1], vec_local[2]);
		//getchar();

		ConvStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_deep[i],
				dimension, NULL, vec_global, vec_local, 0,
				NULL, NULL);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		ConvEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- ConvStart_NDRangeKernelTimer);
	}

	status = clReleaseMemObject(teste);
	ocl.checkError(status, "clReleaseMemObject failed. (output_pad)");

	ou += (ocl.getCurrentTimestamp() - in);
}

void Net::convolutionOcl2(cl_mem &input_conv, cl_mem &output_conv,
		cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols,
		int kernel_rows, int kernel_cols, short kernel_amount,
		short kernel_depth, int featuresX, int featuresY, short strideX,
		short strideY) {
	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	//float *OutConv = NULL;
	cl_int status;
	size_t global_work_size;

	FILE * file = fopen("FileConv.txt", "w");

	for (unsigned i = 0; i < ocl.num_devices; ++i)
		ocl.n_per_device[i] = kernel_amount;

	/*
	 rows = 6;
	 cols = 6;
	 featuresX = 2;
	 featuresY = 2;
	 kernel_rows = 3;
	 kernel_cols = 3;
	 */

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&bias_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				8 * kernel_rows * kernel_cols * sizeof(float), NULL);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		//printf("kernel_amount: %d\nmax_work_itens_size: %d\n", ocl.n_per_device[i], ocl.max_work_itens_size);

		size_t vec_local[3];
		size_t vec_global[3];
		if (ocl.n_per_device[i] > ocl.max_work_itens_size) {
			vec_local[0] = ocl.max_work_itens_size;
			vec_local[1] = 1;
			vec_local[2] = 1;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / ocl.max_work_itens_size)
					* ocl.n_per_device[i];
			vec_global[1] = featuresX;
			vec_global[2] = featuresY;
		} else {
			vec_local[1] = 1;
			vec_local[2] = featuresY;
			vec_local[0] = 8;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = featuresX;
			vec_global[2] = featuresY;
		}

		//size_t vec_global[3] = { ocl.n_per_device[i], featuresX, featuresY };
		//size_t vec_local[3] = { x, 1, featuresY };

		//printf("(%d x %d x %d) - (%d x %d x %d)\n", vec_global[0], vec_global[1], vec_global[2], vec_local[0], vec_local[1], vec_local[2]);
		//getchar();

		//printf("Launching kernel_conv for device %d (%d elements)\n", i, global_work_size);
		ConvStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_conv[i], 3,
				NULL, vec_global, vec_local, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");

		clFinish(ocl.queue[i]);

		ConvEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- ConvStart_NDRangeKernelTimer);
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++) {
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	}
	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	/*
	 float*OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[0], // Corresponding command queue
	 teste, // Buffer to be mapped
	 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
	 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
	 0, // offset
	 sizeof(float) * featuresX * featuresY* ocl.n_per_device[0], // number of bytes mapped
	 0,// number of events in the wait list
	 NULL,// event wait list
	 &ocl.finish_event[0],// event
	 &status);// error
	 ocl.checkError(status, "Failed to clEnqueueMapBuffer");


	 // Wait for all devices to finish.
	 for (int i = 0; i < ocl.num_devices; i++) {
	 clWaitForEvents(ocl.num_devices, &ocl.finish_event[i]);
	 }
	 // Release all events.
	 for (unsigned i = 0; i < ocl.num_devices; ++i) {
	 clReleaseEvent(ocl.finish_event[i]);
	 }*/

}

void Net::convolutionOcl(cl_mem &input_conv, cl_mem &output_conv,
		cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols,
		int kernel_rows, int kernel_cols, short kernel_amount,
		short kernel_depth, int featuresX, int featuresY, short strideX,
		short strideY) {
	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	float *OutConv = NULL;
	cl_int status;
	size_t global_work_size;

	//FILE * file = fopen("FileConv.txt", "w");

	for (unsigned i = 0; i < ocl.num_devices; ++i)
		ocl.n_per_device[i] = kernel_amount;		//*kernel_depth;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		short ondepth = 0;
		const size_t BLOCK = 32;
		const size_t dimension = 3;
		size_t vec_local[dimension];
		size_t vec_global[dimension];

		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		//status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem), &bias_vector);
		//ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		//status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, BLOCK*kernel_rows *kernel_cols*sizeof(float), NULL);
		//ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		//status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, BLOCK*kernel_rows *kernel_cols*sizeof(float), NULL);
		//ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &ondepth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		//printf("kernel_amount: %d\nmax_work_itens_size: %d\n", ocl.n_per_device[i], ocl.max_work_itens_size);

		vec_local[0] = BLOCK;
		vec_local[1] = 1;
		vec_local[2] = 4;

		vec_global[0] = (int) ceil((float) ocl.n_per_device[i] / vec_local[0])
				* vec_local[0];
		vec_global[1] = (int) ceil((float) featuresX / vec_local[1])
				* vec_local[1];
		vec_global[2] = (int) ceil((float) featuresY / vec_local[2])
				* vec_local[2];

		//printf("Launching kernel_conv for device %d (%d elements)\n", i, global_work_size);
		ConvStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_conv[i],
				dimension, NULL, vec_global, vec_local, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		ConvEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- ConvStart_NDRangeKernelTimer);

		/*OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 output_conv, // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");*/
	}

	/*// Wait for all devices to finish.
	 for (int i = 0; i < ocl.num_devices; i++) {
	 clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	 }
	 // Release all events.
	 for (unsigned i = 0; i < ocl.num_devices; ++i) {
	 clReleaseEvent(ocl.kernel_event[i]);
	 }

	 for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++)
	 {
	 if (i % (featuresX * featuresY)) {
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % (featuresY)))
	 printf("\n");
	 }
	 else {
	 printf("%3.4lf\n\n", OutConv[i - 1]);

	 }
	 }

	 printf("\n\nfoi\n\n");
	 getchar();*/

}

void Net::convolutionDepthOcl(cl_mem &input_conv, cl_mem &output_conv,
		cl_mem &kernel_vector, int rows, int cols, int kernel_rows,
		int kernel_cols, short kernel_amount, short kernel_depth, int featuresX,
		int featuresY, short strideX, short strideY) {
	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	float *OutConv = NULL;
	cl_int status;
	size_t global_work_size;

	//FILE * file = fopen("FileConv.txt", "w");

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = kernel_amount;

	}
	/*
	 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[0], // Corresponding command queue
	 output_conv, // Buffer to be mapped
	 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
	 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
	 0, // offset
	 kernel_amount* kernel_depth * sizeof(float) * (rows) * (cols), // number of bytes mapped
	 0,// number of events in the wait list
	 NULL,// event wait list
	 NULL,// event
	 &status);// error
	 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
	 for (int i = 1; i <= kernel_amount*kernel_depth * (rows) * (cols); i++)
	 {
	 if (i % ((rows) * (cols))) {
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % ((cols))))
	 printf("\n");
	 }
	 else {
	 printf("%3.4lf\n\n", OutConv[i - 1]);
	 getchar();
	 }
	 }*/
	//printf("rows %d, cols %d\n",rows,cols);
	//printf("featuresX %d, featuresY %d\n",featuresX,featuresY);
	//printf("kernel_rows %d, kernel_cols %d\n",kernel_rows,kernel_cols);
	//printf("kernel_amount %d, kernel_depth %d\n",kernel_amount,kernel_depth);
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		short ondepth = 1;
		const size_t BLOCK = 32;
		const size_t dimension = 3;
		size_t vec_local[dimension];
		size_t vec_global[dimension];

		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_mem), &input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_mem), &kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_mem), &output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_depth[i], argi_conv++,
				sizeof(cl_short), &ondepth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		//printf("kernel_amount: %d\nmax_work_itens_size: %d\n", ocl.n_per_device[i], ocl.max_work_itens_size);
		//getchar();
		global_work_size = ocl.n_per_device[i];

		vec_local[0] = BLOCK;
		vec_local[1] = 1;
		vec_local[2] = 4;

		vec_global[0] = (int) ceil((float) ocl.n_per_device[i] / vec_local[0])
				* vec_local[0];
		vec_global[1] = (int) ceil((float) featuresX / vec_local[1])
				* vec_local[1];
		vec_global[2] = (int) ceil((float) featuresY / vec_local[2])
				* vec_local[2];

		//printf("Launching kernel_conv for device %d (%d elements)\n", i, global_work_size);
		DepthStart_NDRangeKernelTimer = ocl.getCurrentTimestamp(); // Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_depth[i],
				dimension, NULL, vec_global, vec_local, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		DepthEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- DepthStart_NDRangeKernelTimer);

		/*for (unsigned i = 0; i < ocl.num_devices; ++i) {
		 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 output_conv, // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_amount* kernel_depth * sizeof(float) * (rows-kernel_rows+1) * (cols-kernel_cols+1), // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
		 //clFinish(ocl.queue[i]);
		 }*/
	}
	// Wait for all devices to finish.

	for (int i = 0; i < ocl.num_devices; i++) {
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	}

	// Release all events.
	//for (unsigned i = 0; i < ocl.num_devices; ++i) {
	//    clReleaseEvent(ocl.kernel_event[i]);
	//}

	/*printf("DEPtH\n");
	 for (int i = 1; i <= kernel_amount*kernel_depth * (rows-kernel_rows+1) * (cols-kernel_cols+1); i++)
	 {
	 if (i % ((rows-kernel_rows+1) * (cols-kernel_cols+1))) {
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % ((cols-kernel_cols+1))))
	 printf("\n");
	 }
	 else {
	 printf("%3.4lf\n\n", OutConv[i - 1]);

	 }
	 }
	 getchar();

	 status = clEnqueueUnmapMemObject(ocl.queue[0],
	 output_conv,
	 OutConv,
	 0,
	 NULL,
	 NULL
	 );*/
}

void Net::add_Bias_Ocl(cl_mem &input_add_a, cl_mem &input_add_b, int rows,
		int cols, short deep) {
	cl_int status;
	size_t global_work_size;

	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = deep;
		//printf("deep: %d",deep);
		//getchar();

		// Set kernel arguments.
		unsigned argi_add = 0;
		status = clSetKernelArg(ocl.kernel_add_bias[i], argi_add++,
				sizeof(cl_mem), &input_add_a);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add_bias[i], argi_add++,
				sizeof(cl_mem), &input_add_b);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add_bias[i], argi_add++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add_bias[i], argi_add++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		global_work_size = ocl.n_per_device[i];

		AddBiasStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_add_bias[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		AddBiasEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- AddBiasStart_NDRangeKernelTimer);
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++)
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);

	// Release all events.
	//for (unsigned i = 0; i < ocl.num_devices; ++i) {
	//	clReleaseEvent(ocl.kernel_event[i]);
	//}
}

void Net::cnn_conv_global_local_workitens_kernel_local_layer(cl_mem &input_conv,
		cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector,
		int rows, int cols, int kernel_rows, int kernel_cols,
		short kernel_amount, short kernel_depth, int featuresX, int featuresY,
		short strideX, short strideY) {

	//float *OutConv = NULL;
	cl_int status;
	size_t global_work_size;
	size_t local_work_size;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = kernel_amount;

	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&bias_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				kernel_depth * kernel_rows * kernel_cols * sizeof(cl_float),
				NULL);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		size_t vec_global[] = { kernel_depth * ocl.n_per_device[i], 1, 1 };
		size_t vec_local[] = { kernel_depth, 1, 1 };

		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_conv[i], 1,
				NULL, vec_global, vec_local, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");

	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++) {
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	}
	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

}

void Net::convolutionOcl5(cl_mem &input_conv, cl_mem &output_conv,
		cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols,
		int kernel_rows, int kernel_cols, short kernel_amount,
		short kernel_depth, int featuresX, int featuresY, short strideX,
		short strideY) {

	//float *OutConv = NULL;
	cl_int status;

	cl_mem teste = NULL;

	//printf("\n\n(%d %d)\n\n(%d %d)\n\n", kernel_amount, kernel_depth, featuresX, featuresY);
	//FILE * file = fopen("FileConv.txt", "w");

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = kernel_amount * kernel_depth;

		teste = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
				sizeof(float) * kernel_amount * kernel_depth * featuresY
						* featuresX,
				NULL, &status);
		ocl.checkError(status, "Failed to create buffer for convW");

		cl_float initValue = 0.0;
		clEnqueueFillBuffer(ocl.queue[0], teste, &initValue, sizeof(float), 0,
				sizeof(float) * kernel_amount * kernel_depth * featuresY
						* featuresX, 0, NULL, NULL);
		clFinish(ocl.queue[0]);
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&teste);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				64 * kernel_rows * kernel_cols * sizeof(cl_float), NULL);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		size_t vec_local[3];
		size_t vec_global[3];

		vec_local[0] = 64;
		vec_local[1] = 1;
		vec_local[2] = 1;

		vec_global[0] = (int) ceil(
				(float) kernel_depth * ocl.n_per_device[i] / vec_local[0])
				* vec_local[0];
		vec_global[1] = 1;
		vec_global[2] = 1;

		ConvStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_conv[i], 3,
				NULL, vec_global, vec_local, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		ConvEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- ConvStart_NDRangeKernelTimer);

	}

	//printf("\n\nSAIU\n\n");
	//getchar();
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++, sizeof(cl_mem),
				&teste);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++, sizeof(cl_mem),
				&output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++, sizeof(cl_mem),
				&bias_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_deep[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		ocl.n_per_device[i] = kernel_amount;

		const size_t dimension = 3;
		const int deep_local = 4;

		size_t vec_local[dimension];
		size_t vec_global[dimension];

		if (ocl.n_per_device[i] > ocl.max_work_itens_size) {
			vec_local[0] = deep_local;
			vec_local[1] = 1;
			vec_local[2] = featuresY;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = featuresX;
			vec_global[2] = featuresY;
		} else {
			vec_local[0] = deep_local;
			vec_local[1] = 1;
			vec_local[2] = featuresY;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = featuresX;
			vec_global[2] = featuresY;
		}

		//printf("%d -> (%d x %d x %d) - (%d x %d x %d)\n", ocl.n_per_device[i], vec_global[0], vec_global[1], vec_global[2], vec_local[0], vec_local[1], vec_local[2]);
		//getchar();

		ConvStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_deep[i],
				dimension, NULL, vec_global, vec_local, 0,
				NULL, NULL);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		ConvEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- ConvStart_NDRangeKernelTimer);
	}

	status = clReleaseMemObject(teste);
	ocl.checkError(status, "clReleaseMemObject failed. (output_pad)");
}

void Net::convolution_Canonic_Global_Local_WorkItensOcl(cl_mem &input_conv,
		cl_mem &output_conv, cl_mem &kernel_vector, cl_mem &bias_vector,
		int rows, int cols, int kernel_rows, int kernel_cols,
		short kernel_amount, short kernel_depth, int featuresX, int featuresY,
		short strideX, short strideY) {

	//float *OutConv = NULL;
	cl_int status;
	size_t global_work_size;
	size_t local_work_size;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = kernel_amount;

		/*
		 ocl.input_kernel_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		 kernel_amount * kernel_depth * sizeof(float) * kernel_rows * kernel_cols,
		 kernel_vector, &status);
		 ocl.checkError(status, "Failed to create buffer for input B");

		 ocl.input_bias_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		 kernel_amount * sizeof(float),
		 bias_vector, &status);
		 ocl.checkError(status, "Failed to create buffer for bias_vector");

		 // Output buffer.
		 ocl.output_conv_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
		 kernel_amount * featuresX  * featuresY  * sizeof(cl_float),
		 NULL, &status);
		 ocl.checkError(status, "Failed to create buffer for output");*/
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&bias_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		size_t vec_global[] = { kernel_depth * ocl.n_per_device[i], 1, 1 };
		size_t vec_local[] = { kernel_depth, 1, 1 };

		global_work_size = 1;
		local_work_size = 2;
		//printf("Launching kernel_conv for device %d (%d elements)\n", i, global_work_size);

		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_conv[i], 1,
				NULL, vec_global, vec_local, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");

		/*
		 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 output_conv, // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
		 */

	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++) {
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	}
	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	/*
	 for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++)
	 {
	 if (i % (featuresX * featuresY)) {
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % (featuresY)))
	 printf("\n");
	 }
	 else {
	 printf("%3.4lf\n\n", OutConv[i - 1]);
	 getchar();
	 }
	 }*/

	//printf("\n\nCABOU\n\n");
	//getchar();
	/*
	 for (unsigned i = 0; i < ocl.num_devices; ++i) {

	 status = clReleaseMemObject(input_conv[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_conv)");

	 status = clReleaseMemObject(ocl.input_kernel_buf[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_kernel_buf)");

	 status = clReleaseMemObject(ocl.input_bias_buf[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_bias_buf)");
	 }
	 */
}

void Net::convolution_CanonicOcl(cl_mem &input_conv, cl_mem &output_conv,
		cl_mem &kernel_vector, cl_mem &bias_vector, int rows, int cols,
		int kernel_rows, int kernel_cols, short kernel_amount,
		short kernel_depth, int featuresX, int featuresY, short strideX,
		short strideY) {

	//float *OutConv = NULL;
	cl_int status;
	size_t global_work_size;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = 1;

		/*
		 ocl.input_kernel_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		 kernel_amount * kernel_depth * sizeof(float) * kernel_rows * kernel_cols,
		 kernel_vector, &status);
		 ocl.checkError(status, "Failed to create buffer for input B");

		 ocl.input_bias_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		 kernel_amount * sizeof(float),
		 bias_vector, &status);
		 ocl.checkError(status, "Failed to create buffer for bias_vector");

		 // Output buffer.
		 ocl.output_conv_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
		 kernel_amount * featuresX  * featuresY  * sizeof(cl_float),
		 NULL, &status);
		 ocl.checkError(status, "Failed to create buffer for output");*/
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_conv = 0;
		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&input_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&output_conv);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++, sizeof(cl_mem),
				&bias_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_rows);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		status = clSetKernelArg(ocl.kernel_conv[i], argi_conv++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_conv - 1);

		global_work_size = ocl.n_per_device[i];
		//printf("Launching kernel_conv for device %d (%d elements)\n", i, global_work_size);

		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_conv[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");

		/*
		 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 output_conv, // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
		 */

	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++) {
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	}
	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	/*
	 for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++)
	 {
	 if (i % (featuresX * featuresY)) {
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % (featuresY)))
	 printf("\n");
	 }
	 else {
	 printf("%3.4lf\n\n", OutConv[i - 1]);
	 getchar();
	 }
	 }*/

	/*
	 for (unsigned i = 0; i < ocl.num_devices; ++i) {

	 status = clReleaseMemObject(input_conv[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_conv)");

	 status = clReleaseMemObject(ocl.input_kernel_buf[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_kernel_buf)");

	 status = clReleaseMemObject(ocl.input_bias_buf[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_bias_buf)");
	 }
	 */
}

void Net::addOcl(cl_mem &input_add_a, cl_mem &input_add_b, cl_mem &output_add,
		int rows, int cols, short deep) {
	cl_int status;
	size_t global_work_size;

	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_add = 0;
		status = clSetKernelArg(ocl.kernel_add[i], argi_add++, sizeof(cl_mem),
				&input_add_a);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add[i], argi_add++, sizeof(cl_mem),
				&input_add_b);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add[i], argi_add++, sizeof(cl_mem),
				&output_add);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add[i], argi_add++, sizeof(cl_short),
				&rows);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add[i], argi_add++, sizeof(cl_short),
				&cols);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		status = clSetKernelArg(ocl.kernel_add[i], argi_add++, sizeof(cl_short),
				&deep);
		ocl.checkError(status, "Failed to set argument %d", argi_add - 1);

		ocl.n_per_device[i] = deep;

		const size_t dimension = 3;
		const int deep_local = 4;

		size_t vec_local[dimension];
		size_t vec_global[dimension];

		if (ocl.n_per_device[i] > ocl.max_work_itens_size) {
			vec_local[0] = deep_local;
			vec_local[1] = 1;
			vec_local[2] = cols;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = rows;
			vec_global[2] = cols;
		} else {
			vec_local[0] = deep_local;
			vec_local[1] = 1;
			vec_local[2] = cols;

			vec_global[0] = (int) ceil(
					(float) ocl.n_per_device[i] / vec_local[0]) * vec_local[0];
			vec_global[1] = rows;
			vec_global[2] = cols;
		}

		//printf("(%d x %d x %d) - (%d x %d x %d)\n", vec_global[0], vec_global[1], vec_global[2], vec_local[0], vec_local[1], vec_local[2]);
		//getchar();

		AddStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_add[i],
				dimension, NULL, vec_global, vec_local, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		AddEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- AddStart_NDRangeKernelTimer);
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++)
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);

	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}
}

void Net::poolingOcl(cl_mem &input_pool, cl_mem &output_pool, int rows,
		int cols, int poolingSizeX, int poolingSizeY, short kernel_amount,
		int featuresX, int featuresY, short strideX, short strideY) {
	cl_int status;
	size_t global_work_size;

	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);
	//float *OutConv = NULL;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = kernel_amount;

		//ocl.output_pool_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
		//										kernel_amount * sizeof(cl_float) * featuresX* featuresY,
		//										NULL, &status);
		//ocl.checkError(status, "Failed to create buffer for output");
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_pool = 0;
		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++, sizeof(cl_mem),
				&input_pool);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++, sizeof(cl_mem),
				&output_pool);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &featuresX);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &featuresY);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &poolingSizeX);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &poolingSizeY);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &strideX);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &strideY);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		status = clSetKernelArg(ocl.kernel_pool[i], argi_pool++,
				sizeof(cl_short), &kernel_amount);
		ocl.checkError(status, "Failed to set argument %d", argi_pool - 1);

		global_work_size = ocl.n_per_device[i];
		//printf("Launching kernel_pool for device %d (%d elements)\n", i, global_work_size);

		PoolStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_pool[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		PoolEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- PoolStart_NDRangeKernelTimer);
		/*
		 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 ocl.output_pool_buf[i], // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
		 */

	}

	/*
	 for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++)
	 {
	 if (i % (featuresX * featuresY)){
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % (featuresY)))
	 printf("\n");
	 }
	 else{
	 printf("%3.4lf\n\n", OutConv[i - 1]);
	 getchar();
	 }
	 }*/

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++)
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);

	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	//for (unsigned i = 0; i < ocl.num_devices; ++i) {
	//status = clReleaseMemObject(input_pool[i]);
	//ocl.checkError(status, "clReleaseMemObject failed. (ocl.output_conv_buf)");
	//}
}

void Net::paddingOcl(cl_mem &input_padd, cl_mem &output_padd,
		short kernel_depth, short rows, short cols, int pad_top, int pad_bottom,
		int pad_left, int pad_rigth) {
	cl_int status;
	size_t global_work_size;
	float *OutConv = NULL;
	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	int featuresX = (rows + pad_top + pad_bottom);
	int featuresY = (cols + pad_left + pad_rigth);

	unsigned int id_out;

	if (pad_top != 0)
		id_out = pad_left + pad_top * (cols + pad_left + pad_rigth);
	else
		id_out = pad_left;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		this->ocl.n_per_device[i] = (kernel_depth * rows * cols)/ this->ocl.num_devices; // number of elements handled by this device
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_padding = 0;
		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_mem), &input_padd);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_mem), &output_padd);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_uint), &id_out);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_int), &pad_left);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_int), &pad_rigth);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		status = clSetKernelArg(ocl.kernel_padding[i], argi_padding++,
				sizeof(cl_int), &pad_bottom);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		global_work_size = this->ocl.n_per_device[i];
		//printf("Launching kernel_padd for device %d (%d elements) %d\n", i, global_work_size, id_out);

		PadStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_padding[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		PadEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- PadStart_NDRangeKernelTimer);

		/*
		 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 output_padd, // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_depth * sizeof(float) * featuresX * featuresY, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
		 */
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++) {
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	}

	/*
	 for (int i = 1; i <= kernel_depth * featuresX * featuresY; i++)
	 {
	 if (i % (featuresX * featuresY)) {
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % (featuresY)))
	 printf("\n");
	 }
	 else {
	 printf("%3.4lf\n\n", OutConv[i - 1]);
	 getchar();
	 }
	 }

	 */
	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	//for (unsigned i = 0; i < ocl.num_devices; ++i) {
	//status = clReleaseMemObject(input_padd[i]);
	//ocl.checkError(status, "clReleaseMemObject failed. (input_img_buf)");
	//}
}

void Net::batchNormOcl(cl_mem &input_padd, cl_mem &output_padd,
		cl_mem &kernel_vector, short kernel_depth, short rows, short cols) {
	cl_int status;
	size_t global_work_size;
	float *OutConv = NULL;
	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		this->ocl.n_per_device[i] = 1;//(kernel_depth*rows*cols) / this->ocl.num_devices; // number of elements handled by this device

		// Spread out the remainder of the elements over the first
		// N % num_devices.
		//if (i < ((kernel_depth*rows*cols) % this->ocl.num_devices))
		//	this->ocl.n_per_device[i]++;

		// Output buffer.
		//output_padd = clCreateBuffer(	ocl.context, CL_MEM_READ_WRITE,
		//								kernel_depth * featuresX * featuresY * sizeof(float),
		//								NULL, &status);
		//ocl.checkError(status, "Failed to create buffer for output");
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_batch_norm = 0;
		status = clSetKernelArg(ocl.kernel_batch_norm[i], argi_batch_norm++,
				sizeof(cl_mem), &input_padd);
		ocl.checkError(status, "Failed to set argument %d", argi_batch_norm - 1);

		status = clSetKernelArg(ocl.kernel_batch_norm[i], argi_batch_norm++,
				sizeof(cl_mem), &kernel_vector);
		ocl.checkError(status, "Failed to set argument %d", argi_batch_norm - 1);

		status = clSetKernelArg(ocl.kernel_batch_norm[i], argi_batch_norm++,
				sizeof(cl_mem), &output_padd);
		ocl.checkError(status, "Failed to set argument %d", argi_batch_norm - 1);

		status = clSetKernelArg(ocl.kernel_batch_norm[i], argi_batch_norm++,
				sizeof(cl_short), &rows);
		ocl.checkError(status, "Failed to set argument %d", argi_batch_norm - 1);

		status = clSetKernelArg(ocl.kernel_batch_norm[i], argi_batch_norm++,
				sizeof(cl_short), &cols);
		ocl.checkError(status, "Failed to set argument %d", argi_batch_norm - 1);

		status = clSetKernelArg(ocl.kernel_batch_norm[i], argi_batch_norm++,
				sizeof(cl_short), &kernel_depth);
		ocl.checkError(status, "Failed to set argument %d", argi_batch_norm - 1);

		global_work_size = this->ocl.n_per_device[i];
		//printf("Launching kernel_padd for device %d (%d elements) %d\n", i, global_work_size, id_out);

		PadStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_batch_norm[i],
				1, NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		PadEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- PadStart_NDRangeKernelTimer);

		/*OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 output_padd, // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_depth * sizeof(float) * rows * cols, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");*/

	}

	// Wait for all devices to finish.
	/*for (int i = 0; i < ocl.num_devices; i++)
	 {
	 clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	 }


	 for (int i = 1; i <= kernel_depth * rows * cols; i++)
	 {
	 if (i % (rows * cols)) {
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % (cols)))
	 printf("\n");
	 }
	 else {
	 printf("%3.4lf\n\n", OutConv[i - 1]);

	 }
	 }
	 printf("\n\nNORM\n\n");
	 getchar();
	 status = clEnqueueUnmapMemObject(ocl.queue[0],
	 output_padd,
	 OutConv,
	 0,
	 NULL,
	 NULL
	 );

	 // Release all events.
	 for (unsigned i = 0; i < ocl.num_devices; ++i) {
	 clReleaseEvent(ocl.kernel_event[i]);
	 }*/

	//for (unsigned i = 0; i < ocl.num_devices; ++i) {
	//status = clReleaseMemObject(input_padd[i]);
	//ocl.checkError(status, "clReleaseMemObject failed. (input_img_buf)");
	//}
}

void Net::reluOcl(cl_mem &input_relu, short kernel_amount, int featuresX,
		int featuresY) {
	cl_int status;
	size_t global_work_size;
	//float *OutConv = NULL;

	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		this->ocl.n_per_device[i] = (kernel_amount * featuresX * featuresY)
				/ this->ocl.num_devices; // number of elements handled by this device

		// Spread out the remainder of the elements over the first
		// N % num_devices.
		if (i
				< ((kernel_amount * featuresX * featuresY)
						% this->ocl.num_devices))
			this->ocl.n_per_device[i]++;

		// Set kernel arguments.
		unsigned argi_padding = 0;
		status = clSetKernelArg(ocl.kernel_relu[i], argi_padding++,
				sizeof(cl_mem), &input_relu);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		global_work_size = this->ocl.n_per_device[i];
		//printf("Launching kernel_relu for device %d (%d elements)\n", i, global_work_size);
		ActStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_relu[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		clFinish(ocl.queue[i]);
		ActEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- ActStart_NDRangeKernelTimer);
		/*
		 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 input_relu[i], // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
		 */
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++)
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);

	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	/*
	 for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++)
	 {
	 if (i % (featuresX * featuresY)){
	 printf("%3.4lf ", OutConv[i - 1]);

	 if (!(i % (featuresY)))
	 printf("\n");
	 }
	 else{
	 printf("%3.4lf\n\n", OutConv[i - 1]);
	 //getchar();
	 }
	 }
	 */
}

void Net::eluOcl(vector<cl_mem> &input_elu, short kernel_amount, int featuresX,
		int featuresY) {
	cl_int status;
	size_t global_work_size;
	float *OutConv = NULL;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		this->ocl.n_per_device[i] = (kernel_amount * featuresX * featuresY)
				/ this->ocl.num_devices; // number of elements handled by this device

		// Spread out the remainder of the elements over the first
		// N % num_devices.
		if (i
				< ((kernel_amount * featuresX * featuresY)
						% this->ocl.num_devices))
			this->ocl.n_per_device[i]++;

		// Set kernel arguments.
		unsigned argi_padding = 0;
		status = clSetKernelArg(ocl.kernel_elu[i], argi_padding++,
				sizeof(cl_mem), &input_elu[i]);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		global_work_size = this->ocl.n_per_device[i];
		printf("Launching kernel_elu for device %d (%d elements)\n", i,
				global_work_size);

		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_elu[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");

		OutConv = (cl_float*) clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
				input_elu[i], // Buffer to be mapped
				1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
				CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
				0, // offset
				kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
				1, // number of events in the wait list
				&ocl.kernel_event[i], // event wait list
				&ocl.finish_event[i], // event
				&status); // error
		ocl.checkError(status, "Failed to clEnqueueMapBuffer");
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++)
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);

	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++) {
		if (i % (featuresX * featuresY)) {
			printf("%3.4lf ", OutConv[i - 1]);

			if (!(i % (featuresY)))
				printf("\n");
		} else {
			printf("%3.4lf\n\n", OutConv[i - 1]);
			getchar();
		}
	}
}

void Net::sigmOcl(vector<cl_mem> &input_sigm, short kernel_amount,
		int featuresX, int featuresY) {
	cl_int status;
	size_t global_work_size;
	float *OutConv = NULL;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		this->ocl.n_per_device[i] = (kernel_amount * featuresX * featuresY)
				/ this->ocl.num_devices; // number of elements handled by this device

		// Spread out the remainder of the elements over the first
		// N % num_devices.
		if (i
				< ((kernel_amount * featuresX * featuresY)
						% this->ocl.num_devices))
			this->ocl.n_per_device[i]++;

		// Set kernel arguments.
		unsigned argi_padding = 0;
		status = clSetKernelArg(ocl.kernel_sigm[i], argi_padding++,
				sizeof(cl_mem), &input_sigm[i]);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		global_work_size = this->ocl.n_per_device[i];
		printf("Launching kernel_sigm for device %d (%d elements)\n", i,
				global_work_size);

		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_sigm[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");

		OutConv = (cl_float*) clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
				input_sigm[i], // Buffer to be mapped
				1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
				CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
				0, // offset
				kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
				1, // number of events in the wait list
				&ocl.kernel_event[i], // event wait list
				&ocl.finish_event[i], // event
				&status); // error
		ocl.checkError(status, "Failed to clEnqueueMapBuffer");
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++)
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);

	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++) {
		if (i % (featuresX * featuresY)) {
			printf("%3.4lf ", OutConv[i - 1]);

			if (!(i % (featuresY)))
				printf("\n");
		} else {
			printf("%3.4lf\n\n", OutConv[i - 1]);
			getchar();
		}
	}
}

void Net::tanhOcl(vector<cl_mem> &input_tanh, short kernel_amount,
		int featuresX, int featuresY) {
	cl_int status;
	size_t global_work_size;
	float *OutConv = NULL;

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		this->ocl.n_per_device[i] = (kernel_amount * featuresX * featuresY)
				/ this->ocl.num_devices; // number of elements handled by this device

		// Spread out the remainder of the elements over the first
		// N % num_devices.
		if (i
				< ((kernel_amount * featuresX * featuresY)
						% this->ocl.num_devices))
			this->ocl.n_per_device[i]++;

		// Set kernel arguments.
		unsigned argi_padding = 0;
		status = clSetKernelArg(ocl.kernel_tanh[i], argi_padding++,
				sizeof(cl_mem), &input_tanh[i]);
		ocl.checkError(status, "Failed to set argument %d", argi_padding - 1);

		global_work_size = this->ocl.n_per_device[i];
		printf("Launching kernel_tanh for device %d (%d elements)\n", i,
				global_work_size);

		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_tanh[i], 1,
				NULL, &global_work_size, NULL, 0,
				NULL, &ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");

		OutConv = (cl_float*) clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
				input_tanh[i], // Buffer to be mapped
				1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
				CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
				0, // offset
				kernel_amount * sizeof(float) * featuresX * featuresY, // number of bytes mapped
				1, // number of events in the wait list
				&ocl.kernel_event[i], // event wait list
				&ocl.finish_event[i], // event
				&status); // error
		ocl.checkError(status, "Failed to clEnqueueMapBuffer");
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++)
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);

	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	for (int i = 1; i <= kernel_amount * featuresX * featuresY; i++) {
		if (i % (featuresX * featuresY)) {
			printf("%3.4lf ", OutConv[i - 1]);

			if (!(i % (featuresY)))
				printf("\n");
		} else {
			printf("%3.4lf\n\n", OutConv[i - 1]);
			getchar();
		}
	}
}

void Net::hiddenOcl(cl_mem &input_hidden, cl_mem &output_hidden,
		cl_mem &hidden_weigth, cl_mem &hidden_bias, int hidden_rows,
		int hidden_cols) {

	cl_int status;
	size_t global_work_size;

	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	//float *OutConv = NULL;

	int COL2 = 1;

	/*
	 visualizeMem(input_hidden,
	 hidden_cols, ocl.queue[0]);
	 printf("\n\nCHEGOU\n\n");
	 getchar();
	 */
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		ocl.n_per_device[i] = hidden_rows;

		/*
		 ocl.input_hidden_weigth_buf[i] = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		 hidden_rows * hidden_cols * sizeof(float),
		 hidden_weigth, &status);

		 // Output buffer.
		 ocl.output_hidden_buf[i] = clCreateBuffer(	ocl.context, CL_MEM_READ_WRITE,
		 hidden_rows * sizeof(cl_float),
		 NULL, &status);

		 ocl.input_hidden_bias_buf[i] = clCreateBuffer(	ocl.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		 hidden_rows * sizeof(cl_float),
		 hidden_bias, &status);


		 ocl.checkError(status, "Failed to create buffer for output");
		 */
	}

	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		// Set kernel arguments.
		unsigned argi_hidden = 0;

		status = clSetKernelArg(ocl.kernel_matrix_mult[i], argi_hidden++,
				sizeof(cl_mem), &hidden_weigth);
		ocl.checkError(status, "Failed to set argument %d", argi_hidden - 1);

		status = clSetKernelArg(ocl.kernel_matrix_mult[i], argi_hidden++,
				sizeof(cl_mem), &input_hidden);
		ocl.checkError(status, "Failed to set argument %d", argi_hidden - 1);

		status = clSetKernelArg(ocl.kernel_matrix_mult[i], argi_hidden++,
				sizeof(cl_mem), &hidden_bias);
		ocl.checkError(status, "Failed to set argument %d", argi_hidden - 1);

		status = clSetKernelArg(ocl.kernel_matrix_mult[i], argi_hidden++,
				sizeof(cl_mem), &output_hidden);
		ocl.checkError(status, "Failed to set argument %d", argi_hidden - 1);

		status = clSetKernelArg(ocl.kernel_matrix_mult[i], argi_hidden++,
				sizeof(cl_int), &hidden_cols);
		ocl.checkError(status, "Failed to set argument %d", argi_hidden - 1);

		status = clSetKernelArg(ocl.kernel_matrix_mult[i], argi_hidden++,
				sizeof(cl_int), &COL2);
		ocl.checkError(status, "Failed to set argument %d", argi_hidden - 1);

		global_work_size = ocl.n_per_device[i];

		//printf("Launching kernel_conv for device %d (%d elements)\n", i, global_work_size);
		HiddenStart_NDRangeKernelTimer = ocl.getCurrentTimestamp();
		status = clEnqueueNDRangeKernel(ocl.queue[i], ocl.kernel_matrix_mult[i],
				1, NULL, &global_work_size, NULL, 0, NULL,
				&ocl.kernel_event[i]);
		ocl.checkError(status, "Failed to launch kernel");
		HiddenEnd_NDRangeKernelTimer += (ocl.getCurrentTimestamp()
				- HiddenStart_NDRangeKernelTimer);
		/*
		 OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
		 output_hidden, // Buffer to be mapped
		 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
		 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
		 0, // offset
		 sizeof(float) * hidden_rows, // number of bytes mapped
		 1,// number of events in the wait list
		 &ocl.kernel_event[i],// event wait list
		 &ocl.finish_event[i],// event
		 &status);// error
		 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
		 */
	}

	// Wait for all devices to finish.
	for (int i = 0; i < ocl.num_devices; i++) {
		clWaitForEvents(ocl.num_devices, &ocl.kernel_event[i]);
	}

	// Release all events.
	for (unsigned i = 0; i < ocl.num_devices; ++i) {
		clReleaseEvent(ocl.kernel_event[i]);
	}

	/*
	 for (int i = 0; i < hidden_rows; i++)
	 {
	 printf("%3.4lf\n", OutConv[i]);
	 }
	 */

	//for (unsigned i = 0; i < ocl.num_devices; ++i) {
	/*
	 status = clReleaseMemObject(input_hidden[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_conv)");

	 status = clReleaseMemObject(ocl.input_hidden_weigth_buf[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_kernel_buf)");

	 status = clReleaseMemObject(ocl.input_hidden_bias_buf[i]);
	 ocl.checkError(status, "clReleaseMemObject failed. (input_kernel_buf)");
	 */
	//}
}

bool Net::run_problemInception() {
	cl_int status;

	cl_event ker_evnt;
	cl_event fnsh_evnt;
	ocl.kernel_event.clear();
	ocl.finish_event.clear();
	ocl.kernel_event.push_back(ker_evnt);
	ocl.finish_event.push_back(fnsh_evnt);

	FILE* preds = fopen("files/Preds.txt", "w+");
	size_t batchSize = this->params_.batchSize;
	size_t nsamples = this->params_.number_of_images;

	printf("nsamples: %zu\n", nsamples);
	printf("batchSize: %zu\n", batchSize);

	char* result = (char*) malloc(sizeof(char) * nsamples);
	float* err = (float*) malloc(sizeof(float) * nsamples);

	if (params_.database == MNIST) {
		this->io.file = new ifstream(
				this->params_.dirfileMNIST_images + "t10k-images.idx3-ubyte",
				ios::binary);
		this->io.fileLabels = new ifstream(
				this->params_.dirfileMNIST_images + "t10k-labels.idx1-ubyte",
				ios::binary);
	} else if (params_.database == FASHION) {
		this->io.file = new ifstream(
				this->params_.dirfileMNIST_images + "t10k-images-idx3-ubyte",
				ios::binary);
		this->io.fileLabels = new ifstream(
				this->params_.dirfileMNIST_images + "t10k-labels-idx1-ubyte",
				ios::binary);
	} else if (params_.database == CIFAR) {
		this->io.file = new ifstream(this->params_.dirfileCIFAR, ios::binary);
	}

	this->start = 0.0;
	this->end = 0.0;
	this->ReadStart_DataTimer = 0.0;
	this->ReadEnd_DataTimer = 0.0;
	this->ConvStart_NDRangeKernelTimer = 0.0;
	this->ConvEnd_NDRangeKernelTimer = 0.0;
	this->PoolStart_NDRangeKernelTimer = 0.0;
	this->PoolEnd_NDRangeKernelTimer = 0.0;
	this->PadStart_NDRangeKernelTimer = 0.0;
	this->PadEnd_NDRangeKernelTimer = 0.0;
	this->ActStart_NDRangeKernelTimer = 0.0;
	this->ActEnd_NDRangeKernelTimer = 0.0;
	this->AddStart_NDRangeKernelTimer = 0.0;
	this->AddEnd_NDRangeKernelTimer = 0.0;
	this->HiddenStart_NDRangeKernelTimer = 0.0;
	this->HiddenEnd_NDRangeKernelTimer = 0.0;
	this->in = 0;
	this->ou = 0;

	if (params_.database == USER){
		ReadStart_DataTimer = ocl.getCurrentTimestamp();// Run NeedleMan OCLs
		this->io.readDataSet();//lendo dados do Usuário
		ReadEnd_DataTimer += (ocl.getCurrentTimestamp() - ReadStart_DataTimer);
	}

	start = ocl.getCurrentTimestamp();
	for (size_t index_samples = 0; index_samples < nsamples; index_samples +=
			batchSize) {
		if (nsamples - index_samples < batchSize) {
			batchSize = nsamples - index_samples;
			io.setbatchSize(batchSize);
		}

		if (params_.database == MNIST) {
			ReadStart_DataTimer = ocl.getCurrentTimestamp();
			this->io.readDataMNIST(); //lendo dados MNIST
			ReadEnd_DataTimer += (ocl.getCurrentTimestamp()
					- ReadStart_DataTimer);
		} else if (params_.database == FASHION) {
			ReadStart_DataTimer = ocl.getCurrentTimestamp();
			this->io.readDataFASHION(); //lendo dados FASHION_MNIST
			ReadEnd_DataTimer += (ocl.getCurrentTimestamp()
					- ReadStart_DataTimer);
		} else if (params_.database == CIFAR) {
			ReadStart_DataTimer = ocl.getCurrentTimestamp(); // Run NeedleMan OCLs
			this->io.readDataCIFAR(); //lendo dados CIFAR
			ReadEnd_DataTimer += (ocl.getCurrentTimestamp()
					- ReadStart_DataTimer);
		}

		memcpy(&err[index_samples], this->io.labels.ptr<float>(0),
				batchSize * sizeof(float)); //copiando labels

		printf(
				"\n%zu imagens produzidas\n%zu imagens faltando\nbatchSize: %zu\n\n",
				index_samples, nsamples - index_samples, batchSize);

		in = ocl.getCurrentTimestamp();
		for (int index_batch = 0; index_batch < batchSize; index_batch++) {
			float* samples_vector =
					this->io.samples[index_batch].reshape(0, 1).ptr<float>(0);

			// Conv & Pooling
			status = clEnqueueWriteBuffer(ocl.queue[0], ocl.featuresMaps_buf[0],
					1, 0,
					sizeof(float) * featuresSizesX[0] * featuresSizesY[0]
							* featuresDeep[0], samples_vector, 0, 0, NULL);
			clFinish(ocl.queue[0]);
			ocl.checkError(status, "Failed to clEnqueueWriteBuffer samples_vector");

			//####################### INIT CONVOLUTION #########################
			int id_conf = 0;
			char* layer = io.conf_net[id_conf];

			while (strstr(io.conf_net_in[id_conf][0], "flatten") == NULL) {
				if (strstr(layer, "conv") || strstr(layer, "zero")) {

					int inputLayer = 0;
					for (int j = 0; j < io.conf_net.size(); j++) {
						if (strstr(io.conf_net_in[id_conf][0],
								io.conf_net[j])) {
							inputLayer = j + 1;
						}
					}
					/*
					 printf("inputLayer: %d (%d x %d x %d) %d\n", inputLayer, featuresSizesX[id_conf], featuresSizesY[id_conf], featuresDeep[id_conf], id_conf);
					 //printf("Entrou CONV: %d - %s\n\n", id_conf, io.conf_net_in[id_conf][0]);
					 //getchar();

					 printf("CONV OUT %d: (%d x %d x %d)\n\n", inputLayer, featuresSizesX[id_conf], featuresSizesY[id_conf], featuresDeep[id_conf]);
					 getchar();
					 for (int kerA = 0; kerA < featuresDeep[id_conf]; kerA++)
					 {
					 for (int i = 0; i < featuresSizesX[id_conf] * featuresSizesY[id_conf]; i++)
					 {
					 printf("%3.4lf ", featuresMaps[inputLayer][kerA*featuresSizesX[id_conf] * featuresSizesY[id_conf] + i]);
					 if (((i + 1) % (featuresSizesY[id_conf])) == 0)
					 printf("\n");
					 }
					 printf("\n\n");
					 //getchar();
					 }
					 printf("\n\n");
					 getchar();*/

					convAndPoolingInception(featuresSizesX[id_conf],
							featuresSizesY[id_conf], &id_conf);
				} else if (strstr(layer, "pool")) {
					int inputLayer = 0;
					for (int j = 0; j < io.conf_net.size(); j++) {
						if (strstr(io.conf_net_in[id_conf][0],
								io.conf_net[j])) {
							inputLayer = j + 1;
						}
					}

					/*
					 printf("inputLayer: %d (%d x %d x %d) %d\n", inputLayer, featuresSizesX[id_conf], featuresSizesY[id_conf], featuresDeep[id_conf], id_conf);
					 //printf("Entrou CONV: %d - %s\n\n", id_conf, io.conf_net_in[id_conf][0]);
					 //getchar();

					 printf("CONV OUT %d: (%d x %d x %d)\n\n", inputLayer, featuresSizesX[id_conf], featuresSizesY[id_conf], featuresDeep[id_conf]);
					 getchar();
					 for (int kerA = 0; kerA < featuresDeep[id_conf]; kerA++)
					 {
					 for (int i = 0; i < featuresSizesX[id_conf] * featuresSizesY[id_conf]; i++)
					 {
					 printf("%3.4lf ", featuresMaps[inputLayer][kerA*featuresSizesX[id_conf] * featuresSizesY[id_conf] + i]);
					 if (((i + 1) % (featuresSizesY[id_conf])) == 0)
					 printf("\n");
					 }
					 printf("\n\n");
					 //getchar();
					 }
					 printf("\n\n");
					 getchar();*/

					convAndPoolingInception(featuresSizesX[id_conf],
							featuresSizesY[id_conf], &id_conf);
					//convAndPoolingInception(featuresMaps[inputLayer], featuresSizesX[id_conf], featuresSizesY[id_conf], &id_conf);
				} else if (strstr(layer, "concatenate")) {
					//printf("CONCAT: %d\n\n", id_conf);
					//printf("\n\n->>>>%d %d %d", featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
					//getchar();

					float* concatenate = (float*) malloc(
							sizeof(float) * featuresSizesX[id_conf + 1]
									* featuresSizesY[id_conf + 1]
									* featuresDeep[id_conf + 1]);

					long long unsigned int idconcat = 0;

					for (int i = 0; i < io.conf_net_in[id_conf].size(); i++) {
						for (int j = 0; j < io.conf_net.size(); j++) {

							if (strstr(io.conf_net_in[id_conf][i],
									io.conf_net[j])) {
								short featuresX = featuresSizesX[j + 1];
								short featuresY = featuresSizesY[j + 1];
								short featuresD = featuresDeep[j + 1];

								float * OutConv =
										(cl_float*) clEnqueueMapBuffer(
												ocl.queue[0], // Corresponding command queue
												ocl.featuresMaps_buf[j + 1], // Buffer to be mapped
												1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
												CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
												0, // offset
												featuresX * featuresY
														* featuresD
														* sizeof(cl_float), // number of bytes mapped
												0, // number of events in the wait list
												NULL, // event wait list
												NULL, // event
												&status); // error
								ocl.checkError(status, "Failed to clEnqueueMapBuffer");

								memcpy(&concatenate[idconcat], OutConv,
										sizeof(float)
												* (featuresD * featuresX
														* featuresY));

								status = clEnqueueUnmapMemObject(ocl.queue[0],
										ocl.featuresMaps_buf[j + 1], OutConv, 0,
										NULL,
										NULL);
								ocl.checkError(status, "Failed to clEnqueueMapBuffer");

								idconcat = idconcat
										+ (featuresD * featuresX * featuresY);
							}
						}
					}
					status = clEnqueueWriteBuffer(ocl.queue[0],
							ocl.featuresMaps_buf[id_conf + 1], 1, 0,
							sizeof(float) * featuresSizesX[id_conf + 1]
									* featuresSizesY[id_conf + 1]
									* featuresDeep[id_conf + 1], concatenate, 0,
							0, 0);
					ocl.checkError(status, "Failed to clEnqueueWriteBuffer samples_vector");

					free(concatenate);

					/*
					 float * OutConv = (cl_float*)clEnqueueMapBuffer(ocl.queue[0], // Corresponding command queue
					 ocl.featuresMaps_buf[id_conf+1], // Buffer to be mapped
					 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
					 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
					 0, // offset
					 sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1], // number of bytes mapped
					 0,// number of events in the wait list
					 NULL,// event wait list
					 NULL,// event
					 NULL);// error


					 for (int i = 0; i < featuresDeep[id_conf+1]; i++)
					 {
					 for (int j = 0; j < featuresSizesX[id_conf+1] * featuresSizesY[id_conf+1]; j++)
					 {
					 printf("%3.5lf ", OutConv[featuresSizesX[id_conf+1] * featuresSizesY[id_conf+1] * i + j]);
					 if (!((j + 1) % featuresSizesY[id_conf+1]))
					 printf("\n");
					 }
					 printf("\n\n");
					 }
					 */

				} else if (strstr(layer, "add")) {
					int adds[2];
					int cont = 0;

					for (int i = 0; i < io.conf_net_in[id_conf].size(); i++)
						for (int j = 0; j < io.conf_net.size(); j++)
							if (strstr(io.conf_net_in[id_conf][i],
									io.conf_net[j]))
								adds[cont++] = j + 1;

					addOcl(ocl.featuresMaps_buf[adds[0]],
							ocl.featuresMaps_buf[adds[1]],
							ocl.featuresMaps_buf[id_conf + 1],
							featuresSizesX[id_conf + 1],
							featuresSizesY[id_conf + 1],
							featuresDeep[id_conf + 1]);
				}
				layer = io.conf_net[++id_conf];
			}
			//####################### END CONVOLUTION #########################

			--id_conf;
			//####################### FULLY CONECTED #########################
			{
				short hidden_layers = params_.NumHiddenLayers;

				for (int hl = 0; hl < hidden_layers; hl++) {
					int hidden_rows = this->HiddenFeaturesX[hl];
					int hidden_cols = this->HiddenFeaturesY[hl];

					hiddenOcl(ocl.featuresMaps_buf[id_conf],
							ocl.featuresMaps_buf[id_conf + 1],
							this->HiddenLayers[hl]->W,
							this->HiddenLayers[hl]->b, hidden_rows,
							hidden_cols);

					reluOcl(ocl.featuresMaps_buf[id_conf + 1], 1, hidden_rows,
							1);

					/*visualizeMem(ocl.featuresMaps_buf[ocl.featuresMaps_buf.size() - 1 - params_.NumHiddenLayers - 1 + hl + 1],
					 hidden_rows, ocl.queue[0]);
					 printf("\n\nCHEGOU\n\n");
					 getchar();*/
					id_conf++;
				}
			}
			//####################### END FULLY CONECTED #########################

			//####################### SOFTMAX #########################
			{
				hiddenOcl(ocl.featuresMaps_buf[id_conf],
						ocl.featuresMaps_buf[id_conf + 1], this->smr->W,
						this->smr->b, this->params_.nclasses,
						this->params_.NumHiddenNeurons);

				cl_float* M = NULL;
				for (unsigned i = 0; i < ocl.num_devices; ++i) {
					M = (cl_float*) clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
							ocl.featuresMaps_buf[id_conf + 1], // Buffer to be mapped
							1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
							CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
							0, // offset
							sizeof(float) * this->params_.nclasses, // number of bytes mapped
							0, // number of events in the wait list
							NULL, // event wait list
							&ocl.finish_event[i], // event
							&status); // error
					ocl.checkError(status, "Failed to clEnqueueMapBuffer");
				}

				// Wait for all devices to finish.
				for (int i = 0; i < ocl.num_devices; i++)
					clWaitForEvents(ocl.num_devices, &ocl.finish_event[i]);

				// Release all events.
				for (unsigned i = 0; i < ocl.num_devices; ++i) {
					clReleaseEvent(ocl.finish_event[i]);
				}

				float maximum = -INFINITY;
				for (int i = 0; i < this->params_.nclasses; i++)
					if (M[i] > maximum)
						maximum = M[i];

				for (int i = 0; i < this->params_.nclasses; i++)
					M[i] -= maximum;

				for (int i = 0; i < this->params_.nclasses; i++)
					M[i] = (float) exp((double) M[i]);

				float sum = 0.0;
				for (int i = 0; i < this->params_.nclasses; i++)
					sum += M[i];

				for (int i = 0; i < this->params_.nclasses; i++)
					M[i] = M[i] / sum;

				float maxele = M[0];
				int output = 0;
				for (int i = 1; i < this->params_.nclasses; i++) {
					if (M[i] >= maxele) {
						maxele = M[i];
						output = i;
					}
				}

				result[index_samples + index_batch] = output;

				for (int i = 0; i < this->params_.nclasses; i++)
					printf("%.4lf ", M[i]);
				printf("\n");

				cl_event ker_evnt;
				cl_event fnsh_evnt;
				ocl.kernel_event.clear();
				ocl.finish_event.clear();
				ocl.kernel_event.push_back(ker_evnt);
				ocl.finish_event.push_back(fnsh_evnt);

				for (unsigned i = 0; i < ocl.num_devices; ++i) {
					status =
							clEnqueueUnmapMemObject(ocl.queue[i],
									ocl.featuresMaps_buf[ocl.featuresMaps_buf.size()
											- 1], M, 0,
									NULL, &ocl.finish_event[i]);
					ocl.checkError(status, "Failed to clEnqueueMapBuffer");

					for (int i = 0; i < ocl.num_devices; i++) {
						clWaitForEvents(ocl.num_devices, &ocl.finish_event[i]);
					}

				}
			}
			//####################### END SOFTMAX #########################

		}
		ou += (ocl.getCurrentTimestamp() - in);

	}
	end = ocl.getCurrentTimestamp();

	printf("Tempo without IO: %.2f ms\n", (this->ou * 1000));
	printf("Tempo medio Read_DataTimer: %.2f ms\n",
			(this->ReadEnd_DataTimer * 1000));
	printf("Tempo medio CONV_NDRangeKernelTimer: %.2f ms\n",
			(this->ConvEnd_NDRangeKernelTimer * 1000));
	printf("Tempo medio POLL_NDRangeKernelTimer: %.2f ms\n",
			(this->PoolEnd_NDRangeKernelTimer * 1000));
	printf("Tempo medio PAD_NDRangeKernelTimer: %.2f ms\n",
			(this->PadEnd_NDRangeKernelTimer * 1000));
	printf("Tempo medio ACT_NDRangeKernelTimer: %.2f ms\n",
			(this->ActEnd_NDRangeKernelTimer * 1000));
	printf("Tempo medio ADD_NDRangeKernelTimer: %.2f ms\n",
			(this->AddEnd_NDRangeKernelTimer * 1000));
	printf("Tempo medio HIDDEN_NDRangeKernelTimer: %.2f ms\n",
			(this->HiddenEnd_NDRangeKernelTimer * 1000));

	int correct = nsamples;
	for (int i = 0; i < nsamples; i++) {
		err[i] -= (float) result[i];
		if (err[i] != 0)
			--correct;
	}
	printf("\n\ncorrect: %d, total: %d, accuracy: %.2lf\n", correct, nsamples,
			float(correct) / (float) (nsamples));

	return true;
}

void Net::convAndPoolingInception(int rows, int cols, int* layer) {
	cl_int status;
	cl_float initValue = 0.0;

	int id_conf = *layer;
	char* lay = io.conf_net[id_conf];

	while (strstr(lay, "flatten") == NULL) {
		int pad_top = PadTop[id_conf];
		int pad_bottom = PadBottom[id_conf];
		int pad_left = PadLeft[id_conf];
		int pad_rigth = PadRigth[id_conf];

		int featuresX = featuresSizesX[id_conf + 1];
		int featuresY = featuresSizesY[id_conf + 1];

		int featuresD = featuresDeep[id_conf];

		int kernel_rows = this->KernelSize[id_conf] >> 8;
		int kernel_cols = this->KernelSize[id_conf] & 0xFF;
		int conv_strideX = this->ConvStride[id_conf] >> 8;
		int conv_strideY = this->ConvStride[id_conf] & 0xFF;

		int pdimX = this->PoolingDim[id_conf] >> 8;
		int pdimY = this->PoolingDim[id_conf] & 0xFF;
		int pool_strideX = this->PoolStride[id_conf] >> 8;
		int pool_strideY = this->PoolStride[id_conf] & 0xFF;

		int ker_amount = KernelAmount[id_conf];
		int ker_deep = KernelDepth[id_conf];

		ReadStart_DataTimer = ocl.getCurrentTimestamp(); // Run NeedleMan OCLs
		//Zerando Buffer de Saida
		clEnqueueFillBuffer(ocl.queue[0], ocl.featuresMaps_buf[id_conf + 1],
				&initValue, sizeof(float), 0,
				featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1]
						* featuresSizesY[id_conf + 1] * sizeof(cl_float), 0,
				NULL, NULL);
		clFinish(ocl.queue[0]);
		ReadEnd_DataTimer += (ocl.getCurrentTimestamp() - ReadStart_DataTimer);

		if (strstr(lay, "depth")) {
			//printf("Entrou CONV\n");

			bool check_valid = strstr(this->ConvLayers[id_conf]->padding,
					"valid");

			if (!check_valid) {
				//in = ocl.getCurrentTimestamp();
				cl_mem output_pad = clCreateBuffer(ocl.context,
						CL_MEM_READ_WRITE,
						featuresDeep[id_conf]
								* (featuresSizesX[id_conf] + pad_top
										+ pad_bottom)
								* (featuresSizesY[id_conf] + pad_left
										+ pad_rigth) * sizeof(float),
						NULL, &status);
				ocl.checkError(status, "clCreateBuffer failed. (output_pad)");

				clEnqueueFillBuffer(ocl.queue[0], output_pad, &initValue,
						sizeof(float), 0,
						featuresDeep[id_conf]
								* (featuresSizesX[id_conf] + pad_top
										+ pad_bottom)
								* (featuresSizesY[id_conf] + pad_left
										+ pad_rigth) * sizeof(cl_float), 0,
						NULL, NULL);
				clFinish(ocl.queue[0]);

				//visualizeMem(ocl.featuresMaps_buf[id_conf], featuresDeep[id_conf] * featuresSizesX[id_conf] * featuresSizesY[id_conf], ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				paddingOcl(ocl.featuresMaps_buf[id_conf], output_pad,
						featuresDeep[id_conf], featuresSizesX[id_conf],
						featuresSizesY[id_conf], pad_top, pad_bottom, pad_left,
						pad_rigth);

				//visualizeMem(output_pad, featuresDeep[id_conf] * (featuresSizesX[id_conf] + pad_top + pad_bottom) * (featuresSizesY[id_conf] + pad_left + pad_rigth), ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				convolutionOcl(output_pad, ocl.featuresMaps_buf[id_conf + 1],
						this->ConvLayers[id_conf]->W,
						this->ConvLayers[id_conf]->b,
						featuresSizesX[id_conf] + pad_top + pad_bottom,
						featuresSizesY[id_conf] + pad_left + pad_rigth,
						kernel_rows, kernel_cols, ker_amount,
						featuresDeep[id_conf], featuresX, featuresY,
						conv_strideX, conv_strideY);

				//visualizeMem(this->ConvLayers[id_conf]->W, ker_amount * ker_deep * kernel_rows * kernel_cols, ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				//visualizeMem(this->ConvLayers[id_conf]->b, ker_amount, ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				//convolutionOcl(input_conv, output_conv, kernel_vector, bias_vector, rows, cols, kernel_rows, kernel_cols, kernel_amount, kernel_depth, featuresX, featuresY, strideX, strideY);

				//visualizeMem(ocl.featuresMaps_buf[id_conf + 1], featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1], ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				status = clReleaseMemObject(output_pad);
				ocl.checkError(status, "clReleaseMemObject failed. (output_pad)");

				//ou += (ocl.getCurrentTimestamp() - in);
			} else {
				//in = ocl.getCurrentTimestamp();
				convolutionDepthOcl(ocl.featuresMaps_buf[id_conf],
						ocl.featuresMaps_buf[id_conf + 1],
						this->ConvLayers[id_conf]->W, rows, cols, kernel_rows,
						kernel_cols, ker_amount, ker_deep, featuresX, featuresY,
						conv_strideX, conv_strideY);
				//ou += (ocl.getCurrentTimestamp() - in);
			}

			//reluOcl( ocl.featuresMaps_buf[id_conf + 1], ker_amount, featuresX, featuresY);

			/*
			 float* Out = (cl_float*)clEnqueueMapBuffer(
			 ocl.queue[0], // Corresponding command queue
			 ocl.featuresMaps_buf[id_conf + 1], // Buffer to be mapped
			 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
			 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
			 0, // offset
			 sizeof(float) * featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]* featuresDeep[id_conf + 1], // number of bytes mapped
			 0,// number of events in the wait list
			 NULL,// event wait list
			 NULL,// event
			 &status
			 );// error
			 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
			 */
			/*
			 string filename = "file"  + std::to_string(id_conf+1) + ".txt";
			 FILE * file = fopen(filename.c_str(),"w");
			 for (int kerA = 0; kerA < featuresDeep[id_conf + 1]; kerA++)
			 {
			 for (int i = 0; i < featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]; i++)
			 {
			 fprintf(file, "%.5lf ", Out[featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] *kerA + i]);
			 if (!((i + 1) % featuresSizesY[id_conf + 1]))
			 fprintf(file, "\n");
			 }
			 fprintf(file, "\n");
			 }
			 fclose(file);*/
			//printf("FOI LAYERS: %d\n\n", id_conf + 1);
			//getchar();
			/*
			 visualizeMem(ocl.featuresMaps_buf[id_conf + 1], featuresSizesX[id_conf + 1]* featuresSizesY[id_conf + 1]* featuresDeep[id_conf + 1], ocl.queue[0]);
			 printf("\n\n");
			 getchar();
			 */
			rows = featuresX;
			cols = featuresY;

			//printf("Saiu CONV (%d %d) id_conf: %d\n\n", rows, cols, id_conf + 1);
			//getchar();
		} else if (strstr(lay, "conv")) {
			//printf("Entrou CONV\n");

			bool check_valid = strstr(this->ConvLayers[id_conf]->padding,
					"valid");

			if (!check_valid) {
				//in = ocl.getCurrentTimestamp();
				cl_mem output_pad = clCreateBuffer(ocl.context,
						CL_MEM_READ_WRITE,
						featuresDeep[id_conf]
								* (featuresSizesX[id_conf] + pad_top
										+ pad_bottom)
								* (featuresSizesY[id_conf] + pad_left
										+ pad_rigth) * sizeof(float),
						NULL, &status);
				ocl.checkError(status, "clCreateBuffer failed. (output_pad)");

				clEnqueueFillBuffer(ocl.queue[0], output_pad, &initValue,
						sizeof(float), 0,
						featuresDeep[id_conf]
								* (featuresSizesX[id_conf] + pad_top
										+ pad_bottom)
								* (featuresSizesY[id_conf] + pad_left
										+ pad_rigth) * sizeof(cl_float), 0,
						NULL, NULL);
				clFinish(ocl.queue[0]);

				//visualizeMem(ocl.featuresMaps_buf[id_conf], featuresDeep[id_conf] * featuresSizesX[id_conf] * featuresSizesY[id_conf], ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				paddingOcl(ocl.featuresMaps_buf[id_conf], output_pad,
						featuresDeep[id_conf], featuresSizesX[id_conf],
						featuresSizesY[id_conf], pad_top, pad_bottom, pad_left,
						pad_rigth);

				//visualizeMem(output_pad, featuresDeep[id_conf] * (featuresSizesX[id_conf] + pad_top + pad_bottom) * (featuresSizesY[id_conf] + pad_left + pad_rigth), ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				convolutionOcl(output_pad, ocl.featuresMaps_buf[id_conf + 1],
						this->ConvLayers[id_conf]->W,
						this->ConvLayers[id_conf]->b,
						featuresSizesX[id_conf] + pad_top + pad_bottom,
						featuresSizesY[id_conf] + pad_left + pad_rigth,
						kernel_rows, kernel_cols, ker_amount,
						featuresDeep[id_conf], featuresX, featuresY,
						conv_strideX, conv_strideY);

				//visualizeMem(this->ConvLayers[id_conf]->W, ker_amount * ker_deep * kernel_rows * kernel_cols, ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				//visualizeMem(this->ConvLayers[id_conf]->b, ker_amount, ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				//convolutionOcl(input_conv, output_conv, kernel_vector, bias_vector, rows, cols, kernel_rows, kernel_cols, kernel_amount, kernel_depth, featuresX, featuresY, strideX, strideY);

				//visualizeMem(ocl.featuresMaps_buf[id_conf + 1], featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1], ocl.queue[0]);
				//printf("\n\n");
				//getchar();

				status = clReleaseMemObject(output_pad);
				ocl.checkError(status, "clReleaseMemObject failed. (output_pad)");

				//ou += (ocl.getCurrentTimestamp() - in);
			} else {
				//visualizeMem(ocl.featuresMaps_buf[id_conf ], featuresDeep[id_conf] * featuresSizesX[id_conf] * featuresSizesY[id_conf], ocl.queue[0]);
				//printf("\n\n");
				//getchar();
				//in = ocl.getCurrentTimestamp();
				convolutionOcl(ocl.featuresMaps_buf[id_conf],
						ocl.featuresMaps_buf[id_conf + 1],
						this->ConvLayers[id_conf]->W,
						this->ConvLayers[id_conf]->b, rows, cols, kernel_rows,
						kernel_cols, ker_amount, ker_deep, featuresX, featuresY,
						conv_strideX, conv_strideY);
				//ou += (ocl.getCurrentTimestamp() - in);
			}

			add_Bias_Ocl(ocl.featuresMaps_buf[id_conf + 1],
					this->ConvLayers[id_conf]->b, featuresX, featuresY,
					ker_amount);

			reluOcl(ocl.featuresMaps_buf[id_conf + 1], ker_amount, featuresX,
					featuresY);

			/*
			 float* Out = (cl_float*)clEnqueueMapBuffer(
			 ocl.queue[0], // Corresponding command queue
			 ocl.featuresMaps_buf[id_conf + 1], // Buffer to be mapped
			 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
			 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
			 0, // offset
			 sizeof(float) * featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]* featuresDeep[id_conf + 1], // number of bytes mapped
			 0,// number of events in the wait list
			 NULL,// event wait list
			 NULL,// event
			 &status
			 );// error
			 ocl.checkError(status, "Failed to clEnqueueMapBuffer");
			 */
			/*
			 string filename = "file"  + std::to_string(id_conf+1) + ".txt";
			 FILE * file = fopen(filename.c_str(),"w");
			 for (int kerA = 0; kerA < featuresDeep[id_conf + 1]; kerA++)
			 {
			 for (int i = 0; i < featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]; i++)
			 {
			 fprintf(file, "%.5lf ", Out[featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] *kerA + i]);
			 if (!((i + 1) % featuresSizesY[id_conf + 1]))
			 fprintf(file, "\n");
			 }
			 fprintf(file, "\n");
			 }
			 fclose(file);*/
			//printf("FOI LAYERS: %d\n\n", id_conf + 1);
			//getchar();
			/*
			 visualizeMem(ocl.featuresMaps_buf[id_conf + 1], featuresSizesX[id_conf + 1]* featuresSizesY[id_conf + 1]* featuresDeep[id_conf + 1], ocl.queue[0]);
			 printf("\n\n");
			 getchar();
			 */
			rows = featuresX;
			cols = featuresY;

			//printf("Saiu CONV (%d %d) id_conf: %d\n\n", rows, cols, id_conf + 1);
			//getchar();
		} else if (strstr(lay, "pool")) {

			//printf("Entrou POOL\n");

			bool check_padding = (pad_top != 0 || pad_bottom != 0
					|| pad_left != 0 || pad_rigth != 0);

			if (check_padding) {
				cl_mem output_pad = clCreateBuffer(ocl.context,
						CL_MEM_READ_WRITE,
						featuresDeep[id_conf]
								* (featuresSizesX[id_conf] + pad_top
										+ pad_bottom)
								* (featuresSizesY[id_conf] + pad_left
										+ pad_rigth) * sizeof(float),
						NULL, &status);
				ocl.checkError(status, "clCreateBuffer failed. (output_pad)");

				clEnqueueFillBuffer(ocl.queue[0], output_pad, &initValue,
						sizeof(float), 0,
						featuresDeep[id_conf]
								* (featuresSizesX[id_conf] + pad_top
										+ pad_bottom)
								* (featuresSizesY[id_conf] + pad_left
										+ pad_rigth) * sizeof(cl_float), 0,
						NULL, NULL);
				clFinish(ocl.queue[0]);

				paddingOcl(ocl.featuresMaps_buf[id_conf], output_pad, featuresD,
						featuresSizesX[id_conf], featuresSizesY[id_conf],
						pad_top, pad_bottom, pad_left, pad_rigth);

				poolingOcl(output_pad, ocl.featuresMaps_buf[id_conf + 1],
						featuresSizesX[id_conf] + pad_top + pad_bottom,
						featuresSizesY[id_conf] + pad_left + pad_rigth, pdimX,
						pdimY, featuresD, featuresX, featuresY, pool_strideX,
						pool_strideY);

				status = clReleaseMemObject(output_pad);
				ocl.checkError(status, "clReleaseMemObject failed. (output_pad)");
			} else
				poolingOcl(ocl.featuresMaps_buf[id_conf],
						ocl.featuresMaps_buf[id_conf + 1], rows, cols, pdimX,
						pdimY, featuresD, featuresX, featuresY, pool_strideX,
						pool_strideY);

			//visualizeMem(ocl.featuresMaps_buf[id_conf + 1], featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1], ocl.queue[0]);
			//printf("\n\n");
			//printf("\n\n");
			//printf("\n\n");
			//getchar();

			rows = featuresX;
			cols = featuresY;
			//printf("Saiu POOL (%d %d) id_conf: %d\n\n", rows, cols,id_conf + 1);
			//getchar();

		} else if (strstr(lay, "zero")) {			//CORRIGIR SAIDA DO PADDING

			//printf("Entrou ZERO\n");

			//paddingOcl(input_padd, output_padd, kernel_depth, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth)
			paddingOcl(ocl.featuresMaps_buf[id_conf],
					ocl.featuresMaps_buf[id_conf + 1], featuresDeep[id_conf],
					featuresSizesX[id_conf], featuresSizesY[id_conf], pad_top,
					pad_bottom, pad_left, pad_rigth);
			//printf("Saiu ZERO (%d %d)\n\n", rows, cols);
			//getchar();

			//visualizeMem(ocl.featuresMaps_buf[id_conf + 1], featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1], ocl.queue[0]);
			//getchar();

			rows = featuresX;
			cols = featuresY;

			//printf("Saiu ZERO (%d %d) id_conf: %d\n\n", rows, cols, id_conf +1);
			//getchar();
		} else if (strstr(lay, "batch_norm")) {		//CORRIGIR SAIDA DO PADDING

			//printf("Entrou ZERO\n");

			//paddingOcl(input_padd, output_padd, kernel_depth, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth)
			batchNormOcl(ocl.featuresMaps_buf[id_conf],
					ocl.featuresMaps_buf[id_conf + 1],
					this->ConvLayers[id_conf]->W, featuresDeep[id_conf],
					featuresSizesX[id_conf], featuresSizesY[id_conf]);
			reluOcl(ocl.featuresMaps_buf[id_conf + 1], featuresDeep[id_conf],
					featuresSizesX[id_conf], featuresSizesY[id_conf]);
			//printf("Saiu ZERO (%d %d)\n\n", rows, cols);
			//getchar();

			//visualizeMem(ocl.featuresMaps_buf[id_conf + 1], featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1], ocl.queue[0]);
			//getchar();

			rows = featuresX;
			cols = featuresY;

			//printf("Saiu ZERO (%d %d) id_conf: %d\n\n", rows, cols, id_conf +1);
			//getchar();
		}

		lay = io.conf_net[++id_conf];
	}

	*layer = id_conf;			//Retornando Id de saida
}

bool Net::run_problem() {
	cl_int status;
	size_t global_work_size;

	short kernel_rows;
	short kernel_cols;

	size_t batchSize = this->params_.batchSize;
	size_t nsamples = this->params_.number_of_images;

	printf("nsamples: %d\n", nsamples);
	printf("batchSize: %d\n", batchSize);

	char* result = (char*) malloc(sizeof(char) * nsamples);
	float* err = (float*) malloc(sizeof(float) * nsamples);

	if (params_.database == MNIST) {
		this->io.file = new ifstream(
				this->params_.dirfileMNIST_images + "t10k-images.idx3-ubyte",
				ios::binary);
		this->io.fileLabels = new ifstream(
				this->params_.dirfileMNIST_images + "t10k-labels.idx1-ubyte",
				ios::binary);
	} else if (params_.database == CIFAR) {
		this->io.file = new ifstream(this->params_.dirfileCIFAR, ios::binary);
	}

	for (int index_samples = 0; index_samples < nsamples; index_samples +=
			batchSize) {
		if (nsamples - index_samples < batchSize) {
			batchSize = nsamples - index_samples;
			io.setbatchSize(batchSize);
		}

		if (params_.database == MNIST) {
			this->io.readDataMNIST();			//lendo dados MNIST
		} else if (params_.database == CIFAR) {
			this->io.readDataCIFAR();			//lendo dados CIFAR
		}

		memcpy(&err[index_samples], this->io.labels.ptr<float>(0),
				batchSize * sizeof(float));			//copiando labels

		printf(
				"\n%d imagens produzidas\n%d imagens faltando\nbatchSize: %d\n\n",
				index_samples, nsamples - index_samples, batchSize);

		for (int index_batch = 0; index_batch < batchSize; index_batch++) {

			short rows = this->params_.samples_rows;
			short cols = this->params_.samples_cols;

			int id_conf = 0;
			string layer = io.conf_net[id_conf];
			int cl = 0;
			int pl = 0;

			float* init_sample =
					this->io.samples[index_batch].reshape(0, 1).ptr<float>(0);
			for (unsigned i = 0; i < ocl.num_devices; ++i) {
				// Input buffers.
				ocl.input_img_buf[i] = clCreateBuffer(ocl.context,
						CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
								| CL_MEM_COPY_HOST_PTR,
						params_.samples_inchl * rows * cols * sizeof(float),
						init_sample, &status);
				ocl.checkError(status, "Failed to create buffer for input A");

			}

			//####################### INIT CONVOLUTION #########################
			while (strstr(io.conf_net[id_conf], "flatten") == NULL) {

				short strideX = 1;
				short strideY = 1;

				int pad_top = PadTop[id_conf];
				int pad_bottom = PadBottom[id_conf];
				int pad_left = PadLeft[id_conf];
				int pad_rigth = PadRigth[id_conf];

				short featuresX = featuresSizesX[id_conf];
				short featuresY = featuresSizesY[id_conf];

				int img_size = 0;			// ImageSize[id_conf];

				if (strstr(io.conf_net[id_conf], "conv")) {
					//printf("Entrou CONV\n\n");

					kernel_rows = this->KernelSize[cl] >> 8;
					kernel_cols = this->KernelSize[cl] & 0xFF;
					strideX = this->ConvStride[cl] >> 8;
					strideY = this->ConvStride[cl] & 0xFF;

					bool check_valid = strstr(this->ConvLayers[cl]->padding,
							"valid");

					short kernel_depth = this->KernelDepth[cl];
					short kernel_amount = this->KernelAmount[cl];

					float* bias_vector = (float*) _aligned_malloc(
							sizeof(float) * kernel_amount, ALIGNMENT);
					//for (int kerA = 0; kerA < kernel_amount; kerA++)
					//	bias_vector[kerA] = this->ConvLayers[cl]->layer[kerA]->b;

					float* kernel_vector = (float*) _aligned_malloc(
							sizeof(float) * kernel_amount * kernel_depth
									* kernel_cols * kernel_rows, ALIGNMENT);

					for (int kerA = 0; kerA < kernel_amount; kerA++) {
						for (int kerD = 0; kerD < kernel_depth; kerD++) {
							//memcpy(&kernel_vector[kerA*kernel_depth*kernel_cols*kernel_rows + kerD*kernel_cols*kernel_rows],
							//	this->ConvLayers[cl]->layer[kerA]->W[kerD],
							//	sizeof(float) * kernel_cols * kernel_rows
							//	);
						}
					}

					//paddingOcl(&teste, 3, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);
					if (!check_valid) {
						//paddingOcl(ocl.input_img_buf, img_size*kernel_depth, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);

						//convolutionOcl(ocl.output_padd_buf, kernel_vector, bias_vector,
						//	rows + pad_top + pad_bottom, cols + pad_left + pad_rigth, kernel_rows, kernel_cols,
						//	kernel_amount, kernel_depth, featuresX, featuresY,
						//	strideX, strideY);
					}
					//else
					//	convolutionOcl(ocl.input_img_buf, kernel_vector, bias_vector,
					//	rows, cols, kernel_rows, kernel_cols,
					//	kernel_amount, kernel_depth, featuresX, featuresY,
					//	strideX, strideY);

					//reluOcl(ocl.output_conv_buf, kernel_amount, featuresX, featuresY);

					swap(ocl.output_conv_buf, ocl.input_img_buf);

					rows = featuresX;
					cols = featuresY;
					cl = cl + 1;

					_aligned_free(bias_vector);
					_aligned_free(kernel_vector);
					//printf("Saiu CONV\n\n");
					//getchar();
				} else if (strstr(io.conf_net[id_conf], "pool")) {
					//printf("Entrou POOL %d \n\n", featuresX);

					int pdimX = this->PoolingDim[pl] >> 8;
					int pdimY = this->PoolingDim[pl] & 0xFF;

					strideX = pdimX;
					strideY = pdimY;

					bool check_padding = (pad_top != 0 || pad_bottom != 0
							|| pad_left != 0 || pad_rigth != 0);

					if (check_padding) {
						//printf("ENTROU\n\n\n");
						//paddingOcl(ocl.input_img_buf, img_size, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);
						//poolingOcl(ocl.output_padd_buf, rows, cols, pdimX, pdimY, img_size, featuresX, featuresY, strideX, strideY);
					}
					//else
					//	poolingOcl(ocl.input_img_buf, rows, cols, pdimX, pdimY, img_size, featuresX, featuresY, strideX, strideY);

					swap(ocl.output_pool_buf, ocl.input_img_buf);

					rows = featuresX;
					cols = featuresY;
					pl = pl + 1;

					//printf("Saiu POOL\n\n");
					//getchar();
				}

				id_conf = id_conf + 1;
				layer = io.conf_net[id_conf];
			}

			/*
			 short featuresX = featuresSizes[id_conf-1] >> 16;
			 short featuresY = featuresSizes[id_conf-1] & 0xFFFF;
			 int img_size	= ImageSize[id_conf-1];

			 cl_float* Out = (cl_float*)clEnqueueMapBuffer(	ocl.queue[0], // Corresponding command queue
			 ocl.input_img_buf[0], // Buffer to be mapped
			 1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
			 CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
			 0, // offset
			 img_size * sizeof(float) * featuresX * featuresY, // number of bytes mapped
			 0,// number of events in the wait list
			 NULL,// event wait list
			 NULL,// event
			 &status);// error
			 ocl.checkError(status, "Failed to clEnqueueMapBuffer");

			 for (int i = 1; i <= img_size * featuresX * featuresY; i++)
			 {
			 if (i % (featuresX * featuresY)){
			 printf("%3.4lf ", Out[i - 1]);

			 if (!(i % (featuresY)))
			 printf("\n");
			 }
			 else{
			 printf("%3.4lf\n\n", Out[i - 1]);
			 getchar();
			 }
			 }
			 */

			//####################### FULLY CONECTED #########################
			short hidden_layers = params_.NumHiddenLayers;

			for (int hl = 0; hl < hidden_layers; hl++) {
				int hidden_rows = this->HiddenFeaturesX[hl];
				int hidden_cols = this->HiddenFeaturesY[hl];

				//printf("Entrou Hidden %d x %d \n\n", hidden_rows, hidden_cols);

				//hiddenOcl(ocl.input_img_buf, this->HiddenLayers[hl].W, this->HiddenLayers[hl].b, hidden_rows, hidden_cols);

				//reluOcl(ocl.output_hidden_buf, 1, hidden_rows, 1);

				swap(ocl.output_hidden_buf, ocl.input_img_buf);

				//printf("Saiu Hidden\n\n");
				//getchar();
			}
			//####################### END FULLY CONECTED #########################

			//####################### SOFTMAX #########################

			//printf("Entrou SoftMax %d x %d \n\n", this->params_.nclasses, this->params_.NumHiddenNeurons);

			//hiddenOcl(ocl.input_img_buf, this->smr->W, this->smr->b, this->params_.nclasses, this->params_.NumHiddenNeurons);

			cl_float* M = NULL;
			for (unsigned i = 0; i < ocl.num_devices; ++i) {
				M = (cl_float*) clEnqueueMapBuffer(ocl.queue[i], // Corresponding command queue
						ocl.output_hidden_buf[i], // Buffer to be mapped
						1, // block_map, CL_TRUE: can't be unmapped before at least 1 read
						CL_MAP_READ | CL_MAP_WRITE, // mapped for reading or writing?
						0, // offset
						sizeof(float) * this->params_.nclasses, // number of bytes mapped
						0, // number of events in the wait list
						NULL, // event wait list
						&ocl.finish_event[i], // event
						&status); // error
				ocl.checkError(status, "Failed to clEnqueueMapBuffer");
			}

			// Wait for all devices to finish.
			for (int i = 0; i < ocl.num_devices; i++)
				clWaitForEvents(ocl.num_devices, &ocl.finish_event[i]);

			// Release all events.
			for (unsigned i = 0; i < ocl.num_devices; ++i) {
				clReleaseEvent(ocl.finish_event[i]);
			}

			for (unsigned i = 0; i < ocl.num_devices; ++i) {
				status = clReleaseMemObject(ocl.output_hidden_buf[i]);
				ocl.checkError(status, "clReleaseMemObject failed. (input_conv)");
			}

			double maximum = -INFINITY;
			for (int i = 0; i < this->params_.nclasses; i++)
				if (M[i] > maximum)
					maximum = M[i];

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] -= maximum;

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] = exp((double) M[i]);

			float sum = 0.0;
			for (int i = 0; i < this->params_.nclasses; i++)
				sum += (double) M[i];

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] = (double) M[i] / sum;

			float maxele = M[0];
			int output = 0;
			for (int i = 1; i < this->params_.nclasses; i++) {
				if (M[i] >= maxele) {
					maxele = M[i];
					output = i;
				}
			}

			result[index_samples + index_batch] = output;

			//printf("Saiu SoftMax : %d\n\n", output);
			//getchar();
		}
	}
	//####################### END SOFTMAX #########################

	int correct = params_.number_of_images;
	for (int i = 0; i < params_.number_of_images; i++) {
		err[i] -= (float) result[i];
		if (err[i] != 0)
			--correct;
	}
	printf("correct: %d, total: %d, accuracy: %.2lf\n", correct,
			params_.number_of_images,
			float(correct) / (float) (params_.number_of_images));
	//cout << "\n\n\n\n\n\correct: " << correct << ", total: " << params_.number_of_images << ", accuracy: " << float(correct) / (float)(params_.number_of_images) << endl;

	return true;
}

Net::~Net() {

	KernelSize.clear();
	KernelAmount.clear();
	PoolingDim.clear();

	/*
	 for (int i = 0; i < ConvLayers.size(); i++)
	 {
	 for (int j = 0; j < ConvLayers[i]->layer.size(); j++)
	 if (ConvLayers[i]->layer[j]->W != NULL)
	 _aligned_free(ConvLayers[i]->layer[j]->W);
	 ConvLayers[i]->layer.clear();
	 }
	 ConvLayers.clear();*/

	/*
	 for (int i = 0; i < HiddenLayers.size(); i++)
	 {
	 if (HiddenLayers[i]->W != NULL)
	 _aligned_free(HiddenLayers[i]->W);
	 if (HiddenLayers[i]->b != NULL)
	 _aligned_free(HiddenLayers[i]->b);
	 }
	 HiddenLayers.clear();

	 if (smr->W != NULL)
	 _aligned_free(smr->W);*/
}

