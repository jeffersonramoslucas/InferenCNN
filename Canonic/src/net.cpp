#include "net.h"


void *_aligned_malloc( size_t required_bytes, size_t alignment )
{
	void *p1;
	void **p2;
	int offset = alignment - 1 + sizeof(void*);
	p1 = malloc(required_bytes + offset);
	if (!p1) {
		return 0;
	}
	p2 = (void**) (((size_t)(p1) + offset) & ~(alignment - 1));
	if (!p2) {
		return 0;
	}
	p2[-1] = p1; //line 6
	return p2;
}

void _aligned_free( void *p )
{
    free(((void**)p)[-1]);
}

Net::Net(Params params_, Io io){

	this->params_ = params_;
	this->io = io;

	this->smr = new Fully();
}

void Net::loadNetwork(){

	printf("\nLoading ConvLayers ...\n");
	//loadConvLayers();
	loadConvLayersInception();

	printf("\nLoading HiddenLayers ...\n");
	loadHiddenLayers();

	printf("\nLoading SMR ...\n");
	loadSmrLayers();
}

void Net::loadConvLayersInception() {

	string sW = params_.dirFiles + "/ConvLayersW.txt";
	string sB = params_.dirFiles + "/ConvLayersB.txt";

	FILE* pInW = fopen(sW.c_str(), "r");
	FILE* pInB = fopen(sB.c_str(), "r");

	fscanf(pInW, "%d", &params_.NumConvLayers);

	int input_featuresX = params_.samples_rows;
	int input_featuresY = params_.samples_cols;
	int input_deep = params_.samples_inchl;

	featuresSizesX.push_back(input_featuresX);
	featuresSizesY.push_back(input_featuresY);
	featuresDeep.push_back(input_deep);

	float* feat = (float*)malloc(sizeof(float)*featuresSizesX[0] * featuresSizesY[0] * featuresDeep[0]);
	this->featuresMaps.push_back(feat);//features da entrada

	int ker_amount_conc;
	int id_conf = 0;

	char* layer = io.conf_net[id_conf];

	int featuresX;
	int featuresY;
	int deep;

	while (strstr(io.conf_net_in[id_conf][0], "flatten") == NULL)
	{
		this->PadTop.push_back(0);
		this->PadBottom.push_back(0);
		this->PadLeft.push_back(0);
		this->PadRigth.push_back(0);

		this->featuresMaps.push_back(NULL);//features de todas as camadas
		this->ConvLayers.push_back(NULL);

		featuresSizesX.push_back(0);
		featuresSizesY.push_back(0);
		featuresDeep.push_back(0);//

		KernelAmount.push_back(0);
		KernelDepth.push_back(0);
		KernelSize.push_back(0);
		ConvStride.push_back(0);

		PoolingDim.push_back(0);
		PoolStride.push_back(0);
		PoolingPad.push_back(0);


		for (int j = 0; j < io.conf_net.size(); j++)
		{
			if (strstr(io.conf_net_in[id_conf][0], "input"))
			{
				featuresX	= input_featuresX;
				featuresY	= input_featuresY;
				deep		= input_deep;
			}
			else if (strstr(io.conf_net_in[id_conf + 1][0], io.conf_net[j]))
			{
				featuresX	= featuresSizesX[j];
				featuresY	= featuresSizesY[j];
				deep		= featuresDeep[j];
			}
		}

		if (strstr(layer, "conv"))
		{
			Cvl* tpcvl = new Cvl();

			int tpKernelSizeX, tpKernelSizeY, tpKernelAmount, tpKernelDepth;
			int strideX = 1;
			int strideY = 1;

			char* padding = (char*)malloc(sizeof(char) * 20);

			fscanf(pInW, "%d %d %d %d %s %d %d", &tpKernelSizeX, &tpKernelSizeY, &tpKernelAmount, &tpKernelDepth, padding, &strideX, &strideY);

			KernelSize[id_conf]		= ((tpKernelSizeX << 8) | tpKernelSizeY);
			KernelDepth[id_conf]	= tpKernelDepth;
			KernelAmount[id_conf]	= tpKernelAmount;
			deep					= tpKernelAmount;
			ConvStride[id_conf]		= ((strideX << 8) | strideY);
			tpcvl->padding			= padding;

			if (strstr(tpcvl->padding, "valid"))
			{

#ifdef DEBUG
				printf("\n\nCONV - valid\n\n");
#endif // DEBUG

				featuresX = (int)ceil((float)(featuresX - tpKernelSizeX + 1) / strideX);
				featuresY = (int)ceil((float)(featuresY - tpKernelSizeY + 1) / strideY);
			}
			else {

				int output_heigth = (int)ceil((float)featuresX / strideX);
				int output_width = (int)ceil((float)featuresY / strideY);

				int pad_along_heigth = (output_heigth - 1) * strideX + tpKernelSizeX - featuresX;
				int pad_along_width = (output_width - 1) * strideY + tpKernelSizeY - featuresY;

				PadTop[id_conf] = (short)(floor((float)pad_along_heigth / 2));
				PadBottom[id_conf] = (pad_along_heigth - PadTop.back());
				PadLeft[id_conf] = (short)(floor((float)pad_along_width / 2));
				PadRigth[id_conf] = (pad_along_width - PadLeft.back());

				featuresX = output_heigth;// ceil((float)(featuresX + PadTop.back() + PadBottom.back() - tpKernelSizeX + 1) / strideX);
				featuresY = output_width;// ceil((float)(featuresY + PadLeft.back() + PadRigth.back() - tpKernelSizeY + 1) / strideY);

#ifdef DEBUG
				printf("\n\nCONV - same\n\n");
				printf("\n\nCONV - featuresX: %d - featuresY: %d  (%d %d)  (%d, %d) (%d,%d)\n\n", featuresX, featuresY, tpKernelSizeX, tpKernelSizeY, PadTop.back(), PadBottom.back(), PadLeft.back(), PadRigth.back());
#endif // DEBUG
			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1] = deep;

#ifdef DEBUG
			printf("featuresX: %d\nfeaturesY: %d  -> %d\n\n", featuresX, featuresY, KernelAmount[id_conf]);
#endif // DEBUG


			for (int j = 0; j < KernelAmount[id_conf]; j++)
			{
				ConvK* tmpConvK = new ConvK();
				loadConvParam(pInW, pInB, tmpConvK, KernelSize[id_conf], KernelDepth[id_conf]);

				tpcvl->layer.push_back(tmpConvK);
			}

			this->ConvLayers[id_conf] = (tpcvl);

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			this->featuresMaps[id_conf + 1] = feat;
		}
		else if (strstr(layer, "pool")) 
		{
			char* padding = (char*)malloc(sizeof(char) * 20);

			int tpPoolSizeX, tpPoolSizeY;
			int strideX = 1;
			int strideY = 1;

			fscanf(pInW, "%d %d %s %d %d", &tpPoolSizeX, &tpPoolSizeY, padding, &strideX, &strideY);

			PoolingDim[id_conf] = ((tpPoolSizeX << 8) | tpPoolSizeY);
			PoolStride[id_conf] = ((strideX << 8) | strideY);
			PoolingPad[id_conf] = (padding);


			if (strstr(padding, "valid"))
			{
				

				int output_heigth = (int)ceil(((float)featuresX - tpPoolSizeX + 1) / strideX);
				int output_width = (int)ceil(((float)featuresY - tpPoolSizeY + 1) / strideY);

#ifdef DEBUG
				printf("\n\nPool - valid\n\n");
				printf("\n\nPOOL - featuresX: %d - featuresY: %d  (%d %d) %d %d\n\n", featuresX, featuresY, tpPoolSizeX, tpPoolSizeY, output_heigth, output_width);
#endif // DEBUG
				
				featuresX = output_heigth;
				featuresY = output_width;
			}
			else
			{
				
				int output_heigth = (int)ceil((float)featuresX / strideX);
				int output_width = (int)ceil((float)featuresY / strideY);

				int pad_along_heigth = max(0, (output_heigth - 1) * strideX + tpPoolSizeX - featuresX);
				int pad_along_width = max(0, (output_width - 1) * strideY + tpPoolSizeY - featuresY);

				//printf("\n\npad_along_heigth: %d\n\n", pad_along_heigth);

				PadTop[id_conf] = (short)(floor((float)pad_along_heigth / 2));
				PadBottom[id_conf] = (pad_along_heigth - PadTop.back());
				PadLeft[id_conf] = (short)(floor((float)pad_along_width / 2));
				PadRigth[id_conf] = (pad_along_width - PadLeft.back());

				//AllocateVectorSizes.push_back((featuresX + PadTop.back() + PadBottom.back() << 8) | featuresY + PadLeft.back() + PadRigth.back());

				featuresX = output_heigth;// floor((float)(featuresX + PadTop.back() + PadBottom.back()) / tpPoolSizeX);
				featuresY = output_width;// floor((float)(featuresY + PadLeft.back() + PadRigth.back()) / tpPoolSizeY);

#ifdef DEBUG
				printf("\n\nPool - same\n\n");
				printf("\n\nPOOL - featuresX: %d - featuresY: %d  (%d %d)  (%d, %d) (%d,%d) %d %d\n\n", featuresX, featuresY, tpPoolSizeX, tpPoolSizeY, PadTop.back(), PadBottom.back(), PadLeft.back(), PadRigth.back(), output_heigth, output_width);		
#endif // DEBUG
			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1]	= deep;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			this->featuresMaps[id_conf + 1] = feat;
		}
		else if (strstr(layer, "zero"))
		{
			int pad_top = 0;
			int pad_bottom = 0;
			int pad_left = 0;
			int pad_rigth = 0;

			fscanf(pInW, "%d %d %d %d", &pad_top, &pad_bottom, &pad_left, &pad_rigth);

			PadTop[id_conf] = pad_top;
			PadBottom[id_conf] = pad_bottom;
			PadLeft[id_conf] = pad_left;
			PadRigth[id_conf] = pad_rigth;


			featuresX = featuresX + pad_top + pad_bottom;
			featuresY = featuresY + pad_left + pad_rigth;

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1]	= deep;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			this->featuresMaps[id_conf + 1] = feat;
		}
		else if (strstr(layer, "concatenate"))
		{
			ker_amount_conc = 0;

			for (int i = 0; i < io.conf_net_in[id_conf].size(); i++)
			{
				for (int j = 0; j < io.conf_net.size(); j++)
				{
					if (strstr(io.conf_net_in[id_conf][i], io.conf_net[j]))
					{
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
			featuresDeep[id_conf + 1]	= ker_amount_conc;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			this->featuresMaps[id_conf + 1] = feat;
		}
		else if (strstr(layer, "flatten"))
		{
			if (!strstr(io.conf_net[id_conf + 1], "concatenate"))
			{
				for (int j = 0; j < io.conf_net.size(); j++)
				{
					if (strstr(io.conf_net_in[id_conf + 1][0], "input"))
					{
						featuresSizesX[id_conf + 1] = input_featuresX;
						featuresSizesY[id_conf + 1] = input_featuresY;
						featuresDeep[id_conf + 1] = input_deep;

						this->featuresMaps[id_conf + 1] = featuresMaps[0];
					}
					else if (strstr(io.conf_net_in[id_conf + 1][0], io.conf_net[j]))
					{
						//printf("%d %d %d\n", featuresSizesX[j + 1], featuresSizesY[j + 1], featuresDeep[j + 1]);
						//getchar();
						featuresSizesX[id_conf + 1] = featuresSizesX[j + 1];
						featuresSizesY[id_conf + 1] = featuresSizesY[j + 1];
						featuresDeep[id_conf + 1] = featuresDeep[j + 1];

						this->featuresMaps[id_conf + 1] = featuresMaps[j + 1];
					}
				}
			}

#ifdef DEBUG
			//printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			//float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			//this->featuresMaps[id_conf + 1] = feat;
		}
		else if (strstr(layer, "add"))
		{
			for (int i = 0; i < io.conf_net_in[id_conf].size(); i++)
			{
				for (int j = 0; j < io.conf_net.size(); j++)
				{
					if (strstr(io.conf_net_in[id_conf][i], io.conf_net[j]))
					{
						
						featuresX	= featuresSizesX[j + 1];
						featuresY	= featuresSizesY[j + 1];
						deep		= featuresDeep[j + 1];
					}
				}
			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1]	= deep;

#ifdef DEBUG
			printf("Alloc %s-%d: %d %d %d\n", io.conf_net[id_conf], id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			//getchar();
#endif // DEBUG

			float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			this->featuresMaps[id_conf + 1] = feat;
		}
		layer = io.conf_net[++id_conf];
	}
	
	PadTop.push_back(0);
	PadBottom.push_back(0);
	PadLeft.push_back(0);
	PadRigth.push_back(0);

	this->featuresMaps.pop_back();


	//this->features = featuresSizesX[id_conf] * featuresSizesY[id_conf] * featuresDeep[id_conf];

#ifdef DEBUG
	printf("\n\n\n\n%d %d %d\n\n", featuresSizesX[id_conf], featuresSizesY[id_conf], featuresDeep[id_conf]);
	printf("End Conv %d \n\n", featuresSizesX[id_conf] * featuresSizesY[id_conf] * featuresDeep[id_conf]);
	//printf("End Conv %d \n\n", this->features);

	for (int i = 0; i < featuresSizesX.size(); i++)
	{
		printf("%d (%d %d %d)\n",i, featuresSizesX[i], featuresSizesY[i], featuresDeep[i]);
	}
	printf("\n\n");
	getchar();
#endif // DEBUG
	

	fclose(pInW);
	fclose(pInB);
}

void Net::loadConvLayers(){

	string sW = params_.dirFiles + "/ConvLayersW.txt";
	string sB = params_.dirFiles + "/ConvLayersB.txt";

	FILE* pInW = fopen(sW.c_str(), "r");
	FILE* pInB = fopen(sB.c_str(), "r");

	fscanf(pInW, "%d", &params_.NumConvLayers);

	int featuresX = params_.samples_rows;
	int featuresY = params_.samples_cols;

	featuresSizesX.push_back(featuresX);
	featuresSizesY.push_back(featuresY);
	featuresDeep.push_back(params_.samples_inchl);

	float* feat = (float*)malloc(sizeof(float)*featuresSizesX[0]*featuresSizesY[0]*featuresDeep[0]);
	this->featuresMaps.push_back(feat);//features da entrada

	int id_conf = 0;
	char* layer = io.conf_net[id_conf];

	while (strstr(layer, "flatten") == NULL)
	{
		this->PadTop.push_back(0);
		this->PadBottom.push_back(0);
		this->PadLeft.push_back(0);
		this->PadRigth.push_back(0);

		this->featuresMaps.push_back(NULL);//features para as demais camadas
		this->ConvLayers.push_back(NULL);

		featuresSizesX.push_back(0);
		featuresSizesY.push_back(0);
		featuresDeep.push_back(0);

		KernelAmount.push_back(0);
		KernelDepth.push_back(0);
		KernelSize.push_back(0);
		ConvStride.push_back(0);

		PoolingDim.push_back(0);
		PoolStride.push_back(0);
		PoolingPad.push_back(0);

		if (strstr(layer, "conv"))
		{
			Cvl* tpcvl = new Cvl();

			int tpKernelSizeX, tpKernelSizeY, tpKernelAmount, tpKernelDepth;
			int strideX = 1;
			int strideY = 1;

			char* padding = (char*)malloc(sizeof(char) * 20);

			fscanf(pInW, "%d %d %d %d %s %d %d", &tpKernelSizeX, &tpKernelSizeY, &tpKernelAmount, &tpKernelDepth, padding, &strideX, &strideY);


			KernelSize[id_conf]		= (tpKernelSizeX << 8) | tpKernelSizeY;
			KernelDepth[id_conf]	= tpKernelDepth;
			KernelAmount[id_conf]	= tpKernelAmount;
			ConvStride[id_conf]		= (strideX << 8) | strideY;

			tpcvl->padding = padding;

			if (strstr(tpcvl->padding, "valid"))
			{
				printf("\n\nCONV - valid\n\n");
				//AllocateVectorSizes.push_back((featuresX << 8) | featuresY);

				featuresX = (int)ceil((float)(featuresX - tpKernelSizeX + 1) / strideX);
				featuresY = (int)ceil((float)(featuresY - tpKernelSizeY + 1) / strideY);
			}
			else{
				printf("\n\nCONV - same\n\n");

				int output_heigth = (int)ceil((float)featuresX / strideX);
				int output_width = (int)ceil((float)featuresY / strideY);

				int pad_along_heigth = (output_heigth - 1) * strideX + tpKernelSizeX - featuresX;
				int pad_along_width = (output_width - 1) * strideY + tpKernelSizeY - featuresY;

				PadTop[id_conf] = (short)(floor((float)pad_along_heigth / 2));
				PadBottom[id_conf] = (pad_along_heigth - PadTop.back());
				PadLeft[id_conf] = (short)(floor((float)pad_along_width / 2));
				PadRigth[id_conf] = (pad_along_width - PadLeft.back());

				//AllocateVectorSizes.push_back(((featuresX + PadTop.back() + PadBottom.back()) << 8) | (featuresY + PadLeft.back() + PadRigth.back()));

				featuresX = output_heigth;// ceil((float)(featuresX + PadTop.back() + PadBottom.back() - tpKernelSizeX + 1) / strideX);
				featuresY = output_width;// ceil((float)(featuresY + PadLeft.back() + PadRigth.back() - tpKernelSizeY + 1) / strideY);

				printf("\n\nCONV - featuresX: %d - featuresY: %d  (%d %d)  (%d, %d) (%d,%d)\n\n", featuresX, featuresY, tpKernelSizeX, tpKernelSizeY, PadTop.back(), PadBottom.back(), PadLeft.back(), PadRigth.back());

			}

			
			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1]	= tpKernelAmount;

			printf("featuresX: %d\nfeaturesY: %d\n\n", featuresX, featuresY);
			//getchar();
			
			
			for (int j = 0; j < KernelAmount[id_conf]; j++)
			{
				ConvK* tmpConvK = new ConvK();
				loadConvParam(pInW, pInB, tmpConvK, KernelSize[id_conf], KernelDepth[id_conf]);

				tpcvl->layer.push_back(tmpConvK);
			}

			this->ConvLayers[id_conf] = (tpcvl);

		}
		else if (strstr(layer, "pool")){

			char* padding = (char*)malloc(sizeof(char) * 20);

			int tpPoolSizeX, tpPoolSizeY;
			int strideX = 1;
			int strideY = 1;

			fscanf(pInW, "%d %d %s %d %d", &tpPoolSizeX, &tpPoolSizeY, padding, &strideX, &strideY);

			PoolingDim[id_conf] = (tpPoolSizeX << 8) | tpPoolSizeY;
			PoolStride[id_conf] = (strideX << 8) | strideY;
			PoolingPad[id_conf] = (padding);

			//int strideX = tpPoolSizeX;
			//int strideY = tpPoolSizeY;

			if (strstr(padding, "valid"))
			{
				printf("\n\nPool - valid\n\n");

				int output_heigth = (int)ceil(((float)featuresX - tpPoolSizeX + 1) / strideX);
				int output_width = (int)ceil(((float)featuresY - tpPoolSizeY + 1) / strideY);

				printf("\n\nPOOL - featuresX: %d - featuresY: %d  (%d %d) %d %d\n\n", featuresX, featuresY, tpPoolSizeX, tpPoolSizeY, output_heigth, output_width);

				featuresX = output_heigth;
				featuresY = output_width;
			}
			else
			{
				printf("\n\nPool - same\n\n");

				int output_heigth = (int)ceil((float)featuresX / strideX);
				int output_width = (int)ceil((float)featuresY / strideY);

				int pad_along_heigth = max(0, (output_heigth - 1) * strideX + tpPoolSizeX - featuresX);
				int pad_along_width = max(0, (output_width - 1) * strideY + tpPoolSizeY - featuresY);

				PadTop[id_conf] = (short)(floor((float)pad_along_heigth / 2));
				PadBottom[id_conf] = (pad_along_heigth - PadTop.back());
				PadLeft[id_conf] = (short)(floor((float)pad_along_width / 2));
				PadRigth[id_conf] = (pad_along_width - PadLeft.back());

				featuresX = output_heigth;// floor((float)(featuresX + PadTop.back() + PadBottom.back()) / tpPoolSizeX);
				featuresY = output_width;// floor((float)(featuresY + PadLeft.back() + PadRigth.back()) / tpPoolSizeY);

				printf("\n\nPOOL - featuresX: %d - featuresY: %d  (%d %d)  (%d, %d) (%d,%d) %d %d\n\n", featuresX, featuresY, tpPoolSizeX, tpPoolSizeY, PadTop.back(), PadBottom.back(), PadLeft.back(), PadRigth.back(), output_heigth, output_width);

			}

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1]	= featuresDeep[id_conf];

		}
		else if (strstr(layer, "zero"))
		{
			int pad_top = 0;
			int pad_bottom = 0;
			int pad_left = 0;
			int pad_rigth = 0;


			fscanf(pInW, "%d %d %d %d", &pad_top, &pad_bottom, &pad_left, &pad_rigth);

			PadTop[id_conf] = pad_top;
			PadBottom[id_conf] = pad_bottom;
			PadLeft[id_conf] = pad_left;
			PadRigth[id_conf] = pad_rigth;


			featuresX = featuresX + pad_top + pad_bottom;
			featuresY = featuresY + pad_left + pad_rigth;

			featuresSizesX[id_conf + 1] = featuresX;
			featuresSizesY[id_conf + 1] = featuresY;
			featuresDeep[id_conf + 1]	= featuresDeep[id_conf];
		}
		
		float* feat = (float*)malloc(sizeof(float)*featuresSizesX[id_conf + 1]*featuresSizesY[id_conf + 1]*featuresDeep[id_conf + 1]);
		this->featuresMaps[id_conf + 1] = feat;

		layer = io.conf_net[++id_conf];
	}
	
	PadTop.push_back(0);
	PadBottom.push_back(0);
	PadLeft.push_back(0);
	PadRigth.push_back(0);

	printf("\n\n\n\n%d %d %d\n\n", featuresSizesX[id_conf], featuresSizesY[id_conf], featuresDeep[id_conf]);

	//this->features = featuresSizesX[id_conf] * featuresSizesY[id_conf] * featuresDeep[id_conf];

	printf("End Conv %d \n\n", featuresSizesX[id_conf] * featuresSizesY[id_conf] * featuresDeep[id_conf]);


	fclose(pInW);
	fclose(pInB);
}

void Net::load(FILE* pIn, float* M, int length)
{
	for (int i = 0; i < length; i++)
		fscanf(pIn, "%f", &M[i]);
}

void Net::loadConvParam(FILE* pInW, FILE* pInB, ConvK* convk, int kernelSize, short kernelDepth)
{
	int width = kernelSize >> 8;
	int heigth = kernelSize & 0xFF;

	for (int i = 0; i < kernelDepth; i++)
	{
		float* W = (float*)_aligned_malloc(sizeof(float) * width * heigth, ALIGNMENT);//Mat::ones(width, width, CV_64FC1);
		if (W == NULL)
		{
			printf("Error allocation aligned memory.");
			getchar();
			exit(-1);
		}

		load(pInW, W, width * heigth);

		//loadMat(pInW, convk->W);
		//printf("%3.4lf\n", convk->b);
		//getchar();

		convk->W.push_back(W);
	}

	fscanf(pInB, " %f", &convk->b);
}

void  Net::loadHiddenLayers()
{
	string sW = params_.dirFiles + "/HiddenLayersW.txt";
	string sB = params_.dirFiles + "/HiddenLayersB.txt";

	FILE* pInW = fopen(sW.c_str(), "r");
	FILE* pInB = fopen(sB.c_str(), "r");

	fscanf(pInW, "%d", &params_.NumHiddenLayers);

	printf("NumHiddenLayers: %d\n\n", params_.NumHiddenLayers);
	//params_.NumHiddenLayers = 2;

	// Init Hidden layers
	for (int hl = 0; hl < params_.NumHiddenLayers; hl++){

		int hiddenfeatures;

		fscanf(pInW, "%d %d", &params_.NumHiddenNeurons, &hiddenfeatures);

		printf("%d %d\n\n", params_.NumHiddenNeurons, hiddenfeatures);

		float* feat = (float*)malloc(sizeof(float) * params_.NumHiddenNeurons);
		this->featuresMaps.push_back(feat);//features da entrada

		Fully tpntw;

		HiddenFeaturesX.push_back(params_.NumHiddenNeurons);
		HiddenFeaturesY.push_back(hiddenfeatures);

		loadHiddenParam(pInW, pInB, tpntw, params_.NumHiddenNeurons, hiddenfeatures);
		this->HiddenLayers.push_back(tpntw);
	}

	fclose(pInW);
	fclose(pInB);
}

void Net::loadHiddenParam(FILE* pInW, FILE* pInB, Fully &ntw, int NumHiddenNeurons, int hiddenfeatures)
{
	ntw.W = (float*)_aligned_malloc(sizeof(float) * NumHiddenNeurons * hiddenfeatures, ALIGNMENT);// Mat::ones(NumHiddenNeurons, hiddenfeatures, CV_64FC1);
	if (ntw.W == NULL)
	{
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	ntw.b = (float*)_aligned_malloc(sizeof(float) * NumHiddenNeurons, ALIGNMENT);//Mat::ones(NumHiddenNeurons, 1, CV_64FC1);
	if (ntw.b == NULL)
	{
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	load(pInW, ntw.W, NumHiddenNeurons * hiddenfeatures);

	load(pInB, ntw.b, NumHiddenNeurons);
}

void Net::loadSmrLayers()
{
	string sW = params_.dirFiles + "/SmrLayerW.txt";
	string sB = params_.dirFiles + "/SmrLayerB.txt";

	FILE* pInW = fopen(sW.c_str(), "r");
	FILE* pInB = fopen(sB.c_str(), "r");

	int nclasses, nfeatures;

	fscanf(pInW, "%d %d", &nclasses, &nfeatures);

	float* feat = (float*)malloc(sizeof(float) * nclasses);
	this->featuresMaps.push_back(feat);//features da entrada

	loadSmrParam(pInW, pInB, this->smr, nclasses, nfeatures);

	fclose(pInW);
	fclose(pInB);
}

void Net::loadSmrParam(FILE* pInW, FILE* pInB, Fully *smr, int nclasses, int nfeatures)
{
	smr->W = (float*)_aligned_malloc(sizeof(float) * nclasses * nfeatures, ALIGNMENT); //Mat::ones(nclasses, nfeatures, CV_64FC1);
	if (smr->W == NULL)
	{
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	smr->b = (float*)_aligned_malloc(sizeof(float) * nclasses, ALIGNMENT); //Mat::ones(nclasses, nfeatures, CV_64FC1);
	if (smr->b == NULL)
	{
		printf("Error allocation aligned memory.");
		getchar();
		exit(-1);
	}

	load(pInW, smr->W, nclasses * nfeatures);

	load(pInB, smr->b, nclasses);
}

void Net::MatrixMult(float* src, float* src2, float* dst, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		float sum = 0.0;

		for (int j = 0; j < cols; j++)
			sum += src[i*cols + j] * src2[j];
		dst[i] = sum;
	}
}

char* Net::resultNetwork(){

	//FILE* preds = fopen("Preds.txt", "w+");
	size_t batchSize = this->params_.batchSize;
	size_t nsamples = this->params_.number_of_images;

	printf("nsamples: %zu\n", nsamples);
	printf("batchSize: %zu\n", batchSize);

	char* result = (char*)malloc(sizeof(char)*nsamples);
	float* err = (float*)malloc(sizeof(float)*nsamples);

	if (params_.database == MNIST)
	{
		this->io.file		= new ifstream((this->params_.dirfileMNIST_images + "t10k-images.idx3-ubyte").c_str(), ios::binary);
		this->io.fileLabels = new ifstream((this->params_.dirfileMNIST_images + "t10k-labels.idx1-ubyte").c_str(), ios::binary);
	}
	else if (params_.database == CIFAR)
	{
		this->io.file = new ifstream(this->params_.dirfileCIFAR.c_str(), ios::binary);
	}

	for (size_t index_samples = 0; index_samples < nsamples; index_samples += batchSize)
	{
		if (nsamples - index_samples < batchSize)
		{
			batchSize = nsamples - index_samples;
			io.setbatchSize(batchSize);
		}

		if (params_.database == MNIST)
		{
			this->io.readDataMNIST();//lendo dados MNIST
		}
		else if (params_.database == CIFAR)
		{
			this->io.readDataCIFAR();//lendo dados CIFAR
		}
		
		
		memcpy(&err[index_samples], this->io.labels.ptr< float >(0), batchSize * sizeof(float));//copiando labels

		
		printf("\n%zu imagens produzidas\n%zu imagens faltando\nbatchSize: %zu\n\n", index_samples, nsamples - index_samples, batchSize);

		for (int index_batch = 0; index_batch < batchSize; index_batch++)
		{
			
			//printf("batch_feitos:%d\n", index_batch);
			float* samples_vector = this->io.samples[index_batch].reshape(0, 1).ptr< float >(0);

			// Conv & Pooling
			float* convOut = NULL;
			/*float* convOut = (float*)_aligned_malloc(sizeof(float) * this->features, ALIGNMENT);
			if (convOut == NULL)
			{
				printf_s("Error allocation memory. convOut");
				getchar();
				exit(-1);
			}*/

			//printf("\n\nstart convAndPooling\n");
			convAndPooling(samples_vector, convOut, this->params_.samples_rows, this->params_.samples_cols);
			//printf("end convAndPooling\n\n");

			/*
			for (int i = 0; i < this->features; i++)
			{
				printf("%3.4lf ", convOut[i]);
			}*/

			// FullyConected
			
			for (int hl = 0; hl < params_.NumHiddenLayers; hl++)
			{
				int hidden_rows = this->HiddenFeaturesX[hl];
				int hidden_cols = this->HiddenFeaturesY[hl];

				//printf("\n\n\n%d %d\n\n\n", hidden_rows, hidden_cols);

				/*
				tmpacti = (float*)_aligned_malloc(sizeof(float) * hidden_rows, ALIGNMENT);
				if (tmpacti == NULL)
				{
					printf_s("Error allocation memory. tmpacti");
					getchar();
					exit(-1);
				}*/
				
				float* tmpacti = featuresMaps[featuresMaps.size()-1 - params_.NumHiddenLayers - 1 + hl + 1];

				MatrixMult(this->HiddenLayers[hl].W, featuresMaps[featuresMaps.size()-1 - params_.NumHiddenLayers - 1 + hl], tmpacti, hidden_rows, hidden_cols);



				for (int i = 0; i < hidden_rows; i++)
					tmpacti[i] += this->HiddenLayers[hl].b[i];

				nonLinearity(tmpacti, hidden_rows);

				//swap(convOut, tmpacti);

				//if (tmpacti != NULL)
				//	_aligned_free(tmpacti);
			}
			//printf("end FullyConected\n\n");
			//getchar();
			// SMR
			float* M = featuresMaps[featuresMaps.size()-1];
			
			/*float* M = (float*)malloc(sizeof(float) * this->params_.nclasses);
			if (M == NULL)
			{
				printf_s("Error allocation memory. M");
				getchar();
				exit(-1);
			}*/

			MatrixMult(this->smr->W, featuresMaps[featuresMaps.size() - 1 -1], M, this->params_.nclasses, this->params_.NumHiddenNeurons);

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] += this->smr->b[i];

			//if (convOut != NULL)
			//	_aligned_free(convOut);

			float maximum = -INFINITY;
			for (int i = 0; i < this->params_.nclasses; i++)
				if (M[i] > maximum)
					maximum = M[i];

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] -= maximum;

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] = (float)exp((double)M[i]);


			float sum = 0.0;
			for (int i = 0; i < this->params_.nclasses; i++)
				sum += M[i];

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] = M[i] / sum;

			float maxele = M[0];
			int output = 0;
			for (int i = 1; i < this->params_.nclasses; i++){
				if (M[i] >= maxele){
					maxele = M[i];
					output = i;
				}
			}
			//printf("end SMR %d\n\n", output);
			//getchar();
			//for (int i = 0; i < this->params_.nclasses; i++)
			//	fprintf(preds, "%.1lf ", M[i]);
			//fprintf(preds, "\n");

			result[index_samples + index_batch] = output;

			//if (M != NULL)
			//	free(M);
		}
	}

	int correct = params_.number_of_images;
	for (int i = 0; i<params_.number_of_images; i++){
		err[i] -= (float)result[i];
		if (err[i] != 0) --correct;
	}
	printf("correct: %d, total: %d, accuracy: %.2lf\n", correct, params_.number_of_images, float(correct) / (float)(params_.number_of_images));
	//cout << "correct: " << correct << ", total: " << params_.number_of_images << ", accuracy: " << float(correct) / (float)(params_.number_of_images) << endl;

	return result;
}

char* Net::resultNetworkInception() {

	FILE* preds = fopen("Preds.txt", "w+");
	size_t batchSize = this->params_.batchSize;
	size_t nsamples = this->params_.number_of_images;

	printf("nsamples: %zu\n", nsamples);
	printf("batchSize: %zu\n", batchSize);

	char* result = (char*)malloc(sizeof(char)*nsamples);
	float* err = (float*)malloc(sizeof(float)*nsamples);

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

	if (params_.database == USER){
		this->io.readDataSet();//lendo dados do Usuário
	}

	for (size_t index_samples = 0; index_samples < nsamples; index_samples += batchSize)
	{
		if (nsamples - index_samples < batchSize)
		{
			batchSize = nsamples - index_samples;
			io.setbatchSize(batchSize);
		}

		if (params_.database == MNIST) {
			this->io.readDataMNIST(); //lendo dados MNIST
		} else if (params_.database == FASHION) {
			this->io.readDataFASHION(); //lendo dados FASHION_MNIST
		} else if (params_.database == CIFAR) {
			this->io.readDataCIFAR(); //lendo dados CIFAR
		}


		memcpy(&err[index_samples], this->io.labels.ptr< float >(0), batchSize * sizeof(float));//copiando labels


		printf("\n%zu imagens produzidas\n%zu imagens faltando\nbatchSize: %zu\n\n", index_samples, nsamples - index_samples, batchSize);

		for (int index_batch = 0; index_batch < batchSize; index_batch++)
		{
			float* samples_vector = this->io.samples[index_batch].reshape(0, 1).ptr< float >(0);

			// Conv & Pooling	
			for (int id = 0; id < params_.samples_inchl; id++) {
				memcpy(&featuresMaps[0][id*params_.samples_rows * params_.samples_cols], &samples_vector[id*params_.samples_rows*params_.samples_cols], sizeof(float)*params_.samples_rows*params_.samples_cols);
			}

			int id_conf = 0;
			char* layer = io.conf_net[id_conf];
	
			while (strstr(io.conf_net_in[id_conf][0], "flatten") == NULL)
			{
				if (strstr(layer, "conv") || strstr(layer, "zero"))
				{
					int inputLayer = 0;
					for (int j = 0; j < io.conf_net.size(); j++)
					{
						if (strstr(io.conf_net_in[id_conf][0], io.conf_net[j]))
						{
							inputLayer = j+1;
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

					convAndPoolingInception(featuresMaps[inputLayer], featuresSizesX[id_conf], featuresSizesY[id_conf], &id_conf);
				}
				else if (strstr(layer, "pool"))
				{
					int inputLayer = 0;
					for (int j = 0; j < io.conf_net.size(); j++)
					{
						if (strstr(io.conf_net_in[id_conf][0], io.conf_net[j]))
						{
							inputLayer = j+1;
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
					convAndPoolingInception(featuresMaps[inputLayer],featuresSizesX[id_conf], featuresSizesY[id_conf], &id_conf);
				}
				else if (strstr(layer, "concatenate"))
				{
					//printf("CONCAT: %d\n\n", id_conf);

					long long unsigned int idconcat = 0;

					for (int i = 0; i < io.conf_net_in[id_conf].size(); i++)
					{
						for (int j = 0; j < io.conf_net.size(); j++)
						{

							if (strstr(io.conf_net_in[id_conf][i], io.conf_net[j]))
							{
								short featuresX = featuresSizesX[j+1];
								short featuresY = featuresSizesY[j+1];
								short featuresD = featuresDeep[j+1];

								//printf("featuresX: %d\nfeaturesY: %d\nfeaturesD: %d\n\n\n", featuresX, featuresY, featuresD);
								//getchar();
								memcpy(&featuresMaps[id_conf+1][idconcat], featuresMaps[j+1], sizeof(float) * (featuresD * featuresX * featuresY));

								idconcat = idconcat + (featuresD * featuresX * featuresY);
							}
						}
					}

					
				}
				else if (strstr(layer, "add"))
				{
					
					for (int i = 0; i < featuresDeep[id_conf + 1] * featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]; i++)
						featuresMaps[id_conf + 1][i] = 0.0;
					//printf("ADD: %d\n\n", id_conf);
					//getchar();
					long long unsigned int idconcat = 0;

					for (int i = 0; i < io.conf_net_in[id_conf].size(); i++)
					{
						for (int j = 0; j < io.conf_net.size(); j++)
						{

							if (strstr(io.conf_net_in[id_conf][i], io.conf_net[j]))
							{
								short featuresX = featuresSizesX[j + 1];
								short featuresY = featuresSizesY[j + 1];
								short featuresD = featuresDeep[j + 1];

								//printf("featuresX: %d\nfeaturesY: %d\nfeaturesD: %d\nID:%d\n%s\n", featuresX, featuresY, featuresD, j + 1, io.conf_net[j]);
								//getchar();

								for (int k = 0; k < featuresX*featuresY*featuresD; k++)
									featuresMaps[id_conf + 1][k] += featuresMaps[j + 1][k];
							}
						}
					}

					/*
					for (int kerA = 0; kerA < featuresDeep[id_conf + 1]; kerA++)
					{
						for (int i = 0; i < featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]; i++)
						{
							printf("%3.4lf ", featuresMaps[id_conf + 1][kerA*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] + i]);
							if (((i + 1) % (featuresSizesY[id_conf + 1])) == 0)
								printf("\n");
						}
						printf("\n\n");
						//getchar();
					}
					getchar();
					*/
				}
				layer = io.conf_net[++id_conf];
			}

			/*
			id_conf--;
			printf("id_conf: %d\n\n",id_conf);

			for (int kerA = 0; kerA < featuresDeep[id_conf]; kerA++)
			{
				for (int i = 0; i < featuresSizesX[id_conf] * featuresSizesY[id_conf]; i++)
				{
					printf("%3.4lf ", featuresMaps[id_conf][kerA*featuresSizesX[id_conf] * featuresSizesY[id_conf] + i]);
					if (((i + 1) % (featuresSizesY[id_conf])) == 0)
						printf("\n");
				}
				printf("\n\n");
				//getchar();
			}
			getchar();*/

			//printf("--> %d", featuresMaps.size() - 2 - params_.NumHiddenLayers - 1);
			//getchar();
			
			//printf("\n\n\nend convAndPooling\n\n");
			//getchar();
			

			for (int hl = 0; hl < params_.NumHiddenLayers; hl++)
			{
				int hidden_rows = this->HiddenFeaturesX[hl];
				int hidden_cols = this->HiddenFeaturesY[hl];

				float* tmpacti = featuresMaps[featuresMaps.size() - 1 - params_.NumHiddenLayers - 1 + hl + 1];

				MatrixMult(this->HiddenLayers[hl].W, featuresMaps[featuresMaps.size() - 2 - params_.NumHiddenLayers - 1 + hl], tmpacti, hidden_rows, hidden_cols);

				for (int i = 0; i < hidden_rows; i++)
					tmpacti[i] += this->HiddenLayers[hl].b[i];

				nonLinearity(tmpacti, hidden_rows);


			}
			//printf("end FullyConected\n\n");
			//getchar();


			float* M = featuresMaps[featuresMaps.size() - 1];


			MatrixMult(this->smr->W, featuresMaps[featuresMaps.size() - 1 - 1], M, this->params_.nclasses, this->params_.NumHiddenNeurons);

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] += this->smr->b[i];


			float maximum = -INFINITY;
			for (int i = 0; i < this->params_.nclasses; i++)
				if (M[i] > maximum)
					maximum = M[i];

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] -= maximum;

			for (int i = 0; i < this->params_.nclasses; i++)
				M[i] = (float)exp((double)M[i]);


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
			//printf("end SMR %d\n\n", output);
			//getchar();

			result[index_samples + index_batch] = output;

			for (int i = 0; i < this->params_.nclasses; i++)
				fprintf(preds, "%.4lf ", M[i]);
			fprintf(preds, "\n");
		}
	}

	int correct = params_.number_of_images;
	for (int i = 0; i<params_.number_of_images; i++) {
		err[i] -= (float)result[i];
		if (err[i] != 0) --correct;
	}
	printf("correct: %d, total: %d, accuracy: %.2lf\n", correct, params_.number_of_images, float(correct) / (float)(params_.number_of_images));
	//cout << "correct: " << correct << ", total: " << params_.number_of_images << ", accuracy: " << float(correct) / (float)(params_.number_of_images) << endl;

	fclose(preds);
	return result;
}

void Net::sigmoid(float* M, int length){

	for (int i = 0; i < length; i++)
		M[i] = (float)(1.0 / (1.0 + M[i]));
}

void Net::ReLU(float* M, int length){

	for (int i = 0; i<length; i++)
		if (M[i] < 0.0)
			M[i] = 0.0;
}

void Net::Tanh(float* M, int length){

	for (int i = 0; i<length; i++)
		M[i] = tanh(M[i]);
}

void Net::nonLinearity(float* M, int length){

	if (params_.nonlin == NL_RELU){
		ReLU(M, length);
	}
	else if (params_.nonlin == NL_TANH){
		Tanh(M, length);
	}
	else{
		sigmoid(M, length);
	}
}

void Net::convolution(float* img, float* kernel, float *tmpOutConv, int rows, int cols, int kernel_rows, int kernel_cols, int strideX, int strideY) {

	int i, j, l, k;
	float sum;

	/*
	for (int i = 0; i < kernel_rows*kernel_cols; i++)
	{
	printf("%3.4lf ", kernel[i]);
	if ((i + 1) % kernel_cols == 0)
	printf("\n");
	}
	printf("\n\n");


	for (int i = 0; i < rows*cols; i++)
	{
	printf("%3.4lf ", img[i]);
	if ((i + 1) % cols == 0)
	printf("\n");
	}
	printf("\n\n");
	*/



	//printf("\n\n(%d %d) (%d %d) (%d %d)\n\n", rows, cols, kernel_rows, kernel_cols, strideX, strideY);
	//getchar();


	int idxOut = 0;
	for (i = 0; i < rows - kernel_rows + 1; i = i + strideX)
	{
		for (j = 0; j < cols - kernel_cols + 1; j = j + strideY)
		{
			sum = 0.0;
			for (l = 0; l < kernel_rows; l++)
			{
				const int idxFtmp = l * kernel_cols;
				const int idxIntmp = (i + l)*cols + j;

				for (k = 0; k < kernel_cols; k++)
				{
					//printf("(%d %d)\n", (idxFtmp + k), idxIntmp + k);
					//getchar();
					sum += (float)kernel[(idxFtmp + k)] * img[idxIntmp + k];//THEANO ( kernel_rows*kernel_cols - 1 - ) 
				}
			}
			//const int idxOut = floor((float)(i * ceil((float)(cols - kernel_cols + 1) / strideX) + j) / strideY) + ceil((float)i / strideY);
			//printf("\n(%d %d) -> %d = %f",i,j, idxOut, sum);
			//getchar();
			tmpOutConv[idxOut++] = sum;
		}
	}
}

void Net::Pooling(float* img, float* tmpOutPool, int poolingSizeX, int poolingSizeY, int strideX, int strideY, int rows, int cols){

	float value_max;
	int i, j, l, k;

	int out_pool = 0;
	for (i = 0; i < rows - poolingSizeX + 1; i = i + strideX)
	{
		for (j = 0; j < cols - poolingSizeY + 1; j = j + strideY)
		{
			value_max = 0.0;
			for (l = 0; l < poolingSizeX; l++)
			{
				const int idxIntmp = (i + l)*cols + j;

				for (k = 0; k < poolingSizeY; k++)
					if (img[idxIntmp + k] > value_max)
						value_max = img[idxIntmp + k];
			}
			//tmpOutPool[(i * cols / poolingSizeX + j) / poolingSizeY] = value_max;
			tmpOutPool[out_pool++] = value_max;
		}
	}
}

float* Net::padding(float* &input, int rows, int cols, int pad_top, int pad_bottom, int pad_left, int pad_rigth)
{
	float *newImg = (float*)malloc(sizeof(float) * (rows + pad_top + pad_bottom) * (cols + pad_left + pad_rigth));
	if (newImg == NULL)
	{
		printf("Error allocation memory. tmpOutPool");
		getchar();
		exit(-1);
	}

	//printf("\n\n(%d %d) - (%d %d)\n\n", rows, cols, (rows + pad_top + pad_bottom), (cols + pad_left + pad_rigth));
	//getchar();


	for (int i = 0; i < (rows + pad_top + pad_bottom) * (cols + pad_left + pad_rigth); i++)
		newImg[i] = 0.0;

	int j;
	if (pad_top != 0)
		j = pad_left + pad_top*(cols + pad_left + pad_rigth);
	else
		j = pad_left;

	
	for (int i = 0; i < rows*cols; i++, j++)
	{
		newImg[j] = input[i];

		if ((i + 1) % cols == 0)
			j = j + pad_left + pad_rigth;
	}


	return newImg;
}

void Net::convAndPooling(float* sample, float* &convOut, int rows, int cols)
{
	int id = 0;
	while (strstr(io.conf_net[id], "flatten") == NULL)
	{
		for (int i = 0; i < featuresDeep[id] * featuresSizesX[id] * featuresSizesY[id]; i++)
			featuresMaps[id][i] = 0.0;
		id++;
	}

	for (int id = 0; id < params_.samples_inchl; id++){

		float* tmpsample = (float*)malloc(sizeof(float) * rows * cols);
		if (tmpsample == NULL)
		{
			printf("Error allocation memory. tmpsample");
			getchar();
			exit(-1);
		}


		//memcpy(tmpsample, &sample[0 + id* rows * cols], sizeof(float) * rows * cols);
		memcpy(&featuresMaps[0][id* rows * cols], &sample[id* rows * cols], sizeof(float) * rows * cols);
		//vec.push_back(tmpsample);
	}

	/*
	Mat ch1 = Mat::zeros(32, 32, CV_32FC1);
	Mat ch2 = Mat::zeros(32, 32, CV_32FC1);
	Mat ch3 = Mat::zeros(32, 32, CV_32FC1);
	vector<Mat> channels;

	for (int r = 0; r < 32 * 32; ++r){
	ch1.at<float>(r) = vec[0][r];
	ch2.at<float>(r) = vec[1][r];
	ch3.at<float>(r) = vec[2][r];
	}

	channels.push_back(ch1);
	channels.push_back(ch2);
	channels.push_back(ch3);

	Mat fin_img = Mat::zeros(32, 32, CV_32FC3);
	merge(channels, fin_img);


	imshow("Teste", fin_img);
	waitKey(0);
	*/


	int id_conf = 0;
	char* layer = io.conf_net[id_conf];
	
	while (strstr(layer, "flatten") == NULL)
	{
		int pad_top		= PadTop[id_conf];
		int pad_bottom	= PadBottom[id_conf];
		int pad_left	= PadLeft[id_conf];
		int pad_rigth	= PadRigth[id_conf];

		short featuresX = featuresSizesX[id_conf + 1];
		short featuresY = featuresSizesY[id_conf + 1];

		int kernel_rows		= this->KernelSize[id_conf] >> 8;
		int kernel_cols		= this->KernelSize[id_conf] & 0xFF;
		int conv_strideX	= this->ConvStride[id_conf] >> 8;
		int conv_strideY	= this->ConvStride[id_conf] & 0xFF;

		int pdimX		= this->PoolingDim[id_conf] >> 8;
		int pdimY		= this->PoolingDim[id_conf] & 0xFF;
		int pool_strideX = this->PoolStride[id_conf] >> 8;
		int pool_strideY = this->PoolStride[id_conf] & 0xFF;

		if (strstr(layer, "conv"))
		{
			bool check_valid = strstr(this->ConvLayers[id_conf]->padding, "valid");

			float *tmpOutConv = (float*)malloc(sizeof(float) * featuresX * featuresY);
			if (tmpOutConv == NULL)
			{
				printf("Error allocation memory. tmpOutConv (%d %d)", featuresX, featuresY);
				getchar();
				exit(-1);
			}

			for (int kerA = 0; kerA < KernelAmount[id_conf]; kerA++)
			{
				for (int kerD = 0; kerD < KernelDepth[id_conf]; kerD++)
				{
						
					if (check_valid)
					{
						convolution(&featuresMaps[id_conf][kerD*featuresSizesX[id_conf]*featuresSizesY[id_conf]], //Input
									this->ConvLayers[id_conf]->layer[kerA]->W[kerD], //Kernel
									tmpOutConv, //Output
									rows, cols, //Input Dimension
									kernel_rows, kernel_cols, //Kernel Dimension
									conv_strideX, conv_strideY); //Kernel Stride
					}
					else
					{
						float *inp = &featuresMaps[id_conf][kerD*featuresSizesX[id_conf] * featuresSizesY[id_conf]];
						float* pad = padding(inp, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);
						
						convolution(pad, //Input
									this->ConvLayers[id_conf]->layer[kerA]->W[kerD], //Kernel
									tmpOutConv, //Output
									rows + pad_top + pad_bottom, cols + pad_left + pad_rigth, //Input Dimension
									kernel_rows, kernel_cols, //Kernel Dimension
									conv_strideX, conv_strideY); //Kernel Stride
						
						/*
						printf("\n\n");
						for (int i = 0; i < outConvX * outConvY; i++)
						{
						printf("%3.4lf ", tmpOutConv[i]);
						if (((i + 1) % (outConvY)) == 0)
						printf("\n");
						}
						getchar();*/

						if (pad != NULL)
							delete(pad);
					}

					/*
					for (int i = 0; i < outConvX * outConvY; i++)
					{
					printf("%3.4lf ", tmpOutConv[i]);
					if (((i + 1) % (outConvY)) == 0)
					printf("\n");
					}
					printf("\n\n");
					getchar();*/

					for (int i = 0; i < featuresX * featuresY; i++)
						featuresMaps[id_conf + 1][kerA*featuresSizesX[id_conf + 1]*featuresSizesY[id_conf + 1] + i] += tmpOutConv[i];
				}

				for (int i = 0; i < featuresX * featuresY; i++)
					featuresMaps[id_conf + 1][kerA*featuresSizesX[id_conf + 1]*featuresSizesY[id_conf + 1] + i] += this->ConvLayers[id_conf]->layer[kerA]->b;

				/*
				for (int i = 0; i < featuresX * featuresY; i++)
				{
				printf("%3.4lf ", tmpConvAm[i]);
				if (((i + 1) % (featuresY)) == 0)
				printf("\n");
				}
				printf("\n\n");
				getchar();*/
				//tmpvec.push_back(tmpConvAm);
			}

			nonLinearity(featuresMaps[id_conf + 1], featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]* featuresDeep[id_conf + 1]);

			free(tmpOutConv);

			rows = featuresX;
			cols = featuresY;

			//printf("Saiu CONV (%d %d)\n\n", rows, cols);
			//getchar();
		}
		else if (strstr(layer, "pool")){

			//printf("Entrou POOL\n");

			bool check_padding = (pad_top != 0 || pad_bottom != 0 || pad_left != 0 || pad_rigth != 0);


			for (int tp = 0; tp < featuresDeep[id_conf]; tp++)
			{
				if (check_padding)
				{
					float *inp = &featuresMaps[id_conf][tp*featuresSizesX[id_conf] * featuresSizesY[id_conf]];
					float* pad = padding(inp, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);
					
					Pooling(pad, //input
							&featuresMaps[id_conf + 1][tp*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]], //output
							pdimX, pdimY, //pooling dimension
							pool_strideX, pool_strideY, //pooling stride
							rows + pad_top + pad_bottom, cols + pad_left + pad_rigth); //input dimension

					if (pad != NULL)
						delete(pad);
				}
				else {


					Pooling(&featuresMaps[id_conf][tp*featuresSizesX[id_conf] * featuresSizesY[id_conf]], //input
							&featuresMaps[id_conf + 1][tp*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]], //output
							pdimX, pdimY, //pooling dimension
							pool_strideX, pool_strideY, //pooling stride
							rows + pad_top + pad_bottom, cols + pad_left + pad_rigth); //input dimension
					
					/*
					for (int i = 0; i < (featuresX * featuresY); i++)
					{
						printf("%3.4lf ", tmpOutPool[i]);
						if (((i + 1) % featuresY) == 0)
						printf("\n");
					}
					printf("\n\n");
					getchar();
					*/
				}
			}

			rows = featuresX;
			cols = featuresY;
			//printf("Saiu POOL (%d %d)\n\n", rows, cols);
			//getchar();
		}
		else if (strstr(layer, "zero")){//CORRIGIR SAIDA DO PADDING

			for (int tp = 0; tp < featuresDeep[id_conf]; tp++)
			{
				float *inp = &featuresMaps[id_conf][tp*featuresSizesX[id_conf] * featuresSizesY[id_conf]];
				float* pad = padding(inp, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);// <<<-- AJUSTAR SAIDA DO RESULTADO NA MEMORIA
			}

			rows = featuresX;
			cols = featuresY;

			//printf("Saiu ZERO (%d %d)\n\n", rows, cols);
			//getchar();
		}

		layer = io.conf_net[++id_conf];
	}

}

void Net::convAndPoolingInception(float* &sample, int rows, int cols, int* layer)
{

	int id = *layer;
	while (strstr(io.conf_net[id], "flatten") == NULL)
	{
		//printf("Zerar Saida: %d - (%d x %d x %d)\n", id + 1, featuresDeep[id+1], featuresSizesX[id + 1], featuresSizesY[id + 1]);
		for (int i = 0; i < featuresDeep[id + 1] * featuresSizesX[id + 1] * featuresSizesY[id + 1]; i++)
			featuresMaps[id + 1][i] = 0.0;
		id++;
	}
	
	
	
	/*
	printf("Copiar Entrada: %d - (%d x %d x %d)\n", id_conf, featuresDeep[id_conf], featuresSizesX[id_conf], featuresSizesY[id_conf]);
	for (int id = 0; id < params_.samples_inchl; id++) {
		memcpy(featuresMaps[id_conf], sample, sizeof(float) * featuresSizesX[id_conf] * featuresSizesY[id_conf]* featuresDeep[id_conf]);
	}
	printf("Copiou Entrada\n");
	getchar();
	*/
	
	/*
	printf("CONV IN %d: (%d x %d x %d)\n\n", id_conf, featuresSizesX[id_conf], featuresSizesY[id_conf], featuresDeep[id_conf]);
	getchar();
	for (int kerA = 0; kerA < featuresDeep[id_conf]; kerA++)
	{
		for (int i = 0; i < featuresSizesX[id_conf] * featuresSizesY[id_conf]; i++)
		{
			printf("%3.4lf ", featuresMaps[id_conf][kerA*featuresSizesX[id_conf] * featuresSizesY[id_conf] + i]);
			if (((i + 1) % (featuresSizesY[id_conf])) == 0)
				printf("\n");
		}
		printf("\n\n");
		//getchar();
	}
	printf("\n\n");
	getchar();
	*/


	int id_conf = *layer;
	char* lay = io.conf_net[id_conf];

	while (strstr(lay, "flatten") == NULL)
	{
		int pad_top		= PadTop[id_conf];
		int pad_bottom	= PadBottom[id_conf];
		int pad_left	= PadLeft[id_conf];
		int pad_rigth	= PadRigth[id_conf];

		int featuresX = featuresSizesX[id_conf + 1];
		int featuresY = featuresSizesY[id_conf + 1];

		int featuresD = featuresDeep[id_conf];

		int kernel_rows		= this->KernelSize[id_conf] >> 8;
		int kernel_cols		= this->KernelSize[id_conf] & 0xFF;
		int conv_strideX	= this->ConvStride[id_conf] >> 8;
		int conv_strideY	= this->ConvStride[id_conf] & 0xFF;

		int pdimX			= this->PoolingDim[id_conf] >> 8;
		int pdimY			= this->PoolingDim[id_conf] & 0xFF;
		int pool_strideX	= this->PoolStride[id_conf] >> 8;
		int pool_strideY	= this->PoolStride[id_conf] & 0xFF;


		int ker_amount	= KernelAmount[id_conf];
		int ker_deep	= KernelDepth[id_conf];


		if (strstr(lay, "conv"))
		{
			//printf("Entrou CONV\n");

			bool check_valid = strstr(this->ConvLayers[id_conf]->padding, "valid");

			float *tmpOutConv = (float*)malloc(sizeof(float) * featuresX * featuresY);
			if (tmpOutConv == NULL)
			{
				printf("Error allocation memory. tmpOutConv (%d %d)", featuresX, featuresY);
				getchar();
				exit(-1);
			}


			for (int kerA = 0; kerA < ker_amount; kerA++)
			{
				for (int kerD = 0; kerD < ker_deep; kerD++)
				{
					if (check_valid)
						convolution(&featuresMaps[id_conf][kerD*featuresSizesX[id_conf] * featuresSizesY[id_conf]], //Input
									this->ConvLayers[id_conf]->layer[kerA]->W[kerD], //Kernel
									tmpOutConv, //Output
									rows, cols, //Input Dimension
									kernel_rows, kernel_cols, //Kernel Dimension
									conv_strideX, conv_strideY); //Kernel Stride
					else
					{
						/*
						for (int i = 0; i < rows*cols; i++)
						{
						printf("%3.4lf ", vec[tp + kerD][i]);
						if ((i + 1) % cols == 0)
						printf("\n");
						}
						printf("\n\n");*/

						float *inp = &featuresMaps[id_conf][kerD*featuresSizesX[id_conf] * featuresSizesY[id_conf]];
						float* pad = padding(inp, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);

						convolution(pad, //Input
									this->ConvLayers[id_conf]->layer[kerA]->W[kerD], //Kernel
									tmpOutConv, //Output
									rows + pad_top + pad_bottom, cols + pad_left + pad_rigth, //Input Dimension
									kernel_rows, kernel_cols, //Kernel Dimension
									conv_strideX, conv_strideY); //Kernel Stride
						
						/*
						printf("\n\n");
						for (int i = 0; i < outConvX * outConvY; i++)
						{
						printf("%3.4lf ", tmpOutConv[i]);
						if (((i + 1) % (outConvY)) == 0)
						printf("\n");
						}
						getchar();*/

						if (pad != NULL)
							delete(pad);
					}

					/*
					for (int i = 0; i < outConvX * outConvY; i++)
					{
					printf("%3.4lf ", tmpOutConv[i]);
					if (((i + 1) % (outConvY)) == 0)
					printf("\n");
					}
					printf("\n\n");
					getchar();*/

					for (int i = 0; i < featuresX * featuresY; i++)
						featuresMaps[id_conf + 1][kerA*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] + i] += tmpOutConv[i];
						//tmpConvAm[i] += tmpOutConv[i];
				}

				for (int i = 0; i < featuresX * featuresY; i++)
					featuresMaps[id_conf + 1][kerA*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] + i] += this->ConvLayers[id_conf]->layer[kerA]->b;
					//tmpConvAm[i] += this->ConvLayers[id_conf]->layer[kerA]->b;

				

			
				//tmpvec.push_back(tmpConvAm);
			}
			
			nonLinearity(featuresMaps[id_conf + 1], featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] * featuresDeep[id_conf + 1]);
			//nonLinearity(tmpConvAm, featuresX * featuresY);

			//printf("Layer: %d (%d x %d x %d)\n\n", id_conf + 1, featuresDeep[id_conf + 1], featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1]);
			
			/*
			printf("CONV %d: (%d x %d x %d)\n\n", id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			getchar();
			for (int kerA = 0; kerA < featuresDeep[id_conf + 1]; kerA++)
			{
				for (int i = 0; i < featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]; i++)
				{
					printf("%3.4lf ", featuresMaps[id_conf + 1][kerA*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] + i]);
					if (((i + 1) % (featuresSizesY[id_conf + 1])) == 0)
						printf("\n");
				}
				printf("\n\n");
				//getchar();
			}
			printf("\n\n");
			getchar();
			*/

			free(tmpOutConv);

			rows = featuresX;
			cols = featuresY;

			//printf("Saiu CONV (%d %d)\n\n", rows, cols);
			//getchar();
		}
		else if (strstr(lay, "pool")) {

			
			//printf("Entrou POOL\n");
		
			bool check_padding = (pad_top != 0 || pad_bottom != 0 || pad_left != 0 || pad_rigth != 0);


			for (int tp = 0; tp < featuresD; tp++)
			{
				if (check_padding)
				{
					float *inp = &featuresMaps[id_conf][tp*featuresSizesX[id_conf] * featuresSizesY[id_conf]];
					float* pad = padding(inp, rows, cols, pad_top, pad_bottom, pad_left, pad_rigth);

					

					Pooling(pad, //input
							&featuresMaps[id_conf + 1][tp*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]], //output
							pdimX, pdimY, //pooling dimension
							pool_strideX, pool_strideY, //pooling stride
							rows + pad_top + pad_bottom, cols + pad_left + pad_rigth); //input dimension

					//swap(pad, inp);
					if (pad != NULL)
						delete(pad);
				}
				else{

					/*
					float *tmpOutPool = (float*)malloc(sizeof(float) * featuresX * featuresY);
					if (tmpOutPool == NULL)
					{
						printf_s("Error allocation memory. tmpOutPool");
						getchar();
						exit(-1);
					}*/


					//Pooling(vec[tp], tmpOutPool, pdimX, pdimY, pool_strideX, pool_strideY, rows + pad_top + pad_bottom, cols + pad_left + pad_rigth);
					Pooling(&featuresMaps[id_conf][tp*featuresSizesX[id_conf] * featuresSizesY[id_conf]], //input
							&featuresMaps[id_conf + 1][tp*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]], //output
							pdimX, pdimY, //pooling dimension
							pool_strideX, pool_strideY, //pooling stride
							rows + pad_top + pad_bottom, cols + pad_left + pad_rigth); //input dimension

					/*
					for (int i = 0; i < (featuresX * featuresY); i++)
					{
					printf("%3.4lf ", tmpOutPool[i]);
					if (((i + 1) % featuresY) == 0)
					printf("\n");
					}
					printf("\n\n");
					getchar();
					*/

					//tmpvec.push_back(tmpOutPool);
				}

				
			}

			/*
			printf("POOL %d: (%d x %d x %d)\n\n", id_conf + 1, featuresSizesX[id_conf + 1], featuresSizesY[id_conf + 1], featuresDeep[id_conf + 1]);
			getchar();
			for (int kerA = 0; kerA < featuresDeep[id_conf + 1]; kerA++)
			{
				for (int i = 0; i < featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1]; i++)
				{
					printf("%3.4lf ", featuresMaps[id_conf + 1][kerA*featuresSizesX[id_conf + 1] * featuresSizesY[id_conf + 1] + i]);
					if (((i + 1) % (featuresSizesY[id_conf + 1])) == 0)
						printf("\n");
				}
				printf("\n\n");
				//getchar();
			}

			getchar();*/

			rows = featuresX;
			cols = featuresY;
			//printf("Saiu POOL (%d %d)\n\n", rows, cols);
			//getchar();
			//pl = pl + 1;
		}
		else if (strstr(lay, "zero")) {//CORRIGIR SAIDA DO PADDING

			for (int tp = 0; tp < featuresDeep[id_conf]; tp++)
			{
				float *inp = &featuresMaps[id_conf][tp*featuresSizesX[id_conf] * featuresSizesY[id_conf]];
				float* pad = padding(inp, featuresSizesX[id_conf], featuresSizesY[id_conf], pad_top, pad_bottom, pad_left, pad_rigth);// <<<-- AJUSTAR SAIDA DO RESULTADO NA MEMORIA

				memcpy(&featuresMaps[id_conf+1][tp*featuresSizesX[id_conf+1]*featuresSizesY[id_conf+1]], pad, sizeof(float)*featuresSizesX[id_conf+1]*featuresSizesY[id_conf+1]);

				if (pad != NULL)
					delete(pad);
			}

			rows = featuresX;
			cols = featuresY;

			//printf("Saiu ZERO (%d %d)\n\n", rows, cols);
			//getchar();
		}


		lay = io.conf_net[++id_conf];
	}


	*layer = id_conf;//Retornando Id de saida
}

Net::~Net(){

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

	for (int i = 0; i < HiddenLayers.size(); i++)
	{
		if (HiddenLayers[i].W != NULL)
			_aligned_free(HiddenLayers[i].W);
		if (HiddenLayers[i].b != NULL)
			_aligned_free(HiddenLayers[i].b);
	}
	HiddenLayers.clear();

	if (smr->W != NULL)
		_aligned_free(smr->W);
}


