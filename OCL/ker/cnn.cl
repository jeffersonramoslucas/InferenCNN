 /****************************************************************************************
*                                                                                       *
*                          Projeto BioInfo - Lincs and CIn / UFPE                       *
*                                    09/09/2015                                         *
*																					    *
*****************************************************************************************
*****************************************************************************************
* Responsaveis: Jefferson Ramos L. dos Anjos										   	*
*               Joao Gabriel M. da Silva                 	                        	*
*               Antonyus Pyetro do A. Ferreira                                        	*
*****************************************************************************************/
        

#define MAX_KER_ROWS 7
#define MAX_KER_COLS 7
#define MAX_KER_AMOU 2
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void cnn_matrixMul_layer(
							__global float *in1,
							__global float *in2,
							__global float *restrict bias,
							__global float *outMul,							
							const int COLSROWS,
							const int COLS2
							)
{
	int r = get_global_id(0);

	float sum;
	
	for (int c = 0; c < COLS2; c++)
	{
		sum = 0.0;
		for (int cr = 0; cr < COLSROWS; cr++)
			sum += (float)in1[r * COLSROWS + cr] * in2[cr + c * COLSROWS];
		outMul[r * COLS2 + c] = (float)sum + bias[r];
	}
}

__kernel void
cnn_add_layer(	
				__global float *restrict imgA,
				__global float *restrict imgB,
				__global float *restrict Out,
				const short rows,
				const short cols,
				const short deph
				)
{
	size_t global_id	= get_global_id(0);
	size_t IDRow		= get_global_id(1);
	size_t IDCol		= get_global_id(2);

	if ((global_id >= deph) | (IDRow >= rows) | (IDCol >= cols))
		return; 

	size_t id = global_id*rows*cols + IDCol + IDRow*cols;

	Out[id] = imgA[id] + imgB[id];

	return;
}

__kernel void
cnn_conv_deep_layer(	
				__global float *restrict img,
				__global float *restrict conv_out,
				__constant float *restrict bias,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_amnt,
				const short ker_deph
				)
{
	size_t global_id	= get_global_id(0);
	size_t IDRow		= get_global_id(1);
	size_t IDCol		= get_global_id(2);

	if ((global_id >= ker_amnt) | (IDRow >= conv_out_rows) | (IDCol >= conv_out_cols))
		return; 

	size_t conv_out_rows_vs_conv_out_cols = conv_out_rows * conv_out_cols;

	size_t id_amount	= global_id * conv_out_rows_vs_conv_out_cols * ker_deph;
	size_t IDImgOut		= global_id * conv_out_rows_vs_conv_out_cols;
	size_t id			= IDCol + IDRow * conv_out_cols;

	float bias_private = (float)bias[global_id]/ker_deph;

	for(size_t i = 0; i < ker_deph; i++)
	{
		size_t id_deep = i*conv_out_rows_vs_conv_out_cols;
		
		conv_out[IDImgOut + id] +=  img[id_amount + id_deep + id] + bias_private;
		//barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	}

	return; 
}

__kernel void
cnn_conv_layer6(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__local float *restrict ker_local,
				//__local float *restrict img_local,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	int global_id	= get_global_id(0);
	int local_id	= get_local_id(0);

	int IDRow		= get_global_id(1);
	int IDCol		= get_global_id(2);

	if ((global_id >= ker_amnt*ker_deph) | (IDRow >= conv_out_rows) | (IDCol >= conv_out_cols))
		return; 

	//__private float ker_private[MAX_KER_ROWS * MAX_KER_COLS * MAX_KER_AMOU];

	size_t ker_rows_vs_ker_cols = ker_rows*ker_cols;
	size_t id_kernel_global = ker_rows_vs_ker_cols*global_id;
	size_t id_kernel_local	= ker_rows_vs_ker_cols*local_id;

	int kerD = global_id % ker_deph;
	size_t id_img = img_rows*img_cols*kerD + IDRow*strideX*img_cols + IDCol*strideY;

	size_t IDImgOut = global_id*conv_out_rows*conv_out_cols;

	int rowOffset	= IDRow * conv_out_cols;
	int id			= IDImgOut + IDCol + rowOffset;

	float sum = 0.0;

	
	
	for (int cr = 0; cr < ker_rows_vs_ker_cols; cr = cr + 1) {
		ker_local[id_kernel_local + cr] = ker[id_kernel_global + cr];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	
	
	
	/*
	for (int l = 0; l < ker_rows; l++)
	{
		size_t id_ker = id_kernel_local + l*ker_cols;
		size_t id_img = id_img_deep + (IDRow*strideX + l)*img_cols + IDCol*strideY;
		
		#pragma unroll
		for (int k = 0; k < ker_cols; k++)
			sum += ker_local[id_ker + k] * img[id_img + k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	*/
	
	//#pragma unroll 8
	for (int i = 0; i < ker_rows*ker_cols; i++)
	{
		sum += ker_local[id_kernel_local + i] * img[id_img + (i % ker_cols) + (i / ker_cols)*img_cols];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	conv_out[id] = sum;
	return;
}



__kernel void
cnn_conv_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				__local float *restrict ker_local,
				__local float *restrict img_local,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	size_t global_id	= get_global_id(0);
	size_t IDLocal		= get_local_id(0);
	size_t IDRow		= get_global_id(1);
	size_t IDCol		= get_global_id(2);

	if ((global_id >= ker_amnt) | (IDRow >= conv_out_rows) | (IDCol >= conv_out_cols))
		return;
	
	int ker_rows_vs_cols = ker_rows*ker_cols;
	int img_rows_vs_cols = img_rows*img_cols;

	//size_t IDGroup = get_group_id(0);
	//size_t TamGroup = get_num_groups(0);
	//size_t TamLocal = get_local_size(0);
	//printf("TamLocal: %d\n",TamLocal);

	size_t id_kernel_global = ker_rows_vs_cols*ker_deph*global_id;
		
	size_t IDImgRow = IDRow*strideX*img_cols;
	size_t IDImgCol = IDCol*strideY;
	size_t IDImgOut = global_id*conv_out_rows*conv_out_cols + IDCol + IDRow*conv_out_cols;

	float bias_private = bias[global_id] / ker_deph;

	//__private float image_local[MAX_KER_ROWS * MAX_KER_COLS];

	for (int kerD = 0; kerD < ker_deph; kerD++)
	{
		float sum = 0.0;

		size_t id_kernel = id_kernel_global + ker_rows_vs_cols*kerD;
		size_t id_image = IDImgRow + IDImgCol + img_rows_vs_cols*kerD;


		int ID_ker_local = ker_rows_vs_cols*IDLocal;
		for (int i = 0; i < ker_rows_vs_cols; i++) {
			ker_local[ID_ker_local + i] = ker[id_kernel + i];
			//image_local[i] = img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];
			img_local[ID_ker_local + i] = img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];
		}
		barrier(CLK_LOCAL_MEM_FENCE);


		for (int i = 0; i < ker_rows_vs_cols; i++, ID_ker_local++)
			sum += ker_local[ID_ker_local] * img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];// image_local[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		conv_out[IDImgOut] += sum +  bias_private;
	}

	return;
}

__kernel void
cnn_conv4_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				__local float *restrict tileInput,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	size_t global_id = get_global_id(0);

	if (global_id >= ker_amnt)
		return; 

	size_t id_kernel_global = ker_rows*ker_cols*ker_deph*global_id;

	size_t IDRow	= get_global_id(1);
	size_t IDCol	= get_global_id(2);
	size_t id		= IDCol + IDRow*conv_out_cols;

	size_t IDImgRow = IDRow*strideX*img_cols;
	size_t IDImgCol = IDCol*strideY;
	size_t IDImgOut = global_id*conv_out_rows*conv_out_cols;

	//barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
	float bias_private = bias[global_id] / ker_deph;

	int ker_rows_vs_cols = ker_rows*ker_cols;
	int img_rows_vs_cols = img_rows*img_cols;

	for (int kerD = 0; kerD < ker_deph; kerD++)
	{
		size_t id_kernel = id_kernel_global + ker_rows_vs_cols*kerD;
		size_t id_image = IDImgRow + IDImgCol + img_rows_vs_cols*kerD;

		float sum = 0.0;
		for (int l = 0; l < ker_rows; l++)
		{
			size_t id_img = id_image +  l*img_cols;
			
			for (int k = 0; k < ker_cols; k++)
				sum +=  ker[id_kernel++] * img[id_img++]; 
		}
		conv_out[IDImgOut + id] += sum +  bias_private;
	}

	return;
}

__kernel void
cnn_conv_layer5(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__local float *restrict ker_local,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{

	size_t global_id	= get_global_id (0);
	size_t local_id		= get_local_id(0);

	if (global_id >= ker_amnt*ker_deph)
		return;

	//printf("%d\n", local_id);

	long long unsigned int i = 0, j = 0, l = 0, k = 0;
	double sum = 0.0;

	size_t id_amount = global_id*conv_out_rows*conv_out_cols;
	size_t id_kernel_global = ker_rows*ker_cols*global_id;
	
	int kerD = global_id % ker_deph;

	size_t id_img_local = img_rows*img_cols*kerD;
	size_t idxOut = 0;
	
	for (int cr = 0; cr < ker_rows*ker_cols; cr = cr + 1)
		ker_local[local_id*ker_rows*ker_cols + cr] = ker[id_kernel_global + cr];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (i = 0; i < img_rows - ker_rows + 1; i = i + strideX)
	{		
		for (j = 0; j < img_cols - ker_cols + 1; j = j + strideY)
		{
			sum = 0;
			for (l = 0; l < ker_rows; l++)
			{
				size_t id_ker = local_id*ker_rows*ker_cols + l*ker_cols;
				size_t id_img = id_img_local + (i + l)*img_cols + j;

				for (k = 0; k < ker_cols; k++)
					sum += (double)ker_local[(id_ker + k)] * img[id_img + k];  //THEANO ( ker_rows*ker_cols - 1 - ) 
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			conv_out[id_amount + idxOut++] += (float) sum;
		}
	}


	return;
}

__kernel void
cnn_conv2_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{

	size_t global_id = get_global_id (0);

	long long unsigned int i = 0, j = 0, l = 0, k = 0;
	double sum = 0.0;

	size_t id_amount = global_id*conv_out_rows*conv_out_cols;
	size_t id_kernel_global = ker_rows*ker_cols*ker_deph*global_id;

	for (int kerD = 0; kerD < ker_deph; kerD++)
	{
		size_t id_img_local = img_rows*img_cols*kerD;
		size_t id_kernel_local = ker_rows*ker_cols*kerD;
		size_t idxOut = 0;

			
		for (i = 0; i < img_rows - ker_rows + 1; i = i + strideX)
		{
			for (j = 0; j < img_cols - ker_cols + 1; j = j + strideY)
			{
				sum = 0;
				for (l = 0; l < ker_rows; l++)
				{
					size_t id_ker = id_kernel_global + id_kernel_local + l*ker_cols;
					size_t id_img = id_img_local + (i + l)*img_cols + j;

					for (k = 0; k < ker_cols; k++)
						sum += (double) ker[(id_ker + k)] * img[id_img + k];  //THEANO ( ker_rows*ker_cols - 1 - ) 
				}
				//atomic_xchg (&conv_out[id_amount + idxOut], conv_out[id_amount + idxOut] + (float) ((double)sum + bias[global_id] / ker_deph));

				//idxOut++;

				//atom_add(&conv_out[id_amount + idxOut++], ((double)sum + bias[global_id] / ker_deph))
				conv_out[id_amount + idxOut++] += (float) ((double)sum + bias[global_id] / ker_deph);//sum;//
			}
		}
	}
	return;
}

__kernel void
cnn_conv_global_local_workitens_kernel_local_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				__local float *ker_local,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	size_t global_id = get_group_id (0);
	size_t local_id = get_local_id(0);
	int nlocal = get_local_size( 0 );

	size_t id_amount = global_id*conv_out_rows*conv_out_cols;

	size_t id_kernel_global = ker_rows*ker_cols*ker_deph*global_id;
	size_t id_kernel_local = ker_rows*ker_cols*local_id;

	size_t id_img_local = img_rows*img_cols*local_id;

	long long unsigned int i = 0, j = 0, l = 0, k = 0;
	float sum = 0.0;

	for (i = 0; i < conv_out_rows * conv_out_cols; i++)
		conv_out[id_amount + i] = 0.0;
	
	 
	for( int cr = 0; cr < ker_rows*ker_cols; cr = cr + 1 )              
		ker_local[id_kernel_local + cr] = ker[id_kernel_global + id_kernel_local + cr]; 
	barrier( CLK_LOCAL_MEM_FENCE );

	size_t idxOut = 0;
	for (i = 0; i < img_rows - ker_rows + 1; i = i + strideX)
	{
		for (j = 0; j < img_cols - ker_cols + 1; j = j + strideY)
		{
			sum = 0;
			for (l = 0; l < ker_rows; l++)
			{
				size_t id_ker = id_kernel_local + l*ker_cols;
				size_t id_img = id_img_local + (i + l)*img_cols + j;

				for (k = 0; k < ker_cols; k++)
					sum += (float) ker_local[(id_ker + k)] * img[id_img + k];  //THEANO ( ker_rows*ker_cols - 1 - )
			}
			conv_out[id_amount + idxOut++] +=  (sum + bias[global_id] / ker_deph);//sum;//
		}
	}
	

	
	return;
}

__kernel void
cnn_conv_kernel_local_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				__local float *colbuf,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	size_t global_id = get_global_id (0);
	
	size_t id_amount = global_id*conv_out_rows*conv_out_cols;
	size_t id_kernel_global = ker_rows*ker_cols*ker_deph*global_id;

	size_t idlocal = get_local_id( 0 );   
    size_t nlocal = get_local_size( 0 );
	//printf("nlocal: %d\n",nlocal);

	long long unsigned int i = 0, j = 0, l = 0, k = 0;
	float sum = 0.0;

	for (i = 0; i < conv_out_rows * conv_out_cols; i++)
		conv_out[id_amount + i] = 0.0;

	for (int kerD = 0; kerD < ker_deph; kerD++)
	{
		size_t idxOut = 0;
		size_t id_kernel_deph = ker_rows*ker_cols*kerD;
		size_t id_img_deph = img_rows*img_cols*kerD;

		
		for( int cr = idlocal; cr < ker_rows*ker_cols; cr = cr + nlocal )              
			colbuf[ cr ] = ker[id_kernel_global + id_kernel_deph + cr]; 
		barrier( CLK_LOCAL_MEM_FENCE ); 

		for (i = 0; i < img_rows - ker_rows + 1; i = i + strideX)
		{
			for (j = 0; j < img_cols - ker_cols + 1; j = j + strideY)
			{
				sum = 0;
				for (l = 0; l < ker_rows; l++)
				{
					size_t id_ker = l*ker_cols;
					size_t id_img = id_img_deph + (i + l)*img_cols + j;
		
					for (k = 0; k < ker_cols; k++)
						sum += (float) colbuf[(id_ker + k)] * img[id_img + k];  //THEANO ( ker_rows*ker_cols - 1 - )
					
				}
				conv_out[id_amount + idxOut++] +=  (sum + bias[global_id] / ker_deph);//sum;//
			}
		}
	}


	return;
}

__kernel void
cnn_conv_global_local_workitens_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	size_t global_id = get_group_id (0);
	size_t local_id = get_local_id(0);

	size_t id_amount = global_id*conv_out_rows*conv_out_cols;

	size_t id_kernel_global = ker_rows*ker_cols*ker_deph*global_id;
	size_t id_kernel_local = ker_rows*ker_cols*local_id;

	size_t id_img_local = img_rows*img_cols*local_id;

	long long unsigned int i = 0, j = 0, l = 0, k = 0;
	float sum = 0.0;

	for (i = 0; i < conv_out_rows * conv_out_cols; i++)
		conv_out[id_amount + i] = 0.0;

	size_t idxOut = 0;
	for (i = 0; i < img_rows - ker_rows + 1; i = i + strideX)
	{
		for (j = 0; j < img_cols - ker_cols + 1; j = j + strideY)
		{
			sum = 0;
			for (l = 0; l < ker_rows; l++)
			{
				size_t id_ker = id_kernel_global + id_kernel_local + l*ker_cols;
				size_t id_img = id_img_local + (i + l)*img_cols + j;

				for (k = 0; k < ker_cols; k++)
					sum += (float) ker[(id_ker + k)] * img[id_img + k];  //THEANO ( ker_rows*ker_cols - 1 - )
					
			}
			barrier( CLK_LOCAL_MEM_FENCE );

			conv_out[id_amount + idxOut++] +=  (sum + bias[global_id] / ker_deph);//sum;//
		}
	}
	

	
	return;
}

__kernel void
cnn_conv_canonic_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	long long unsigned int i = 0, j = 0, l = 0, k = 0;
	float sum = 0.0;

	/*
	for (int kerA = 0; kerA < ker_amnt; kerA++)
	{
		for (int kerD = 0; kerD < ker_deph; kerD++)
		{
			for (i = 0; i < ker_rows * ker_cols; i++)
			{
				printf("%lf ",ker[ker_rows*ker_cols*ker_deph*kerA + ker_rows*ker_cols*kerD + i]);
				if(!((i+1)%ker_cols))
					printf("\n");
			}
			printf("\n\n");
		}
	}*/


	//printf("\n\n(%d %d) x ( %d %d) = (%d %d)  - (%d %d %d %d)\n\n",img_rows,img_cols,ker_rows,ker_cols,conv_out_rows,conv_out_cols,ker_amnt,ker_deph,strideX,strideY);

	for (i = 0; i < ker_amnt* conv_out_rows * conv_out_cols; i++)
		conv_out[i] = 0.0;

	for (int kerA = 0; kerA < ker_amnt; kerA++)
	{
		for (int kerD = 0; kerD < ker_deph; kerD++)
		{
			size_t idxOut = 0;
			for (i = 0; i < img_rows - ker_rows + 1; i = i + strideX)
			{
				for (j = 0; j < img_cols - ker_cols + 1; j = j + strideY)
				{
					sum = 0;
					for (l = 0; l < ker_rows; l++)
					{
						size_t id_ker = ker_rows*ker_cols*ker_deph*kerA + ker_rows*ker_cols*kerD + l*ker_cols;
						size_t id_img = img_rows*img_cols*kerD + (i + l)*img_cols + j;

						for (k = 0; k < ker_cols; k++)
							sum += (float) ker[(id_ker + k)] * img[id_img + k];  //THEANO ( ker_rows*ker_cols - 1 - ) 
					}
					conv_out[conv_out_rows*conv_out_cols*kerA + idxOut++] +=  (sum + bias[kerA] / ker_deph);//sum;//
				}
			}
		}
	}

	/*
	for (int kerA = 0; kerA < ker_amnt; kerA++)
	{
		for (i = 0; i < conv_out_rows * conv_out_cols; i++)
		{
			printf("%lf ",conv_out[conv_out_rows*conv_out_cols*kerA  + i]);
			if(!((i+1)%conv_out_cols))
				printf("\n");
		}
		printf("\n\n");
	}*/


	return;
}


__kernel void 
cnn_padding_layer(
					__global float *restrict x,
					__global float *restrict z,
					const short img_rows,
					const short img_cols,
					const short img_depth,
					unsigned int id_out,
					const int pad_left,
					const int pad_rigth,
					const int pad_bottom
					)
{
	
	const int globalId = get_global_id(0);

	
	int featuresY = (img_cols + pad_left + pad_rigth);
	

	int i = (globalId / (img_rows*img_cols));

	id_out = (i + 1)*id_out + globalId + (globalId / img_cols)*(pad_left + pad_rigth) + featuresY*pad_bottom*i - i*pad_left;


	z[id_out] = x[globalId];

	return;
}



__kernel void 
cnn_relu_layer( __global float *restrict x )
{
	const int globalId = get_global_id(0);

	x[globalId] = (x[globalId] > 0 ? x[globalId] : 0);
	return;
}

__kernel void
cnn_tanh_layer(__global float *restrict x)
{
	const int globalId = get_global_id(0);

	x[globalId] = tanh(x[globalId]);
	return;
}

__kernel void
cnn_sigm_layer(__global float *restrict x)
{
	const int globalId = get_global_id(0);

	x[globalId] = (1.0f / (1 + exp(-x[globalId])));
	return;
}

__kernel void
cnn_elu_layer(__global float *restrict x)
{
	const int globalId = get_global_id(0);

	x[globalId] = (x[globalId] > 0 ? x[globalId] : exp(x[globalId]) - 1);
	return;
}

__kernel void
cnn_pool_layer(
				__global float *restrict pool_in,
				__global float *restrict pool_out,
				const short img_rows,
				const short img_cols,
				const short featuresX,
				const short featuresY,
				const short poolingSizeX,
				const short poolingSizeY,
				const short strideX,
				const short strideY,
				const short ker_amnt
)
{	
	size_t global_id = get_global_id (0);
	//size_t id_amount = global_id*featuresX*featuresY;

	//size_t id_img_global = img_rows*img_cols*global_id;

	float value_max;
	int i, j, l, k;
	
	short pool_out_rows = (img_rows - poolingSizeX + 1);
	short pool_out_cols = (img_cols - poolingSizeY + 1);


	//for (int kerA = 0; kerA < ker_amnt; kerA++)
	{
		int out_pool = 0;
		for (i = 0; i < pool_out_rows; i = i + strideX)
		{
			for (j = 0; j < pool_out_cols; j = j + strideY)
			{
				value_max = 0.0;
				for (l = 0; l < poolingSizeX; l++)
				{
					const int id_img = img_rows*img_cols*global_id + (i + l)*img_cols + j;
					for (k = 0; k < poolingSizeY; k++)
						if (pool_in[id_img + k] > value_max)
							value_max = pool_in[id_img + k];
				}
				pool_out[global_id*featuresX*featuresY + out_pool++] = value_max;
			}
		}
	}
	
	
	return;
}


__kernel void 
cnn_batch_norm_layer(
					__global float *restrict img,
					__global float *restrict ker,
					__global float *restrict out,
					const short img_rows,
					const short img_cols,
					const short img_depth
					)
{
	
	const int globalId = get_global_id(0);

	//printf("\n\n\n\n");
	
	for(int d = 0; d < img_depth; d++)
	{
		for(int i = 0; i< img_rows; i++)
		{
			for(int j = 0; j < img_cols; j++)
			{
				out[d*img_rows*img_cols + i*img_cols + j] = ((img[d*img_rows*img_cols + i*img_cols + j] - 
						ker[j + img_cols]) * ker[j]) / sqrt(ker[j + img_cols + img_cols]);
				//printf("%f ", out[d*img_rows*img_cols + i*img_cols + j] );
			}
		}
	}
	/*
	printf("img_rows: %d\n", img_rows);
	printf("img_cols: %d\n", img_cols);
	printf("img_depth: %d\n", img_depth);
	
	printf("\n\n\n\n");
	*/
	return;
}


/*
__kernel void
cnn_depth_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				//__global float *restrict bias,
				__local float *restrict ker_local,
				__local float *restrict img_local,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	size_t global_id	= get_global_id(0);
	size_t IDLocal		= get_local_id(0);
	size_t IDRow		= get_global_id(1);
	size_t IDCol		= get_global_id(2);

	if ((global_id >= ker_amnt) | (IDRow >= conv_out_rows) | (IDCol >= conv_out_cols))
		return;
	
	int ker_rows_vs_cols = ker_rows*ker_cols;
	int img_rows_vs_cols = img_rows*img_cols;

	//size_t IDGroup = get_group_id(0);
	//size_t TamGroup = get_num_groups(0);
	//size_t TamLocal = get_local_size(0);
	//printf("TamLocal: %d\n",TamLocal);

	size_t id_kernel_global = ker_rows_vs_cols*ker_deph*global_id;
		
	size_t IDImgRow = IDRow*strideX*img_cols;
	size_t IDImgCol = IDCol*strideY;
	size_t IDImgOut = global_id*conv_out_rows*conv_out_cols + IDCol + IDRow*conv_out_cols;

	//float bias_private = bias[global_id] / ker_deph;

	//__private float image_local[MAX_KER_ROWS * MAX_KER_COLS];

	for (int kerD = 0; kerD < ker_deph; kerD++)
	{
		float sum = 0.0;

		size_t id_kernel = id_kernel_global + ker_rows_vs_cols*kerD;
		size_t id_image = IDImgRow + IDImgCol + img_rows_vs_cols*kerD;


		int ID_ker_local = ker_rows_vs_cols*IDLocal;
		for (int i = 0; i < ker_rows_vs_cols; i++) {
			ker_local[ID_ker_local + i] = ker[id_kernel + i];
			//image_local[i] = img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];
			img_local[ID_ker_local + i] = img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		


		for (int i = 0; i < ker_rows_vs_cols; i++, ID_ker_local++)
			sum += ker_local[ID_ker_local] * img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];// image_local[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		conv_out[kerD*ker_amnt*conv_out_rows*conv_out_cols +  IDImgOut] = sum;// +  bias_private;
	}

	return;
}*/

__kernel void
cnn_depth_layer2(
                __global float *restrict img,
                __global float *restrict ker,
                __global float *restrict conv_out,
                const short img_rows,
                const short img_cols,
                const short conv_out_rows,
                const short conv_out_cols,
                const short ker_rows,
                const short ker_cols,
                const short ker_amnt,
                const short ker_deph,
                const short strideX,
                const short strideY
                )
{
    size_t global_id = get_global_id (0);
    
    long long unsigned int i = 0, j = 0, l = 0, k = 0;
    float sum = 0.0;
    
    size_t irows_vs_icols = img_rows*img_cols;
    size_t krows_vs_kcols = ker_rows*ker_cols;
    size_t id_amount = conv_out_rows*conv_out_cols;
    size_t id_kernel_global = krows_vs_kcols*ker_deph*global_id;
    
    //for (int kerA = 0; kerA < ker_amnt; kerA++)
    {
        for (int kerD = 0; kerD < ker_deph; kerD++)
        {
            size_t id_img_local = irows_vs_icols * kerD;
            size_t id_kernel_local = krows_vs_kcols * kerD;
            size_t id_depth = id_amount*(kerD*ker_amnt + global_id);
            size_t idxOut = 0;
            for (i = 0; i < img_rows - ker_rows + 1; i = i + strideX)
            {
                for (j = 0; j < img_cols - ker_cols + 1; j = j + strideY)
                {
                    sum = 0;
                    for (l = 0; l < ker_rows; l++)
                    {
                        size_t id_ker = id_kernel_global + id_kernel_local + l*ker_cols;
                        size_t id_img = id_img_local + (i + l)*img_cols + j;
                        
                        for (k = 0; k < ker_cols; k++)
                            sum += (float) ker[(id_ker + k)] * img[id_img + k];
                    }
                    conv_out[id_depth + idxOut++] = sum;
                }
            }
        }
    }
    
    
    return;
}

__kernel void
cnn_depth_layer(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY,
				const short ondepth
				)
{
	size_t global_id	= get_global_id(0);
	size_t IDLocal		= get_local_id(0);
	size_t IDRow		= get_global_id(1);
	size_t IDCol		= get_global_id(2);
	
	if ((global_id >= ker_amnt*ker_deph) | (IDRow >= conv_out_rows) | (IDCol >= conv_out_cols))
		return;
	
	int ker_rows_vs_cols = ker_rows*ker_cols;
	int img_rows_vs_cols = img_rows*img_cols;
	int out_rows_vs_cols = conv_out_rows*conv_out_cols;

	size_t id_kernel_global = ker_rows_vs_cols*ker_deph*global_id;
		
	size_t IDImgRow = IDRow*strideX*img_cols;
	size_t IDImgCol = IDCol*strideY;
	size_t IDImgOut = global_id*out_rows_vs_cols + IDCol + IDRow*conv_out_cols;


	for (int kerD = 0; kerD < ker_deph; kerD++)
	{
		float sum = 0.0;

		size_t id_kernel = id_kernel_global + ker_rows_vs_cols*kerD;
		size_t id_image = IDImgRow + IDImgCol + img_rows_vs_cols*kerD;

		int ID_ker_local = ker_rows_vs_cols*IDLocal;
		

		for (int i = 0; i < ker_rows_vs_cols; i++, ID_ker_local++)
			sum += ker[id_kernel + i] * img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];// image_local[i];
		
		if (ondepth){
			size_t id_depth = out_rows_vs_cols*(kerD*ker_amnt + global_id);
			conv_out[id_depth + IDImgOut] = sum;
		}
		else
			conv_out[IDImgOut] += sum;
	}

	return;
}

__kernel void
cnn_add_bias_layer(
                               	__global float *restrict imgA,
                               	__global float *restrict imgB,
                                const short rows,
                               	const short cols
                               	)
{
        size_t global_id = get_global_id(0);

        float bias = imgB[global_id];

        size_t irows_vs_icols = rows * cols;
        size_t id = global_id * irows_vs_icols;

        for(size_t i = 0; i < irows_vs_icols; i++)
        {
               	imgA[id + i] += bias;
        }
}

__kernel void
cnn_conv_teste(	
				__global float *restrict img,
				__global float *restrict ker,
				__global float *restrict conv_out,
				__global float *restrict bias,
				const short img_rows,
				const short img_cols,
				const short conv_out_rows,
				const short conv_out_cols,
				const short ker_rows,
				const short ker_cols,
				const short ker_amnt,
				const short ker_deph,
				const short strideX,
				const short strideY
				)
{
	size_t global_id	= get_global_id(0);
	size_t IDLocal		= get_local_id(0);
	size_t IDRow		= get_global_id(1);
	size_t IDCol		= get_global_id(2);
	
	if ((global_id >= ker_amnt*ker_deph) | (IDRow >= conv_out_rows) | (IDCol >= conv_out_cols))
		return;
	
	int ker_rows_vs_cols = ker_rows*ker_cols;
	int img_rows_vs_cols = img_rows*img_cols;


	size_t id_kernel_global = ker_rows_vs_cols*ker_deph*global_id;
		
	size_t IDImgRow = IDRow*strideX*img_cols;
	size_t IDImgCol = IDCol*strideY;
	size_t IDImgOut = global_id*conv_out_rows*conv_out_cols + IDCol + IDRow*conv_out_cols;

	float bias_private = bias[global_id] / ker_deph;


	for (int kerD = 0; kerD < ker_deph; kerD++)
	{
		float sum = 0.0;

		size_t id_kernel = id_kernel_global + ker_rows_vs_cols*kerD;
		size_t id_image = IDImgRow + IDImgCol + img_rows_vs_cols*kerD;


		int ID_ker_local = ker_rows_vs_cols*IDLocal;

		for (int i = 0; i < ker_rows_vs_cols; i++, ID_ker_local++)
			sum += ker[id_kernel + i] * img[id_image + (i % ker_cols) + (i / ker_cols)*img_cols];// image_local[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		conv_out[IDImgOut] += sum +  bias_private;
	}

	return;
}



__kernel 
void cnn_conv_teste2(
		__global float *restrict input, //bottom,
		__global float *restrict weights,//,
		__global float *restrict output, // top
		__global float *restrict bias,
		       
		const short in_h,
		const short in_w,
		const short out_h,
		const short out_w,
		const short K_h,
		const short K_w,
		const short out_c,
		const short in_c,
		const short S_h,
		const short S_w
 )
 {

     // Local ID index (offset within a block)
      int local_id_x = get_local_id(0);
      int local_id_y = get_local_id(1);
      int local_id_z = get_local_id(2);
     
      // Global ID index (offset within the NDRange)
      int global_id_x = get_global_id(0);
      int global_id_y = get_global_id(1);
      int global_id_z = get_global_id(2);
     
      unsigned weight_size_2d = K_h*K_w;
      unsigned weight_size_3d = weight_size_2d*in_c;
      unsigned ifm_size = in_h*in_w;
      unsigned ofm_size = out_h*out_w;
     
      unsigned ifm_idx = 0;
      unsigned row_idx = 0;
      unsigned cnt_ifm = 1;
      unsigned cnt_row = 1;
     
      float running_sum = 0.0f;
     
      for (int k = 0; k < weight_size_3d; ++k)
      {
          running_sum += weights[weight_size_3d*global_id_z +k] * input[ifm_size*ifm_idx + (S_h*local_id_y + row_idx)*in_w + (S_w*local_id_x + cnt_row-1)];
         
          if (cnt_ifm == weight_size_2d) {
              row_idx = 0;
              ifm_idx++;
              cnt_ifm = 1;
              cnt_row = 1;
              }
          else if (cnt_row == K_w) {
              row_idx++;
              cnt_row = 1;
              cnt_ifm++;
              }
          else {
              cnt_row++;
              cnt_ifm++;
              }
      }
    running_sum += bias[global_id_z];
     // Store result in output
    output[ofm_size*global_id_z + local_id_y*out_w + local_id_x] = running_sum;
  }