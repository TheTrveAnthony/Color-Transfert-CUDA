#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include "function.h"
#include "matrix.h"

#define TAB 500000		
#define C 3
 
 
using namespace std; //::cout
//using namespace std::endl;
//using namespace std::string;
using namespace cv;

//// our tranfor matrices

const float *r_l;
const float *l_a;
const float *a_l;
const float *l_r;


__global__ void rgb2alpha(unsigned char* ipt, float* opt, const float *rgb_l, const float *l_alpha, int width, int height, int step){

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height)){

		//Location of pixel 
		const int pix = yIndex * step + (3 * xIndex);

		/// Now to access the diferent values, we do:
		/// ipt[pix] = R, ipt[pix+1] = G, ipt[pix] = B
		/// So let's perform the transform now
		// LMS transform :

		float L, M, S;

		L = log10(ipt[pix] * rgb_l[0] + ipt[pix+1] * rgb_l[1] + ipt[pix+2] * rgb_l[2]);
		M = log10(ipt[pix] * rgb_l[3] + ipt[pix+1] * rgb_l[4] + ipt[pix+2] * rgb_l[5]);
		S = log10(ipt[pix] * rgb_l[6] + ipt[pix+1] * rgb_l[7] + ipt[pix+2] * rgb_l[8]);

		// l alpha beta transform :

		opt[pix] = L * l_alpha[0] + M * l_alpha[1] + S * l_alpha[2];
		opt[pix+1] = L * l_alpha[3] + M * l_alpha[4] + S * l_alpha[5];
		opt[pix+2] = L * l_alpha[6] + M * l_alpha[7] + S * l_alpha[8];
	}
}

__global__ void alpha2rgb(float* ipt, unsigned char* opt, const float *alpha_l, const float *l_rgb, int width, int height, int step){

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height)){

		//Location of pixel 
		const int pix = yIndex * step + (3 * xIndex);

		/// Same principle as in rgb2alpha
		// LMS transform :

		float L, M, S;

		L = pow(10, ipt[pix] * alpha_l[0] + ipt[pix+1] * alpha_l[1] + ipt[pix+2] * alpha_l[2]);
		M = pow(10, ipt[pix] * alpha_l[3] + ipt[pix+1] * alpha_l[4] + ipt[pix+2] * alpha_l[5]);
		S = pow(10, ipt[pix] * alpha_l[6] + ipt[pix+1] * alpha_l[7] + ipt[pix+2] * alpha_l[8]);

		// RGB transform :

		opt[pix] = static_cast<unsigned char>(L * l_rgb[0] + M * l_rgb[1] + S * l_rgb[2]);
		opt[pix+1] = static_cast<unsigned char>(L * l_rgb[3] + M * l_rgb[4] + S * l_rgb[5]);
		opt[pix+2] = static_cast<unsigned char>(L * l_rgb[6] + M * l_rgb[7] + S * l_rgb[8]);
	}
}

__global__ void make_up(float* ipt, float* opt, const float *mean_t, const float *mean_s, const float *std_t, const float *std_s, int width, int height, int step){

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height)){

		//Location of pixel 
		const int pix = yIndex * step + (3 * xIndex);

		/// Always the same principle

		opt[pix] = (std_s[0]/std_t[0]) * (ipt[pix] - mean_t[0]) + mean_s[0];
		opt[pix+1] = (std_s[1]/std_t[1]) * (ipt[pix+1] - mean_t[1]) + mean_s[1];
		opt[pix+2] = (std_s[2]/std_t[2]) * (ipt[pix+2] - mean_t[2]) + mean_s[2];
	}
}

__global__ void channel_mean(float* tabz, float* meanz, int width, int height, int step){

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// We gonna need this to get the results from each thread

	const int N = width * height;
	__shared__ float sum[TAB][C];

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height)){

		//Location of pixel 
		const int pix = yIndex * step + (3 * xIndex);

		//equivalent in the table
		const int idx = yIndex * width + xIndex;

		// We put the values into sum

		sum[idx][0] = tabz[pix];
		sum[idx][1] = tabz[pix+1];
		sum[idx][2] = tabz[pix+2];
		// threads syncroniation to make sure that they are all done

		__syncthreads();

		float s0 = 0;
		float s1 = 0;
		float s2 = 0;

		// We compute the mean within the 0 thread

		if ( 0 == threadIdx.x ){
			

			for (int i = 0; i < N ; i++){

				s0 += sum[i][0];
				s1 += sum[i][1];
				s2 += sum[i][2];

			}

		}

			// Here they are

			s0 /= N;
			s1 /= N;
			s2 /= N;

			meanz[0] = s0;
			meanz[1] = s1;
			meanz[2] = s2;
	}
}

__global__ void channel_std(float* tabz, float* stdz, float* meanz, int width, int height, int step){

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// We gonna need this to get the results from each thread

	const int N = width * height;

	__shared__ float sum[TAB][C];

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height)){

		//Location of pixel 
		const int pix = yIndex * step + (3 * xIndex);

		//equivalent in the table
		const int idx = yIndex * width + xIndex;

		// We put the values into sum

		sum[idx][0] = pow(tabz[pix] - meanz[0], 2);
		sum[idx][1] = pow(tabz[pix+1] - meanz[1], 2);
		sum[idx][2] = pow(tabz[pix+2] - meanz[2], 2);

		// threads syncroniation to make sure that they are all done

		__syncthreads();

		float s0 = 0;
		float s1 = 0;
		float s2 = 0;

		// We compute the mean within the 0 thread

		if ( 0 == threadIdx.x ){
			


			for (int i = 0; i < N ; i++){

				s0 += sum[i][0];
				s1 += sum[i][1];
				s2 += sum[i][2];

			}

		}

		// Here they are

		s0 /= N;
		s1 /= N;
		s2 /= N;

		stdz[0] = sqrt(s0);
		stdz[1] = sqrt(s1);
		stdz[2] = sqrt(s2);
	}

}

void transfert(string nom1, string nom2){

	/////////// First of all let's load the images we'll use

	Mat target = imread(nom1, 1);
	Mat source = imread(nom2, 1);


	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole images
	const dim3 grid_t((target.cols + block.x - 1)/block.x, (target.rows + block.y - 1)/block.y);
	const dim3 grid_s((source.cols + block.x - 1)/block.x, (source.rows + block.y - 1)/block.y);


	//////////////////////// Phase 1: RGB to l alpha beta Conversion /////////////////////

	// Calculate the size of each image

	const int size_target = target.step * target.rows;
	const int size_source = source.step * source.rows;			// Here step = cols * nb of channels = cols *3

	// We also gonna use floats, so:

	const int size_target_f = size_target * sizeof(float);
	const int size_source_f = size_source * sizeof(float);

	// let's create copies of our images now and copy them to the gpu memory

	unsigned char *gpu_target, *gpu_source;

	cudaMalloc((void **)&gpu_target, size_target);
	cudaMalloc((void **)&gpu_source, size_source);

	cudaMemcpy(gpu_target, target.ptr(), size_target, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_source, source.ptr(), size_source, cudaMemcpyHostToDevice);

	// Now the images in the alpha space :

	float *target_a, *source_a ;

	cudaMalloc((void **)&target_a, size_target_f);
	cudaMalloc((void **)&source_a, size_source_f);

	// We are ready for the conversion now

	rgb2alpha<<<grid_t, block>>>(gpu_target, target_a, r_l, l_a, target.cols, target_a.rows, target.step);
	rgb2alpha<<<grid_s, block>>>(gpu_source, source_a, r_l, l_a, source.cols, source.rows, source.step);

	// Free The Memory !!!!!

	cudaFree(gpu_source);
	cudaFree(gpu_target);


	/////////////////////// Phase 2 : Making up /////////////////////////////////////

	// We'll have to compute the mean and standart deviation by ourselves so let's go

	float *mean_target, *mean_source, *std_target, *std_source;		// those tables shall contain the means and std of each channel,
																	// consequently they will contain 3 values
	const int tab_size = sizeof(float) * 3;

	cudaMalloc((void **)&mean_target, tab_size);
	cudaMalloc((void **)&mean_source, tab_size);
	cudaMalloc((void **)&std_source, tab_size);
	cudaMalloc((void **)&std_target, tab_size);

	// Ready for computations

	channel_mean<<<grid_t, block>>>(target_a, mean_target, target.cols, target_a.rows, target.step);
	channel_mean<<<grid_s, block>>>(source_a, mean_source, source.cols, source.rows, source.step);

	channel_std<<<grid_t, block>>>(target_a, std_target, mean_target, target.cols, target_a.rows, target.step);
	channel_std<<<grid_s, block>>>(source_a, std_source, mean_source, source.cols, source.rows, source.step);

	cudaFree(source_a);		// We don't need it no more

	// Now we can make up our image, we gotta create the result table first

	float *result_a;
	cudaMalloc((void **)&result_a, size_target_f);

	make_up<<<grid_t, block>>>(target_a, result_a, mean_target, mean_source, std_target, std_source, target.cols, target_a.rows, target.step);

	// Now let's throw away what we don't need no more

	cudaFree(target_a);
	cudaFree(mean_target);
	cudaFree(mean_source);
	cudaFree(std_source);
	cudaFree(std_target);

	////////////////////// Phase 3 : l aplha beta to RGB conversion //////////////////////////

	// Our result image in RGB space :

	unsigned char *gpu_result;
	cudaMalloc((void **)&gpu_result, size_target);

	alpha2rgb<<<grid_t, block>>>(result_a, gpu_result, a_l, l_r, target.cols, target_a.rows, target.step);

	cudaFree(result_a);

	// Now our actual result

	Mat result(target.rows, target.cols, CV_8UC3);
	cudaMemcpyHostToDevice(result.ptr(), gpu_result, size_target, cudaMemcpyDeviceToHost);

	cudaFree(gpu_result);

	///////////////// Now the easy part //////////

	string name;
	cout << "name of the new image ?" << endl << endl ;
	cin >> name;

	name += ".png";

	imwrite(name, result); // Again, you can post your new pic to instacrap and let your subscribers believe that
							// you can do miracles with the camera of your phone.
}