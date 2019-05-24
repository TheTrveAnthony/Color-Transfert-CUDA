#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include "function.h"


#define C 3
#define TAB 4096 // This is the whole shared memory avaiable for my GPU (4096*3 bytes), you can raise it
				 // if yours has a compute capability higher than 3.0
#define XX 50000000

using namespace std; //::cout
//using namespace std::endl;
//using namespace std::string;
using namespace cv;


__global__ void rgb2alpha(unsigned char* ipt, float* opt, int width, int height, int step)
{

	// our transform matrices:
	/* 
	I wrote them as simple 1D basic c++ arrays to make things easier 
	once we get inside the GPU, basically, instead of doing a product of matrix
	we gonna directly write the expression of each channels for transforms.

	ex: instead of (L M S) = r_l * (R G B), 
	we gonna do: L = r_l[0]*R + r_l[1]*G + r_l[2]*B
				 M = .....
				 S = .....

	*/

	const float r_l[9] = {
						0.3811, 0.5783, 0.0402,
						0.1967, 0.7244, 0.0782,
						0.0241, 0.1288, 0.8444
           			 };

    const float l_a[9] = {
						0.57735027, 0.57735027, 0.57735027,
						0.40824829, 0.40824829, -0.81649658,
						0.70710678, -0.70710678, 0
					};


	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O

	if(xIndex < width && yIndex < height){

		//Location of pixel 
		const int pix = yIndex * step + (3 * xIndex);
		
		/// Now to access the diferent values, we do:
		/// ipt[pix] = R, ipt[pix+1] = G, ipt[pix] = B
		/// So let's perform the transform now
		// LMS transform :
		
		float L, M, S;

		L = log10(ipt[pix] * r_l[0] + ipt[pix+1] * r_l[1] + ipt[pix+2] * r_l[2]);
		M = log10(ipt[pix] * r_l[3] + ipt[pix+1] * r_l[4] + ipt[pix+2] * r_l[5]);
		S = log10(ipt[pix] * r_l[6] + ipt[pix+1] * r_l[7] + ipt[pix+2] * r_l[8]);

		// l alpha beta transform :

		opt[pix] = L * l_a[0] + M * l_a[1] + S * l_a[2];
		opt[pix+1] = L * l_a[3] + M * l_a[4] + S * l_a[5];
		opt[pix+2] = L * l_a[6] + M * l_a[7] + S * l_a[8];
	}
}

__global__ void alpha2rgb(float* ipt, unsigned char* opt, int width, int height, int step)
{

	// The matrices

	float a_l[9] = {
						0.57735027, 0.40824829, 0.70710678,
						0.57735027, 0.40824829, -0.70710678,
						0.57735027, -0.81649658, 0
					};

	float l_r[9] = { 					
						4.4679, -3.5873, 0.1193,
						-1.2186, 2.3809, -0.1624,
						0.0497, -0.2439, 1.2045
					};
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

		L = pow(10, ipt[pix] * a_l[0] + ipt[pix+1] * a_l[1] + ipt[pix+2] * a_l[2]);
		M = pow(10, ipt[pix] * a_l[3] + ipt[pix+1] * a_l[4] + ipt[pix+2] * a_l[5]);
		S = pow(10, ipt[pix] * a_l[6] + ipt[pix+1] * a_l[7] + ipt[pix+2] * a_l[8]);

		// RGB transform :

		opt[pix] = static_cast<unsigned char>(L * l_r[0] + M * l_r[1] + S * l_r[2]);
		opt[pix+1] = static_cast<unsigned char>(L * l_r[3] + M * l_r[4] + S * l_r[5]);
		opt[pix+2] = static_cast<unsigned char>(L * l_r[6] + M * l_r[7] + S * l_r[8]);
	}
}

__global__ void make_up(float* ipt, float* opt, const float *mean_t,
						 const float *mean_s, const float *std_t, const float *std_s, int width, int height, int step)
{

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


__global__ void channel_std(float* tabz, float* stdz, float* meanz, int width, int height, int step)
{

	/* I can't find a way to write ot in "massive" parallelism mode, so it's with just 1 block and thread. */

	const int N = width * height;

	float s0(0);
	float s1(0);
	float s2(0);

	for (int x = 0; x < width; x ++){
		for (int y = 0; y < height; y++){

			int indx = y*step + 3*x ;

			s0 += pow(tabz[indx] - meanz[0], 2);
			s1 += pow(tabz[indx+1] - meanz[1], 2);
			s2 += pow(tabz[indx+2] - meanz[2], 2);

		}
	}

	s0 /= N;
	s1 /= N;
	s2 /= N;
	
	stdz[0] = sqrt(s0);
	stdz[1] = sqrt(s1);
	stdz[2] = sqrt(s2);

}



__global__ void channel_mean(float* tabz, float* meanz, const int width, const int height, const int step)
{
	/* I can't find a way to write ot in "massive" parallelism mode, so it's with just 1 block and thread. */

	const int N = width * height;

	float s0(0);
	float s1(0);
	float s2(0);

	for (int x = 0; x < width; x ++){
		for (int y = 0; y < height; y++){

			int indx = y*step + 3*x ;

			s0 += tabz[indx];
			s1 += tabz[indx+1];
			s2 += tabz[indx+2];

		}
	}

	s0 /= N;
	s1 /= N;
	s2 /= N;
	
	meanz[0] = s0;
	meanz[1] = s1;
	meanz[2] = s2;
	
}


void transfert(string nom1, string nom2){

	/////////// First of all let's load the images we'll use

	Mat target = imread(nom1, 1);
	Mat source = imread(nom2, 1);

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole images
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

	cudaMalloc((unsigned char **)&gpu_target, size_target);
	cudaMalloc((unsigned char **)&gpu_source, size_source);

	cudaMemcpy(gpu_target, target.ptr(), size_target, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_source, source.ptr(), size_source, cudaMemcpyHostToDevice);

	// Now the images in the alpha space :

	float *target_a, *source_a ;

	cudaMalloc((float **)&target_a, size_target_f);
	cudaMalloc((float **)&source_a, size_source_f);

	// We are ready for the conversion now

	rgb2alpha<<<grid_t, block>>>(gpu_target, target_a, target.cols, target.rows, target.step);
	rgb2alpha<<<grid_s, block>>>(gpu_source, source_a, source.cols, source.rows, source.step);

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

	channel_mean<<<1, 1>>>(target_a, mean_target, target.cols, target.rows, target.step);
	channel_mean<<<1, 1>>>(source_a, mean_source, source.cols, source.rows, source.step);

	channel_std<<<1, 1>>>(target_a, std_target, mean_target, target.cols, target.rows, target.step);
	channel_std<<<1, 1>>>(source_a, std_source, mean_source, source.cols, source.rows, source.step);

	
	cudaFree(source_a);		// We don't need it no more

	// Now we can make up our image, we gotta create the result table first

	float *result_a;
	cudaMalloc((void **)&result_a, size_target_f);

	make_up<<<grid_t, block>>>(target_a, result_a, mean_target, mean_source, std_target, std_source, target.cols, target.rows, target.step);

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

	alpha2rgb<<<grid_t, block>>>(result_a, gpu_result, target.cols, target.rows, target.step);

	cudaFree(result_a);

	// Now our actual result

	Mat result(target.rows, target.cols, CV_8UC3);
	cudaMemcpy(result.ptr(), gpu_result, size_target, cudaMemcpyDeviceToHost);
	
	cudaFree(gpu_result);

	///////////////// Now the easy part //////////

	string name;
	cout << "name of the new image ?" << endl << endl ;
	cin >> name;

	name += ".png";

	imwrite(name, result); // Again, you can post your new pic to instacrap and let your subscribers believe that
							// you can do miracles with the camera of your phone.
}


/*
void transfert(string nom1, string nom2){

	/////////// First of all let's load the images we'll use

	Mat target = imread(nom1, 1);
	

	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole images
	const dim3 grid_t((target.cols + block.x - 1)/block.x, (target.rows + block.y - 1)/block.y);

	//////////////////////// Phase 1: RGB to l alpha beta Conversion /////////////////////

	// Calculate the size of each image

	const int size_target = target.step * target.rows;

	// We also gonna use floats, so:

	const int size_target_f = size_target * sizeof(float);

	// let's create copies of our images now and copy them to the gpu memory

	unsigned char *gpu_target;//, *gpu_source;

	cudaMalloc((void **)&gpu_target, size_target);


	cudaMemcpy(gpu_target, target.ptr(), size_target, cudaMemcpyHostToDevice);

	// Now the images in the alpha space :

	float *target_a;

	cudaMalloc((void **)&target_a, size_target_f);



	// We are ready for the conversion now

	rgb2alpha<<<grid_t, block>>>(gpu_target, target_a, target.cols, target.rows, target.step);


	// Free The Memory !!!!!

	
	cudaFree(gpu_target);


	///////////////////////////////////////////////////////////////////////////////////////

	float *mean_target;		// those tables shall contain the means and std of each channel,
																	// consequently they will contain 3 values
	const int tab_size = sizeof(float) * 3;

	cudaMalloc((void **)&mean_target, tab_size);

	const int size_a = target.cols * target.rows * sizeof(float);

	float *a0, *a1, *a2 ;
	cudaMalloc((void **)&a0, size_a);
	cudaMalloc((void **)&a1, size_a);
	cudaMalloc((void **)&a2, size_a);


	// Ready for computations

	channel_mean<<<1, 1>>>(target_a, mean_target, target.cols, target.rows, target.step, a0, a1, a2);
	//cout << mean_target[0] << mean_target[1] << mean_target[2] << endl ;



	////////////////////// Phase 3 : l aplha beta to RGB conversion //////////////////////////

	// Our result image in RGB space :

	unsigned char *gpu_result;
	cudaMalloc((void **)&gpu_result, size_target);

	alpha2rgb<<<grid_t, block>>>(target_a, gpu_result, target.cols, target.rows, target.step);

	cudaFree(target_a);


	// Now our actual result

	Mat result(target.rows, target.cols, CV_8UC3);
	cudaMemcpy(result.ptr(), gpu_result, size_target, cudaMemcpyDeviceToHost);
	
	//cudaFree(gpu_result);

	// Put it at the right size

	//Mat trve_result;
	//resize(result, trve_result, Size(t_cols, t_rows), 0, 0);

	///////////////// Now the easy part //////////

	string name;
	cout << "name of the new image ?" << endl << endl ;
	cin >> name;

	name += ".png";

	imwrite(name, result); // Again, you can post your new pic to instacrap and let your subscribers believe that
							// you can do miracles with the camera of your phone.
}
*/