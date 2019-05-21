#ifndef PLUS_H_INCLUDED
#define PLUS_H_INCLUDED

using namespace std;

// Called by CPU :
void transfert(string nom1, string nom2);

/*
// CAlled by GPU :
__global__ void rgb2alpha(unsigned char* ipt, float* opt, const float *rgb_l,
							const float *l_alpha, int width, int height, int step);

__global__ void alpha2rgb(float* ipt, unsigned char* opt, const float *alpha_l,
							const float *l_rgb, int width, int height, int step);

__global__ void make_up(float* ipt, float* opt, const float *mean_t,
							const float *mean_s, const float *std_t, const float *std_s,
							int width, int height, int step);

__global__ void channel_mean(float* tabz, float* meanz, int width, int height, int step);

__global__ void channel_std(float* tabz, float* stdz, float* meanz, int width, int height, int step);

*/

#endif