#ifndef PLUS_H_INCLUDED
#define PLUS_H_INCLUDED

using namespace std;

// Called by CPU :
void transfert(string nom1, string nom2);


// CAlled by GPU :
__global__ void rgb2alpha();
__global__ void alpha2rgb();
__global__ void make_up();
__global__ void channel_mean();
__global__ void channel_std();



#endif