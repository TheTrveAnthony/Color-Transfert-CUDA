#ifndef PLUS_H_INCLUDED
#define PLUS_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>




/* Here are the matrix required to make the different transforms.
	I wrote them as simple 1D basic c++ arrays to make things easier 
	once we get inside the GPU, basically, instead of doing a product of matrix
	we gonna directly write the expression of each channels for transforms.

	ex: instead of (L M S) = r_l * (R G B), 
	we gonna do: L = r_l[0]*R + r_l[1]*G + r_l[2]*B
				 M = .....
				 S = .....

*/

////////////////////// From RGB to LMS ////////////////////////////////

extern const float r_l[9] = {
						0.3811, 0.5783, 0.0402,
						0.1967, 0.7244, 0.0782,
						0.0241, 0.1288, 0.8444
           			 };


//////////////////// From LMS to RGB ///////////////////////////////////

extern const float l_r[9] = { 					
						4.4679, -3.5873, 0.1193,
						-1.2186, 2.3809, -0.1624,
						0.0497, -0.2439, 1.2045
					};


//////////////////// From LMS to l alpha beta //////////////////////

extern const float l_a[9] = {
						0.57735027, 0.57735027, 0.57735027,
						0.40824829, 0.40824829, -0.81649658,
						0.70710678, -0.70710678, 0
					};

/////////////////// From l alpha beta to LMS ////////////////////////////

extern const float a_l[9] = {
						0.57735027, 0.40824829, 0.70710678,
						0.57735027, 0.40824829, -0.70710678,
						0.57735027, -0.81649658, 0
					};


#endif