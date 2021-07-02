#ifndef GPUALLOCHEADERDEF
#define GPUALLOCHEADERDEF

#include "param.h"
#define CUMAKE(var)  (cudaMalloc((void**) &var, sizeof(ptype)*Nt_seg))
#define CUKILL(var)  (cudaFree(var))

class DevAlloc
{
public:
	DevAlloc();
	~DevAlloc();
 	ptype *d_W0,   *d_W1,   *d_W2,   *d_W3,   *d_W4;
 	ptype *d_Wl0,  *d_Wl1,  *d_Wl2,  *d_Wl3,  *d_Wl4;
 	ptype *d_Wr0,  *d_Wr1,  *d_Wr2,  *d_Wr3,  *d_Wr4;
	ptype *d_F0x,  *d_F1x,  *d_F2x,  *d_F3x,  *d_F4x;
	ptype *d_F0y,  *d_F1y,  *d_F2y,  *d_F3y,  *d_F4y;
	ptype *d_F0z,  *d_F1z,  *d_F2z,  *d_F3z,  *d_F4z;
	ptype *d_DW0xl, *d_DW1xl, *d_DW2xl, *d_DW3xl, *d_DW4xl;
	ptype *d_DW0yl, *d_DW1yl, *d_DW2yl, *d_DW3yl, *d_DW4yl;
	ptype *d_DW0zl, *d_DW1zl, *d_DW2zl, *d_DW3zl, *d_DW4zl;
	ptype *d_DW0xr, *d_DW1xr, *d_DW2xr, *d_DW3xr, *d_DW4xr;
	ptype *d_DW0yr, *d_DW1yr, *d_DW2yr, *d_DW3yr, *d_DW4yr;
	ptype *d_DW0zr, *d_DW1zr, *d_DW2zr, *d_DW3zr, *d_DW4zr;
	
};

#endif
