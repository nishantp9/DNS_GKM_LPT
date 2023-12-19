#include "GPUAlloc.h"

DevAlloc::DevAlloc()
{

	CUMAKE(d_W0);
	CUMAKE(d_W1);
	CUMAKE(d_W2);
	CUMAKE(d_W3);
	CUMAKE(d_W4);
	CUMAKE(d_Wl0);
	CUMAKE(d_Wl1);
	CUMAKE(d_Wl2);
	CUMAKE(d_Wl3);
	CUMAKE(d_Wl4);
	CUMAKE(d_Wr0);
	CUMAKE(d_Wr1);
	CUMAKE(d_Wr2);
	CUMAKE(d_Wr3);
	CUMAKE(d_Wr4);

	CUMAKE(d_F0x);
	CUMAKE(d_F1x);
	CUMAKE(d_F2x);
	CUMAKE(d_F3x);
	CUMAKE(d_F4x);
	CUMAKE(d_F0y);
	CUMAKE(d_F1y);
	CUMAKE(d_F2y);
	CUMAKE(d_F3y);
	CUMAKE(d_F4y);
	CUMAKE(d_F0z);
	CUMAKE(d_F1z);
	CUMAKE(d_F2z);
	CUMAKE(d_F3z);
	CUMAKE(d_F4z);

	CUMAKE(d_DW0xl);
	CUMAKE(d_DW1xl);
	CUMAKE(d_DW2xl);
	CUMAKE(d_DW3xl);
	CUMAKE(d_DW4xl);
	CUMAKE(d_DW0yl);
	CUMAKE(d_DW1yl);
	CUMAKE(d_DW2yl);
	CUMAKE(d_DW3yl);
	CUMAKE(d_DW4yl);
	CUMAKE(d_DW0zl);
	CUMAKE(d_DW1zl);
	CUMAKE(d_DW2zl);
	CUMAKE(d_DW3zl);
	CUMAKE(d_DW4zl);

	CUMAKE(d_DW0xr);
	CUMAKE(d_DW1xr);
	CUMAKE(d_DW2xr);
	CUMAKE(d_DW3xr);
	CUMAKE(d_DW4xr);
	CUMAKE(d_DW0yr);
	CUMAKE(d_DW1yr);
	CUMAKE(d_DW2yr);
	CUMAKE(d_DW3yr);
	CUMAKE(d_DW4yr);
	CUMAKE(d_DW0zr);
	CUMAKE(d_DW1zr);
	CUMAKE(d_DW2zr);
	CUMAKE(d_DW3zr);
	CUMAKE(d_DW4zr);
}

DevAlloc::~DevAlloc()
{
	CUKILL(d_W0);
	CUKILL(d_W1);
	CUKILL(d_W2);
	CUKILL(d_W3);
	CUKILL(d_W4);
	CUKILL(d_Wl0);
	CUKILL(d_Wl1);
	CUKILL(d_Wl2);
	CUKILL(d_Wl3);
	CUKILL(d_Wl4);
	CUKILL(d_Wr0);
	CUKILL(d_Wr1);
	CUKILL(d_Wr2);
	CUKILL(d_Wr3);
	CUKILL(d_Wr4);
	CUKILL(d_F0x);
	CUKILL(d_F1x);
	CUKILL(d_F2x);
	CUKILL(d_F3x);
	CUKILL(d_F4x);
	CUKILL(d_F0y);
	CUKILL(d_F1y);
	CUKILL(d_F2y);
	CUKILL(d_F3y);
	CUKILL(d_F4y);
	CUKILL(d_F0z);
	CUKILL(d_F1z);
	CUKILL(d_F2z);
	CUKILL(d_F3z);
	CUKILL(d_F4z);
	CUKILL(d_DW0xl);
	CUKILL(d_DW1xl);
	CUKILL(d_DW2xl);
	CUKILL(d_DW3xl);
	CUKILL(d_DW4xl);
	CUKILL(d_DW0yl);
	CUKILL(d_DW1yl);
	CUKILL(d_DW2yl);
	CUKILL(d_DW3yl);
	CUKILL(d_DW4yl);
	CUKILL(d_DW0zl);
	CUKILL(d_DW1zl);
	CUKILL(d_DW2zl);
	CUKILL(d_DW3zl);
	CUKILL(d_DW4zl);
	CUKILL(d_DW0xr);
	CUKILL(d_DW1xr);
	CUKILL(d_DW2xr);
	CUKILL(d_DW3xr);
	CUKILL(d_DW4xr);
	CUKILL(d_DW0yr);
	CUKILL(d_DW1yr);
	CUKILL(d_DW2yr);
	CUKILL(d_DW3yr);
	CUKILL(d_DW4yr);
	CUKILL(d_DW0zr);
	CUKILL(d_DW1zr);
	CUKILL(d_DW2zr);
	CUKILL(d_DW3zr);
	CUKILL(d_DW4zr);
}
