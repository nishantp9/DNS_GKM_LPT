/**************************************************************************************************************************
 * ----------------------------------     Updating Flow-Field on GPU   ---------------------------------------------------*
 * -----------------------------------------------------------------------------------------------------------------------*
 **************************************************************************************************************************/
#include "GPUupdate.h"


/** -----------------------------------------*
 * HANDLING ERROR 
 * ------------------------------------------*/
/*
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
    file, line );
    exit( EXIT_FAILURE );
  }
}
*/

/* 
 * global : GPU GLOBAL FUNCTION 
 * device : GPU DEVICE FUNCTION (Called from global function)
 */
__global__ void derivsX_Kernel(ptype *d_W0, ptype *d_Wl0, ptype *d_Wr0, ptype *d_DW0xl, ptype *d_DW0yl, ptype *d_DW0zl, ptype *d_DW0xr, ptype *d_DW0yr, ptype *d_DW0zr,
							   ptype *d_W1, ptype *d_Wl1, ptype *d_Wr1, ptype *d_DW1xl, ptype *d_DW1yl, ptype *d_DW1zl, ptype *d_DW1xr, ptype *d_DW1yr, ptype *d_DW1zr,
							   ptype *d_W2, ptype *d_Wl2, ptype *d_Wr2, ptype *d_DW2xl, ptype *d_DW2yl, ptype *d_DW2zl, ptype *d_DW2xr, ptype *d_DW2yr, ptype *d_DW2zr,
							   ptype *d_W3, ptype *d_Wl3, ptype *d_Wr3, ptype *d_DW3xl, ptype *d_DW3yl, ptype *d_DW3zl, ptype *d_DW3xr, ptype *d_DW3yr, ptype *d_DW3zr,
							   ptype *d_W4, ptype *d_Wl4, ptype *d_Wr4, ptype *d_DW4xl, ptype *d_DW4yl, ptype *d_DW4zl, ptype *d_DW4xr, ptype *d_DW4yr, ptype *d_DW4zr);

__global__ void derivsY_Kernel(ptype *d_W0, ptype *d_Wl0, ptype *d_Wr0, ptype *d_DW0xl, ptype *d_DW0yl, ptype *d_DW0zl, ptype *d_DW0xr, ptype *d_DW0yr, ptype *d_DW0zr,
							   ptype *d_W1, ptype *d_Wl1, ptype *d_Wr1, ptype *d_DW1xl, ptype *d_DW1yl, ptype *d_DW1zl, ptype *d_DW1xr, ptype *d_DW1yr, ptype *d_DW1zr,
							   ptype *d_W2, ptype *d_Wl2, ptype *d_Wr2, ptype *d_DW2xl, ptype *d_DW2yl, ptype *d_DW2zl, ptype *d_DW2xr, ptype *d_DW2yr, ptype *d_DW2zr,
							   ptype *d_W3, ptype *d_Wl3, ptype *d_Wr3, ptype *d_DW3xl, ptype *d_DW3yl, ptype *d_DW3zl, ptype *d_DW3xr, ptype *d_DW3yr, ptype *d_DW3zr,
							   ptype *d_W4, ptype *d_Wl4, ptype *d_Wr4, ptype *d_DW4xl, ptype *d_DW4yl, ptype *d_DW4zl, ptype *d_DW4xr, ptype *d_DW4yr, ptype *d_DW4zr);

__global__ void derivsZ_Kernel(ptype *d_W0, ptype *d_Wl0, ptype *d_Wr0, ptype *d_DW0xl, ptype *d_DW0yl, ptype *d_DW0zl, ptype *d_DW0xr, ptype *d_DW0yr, ptype *d_DW0zr,
							   ptype *d_W1, ptype *d_Wl1, ptype *d_Wr1, ptype *d_DW1xl, ptype *d_DW1yl, ptype *d_DW1zl, ptype *d_DW1xr, ptype *d_DW1yr, ptype *d_DW1zr,
							   ptype *d_W2, ptype *d_Wl2, ptype *d_Wr2, ptype *d_DW2xl, ptype *d_DW2yl, ptype *d_DW2zl, ptype *d_DW2xr, ptype *d_DW2yr, ptype *d_DW2zr,
							   ptype *d_W3, ptype *d_Wl3, ptype *d_Wr3, ptype *d_DW3xl, ptype *d_DW3yl, ptype *d_DW3zl, ptype *d_DW3xr, ptype *d_DW3yr, ptype *d_DW3zr,
							   ptype *d_W4, ptype *d_Wl4, ptype *d_Wr4, ptype *d_DW4xl, ptype *d_DW4yl, ptype *d_DW4zl, ptype *d_DW4xr, ptype *d_DW4yr, ptype *d_DW4zr);

__global__ void flip_Kernel(ptype *d_Wl1, ptype *d_Wl2, ptype *d_DW1xl, ptype *d_DW2xl, ptype *d_DW1yl, ptype *d_DW2yl, ptype *d_DW1zl, ptype *d_DW2zl,
                            ptype *d_Wr1, ptype *d_Wr2, ptype *d_DW1xr, ptype *d_DW2xr, ptype *d_DW1yr, ptype *d_DW2yr, ptype *d_DW1zr, ptype *d_DW2zr);

__global__ void flipBack_Kernel(ptype *d_F1, ptype *d_F2);

__global__ void update_Kernel (ptype *d_W, ptype *d_Fx, ptype *d_Fy, ptype *d_Fz);								  
										  
__global__ void flux(ptype *d_W0, ptype *d_W1, ptype *d_W2, ptype *d_W3, ptype *d_W4,
					 ptype *d_Wl0, ptype *d_Wl1, ptype *d_Wl2, ptype *d_Wl3, ptype *d_Wl4,
					 ptype *d_Wr0, ptype *d_Wr1, ptype *d_Wr2, ptype *d_Wr3, ptype *d_Wr4,
					 ptype *d_DW0xl, ptype *d_DW1xl, ptype *d_DW2xl, ptype *d_DW3xl, ptype *d_DW4xl,
					 ptype *d_DW0xr, ptype *d_DW1xr, ptype *d_DW2xr, ptype *d_DW3xr, ptype *d_DW4xr,
					 ptype *d_DW0yl, ptype *d_DW1yl, ptype *d_DW2yl, ptype *d_DW3yl, ptype *d_DW4yl,
					 ptype *d_DW0yr, ptype *d_DW1yr, ptype *d_DW2yr, ptype *d_DW3yr, ptype *d_DW4yr,
					 ptype *d_DW0zl, ptype *d_DW1zl, ptype *d_DW2zl, ptype *d_DW3zl, ptype *d_DW4zl,
					 ptype *d_DW0zr, ptype *d_DW1zr, ptype *d_DW2zr, ptype *d_DW3zr, ptype *d_DW4zr,
					 ptype *d_F0, ptype *d_F1, ptype *d_F2, ptype *d_F3, ptype *d_F4, int TAG);

__device__ void d_slopesolver(ptype b[5], ptype U[3], ptype lam, ptype a[5]);
__device__ void d_MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m, ptype ax[5]);
__device__ void d_MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m);

__device__ void d_c2p(ptype W[5], ptype &den, ptype U[3], ptype &P);

__device__ void d_ApplyWENO(ptype WLLL[5], ptype WLL[5], ptype WL[5], ptype WR[5], ptype WRR[5],  ptype WRRR[5], ptype Wl[5], ptype Wr[5]);

__global__ void W2T3D(ptype *d_W0, ptype *d_W1, ptype *d_W2, ptype *d_W3, ptype *d_W4, ptype *d_T);

__constant__ int d_nt_segx, d_nc_segx, d_nt_segy, d_nc_segy, d_nt_segz, d_nc_segz, d_Nt_seg, d_Nc_seg, d_K;

__constant__ ptype d_dx, d_dt, d_mu0, d_T0, GAM;

/**----------------------------------------------------------------------------------------------------------------------*
 * Function : Evolves Flow Field over the whole 3-D domain
 *-----------------------------------------------------------------------------------------------------------------------*/
void evolve(ptype *W[5], DevAlloc *dev, ptype dt)
{
	//cudaSetDevice(int(myrank_3d%2));
	int blkz  = (Nt_seg-1)/Bsz3D + 1;
	int blkf  = (Nt_seg-1)/Bszf  + 1;
	int blkst = (Nt_seg-1)/Bszs  + 1;

/* ----------------------------------------------------*
 * 	Host (CPU) to Device (GPU) Flow Field data Transfer
 * ----------------------------------------------------*/
	HtoD(dev->d_W0, W[0]);
	HtoD(dev->d_W1, W[1]); 
	HtoD(dev->d_W2, W[2]); 
	HtoD(dev->d_W3, W[3]); 
	HtoD(dev->d_W4, W[4]); 
		
	cudaDeviceSynchronize();
	
	ptype Dx = dx;
	ptype Mu = mu0;
	ptype T	= T0;	
	ptype Ga = gam;
	ptype Dt = dt;
	(cudaMemcpyToSymbol(d_nt_segx, &nt_segx, sizeof(int)));
	(cudaMemcpyToSymbol(d_nc_segx, &nc_segx, sizeof(int)));
	(cudaMemcpyToSymbol(d_nt_segy, &nt_segy, sizeof(int)));
	(cudaMemcpyToSymbol(d_nc_segy, &nc_segy, sizeof(int)));
	(cudaMemcpyToSymbol(d_nt_segz, &nt_segz, sizeof(int)));
	(cudaMemcpyToSymbol(d_nc_segz, &nc_segz, sizeof(int)));
	(cudaMemcpyToSymbol(d_Nt_seg,  &Nt_seg, sizeof(int)));
	(cudaMemcpyToSymbol(d_Nc_seg,  &Nc_seg, sizeof(int)));
	(cudaMemcpyToSymbol(d_K,       &K,      sizeof(int)));
	(cudaMemcpyToSymbol(d_dx,      &Dx, sizeof(ptype)));
	(cudaMemcpyToSymbol(d_dt,      &Dt, sizeof(ptype)));	
	(cudaMemcpyToSymbol(d_mu0,     &Mu, sizeof(ptype)));
	(cudaMemcpyToSymbol(GAM,       &Ga, sizeof(ptype)));
	(cudaMemcpyToSymbol(d_T0,      &T,  sizeof(ptype)));
	
	cudaDeviceSynchronize();
	
	derivsX_Kernel<<<blkst, Bszs>>>(dev->d_W0, dev->d_Wl0, dev->d_Wr0, dev->d_DW0xl, dev->d_DW0yl, dev->d_DW0zl, dev->d_DW0xr, dev->d_DW0yr, dev->d_DW0zr,
									dev->d_W1, dev->d_Wl1, dev->d_Wr1, dev->d_DW1xl, dev->d_DW1yl, dev->d_DW1zl, dev->d_DW1xr, dev->d_DW1yr, dev->d_DW1zr,
									dev->d_W2, dev->d_Wl2, dev->d_Wr2, dev->d_DW2xl, dev->d_DW2yl, dev->d_DW2zl, dev->d_DW2xr, dev->d_DW2yr, dev->d_DW2zr,
									dev->d_W3, dev->d_Wl3, dev->d_Wr3, dev->d_DW3xl, dev->d_DW3yl, dev->d_DW3zl, dev->d_DW3xr, dev->d_DW3yr, dev->d_DW3zr,
									dev->d_W4, dev->d_Wl4, dev->d_Wr4, dev->d_DW4xl, dev->d_DW4yl, dev->d_DW4zl, dev->d_DW4xr, dev->d_DW4yr, dev->d_DW4zr);

	cudaDeviceSynchronize();

	flux<<<blkf, Bszf>>>(dev->d_W0, dev->d_W1, dev->d_W2, dev->d_W3, dev->d_W4,
						 dev->d_Wl0, dev->d_Wl1, dev->d_Wl2, dev->d_Wl3, dev->d_Wl4,
						 dev->d_Wr0, dev->d_Wr1, dev->d_Wr2, dev->d_Wr3, dev->d_Wr4,
						 dev->d_DW0xl, dev->d_DW1xl, dev->d_DW2xl, dev->d_DW3xl, dev->d_DW4xl,
						 dev->d_DW0xr, dev->d_DW1xr, dev->d_DW2xr, dev->d_DW3xr, dev->d_DW4xr,  
						 dev->d_DW0yl, dev->d_DW1yl, dev->d_DW2yl, dev->d_DW3yl, dev->d_DW4yl, 
						 dev->d_DW0yr, dev->d_DW1yr, dev->d_DW2yr, dev->d_DW3yr, dev->d_DW4yr, 
						 dev->d_DW0zl, dev->d_DW1zl, dev->d_DW2zl, dev->d_DW3zl, dev->d_DW4zl, 
						 dev->d_DW0zr, dev->d_DW1zr, dev->d_DW2zr, dev->d_DW3zr, dev->d_DW4zr, 
						 dev->d_F0x, dev->d_F1x, dev->d_F2x, dev->d_F3x, dev->d_F4x, 1);
	
	cudaDeviceSynchronize();
	
	derivsY_Kernel<<<blkst, Bszs>>>(dev->d_W0, dev->d_Wl0, dev->d_Wr0, dev->d_DW0xl, dev->d_DW0yl, dev->d_DW0zl, dev->d_DW0xr, dev->d_DW0yr, dev->d_DW0zr,
									dev->d_W1, dev->d_Wl1, dev->d_Wr1, dev->d_DW1xl, dev->d_DW1yl, dev->d_DW1zl, dev->d_DW1xr, dev->d_DW1yr, dev->d_DW1zr,
									dev->d_W2, dev->d_Wl2, dev->d_Wr2, dev->d_DW2xl, dev->d_DW2yl, dev->d_DW2zl, dev->d_DW2xr, dev->d_DW2yr, dev->d_DW2zr,
									dev->d_W3, dev->d_Wl3, dev->d_Wr3, dev->d_DW3xl, dev->d_DW3yl, dev->d_DW3zl, dev->d_DW3xr, dev->d_DW3yr, dev->d_DW3zr,
									dev->d_W4, dev->d_Wl4, dev->d_Wr4, dev->d_DW4xl, dev->d_DW4yl, dev->d_DW4zl, dev->d_DW4xr, dev->d_DW4yr, dev->d_DW4zr);

	cudaDeviceSynchronize();
		
	
	flip_Kernel<<<blkz, Bsz3D>>>(dev->d_Wl1, dev->d_Wl2, dev->d_DW1xl, dev->d_DW2xl, dev->d_DW1yl, dev->d_DW2yl, dev->d_DW1zl, dev->d_DW2zl, 
	                             dev->d_Wr1, dev->d_Wr2, dev->d_DW1xr, dev->d_DW2xr, dev->d_DW1yr, dev->d_DW2yr, dev->d_DW1zr, dev->d_DW2zr);
	(cudaDeviceSynchronize());
					
	flux<<<blkf, Bszf>>>(dev->d_W0, dev->d_W1, dev->d_W2, dev->d_W3, dev->d_W4,
						 dev->d_Wl0, dev->d_Wl1, dev->d_Wl2, dev->d_Wl3, dev->d_Wl4,
						 dev->d_Wr0, dev->d_Wr1, dev->d_Wr2, dev->d_Wr3, dev->d_Wr4,
						 dev->d_DW0xl, dev->d_DW1xl, dev->d_DW2xl, dev->d_DW3xl, dev->d_DW4xl,
						 dev->d_DW0xr, dev->d_DW1xr, dev->d_DW2xr, dev->d_DW3xr, dev->d_DW4xr,  
						 dev->d_DW0yl, dev->d_DW1yl, dev->d_DW2yl, dev->d_DW3yl, dev->d_DW4yl, 
						 dev->d_DW0yr, dev->d_DW1yr, dev->d_DW2yr, dev->d_DW3yr, dev->d_DW4yr, 
						 dev->d_DW0zl, dev->d_DW1zl, dev->d_DW2zl, dev->d_DW3zl, dev->d_DW4zl, 
						 dev->d_DW0zr, dev->d_DW1zr, dev->d_DW2zr, dev->d_DW3zr, dev->d_DW4zr, 
						 dev->d_F0y, dev->d_F1y, dev->d_F2y, dev->d_F3y, dev->d_F4y, 2);
			
	cudaDeviceSynchronize();

	flipBack_Kernel<<<blkz, Bsz3D>>>(dev->d_F1y, dev->d_F2y);
					
	cudaDeviceSynchronize();
			
	derivsZ_Kernel<<<blkst, Bszs>>>(dev->d_W0, dev->d_Wl0, dev->d_Wr0, dev->d_DW0xl, dev->d_DW0yl, dev->d_DW0zl, dev->d_DW0xr, dev->d_DW0yr, dev->d_DW0zr,
									dev->d_W1, dev->d_Wl1, dev->d_Wr1, dev->d_DW1xl, dev->d_DW1yl, dev->d_DW1zl, dev->d_DW1xr, dev->d_DW1yr, dev->d_DW1zr,
									dev->d_W2, dev->d_Wl2, dev->d_Wr2, dev->d_DW2xl, dev->d_DW2yl, dev->d_DW2zl, dev->d_DW2xr, dev->d_DW2yr, dev->d_DW2zr,
									dev->d_W3, dev->d_Wl3, dev->d_Wr3, dev->d_DW3xl, dev->d_DW3yl, dev->d_DW3zl, dev->d_DW3xr, dev->d_DW3yr, dev->d_DW3zr,
									dev->d_W4, dev->d_Wl4, dev->d_Wr4, dev->d_DW4xl, dev->d_DW4yl, dev->d_DW4zl, dev->d_DW4xr, dev->d_DW4yr, dev->d_DW4zr);

	cudaDeviceSynchronize();
		
	flip_Kernel<<<blkz, Bsz3D>>>(dev->d_Wl1, dev->d_Wl3, dev->d_DW1xl, dev->d_DW3xl, dev->d_DW1yl, dev->d_DW3yl, dev->d_DW1zl, dev->d_DW3zl, 
								 dev->d_Wr1, dev->d_Wr3, dev->d_DW1xr, dev->d_DW3xr, dev->d_DW1yr, dev->d_DW3yr, dev->d_DW1zr, dev->d_DW3zr);

	cudaDeviceSynchronize();
					
	flux<<<blkf, Bszf>>>(dev->d_W0, dev->d_W1, dev->d_W2, dev->d_W3, dev->d_W4,
						 dev->d_Wl0, dev->d_Wl1, dev->d_Wl2, dev->d_Wl3, dev->d_Wl4,
						 dev->d_Wr0, dev->d_Wr1, dev->d_Wr2, dev->d_Wr3, dev->d_Wr4,
						 dev->d_DW0xl, dev->d_DW1xl, dev->d_DW2xl, dev->d_DW3xl, dev->d_DW4xl,
						 dev->d_DW0xr, dev->d_DW1xr, dev->d_DW2xr, dev->d_DW3xr, dev->d_DW4xr,  
						 dev->d_DW0yl, dev->d_DW1yl, dev->d_DW2yl, dev->d_DW3yl, dev->d_DW4yl, 
						 dev->d_DW0yr, dev->d_DW1yr, dev->d_DW2yr, dev->d_DW3yr, dev->d_DW4yr, 
						 dev->d_DW0zl, dev->d_DW1zl, dev->d_DW2zl, dev->d_DW3zl, dev->d_DW4zl, 
						 dev->d_DW0zr, dev->d_DW1zr, dev->d_DW2zr, dev->d_DW3zr, dev->d_DW4zr, 
						 dev->d_F0z, dev->d_F1z, dev->d_F2z, dev->d_F3z, dev->d_F4z, 3);
	cudaDeviceSynchronize();

	flipBack_Kernel<<<blkz, Bsz3D>>>(dev->d_F1z, dev->d_F3z);
																					
	cudaDeviceSynchronize();

	update_Kernel<<<blkst, Bszs>>>(dev->d_W0, dev->d_F0x, dev->d_F0y, dev->d_F0z);
	update_Kernel<<<blkst, Bszs>>>(dev->d_W1, dev->d_F1x, dev->d_F1y, dev->d_F1z);
	update_Kernel<<<blkst, Bszs>>>(dev->d_W2, dev->d_F2x, dev->d_F2y, dev->d_F2z);
	update_Kernel<<<blkst, Bszs>>>(dev->d_W3, dev->d_F3x, dev->d_F3y, dev->d_F3z);
	update_Kernel<<<blkst, Bszs>>>(dev->d_W4, dev->d_F4x, dev->d_F4y, dev->d_F4z);
	
	cudaDeviceSynchronize();
					
/* -------------------------------------------------------*
 * 	Applying Peridic BC and Printing Turbulent stats
 * -------------------------------------------------------*/
	DtoH(dev->d_W0, W[0]);	
	DtoH(dev->d_W1, W[1]); 	
	DtoH(dev->d_W2, W[2]);	
	DtoH(dev->d_W3, W[3]);
	DtoH(dev->d_W4, W[4]);

	cudaDeviceSynchronize();
/** ---------------------------------------------------------------------------------------------------*
 * 	Main Iteration Loop ENDS  [Field Evolved]
 * ----------------------------------------------------------------------------------------------------*/	
}


/**-----------------------------------------------------------------------------------------*
 * Global Function : Calculates Derivatives required for calculation of flux in X-direction
 *------------------------------------------------------------------------------------------*/

__global__ void derivsX_Kernel(ptype *d_W0, ptype *d_Wl0, ptype *d_Wr0, ptype *d_DW0xl, ptype *d_DW0yl, ptype *d_DW0zl, ptype *d_DW0xr, ptype *d_DW0yr, ptype *d_DW0zr,
							   ptype *d_W1, ptype *d_Wl1, ptype *d_Wr1, ptype *d_DW1xl, ptype *d_DW1yl, ptype *d_DW1zl, ptype *d_DW1xr, ptype *d_DW1yr, ptype *d_DW1zr,
							   ptype *d_W2, ptype *d_Wl2, ptype *d_Wr2, ptype *d_DW2xl, ptype *d_DW2yl, ptype *d_DW2zl, ptype *d_DW2xr, ptype *d_DW2yr, ptype *d_DW2zr,
							   ptype *d_W3, ptype *d_Wl3, ptype *d_Wr3, ptype *d_DW3xl, ptype *d_DW3yl, ptype *d_DW3zl, ptype *d_DW3xr, ptype *d_DW3yr, ptype *d_DW3zr,
							   ptype *d_W4, ptype *d_Wl4, ptype *d_Wr4, ptype *d_DW4xl, ptype *d_DW4yl, ptype *d_DW4zl, ptype *d_DW4xr, ptype *d_DW4yr, ptype *d_DW4zr)
{
	int I = blockIdx.x*(blockDim.x) + threadIdx.x;

	int iy =  I/(d_nt_segx*d_nt_segz);
	int ix =  (I%(d_nt_segx*d_nt_segz))/d_nt_segz;
	int iz =  I%d_nt_segz;

	ptype WLLL[5], WLL[5], WL[5], WR[5], WRR[5], WRRR[5];
	ptype WLLLN[5], WLLN[5], WLN[5], WRN[5], WRRN[5], WRRRN[5];
	ptype WLLLS[5], WLLS[5], WLS[5], WRS[5], WRRS[5], WRRRS[5];
	ptype WLLLF[5], WLLF[5], WLF[5], WRF[5], WRRF[5], WRRRF[5];
	ptype WLLLB[5], WLLB[5], WLB[5], WRB[5], WRRB[5], WRRRB[5];
	ptype Wl[5], Wr[5], WlN[5], WrN[5], WlS[5], WrS[5], WlF[5], WrF[5], WlB[5], WrB[5];
	int q = 0;	
	if(ix < d_nt_segx-3 && iy < d_nt_segy-3 && iz < d_nt_segz-3 && iy > 2 && iz > 2 && ix > 1)
	{
		q = 0;
		WLLL[q] = d_W0[Id(iy,ix-2,iz)];
		WLL[q]  = d_W0[Id(iy,ix-1,iz)];
		WL[q]   = d_W0[Id(iy,ix,iz)];
		WR[q]   = d_W0[Id(iy,ix+1,iz)];
		WRR[q]  = d_W0[Id(iy,ix+2,iz)];
		WRRR[q] = d_W0[Id(iy,ix+3,iz)];
		
		WLLLN[q] = d_W0[Id(iy+1,ix-2,iz)];
		WLLN[q]  = d_W0[Id(iy+1,ix-1,iz)];
		WLN[q]   = d_W0[Id(iy+1,ix,iz)];
		WRN[q]   = d_W0[Id(iy+1,ix+1,iz)];
		WRRN[q]  = d_W0[Id(iy+1,ix+2,iz)];
		WRRRN[q] = d_W0[Id(iy+1,ix+3,iz)];
		
		WLLLS[q] = d_W0[Id(iy-1,ix-2,iz)];
		WLLS[q]  = d_W0[Id(iy-1,ix-1,iz)];
		WLS[q]   = d_W0[Id(iy-1,ix,iz)];
		WRS[q]   = d_W0[Id(iy-1,ix+1,iz)];
		WRRS[q]  = d_W0[Id(iy-1,ix+2,iz)];
		WRRRS[q] = d_W0[Id(iy-1,ix+3,iz)];

		WLLLF[q] = d_W0[Id(iy,ix-2,iz+1)];
		WLLF[q]  = d_W0[Id(iy,ix-1,iz+1)];
		WLF[q]   = d_W0[Id(iy,ix,iz+1)];
		WRF[q]   = d_W0[Id(iy,ix+1,iz+1)];
		WRRF[q]  = d_W0[Id(iy,ix+2,iz+1)];
		WRRRF[q] = d_W0[Id(iy,ix+3,iz+1)];

		WLLLB[q] = d_W0[Id(iy,ix-2,iz-1)];
		WLLB[q]  = d_W0[Id(iy,ix-1,iz-1)];
		WLB[q]   = d_W0[Id(iy,ix,iz-1)];
		WRB[q]   = d_W0[Id(iy,ix+1,iz-1)];
		WRRB[q]  = d_W0[Id(iy,ix+2,iz-1)];
		WRRRB[q] = d_W0[Id(iy,ix+3,iz-1)];
///////////////////////////////////////////
		q = 1;
		WLLL[q] = d_W1[Id(iy,ix-2,iz)];
		WLL[q]  = d_W1[Id(iy,ix-1,iz)];
		WL[q]   = d_W1[Id(iy,ix,iz)];
		WR[q]   = d_W1[Id(iy,ix+1,iz)];
		WRR[q]  = d_W1[Id(iy,ix+2,iz)];
		WRRR[q] = d_W1[Id(iy,ix+3,iz)];
		
		WLLLN[q] = d_W1[Id(iy+1,ix-2,iz)];
		WLLN[q]  = d_W1[Id(iy+1,ix-1,iz)];
		WLN[q]   = d_W1[Id(iy+1,ix,iz)];
		WRN[q]   = d_W1[Id(iy+1,ix+1,iz)];
		WRRN[q]  = d_W1[Id(iy+1,ix+2,iz)];
		WRRRN[q] = d_W1[Id(iy+1,ix+3,iz)];
		
		WLLLS[q] = d_W1[Id(iy-1,ix-2,iz)];
		WLLS[q]  = d_W1[Id(iy-1,ix-1,iz)];
		WLS[q]   = d_W1[Id(iy-1,ix,iz)];
		WRS[q]   = d_W1[Id(iy-1,ix+1,iz)];
		WRRS[q]  = d_W1[Id(iy-1,ix+2,iz)];
		WRRRS[q] = d_W1[Id(iy-1,ix+3,iz)];

		WLLLF[q] = d_W1[Id(iy,ix-2,iz+1)];
		WLLF[q]  = d_W1[Id(iy,ix-1,iz+1)];
		WLF[q]   = d_W1[Id(iy,ix,iz+1)];
		WRF[q]   = d_W1[Id(iy,ix+1,iz+1)];
		WRRF[q]  = d_W1[Id(iy,ix+2,iz+1)];
		WRRRF[q] = d_W1[Id(iy,ix+3,iz+1)];

		WLLLB[q] = d_W1[Id(iy,ix-2,iz-1)];
		WLLB[q]  = d_W1[Id(iy,ix-1,iz-1)];
		WLB[q]   = d_W1[Id(iy,ix,iz-1)];
		WRB[q]   = d_W1[Id(iy,ix+1,iz-1)];
		WRRB[q]  = d_W1[Id(iy,ix+2,iz-1)];
		WRRRB[q] = d_W1[Id(iy,ix+3,iz-1)];

///////////////////////////////////////////
		q = 2;
		WLLL[q] = d_W2[Id(iy,ix-2,iz)];
		WLL[q]  = d_W2[Id(iy,ix-1,iz)];
		WL[q]   = d_W2[Id(iy,ix,iz)];
		WR[q]   = d_W2[Id(iy,ix+1,iz)];
		WRR[q]  = d_W2[Id(iy,ix+2,iz)];
		WRRR[q] = d_W2[Id(iy,ix+3,iz)];
		
		WLLLN[q] = d_W2[Id(iy+1,ix-2,iz)];
		WLLN[q]  = d_W2[Id(iy+1,ix-1,iz)];
		WLN[q]   = d_W2[Id(iy+1,ix,iz)];
		WRN[q]   = d_W2[Id(iy+1,ix+1,iz)];
		WRRN[q]  = d_W2[Id(iy+1,ix+2,iz)];
		WRRRN[q] = d_W2[Id(iy+1,ix+3,iz)];
		
		WLLLS[q] = d_W2[Id(iy-1,ix-2,iz)];
		WLLS[q]  = d_W2[Id(iy-1,ix-1,iz)];
		WLS[q]   = d_W2[Id(iy-1,ix,iz)];
		WRS[q]   = d_W2[Id(iy-1,ix+1,iz)];
		WRRS[q]  = d_W2[Id(iy-1,ix+2,iz)];
		WRRRS[q] = d_W2[Id(iy-1,ix+3,iz)];

		WLLLF[q] = d_W2[Id(iy,ix-2,iz+1)];
		WLLF[q]  = d_W2[Id(iy,ix-1,iz+1)];
		WLF[q]   = d_W2[Id(iy,ix,iz+1)];
		WRF[q]   = d_W2[Id(iy,ix+1,iz+1)];
		WRRF[q]  = d_W2[Id(iy,ix+2,iz+1)];
		WRRRF[q] = d_W2[Id(iy,ix+3,iz+1)];

		WLLLB[q] = d_W2[Id(iy,ix-2,iz-1)];
		WLLB[q]  = d_W2[Id(iy,ix-1,iz-1)];
		WLB[q]   = d_W2[Id(iy,ix,iz-1)];
		WRB[q]   = d_W2[Id(iy,ix+1,iz-1)];
		WRRB[q]  = d_W2[Id(iy,ix+2,iz-1)];
		WRRRB[q] = d_W2[Id(iy,ix+3,iz-1)];		

///////////////////////////////////////////
		q = 3;
		WLLL[q] = d_W3[Id(iy,ix-2,iz)];
		WLL[q]  = d_W3[Id(iy,ix-1,iz)];
		WL[q]   = d_W3[Id(iy,ix,iz)];
		WR[q]   = d_W3[Id(iy,ix+1,iz)];
		WRR[q]  = d_W3[Id(iy,ix+2,iz)];
		WRRR[q] = d_W3[Id(iy,ix+3,iz)];
	
		WLLLN[q] = d_W3[Id(iy+1,ix-2,iz)];
		WLLN[q]  = d_W3[Id(iy+1,ix-1,iz)];
		WLN[q]   = d_W3[Id(iy+1,ix,iz)];
		WRN[q]   = d_W3[Id(iy+1,ix+1,iz)];
		WRRN[q]  = d_W3[Id(iy+1,ix+2,iz)];
		WRRRN[q] = d_W3[Id(iy+1,ix+3,iz)];

		WLLLS[q] = d_W3[Id(iy-1,ix-2,iz)];
		WLLS[q]  = d_W3[Id(iy-1,ix-1,iz)];
		WLS[q]   = d_W3[Id(iy-1,ix,iz)];
		WRS[q]   = d_W3[Id(iy-1,ix+1,iz)];
		WRRS[q]  = d_W3[Id(iy-1,ix+2,iz)];
		WRRRS[q] = d_W3[Id(iy-1,ix+3,iz)];

		WLLLF[q] = d_W3[Id(iy,ix-2,iz+1)];
		WLLF[q]  = d_W3[Id(iy,ix-1,iz+1)];
		WLF[q]   = d_W3[Id(iy,ix,iz+1)];
		WRF[q]   = d_W3[Id(iy,ix+1,iz+1)];
		WRRF[q]  = d_W3[Id(iy,ix+2,iz+1)];
		WRRRF[q] = d_W3[Id(iy,ix+3,iz+1)];

		WLLLB[q] = d_W1[Id(iy,ix-2,iz-1)];
		WLLB[q]  = d_W1[Id(iy,ix-1,iz-1)];
		WLB[q]   = d_W1[Id(iy,ix,iz-1)];
		WRB[q]   = d_W1[Id(iy,ix+1,iz-1)];
		WRRB[q]  = d_W1[Id(iy,ix+2,iz-1)];
		WRRRB[q] = d_W1[Id(iy,ix+3,iz-1)];

///////////////////////////////////////////
		q = 4;
		WLLL[q] = d_W4[Id(iy,ix-2,iz)];
		WLL[q]  = d_W4[Id(iy,ix-1,iz)];
		WL[q]   = d_W4[Id(iy,ix,iz)];
		WR[q]   = d_W4[Id(iy,ix+1,iz)];
		WRR[q]  = d_W4[Id(iy,ix+2,iz)];
		WRRR[q] = d_W4[Id(iy,ix+3,iz)];
		
		WLLLN[q] = d_W4[Id(iy+1,ix-2,iz)];
		WLLN[q]  = d_W4[Id(iy+1,ix-1,iz)];
		WLN[q]   = d_W4[Id(iy+1,ix,iz)];
		WRN[q]   = d_W4[Id(iy+1,ix+1,iz)];
		WRRN[q]  = d_W4[Id(iy+1,ix+2,iz)];
		WRRRN[q] = d_W4[Id(iy+1,ix+3,iz)];
		
		WLLLS[q] = d_W4[Id(iy-1,ix-2,iz)];
		WLLS[q]  = d_W4[Id(iy-1,ix-1,iz)];
		WLS[q]   = d_W4[Id(iy-1,ix,iz)];
		WRS[q]   = d_W4[Id(iy-1,ix+1,iz)];
		WRRS[q]  = d_W4[Id(iy-1,ix+2,iz)];
		WRRRS[q] = d_W4[Id(iy-1,ix+3,iz)];

		WLLLF[q] = d_W4[Id(iy,ix-2,iz+1)];
		WLLF[q]  = d_W4[Id(iy,ix-1,iz+1)];
		WLF[q]   = d_W4[Id(iy,ix,iz+1)];
		WRF[q]   = d_W4[Id(iy,ix+1,iz+1)];
		WRRF[q]  = d_W4[Id(iy,ix+2,iz+1)];
		WRRRF[q] = d_W4[Id(iy,ix+3,iz+1)];

		WLLLB[q] = d_W4[Id(iy,ix-2,iz-1)];
		WLLB[q]  = d_W4[Id(iy,ix-1,iz-1)];
		WLB[q]   = d_W4[Id(iy,ix,iz-1)];
		WRB[q]   = d_W4[Id(iy,ix+1,iz-1)];
		WRRB[q]  = d_W4[Id(iy,ix+2,iz-1)];
		WRRRB[q] = d_W4[Id(iy,ix+3,iz-1)];		
		
		d_ApplyWENO(WLLL, WLL, WL, WR, WRR, WRRR, Wl, Wr);
		d_ApplyWENO(WLLLN, WLLN, WLN, WRN, WRRN, WRRRN, WlN, WrN);
		d_ApplyWENO(WLLLS, WLLS, WLS, WRS, WRRS, WRRRS, WlS, WrS);
		d_ApplyWENO(WLLLF, WLLF, WLF, WRF, WRRF, WRRRF, WlF, WrF);
		d_ApplyWENO(WLLLB, WLLB, WLB, WRB, WRRB, WRRRB, WlB, WrB);
		
		d_Wl0[Id(iy,ix,iz)] = Wl[0];
		d_Wl1[Id(iy,ix,iz)] = Wl[1];
		d_Wl2[Id(iy,ix,iz)] = Wl[2];
		d_Wl3[Id(iy,ix,iz)] = Wl[3];
		d_Wl4[Id(iy,ix,iz)] = Wl[4];
		
		d_Wr0[Id(iy,ix,iz)] = Wr[0];
		d_Wr1[Id(iy,ix,iz)] = Wr[1];
		d_Wr2[Id(iy,ix,iz)] = Wr[2];
		d_Wr3[Id(iy,ix,iz)] = Wr[3];
		d_Wr4[Id(iy,ix,iz)] = Wr[4];
		
		d_DW0xl[Id(iy,ix,iz)] = 2*(Wl[0] - WL[0]) / d_dx;
		d_DW1xl[Id(iy,ix,iz)] = 2*(Wl[1] - WL[1]) / d_dx;
		d_DW2xl[Id(iy,ix,iz)] = 2*(Wl[2] - WL[2]) / d_dx;
		d_DW3xl[Id(iy,ix,iz)] = 2*(Wl[3] - WL[3]) / d_dx;
		d_DW4xl[Id(iy,ix,iz)] = 2*(Wl[4] - WL[4]) / d_dx;
	
		d_DW0xr[Id(iy,ix,iz)] = 2*(WR[0] - Wr[0]) / d_dx;
		d_DW1xr[Id(iy,ix,iz)] = 2*(WR[1] - Wr[1]) / d_dx;
		d_DW2xr[Id(iy,ix,iz)] = 2*(WR[2] - Wr[2]) / d_dx;
		d_DW3xr[Id(iy,ix,iz)] = 2*(WR[3] - Wr[3]) / d_dx;
		d_DW4xr[Id(iy,ix,iz)] = 2*(WR[4] - Wr[4]) / d_dx;
		
		d_DW0yl[Id(iy,ix,iz)] = 0.5*(WlN[0] - WlS[0]) / d_dx;
		d_DW1yl[Id(iy,ix,iz)] = 0.5*(WlN[1] - WlS[1]) / d_dx;
		d_DW2yl[Id(iy,ix,iz)] = 0.5*(WlN[2] - WlS[2]) / d_dx;
		d_DW3yl[Id(iy,ix,iz)] = 0.5*(WlN[3] - WlS[3]) / d_dx;
		d_DW4yl[Id(iy,ix,iz)] = 0.5*(WlN[4] - WlS[4]) / d_dx;
		
		d_DW0yr[Id(iy,ix,iz)] = 0.5*(WrN[0] - WrS[0]) / d_dx;
		d_DW1yr[Id(iy,ix,iz)] = 0.5*(WrN[1] - WrS[1]) / d_dx;
		d_DW2yr[Id(iy,ix,iz)] = 0.5*(WrN[2] - WrS[2]) / d_dx;
		d_DW3yr[Id(iy,ix,iz)] = 0.5*(WrN[3] - WrS[3]) / d_dx;
		d_DW4yr[Id(iy,ix,iz)] = 0.5*(WrN[4] - WrS[4]) / d_dx;

		d_DW0zl[Id(iy,ix,iz)] = 0.5*(WlF[0] - WlB[0]) / d_dx;
		d_DW1zl[Id(iy,ix,iz)] = 0.5*(WlF[1] - WlB[1]) / d_dx;
		d_DW2zl[Id(iy,ix,iz)] = 0.5*(WlF[2] - WlB[2]) / d_dx;
		d_DW3zl[Id(iy,ix,iz)] = 0.5*(WlF[3] - WlB[3]) / d_dx;
		d_DW4zl[Id(iy,ix,iz)] = 0.5*(WlF[4] - WlB[4]) / d_dx;
		
		d_DW0zr[Id(iy,ix,iz)] = 0.5*(WrF[0] - WrB[0]) / d_dx;
		d_DW1zr[Id(iy,ix,iz)] = 0.5*(WrF[1] - WrB[1]) / d_dx;
		d_DW2zr[Id(iy,ix,iz)] = 0.5*(WrF[2] - WrB[2]) / d_dx;
		d_DW3zr[Id(iy,ix,iz)] = 0.5*(WrF[3] - WrB[3]) / d_dx;
		d_DW4zr[Id(iy,ix,iz)] = 0.5*(WrF[4] - WrB[4]) / d_dx;
	}	
}


/**-----------------------------------------------------------------------------------------*
 * Global Function : Calculates Derivatives required for calculation of flux in Y-direction
 *------------------------------------------------------------------------------------------*/

__global__ void derivsY_Kernel(ptype *d_W0, ptype *d_Wl0, ptype *d_Wr0, ptype *d_DW0xl, ptype *d_DW0yl, ptype *d_DW0zl, ptype *d_DW0xr, ptype *d_DW0yr, ptype *d_DW0zr,
							   ptype *d_W1, ptype *d_Wl1, ptype *d_Wr1, ptype *d_DW1xl, ptype *d_DW1yl, ptype *d_DW1zl, ptype *d_DW1xr, ptype *d_DW1yr, ptype *d_DW1zr,
							   ptype *d_W2, ptype *d_Wl2, ptype *d_Wr2, ptype *d_DW2xl, ptype *d_DW2yl, ptype *d_DW2zl, ptype *d_DW2xr, ptype *d_DW2yr, ptype *d_DW2zr,
							   ptype *d_W3, ptype *d_Wl3, ptype *d_Wr3, ptype *d_DW3xl, ptype *d_DW3yl, ptype *d_DW3zl, ptype *d_DW3xr, ptype *d_DW3yr, ptype *d_DW3zr,
							   ptype *d_W4, ptype *d_Wl4, ptype *d_Wr4, ptype *d_DW4xl, ptype *d_DW4yl, ptype *d_DW4zl, ptype *d_DW4xr, ptype *d_DW4yr, ptype *d_DW4zr)
{
	int I = blockIdx.x*(blockDim.x) + threadIdx.x;

	int iy =  I/(d_nt_segx*d_nt_segz);
	int ix =  (I%(d_nt_segx*d_nt_segz))/d_nt_segz;
	int iz =  I%d_nt_segz;

	ptype WLLL[5], WLL[5], WL[5], WR[5], WRR[5], WRRR[5];
	ptype WLLLN[5], WLLN[5], WLN[5], WRN[5], WRRN[5], WRRRN[5];
	ptype WLLLS[5], WLLS[5], WLS[5], WRS[5], WRRS[5], WRRRS[5];
	ptype WLLLF[5], WLLF[5], WLF[5], WRF[5], WRRF[5], WRRRF[5];
	ptype WLLLB[5], WLLB[5], WLB[5], WRB[5], WRRB[5], WRRRB[5];
	ptype Wl[5], Wr[5], WlN[5], WrN[5], WlS[5], WrS[5], WlF[5], WrF[5], WlB[5], WrB[5];
	int q = 0;	
			
	if(ix < d_nt_segx-3 && iy < d_nt_segy-3 && iz < d_nt_segz-3 && iy > 1 && iz > 2 && ix > 2)
	{	
		q = 0;
		WLLL[q] = d_W0[Id(iy-2,ix,iz)];
		WLL[q]  = d_W0[Id(iy-1,ix,iz)];
		WL[q]   = d_W0[Id(iy,ix,iz)];
		WR[q]   = d_W0[Id(iy+1,ix,iz)];
		WRR[q]  = d_W0[Id(iy+2,ix,iz)];
		WRRR[q] = d_W0[Id(iy+3,ix,iz)];
		
		WLLLN[q] = d_W0[Id(iy-2,ix-1,iz)];
		WLLN[q]  = d_W0[Id(iy-1,ix-1,iz)];
		WLN[q]   = d_W0[Id(iy,ix-1,iz)];
		WRN[q]   = d_W0[Id(iy+1,ix-1,iz)];
		WRRN[q]  = d_W0[Id(iy+2,ix-1,iz)];
		WRRRN[q] = d_W0[Id(iy+3,ix-1,iz)];
		
		WLLLS[q] = d_W0[Id(iy-2,ix+1,iz)];
		WLLS[q]  = d_W0[Id(iy-1,ix+1,iz)];
		WLS[q]   = d_W0[Id(iy,ix+1,iz)];
		WRS[q]   = d_W0[Id(iy+1,ix+1,iz)];
		WRRS[q]  = d_W0[Id(iy+2,ix+1,iz)];
		WRRRS[q] = d_W0[Id(iy+3,ix+1,iz)];

		WLLLF[q] = d_W0[Id(iy-2,ix,iz+1)];
		WLLF[q]  = d_W0[Id(iy-1,ix,iz+1)];
		WLF[q]   = d_W0[Id(iy,ix,iz+1)];
		WRF[q]   = d_W0[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W0[Id(iy+2,ix,iz+1)];
		WRRRF[q] = d_W0[Id(iy+3,ix,iz+1)];

		WLLLB[q] = d_W0[Id(iy-2,ix,iz-1)];
		WLLB[q]  = d_W0[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W0[Id(iy,ix,iz-1)];
		WRB[q]   = d_W0[Id(iy+1,ix,iz-1)];
		WRRB[q]  = d_W0[Id(iy+2,ix,iz-1)];
		WRRRB[q] = d_W0[Id(iy+3,ix,iz-1)];
////////////////////////////////////////////
		q = 1;
		WLLL[q] = d_W1[Id(iy-2,ix,iz)];
		WLL[q]  = d_W1[Id(iy-1,ix,iz)];
		WL[q]   = d_W1[Id(iy,ix,iz)];
		WR[q]   = d_W1[Id(iy+1,ix,iz)];
		WRR[q]  = d_W1[Id(iy+2,ix,iz)];
		WRRR[q] = d_W1[Id(iy+3,ix,iz)];
		
		WLLLN[q] = d_W1[Id(iy-2,ix-1,iz)];
		WLLN[q]  = d_W1[Id(iy-1,ix-1,iz)];
		WLN[q]   = d_W1[Id(iy,ix-1,iz)];
		WRN[q]   = d_W1[Id(iy+1,ix-1,iz)];
		WRRN[q]  = d_W1[Id(iy+2,ix-1,iz)];
		WRRRN[q] = d_W1[Id(iy+3,ix-1,iz)];
		
		WLLLS[q] = d_W1[Id(iy-2,ix+1,iz)];
		WLLS[q]  = d_W1[Id(iy-1,ix+1,iz)];
		WLS[q]   = d_W1[Id(iy,ix+1,iz)];
		WRS[q]   = d_W1[Id(iy+1,ix+1,iz)];
		WRRS[q]  = d_W1[Id(iy+2,ix+1,iz)];
		WRRRS[q] = d_W1[Id(iy+3,ix+1,iz)];

		WLLLF[q] = d_W1[Id(iy-2,ix,iz+1)];
		WLLF[q]  = d_W1[Id(iy-1,ix,iz+1)];
		WLF[q]   = d_W1[Id(iy,ix,iz+1)];
		WRF[q]   = d_W1[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W1[Id(iy+2,ix,iz+1)];
		WRRRF[q] = d_W1[Id(iy+3,ix,iz+1)];

		WLLLB[q] = d_W1[Id(iy-2,ix,iz-1)];
		WLLB[q]  = d_W1[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W1[Id(iy,ix,iz-1)];
		WRB[q]   = d_W1[Id(iy+1,ix,iz-1)];
		WRRB[q]  = d_W1[Id(iy+2,ix,iz-1)];
		WRRRB[q] = d_W1[Id(iy+3,ix,iz-1)];
////////////////////////////////////////////
		q = 2;
		WLLL[q] = d_W2[Id(iy-2,ix,iz)];
		WLL[q]  = d_W2[Id(iy-1,ix,iz)];
		WL[q]   = d_W2[Id(iy,ix,iz)];
		WR[q]   = d_W2[Id(iy+1,ix,iz)];
		WRR[q]  = d_W2[Id(iy+2,ix,iz)];
		WRRR[q] = d_W2[Id(iy+3,ix,iz)];
		
		WLLLN[q] = d_W2[Id(iy-2,ix-1,iz)];
		WLLN[q]  = d_W2[Id(iy-1,ix-1,iz)];
		WLN[q]   = d_W2[Id(iy,ix-1,iz)];
		WRN[q]   = d_W2[Id(iy+1,ix-1,iz)];
		WRRN[q]  = d_W2[Id(iy+2,ix-1,iz)];
		WRRRN[q] = d_W2[Id(iy+3,ix-1,iz)];
		
		WLLLS[q] = d_W2[Id(iy-2,ix+1,iz)];
		WLLS[q]  = d_W2[Id(iy-1,ix+1,iz)];
		WLS[q]   = d_W2[Id(iy,ix+1,iz)];
		WRS[q]   = d_W2[Id(iy+1,ix+1,iz)];
		WRRS[q]  = d_W2[Id(iy+2,ix+1,iz)];
		WRRRS[q] = d_W2[Id(iy+3,ix+1,iz)];

		WLLLF[q] = d_W2[Id(iy-2,ix,iz+1)];
		WLLF[q]  = d_W2[Id(iy-1,ix,iz+1)];
		WLF[q]   = d_W2[Id(iy,ix,iz+1)];
		WRF[q]   = d_W2[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W2[Id(iy+2,ix,iz+1)];
		WRRRF[q] = d_W2[Id(iy+3,ix,iz+1)];

		WLLLB[q] = d_W2[Id(iy-2,ix,iz-1)];
		WLLB[q]  = d_W2[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W2[Id(iy,ix,iz-1)];
		WRB[q]   = d_W2[Id(iy+1,ix,iz-1)];
		WRRB[q]  = d_W2[Id(iy+2,ix,iz-1)];
		WRRRB[q] = d_W2[Id(iy+3,ix,iz-1)];
////////////////////////////////////////////
		q = 3;
		WLLL[q] = d_W3[Id(iy-2,ix,iz)];
		WLL[q]  = d_W3[Id(iy-1,ix,iz)];
		WL[q]   = d_W3[Id(iy,ix,iz)];
		WR[q]   = d_W3[Id(iy+1,ix,iz)];
		WRR[q]  = d_W3[Id(iy+2,ix,iz)];
		WRRR[q] = d_W3[Id(iy+3,ix,iz)];
		
		WLLLN[q] = d_W3[Id(iy-2,ix-1,iz)];
		WLLN[q]  = d_W3[Id(iy-1,ix-1,iz)];
		WLN[q]   = d_W3[Id(iy,ix-1,iz)];
		WRN[q]   = d_W3[Id(iy+1,ix-1,iz)];
		WRRN[q]  = d_W3[Id(iy+2,ix-1,iz)];
		WRRRN[q] = d_W3[Id(iy+3,ix-1,iz)];
		
		WLLLS[q] = d_W3[Id(iy-2,ix+1,iz)];
		WLLS[q]  = d_W3[Id(iy-1,ix+1,iz)];
		WLS[q]   = d_W3[Id(iy,ix+1,iz)];
		WRS[q]   = d_W3[Id(iy+1,ix+1,iz)];
		WRRS[q]  = d_W3[Id(iy+2,ix+1,iz)];
		WRRRS[q] = d_W3[Id(iy+3,ix+1,iz)];

		WLLLF[q] = d_W3[Id(iy-2,ix,iz+1)];
		WLLF[q]  = d_W3[Id(iy-1,ix,iz+1)];
		WLF[q]   = d_W3[Id(iy,ix,iz+1)];
		WRF[q]   = d_W3[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W3[Id(iy+2,ix,iz+1)];
		WRRRF[q] = d_W3[Id(iy+3,ix,iz+1)];

		WLLLB[q] = d_W3[Id(iy-2,ix,iz-1)];
		WLLB[q]  = d_W3[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W3[Id(iy,ix,iz-1)];
		WRB[q]   = d_W3[Id(iy+1,ix,iz-1)];
		WRRB[q]  = d_W3[Id(iy+2,ix,iz-1)];
		WRRRB[q] = d_W3[Id(iy+3,ix,iz-1)];
////////////////////////////////////////////
		q = 4;
		WLLL[q] = d_W4[Id(iy-2,ix,iz)];
		WLL[q]  = d_W4[Id(iy-1,ix,iz)];
		WL[q]   = d_W4[Id(iy,ix,iz)];
		WR[q]   = d_W4[Id(iy+1,ix,iz)];
		WRR[q]  = d_W4[Id(iy+2,ix,iz)];
		WRRR[q] = d_W4[Id(iy+3,ix,iz)];
		
		WLLLN[q] = d_W4[Id(iy-2,ix-1,iz)];
		WLLN[q]  = d_W4[Id(iy-1,ix-1,iz)];
		WLN[q]   = d_W4[Id(iy,ix-1,iz)];
		WRN[q]   = d_W4[Id(iy+1,ix-1,iz)];
		WRRN[q]  = d_W4[Id(iy+2,ix-1,iz)];
		WRRRN[q] = d_W4[Id(iy+3,ix-1,iz)];
		
		WLLLS[q] = d_W4[Id(iy-2,ix+1,iz)];
		WLLS[q]  = d_W4[Id(iy-1,ix+1,iz)];
		WLS[q]   = d_W4[Id(iy,ix+1,iz)];
		WRS[q]   = d_W4[Id(iy+1,ix+1,iz)];
		WRRS[q]  = d_W4[Id(iy+2,ix+1,iz)];
		WRRRS[q] = d_W4[Id(iy+3,ix+1,iz)];

		WLLLF[q] = d_W4[Id(iy-2,ix,iz+1)];
		WLLF[q]  = d_W4[Id(iy-1,ix,iz+1)];
		WLF[q]   = d_W4[Id(iy,ix,iz+1)];
		WRF[q]   = d_W4[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W4[Id(iy+2,ix,iz+1)];
		WRRRF[q] = d_W4[Id(iy+3,ix,iz+1)];

		WLLLB[q] = d_W4[Id(iy-2,ix,iz-1)];
		WLLB[q]  = d_W4[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W4[Id(iy,ix,iz-1)];
		WRB[q]   = d_W4[Id(iy+1,ix,iz-1)];
		WRRB[q]  = d_W4[Id(iy+2,ix,iz-1)];
		WRRRB[q] = d_W4[Id(iy+3,ix,iz-1)];
		
////////////////////////////////////////////

		d_ApplyWENO(WLLL, WLL, WL, WR, WRR, WRRR, Wl, Wr);
		d_ApplyWENO(WLLLN, WLLN, WLN, WRN, WRRN, WRRRN, WlN, WrN);
		d_ApplyWENO(WLLLS, WLLS, WLS, WRS, WRRS, WRRRS, WlS, WrS);
		d_ApplyWENO(WLLLF, WLLF, WLF, WRF, WRRF, WRRRF, WlF, WrF);
		d_ApplyWENO(WLLLB, WLLB, WLB, WRB, WRRB, WRRRB, WlB, WrB);
		
		d_Wl0[Id(iy,ix,iz)] = Wl[0];
		d_Wl1[Id(iy,ix,iz)] = Wl[1];
		d_Wl2[Id(iy,ix,iz)] = Wl[2];
		d_Wl3[Id(iy,ix,iz)] = Wl[3];
		d_Wl4[Id(iy,ix,iz)] = Wl[4];
		
		d_Wr0[Id(iy,ix,iz)] = Wr[0];
		d_Wr1[Id(iy,ix,iz)] = Wr[1];
		d_Wr2[Id(iy,ix,iz)] = Wr[2];
		d_Wr3[Id(iy,ix,iz)] = Wr[3];
		d_Wr4[Id(iy,ix,iz)] = Wr[4];
		
		d_DW0xl[Id(iy,ix,iz)] = 2*(Wl[0] - WL[0]) / d_dx;
		d_DW1xl[Id(iy,ix,iz)] = 2*(Wl[1] - WL[1]) / d_dx;
		d_DW2xl[Id(iy,ix,iz)] = 2*(Wl[2] - WL[2]) / d_dx;
		d_DW3xl[Id(iy,ix,iz)] = 2*(Wl[3] - WL[3]) / d_dx;
		d_DW4xl[Id(iy,ix,iz)] = 2*(Wl[4] - WL[4]) / d_dx;
	
		d_DW0xr[Id(iy,ix,iz)] = 2*(WR[0] - Wr[0]) / d_dx;
		d_DW1xr[Id(iy,ix,iz)] = 2*(WR[1] - Wr[1]) / d_dx;
		d_DW2xr[Id(iy,ix,iz)] = 2*(WR[2] - Wr[2]) / d_dx;
		d_DW3xr[Id(iy,ix,iz)] = 2*(WR[3] - Wr[3]) / d_dx;
		d_DW4xr[Id(iy,ix,iz)] = 2*(WR[4] - Wr[4]) / d_dx;
		
		d_DW0yl[Id(iy,ix,iz)] = -0.5*(WlN[0] - WlS[0]) / d_dx;
		d_DW1yl[Id(iy,ix,iz)] = -0.5*(WlN[1] - WlS[1]) / d_dx;
		d_DW2yl[Id(iy,ix,iz)] = -0.5*(WlN[2] - WlS[2]) / d_dx;
		d_DW3yl[Id(iy,ix,iz)] = -0.5*(WlN[3] - WlS[3]) / d_dx;
		d_DW4yl[Id(iy,ix,iz)] = -0.5*(WlN[4] - WlS[4]) / d_dx;
		
		d_DW0yr[Id(iy,ix,iz)] = -0.5*(WrN[0] - WrS[0]) / d_dx;
		d_DW1yr[Id(iy,ix,iz)] = -0.5*(WrN[1] - WrS[1]) / d_dx;
		d_DW2yr[Id(iy,ix,iz)] = -0.5*(WrN[2] - WrS[2]) / d_dx;
		d_DW3yr[Id(iy,ix,iz)] = -0.5*(WrN[3] - WrS[3]) / d_dx;
		d_DW4yr[Id(iy,ix,iz)] = -0.5*(WrN[4] - WrS[4]) / d_dx;

		d_DW0zl[Id(iy,ix,iz)] = 0.5*(WlF[0] - WlB[0]) / d_dx;
		d_DW1zl[Id(iy,ix,iz)] = 0.5*(WlF[1] - WlB[1]) / d_dx;
		d_DW2zl[Id(iy,ix,iz)] = 0.5*(WlF[2] - WlB[2]) / d_dx;
		d_DW3zl[Id(iy,ix,iz)] = 0.5*(WlF[3] - WlB[3]) / d_dx;
		d_DW4zl[Id(iy,ix,iz)] = 0.5*(WlF[4] - WlB[4]) / d_dx;
		
		d_DW0zr[Id(iy,ix,iz)] = 0.5*(WrF[0] - WrB[0]) / d_dx;
		d_DW1zr[Id(iy,ix,iz)] = 0.5*(WrF[1] - WrB[1]) / d_dx;
		d_DW2zr[Id(iy,ix,iz)] = 0.5*(WrF[2] - WrB[2]) / d_dx;
		d_DW3zr[Id(iy,ix,iz)] = 0.5*(WrF[3] - WrB[3]) / d_dx;
		d_DW4zr[Id(iy,ix,iz)] = 0.5*(WrF[4] - WrB[4]) / d_dx;
	}	
}	


/**-----------------------------------------------------------------------------------------*
 * Global Function : Calculates Derivatives required for calculation of flux in Z-direction
 *------------------------------------------------------------------------------------------*/
__global__ void derivsZ_Kernel(ptype *d_W0, ptype *d_Wl0, ptype *d_Wr0, ptype *d_DW0xl, ptype *d_DW0yl, ptype *d_DW0zl, ptype *d_DW0xr, ptype *d_DW0yr, ptype *d_DW0zr,
							   ptype *d_W1, ptype *d_Wl1, ptype *d_Wr1, ptype *d_DW1xl, ptype *d_DW1yl, ptype *d_DW1zl, ptype *d_DW1xr, ptype *d_DW1yr, ptype *d_DW1zr,
							   ptype *d_W2, ptype *d_Wl2, ptype *d_Wr2, ptype *d_DW2xl, ptype *d_DW2yl, ptype *d_DW2zl, ptype *d_DW2xr, ptype *d_DW2yr, ptype *d_DW2zr,
							   ptype *d_W3, ptype *d_Wl3, ptype *d_Wr3, ptype *d_DW3xl, ptype *d_DW3yl, ptype *d_DW3zl, ptype *d_DW3xr, ptype *d_DW3yr, ptype *d_DW3zr,
							   ptype *d_W4, ptype *d_Wl4, ptype *d_Wr4, ptype *d_DW4xl, ptype *d_DW4yl, ptype *d_DW4zl, ptype *d_DW4xr, ptype *d_DW4yr, ptype *d_DW4zr)
{
	int I = blockIdx.x*(blockDim.x) + threadIdx.x;

	int iy =  I/(d_nt_segx*d_nt_segz);
	int ix =  (I%(d_nt_segx*d_nt_segz))/d_nt_segz;
	int iz =  I%d_nt_segz;

	ptype WLLL[5], WLL[5], WL[5], WR[5], WRR[5], WRRR[5];
	ptype WLLLN[5], WLLN[5], WLN[5], WRN[5], WRRN[5], WRRRN[5];
	ptype WLLLS[5], WLLS[5], WLS[5], WRS[5], WRRS[5], WRRRS[5];
	ptype WLLLF[5], WLLF[5], WLF[5], WRF[5], WRRF[5], WRRRF[5];
	ptype WLLLB[5], WLLB[5], WLB[5], WRB[5], WRRB[5], WRRRB[5];
	ptype Wl[5], Wr[5], WlN[5], WrN[5], WlS[5], WrS[5], WlF[5], WrF[5], WlB[5], WrB[5];
	int q = 0;	
			
	if(ix < d_nt_segx-3 && iy < d_nt_segy-3 && iz < d_nt_segz-3 && iy > 2 && iz > 1 && ix > 2)
	{	
		q = 0;
		WLLL[q] = d_W0[Id(iy,ix,iz-2)];
		WLL[q]  = d_W0[Id(iy,ix,iz-1)];
		WL[q]   = d_W0[Id(iy,ix,iz)];
		WR[q]   = d_W0[Id(iy,ix,iz+1)];
		WRR[q]  = d_W0[Id(iy,ix,iz+2)];
		WRRR[q] = d_W0[Id(iy,ix,iz+3)];
		
		WLLLN[q] = d_W0[Id(iy,ix+1,iz-2)];
		WLLN[q]  = d_W0[Id(iy,ix+1,iz-1)];
		WLN[q]   = d_W0[Id(iy,ix+1,iz)];
		WRN[q]   = d_W0[Id(iy,ix+1,iz+1)];
		WRRN[q]  = d_W0[Id(iy,ix+1,iz+2)];
		WRRRN[q] = d_W0[Id(iy,ix+1,iz+3)];
		
		WLLLS[q] = d_W0[Id(iy,ix-1,iz-2)];
		WLLS[q]  = d_W0[Id(iy,ix-1,iz-1)];
		WLS[q]   = d_W0[Id(iy,ix-1,iz)];
		WRS[q]   = d_W0[Id(iy,ix-1,iz+1)];
		WRRS[q]  = d_W0[Id(iy,ix-1,iz+2)];
		WRRRS[q] = d_W0[Id(iy,ix-1,iz+3)];

		WLLLF[q] = d_W0[Id(iy+1,ix,iz-2)];
		WLLF[q]  = d_W0[Id(iy+1,ix,iz-1)];
		WLF[q]   = d_W0[Id(iy+1,ix,iz)];
		WRF[q]   = d_W0[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W0[Id(iy+1,ix,iz+2)];
		WRRRF[q] = d_W0[Id(iy+1,ix,iz+3)];

		WLLLB[q] = d_W0[Id(iy-1,ix,iz-2)];
		WLLB[q]  = d_W0[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W0[Id(iy-1,ix,iz)];
		WRB[q]   = d_W0[Id(iy-1,ix,iz+1)];
		WRRB[q]  = d_W0[Id(iy-1,ix,iz+2)];
		WRRRB[q] = d_W0[Id(iy-1,ix,iz+3)];
///////////////////////////////////////////////////////
		q = 1;
		WLLL[q] = d_W1[Id(iy,ix,iz-2)];
		WLL[q]  = d_W1[Id(iy,ix,iz-1)];
		WL[q]   = d_W1[Id(iy,ix,iz)];
		WR[q]   = d_W1[Id(iy,ix,iz+1)];
		WRR[q]  = d_W1[Id(iy,ix,iz+2)];
		WRRR[q] = d_W1[Id(iy,ix,iz+3)];
		
		WLLLN[q] = d_W1[Id(iy,ix+1,iz-2)];
		WLLN[q]  = d_W1[Id(iy,ix+1,iz-1)];
		WLN[q]   = d_W1[Id(iy,ix+1,iz)];
		WRN[q]   = d_W1[Id(iy,ix+1,iz+1)];
		WRRN[q]  = d_W1[Id(iy,ix+1,iz+2)];
		WRRRN[q] = d_W1[Id(iy,ix+1,iz+3)];
		
		WLLLS[q] = d_W1[Id(iy,ix-1,iz-2)];
		WLLS[q]  = d_W1[Id(iy,ix-1,iz-1)];
		WLS[q]   = d_W1[Id(iy,ix-1,iz)];
		WRS[q]   = d_W1[Id(iy,ix-1,iz+1)];
		WRRS[q]  = d_W1[Id(iy,ix-1,iz+2)];
		WRRRS[q] = d_W1[Id(iy,ix-1,iz+3)];

		WLLLF[q] = d_W1[Id(iy+1,ix,iz-2)];
		WLLF[q]  = d_W1[Id(iy+1,ix,iz-1)];
		WLF[q]   = d_W1[Id(iy+1,ix,iz)];
		WRF[q]   = d_W1[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W1[Id(iy+1,ix,iz+2)];
		WRRRF[q] = d_W1[Id(iy+1,ix,iz+3)];

		WLLLB[q] = d_W1[Id(iy-1,ix,iz-2)];
		WLLB[q]  = d_W1[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W1[Id(iy-1,ix,iz)];
		WRB[q]   = d_W1[Id(iy-1,ix,iz+1)];
		WRRB[q]  = d_W1[Id(iy-1,ix,iz+2)];
		WRRRB[q] = d_W1[Id(iy-1,ix,iz+3)];
///////////////////////////////////////////////////////
		q = 2;
		WLLL[q] = d_W2[Id(iy,ix,iz-2)];
		WLL[q]  = d_W2[Id(iy,ix,iz-1)];
		WL[q]   = d_W2[Id(iy,ix,iz)];
		WR[q]   = d_W2[Id(iy,ix,iz+1)];
		WRR[q]  = d_W2[Id(iy,ix,iz+2)];
		WRRR[q] = d_W2[Id(iy,ix,iz+3)];
		
		WLLLN[q] = d_W2[Id(iy,ix+1,iz-2)];
		WLLN[q]  = d_W2[Id(iy,ix+1,iz-1)];
		WLN[q]   = d_W2[Id(iy,ix+1,iz)];
		WRN[q]   = d_W2[Id(iy,ix+1,iz+1)];
		WRRN[q]  = d_W2[Id(iy,ix+1,iz+2)];
		WRRRN[q] = d_W2[Id(iy,ix+1,iz+3)];
		
		WLLLS[q] = d_W2[Id(iy,ix-1,iz-2)];
		WLLS[q]  = d_W2[Id(iy,ix-1,iz-1)];
		WLS[q]   = d_W2[Id(iy,ix-1,iz)];
		WRS[q]   = d_W2[Id(iy,ix-1,iz+1)];
		WRRS[q]  = d_W2[Id(iy,ix-1,iz+2)];
		WRRRS[q] = d_W2[Id(iy,ix-1,iz+3)];

		WLLLF[q] = d_W2[Id(iy+1,ix,iz-2)];
		WLLF[q]  = d_W2[Id(iy+1,ix,iz-1)];
		WLF[q]   = d_W2[Id(iy+1,ix,iz)];
		WRF[q]   = d_W2[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W2[Id(iy+1,ix,iz+2)];
		WRRRF[q] = d_W2[Id(iy+1,ix,iz+3)];

		WLLLB[q] = d_W2[Id(iy-1,ix,iz-2)];
		WLLB[q]  = d_W2[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W2[Id(iy-1,ix,iz)];
		WRB[q]   = d_W2[Id(iy-1,ix,iz+1)];
		WRRB[q]  = d_W2[Id(iy-1,ix,iz+2)];
		WRRRB[q] = d_W2[Id(iy-1,ix,iz+3)];
///////////////////////////////////////////////////////
		q = 3;
		WLLL[q] = d_W3[Id(iy,ix,iz-2)];
		WLL[q]  = d_W3[Id(iy,ix,iz-1)];
		WL[q]   = d_W3[Id(iy,ix,iz)];
		WR[q]   = d_W3[Id(iy,ix,iz+1)];
		WRR[q]  = d_W3[Id(iy,ix,iz+2)];
		WRRR[q] = d_W3[Id(iy,ix,iz+3)];
		
		WLLLN[q] = d_W3[Id(iy,ix+1,iz-2)];
		WLLN[q]  = d_W3[Id(iy,ix+1,iz-1)];
		WLN[q]   = d_W3[Id(iy,ix+1,iz)];
		WRN[q]   = d_W3[Id(iy,ix+1,iz+1)];
		WRRN[q]  = d_W3[Id(iy,ix+1,iz+2)];
		WRRRN[q] = d_W3[Id(iy,ix+1,iz+3)];
		
		WLLLS[q] = d_W3[Id(iy,ix-1,iz-2)];
		WLLS[q]  = d_W3[Id(iy,ix-1,iz-1)];
		WLS[q]   = d_W3[Id(iy,ix-1,iz)];
		WRS[q]   = d_W3[Id(iy,ix-1,iz+1)];
		WRRS[q]  = d_W3[Id(iy,ix-1,iz+2)];
		WRRRS[q] = d_W3[Id(iy,ix-1,iz+3)];

		WLLLF[q] = d_W3[Id(iy+1,ix,iz-2)];
		WLLF[q]  = d_W3[Id(iy+1,ix,iz-1)];
		WLF[q]   = d_W3[Id(iy+1,ix,iz)];
		WRF[q]   = d_W3[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W3[Id(iy+1,ix,iz+2)];
		WRRRF[q] = d_W3[Id(iy+1,ix,iz+3)];

		WLLLB[q] = d_W3[Id(iy-1,ix,iz-2)];
		WLLB[q]  = d_W3[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W3[Id(iy-1,ix,iz)];
		WRB[q]   = d_W3[Id(iy-1,ix,iz+1)];
		WRRB[q]  = d_W3[Id(iy-1,ix,iz+2)];
		WRRRB[q] = d_W3[Id(iy-1,ix,iz+3)];
///////////////////////////////////////////////////////
		q = 4;
		WLLL[q] = d_W4[Id(iy,ix,iz-2)];
		WLL[q]  = d_W4[Id(iy,ix,iz-1)];
		WL[q]   = d_W4[Id(iy,ix,iz)];
		WR[q]   = d_W4[Id(iy,ix,iz+1)];
		WRR[q]  = d_W4[Id(iy,ix,iz+2)];
		WRRR[q] = d_W4[Id(iy,ix,iz+3)];
		
		WLLLN[q] = d_W4[Id(iy,ix+1,iz-2)];
		WLLN[q]  = d_W4[Id(iy,ix+1,iz-1)];
		WLN[q]   = d_W4[Id(iy,ix+1,iz)];
		WRN[q]   = d_W4[Id(iy,ix+1,iz+1)];
		WRRN[q]  = d_W4[Id(iy,ix+1,iz+2)];
		WRRRN[q] = d_W4[Id(iy,ix+1,iz+3)];
		
		WLLLS[q] = d_W4[Id(iy,ix-1,iz-2)];
		WLLS[q]  = d_W4[Id(iy,ix-1,iz-1)];
		WLS[q]   = d_W4[Id(iy,ix-1,iz)];
		WRS[q]   = d_W4[Id(iy,ix-1,iz+1)];
		WRRS[q]  = d_W4[Id(iy,ix-1,iz+2)];
		WRRRS[q] = d_W4[Id(iy,ix-1,iz+3)];

		WLLLF[q] = d_W4[Id(iy+1,ix,iz-2)];
		WLLF[q]  = d_W4[Id(iy+1,ix,iz-1)];
		WLF[q]   = d_W4[Id(iy+1,ix,iz)];
		WRF[q]   = d_W4[Id(iy+1,ix,iz+1)];
		WRRF[q]  = d_W4[Id(iy+1,ix,iz+2)];
		WRRRF[q] = d_W4[Id(iy+1,ix,iz+3)];

		WLLLB[q] = d_W4[Id(iy-1,ix,iz-2)];
		WLLB[q]  = d_W4[Id(iy-1,ix,iz-1)];
		WLB[q]   = d_W4[Id(iy-1,ix,iz)];
		WRB[q]   = d_W4[Id(iy-1,ix,iz+1)];
		WRRB[q]  = d_W4[Id(iy-1,ix,iz+2)];
		WRRRB[q] = d_W4[Id(iy-1,ix,iz+3)];
 
///////////////////////////////////////////////////////

		d_ApplyWENO(WLLL, WLL, WL, WR, WRR, WRRR, Wl, Wr);
		d_ApplyWENO(WLLLN, WLLN, WLN, WRN, WRRN, WRRRN, WlN, WrN);
		d_ApplyWENO(WLLLS, WLLS, WLS, WRS, WRRS, WRRRS, WlS, WrS);
		d_ApplyWENO(WLLLF, WLLF, WLF, WRF, WRRF, WRRRF, WlF, WrF);
		d_ApplyWENO(WLLLB, WLLB, WLB, WRB, WRRB, WRRRB, WlB, WrB);
		
		d_Wl0[Id(iy,ix,iz)] = Wl[0];
		d_Wl1[Id(iy,ix,iz)] = Wl[1];
		d_Wl2[Id(iy,ix,iz)] = Wl[2];
		d_Wl3[Id(iy,ix,iz)] = Wl[3];
		d_Wl4[Id(iy,ix,iz)] = Wl[4];
		
		d_Wr0[Id(iy,ix,iz)] = Wr[0];
		d_Wr1[Id(iy,ix,iz)] = Wr[1];
		d_Wr2[Id(iy,ix,iz)] = Wr[2];
		d_Wr3[Id(iy,ix,iz)] = Wr[3];
		d_Wr4[Id(iy,ix,iz)] = Wr[4];
		
		d_DW0xl[Id(iy,ix,iz)] = 2*(Wl[0] - WL[0]) / d_dx;
		d_DW1xl[Id(iy,ix,iz)] = 2*(Wl[1] - WL[1]) / d_dx;
		d_DW2xl[Id(iy,ix,iz)] = 2*(Wl[2] - WL[2]) / d_dx;
		d_DW3xl[Id(iy,ix,iz)] = 2*(Wl[3] - WL[3]) / d_dx;
		d_DW4xl[Id(iy,ix,iz)] = 2*(Wl[4] - WL[4]) / d_dx;
	
		d_DW0xr[Id(iy,ix,iz)] = 2*(WR[0] - Wr[0]) / d_dx;
		d_DW1xr[Id(iy,ix,iz)] = 2*(WR[1] - Wr[1]) / d_dx;
		d_DW2xr[Id(iy,ix,iz)] = 2*(WR[2] - Wr[2]) / d_dx;
		d_DW3xr[Id(iy,ix,iz)] = 2*(WR[3] - Wr[3]) / d_dx;
		d_DW4xr[Id(iy,ix,iz)] = 2*(WR[4] - Wr[4]) / d_dx;
		
		d_DW0yl[Id(iy,ix,iz)] = 0.5*(WlN[0] - WlS[0]) / d_dx;
		d_DW1yl[Id(iy,ix,iz)] = 0.5*(WlN[1] - WlS[1]) / d_dx;
		d_DW2yl[Id(iy,ix,iz)] = 0.5*(WlN[2] - WlS[2]) / d_dx;
		d_DW3yl[Id(iy,ix,iz)] = 0.5*(WlN[3] - WlS[3]) / d_dx;
		d_DW4yl[Id(iy,ix,iz)] = 0.5*(WlN[4] - WlS[4]) / d_dx;
		
		d_DW0yr[Id(iy,ix,iz)] = 0.5*(WrN[0] - WrS[0]) / d_dx;
		d_DW1yr[Id(iy,ix,iz)] = 0.5*(WrN[1] - WrS[1]) / d_dx;
		d_DW2yr[Id(iy,ix,iz)] = 0.5*(WrN[2] - WrS[2]) / d_dx;
		d_DW3yr[Id(iy,ix,iz)] = 0.5*(WrN[3] - WrS[3]) / d_dx;
		d_DW4yr[Id(iy,ix,iz)] = 0.5*(WrN[4] - WrS[4]) / d_dx;

		d_DW0zl[Id(iy,ix,iz)] = 0.5*(WlF[0] - WlB[0]) / d_dx;
		d_DW1zl[Id(iy,ix,iz)] = 0.5*(WlF[1] - WlB[1]) / d_dx;
		d_DW2zl[Id(iy,ix,iz)] = 0.5*(WlF[2] - WlB[2]) / d_dx;
		d_DW3zl[Id(iy,ix,iz)] = 0.5*(WlF[3] - WlB[3]) / d_dx;
		d_DW4zl[Id(iy,ix,iz)] = 0.5*(WlF[4] - WlB[4]) / d_dx;
		
		d_DW0zr[Id(iy,ix,iz)] = 0.5*(WrF[0] - WrB[0]) / d_dx;
		d_DW1zr[Id(iy,ix,iz)] = 0.5*(WrF[1] - WrB[1]) / d_dx;
		d_DW2zr[Id(iy,ix,iz)] = 0.5*(WrF[2] - WrB[2]) / d_dx;
		d_DW3zr[Id(iy,ix,iz)] = 0.5*(WrF[3] - WrB[3]) / d_dx;
		d_DW4zr[Id(iy,ix,iz)] = 0.5*(WrF[4] - WrB[4]) / d_dx;
	}
}

/**-------------------------------------------------------------------------------------------------*
 * Global Function : Flips the momentum and derivatives [for flux calculation in Y and Z direction]
 *--------------------------------------------------------------------------------------------------*/
__global__ void flip_Kernel(ptype *d_Wl1, ptype *d_Wl2, ptype *d_DW1xl, ptype *d_DW2xl, ptype *d_DW1yl, ptype *d_DW2yl, ptype *d_DW1zl, ptype *d_DW2zl,
                            ptype *d_Wr1, ptype *d_Wr2, ptype *d_DW1xr, ptype *d_DW2xr, ptype *d_DW1yr, ptype *d_DW2yr, ptype *d_DW1zr, ptype *d_DW2zr)

{
	int I = blockIdx.x*(blockDim.x) + threadIdx.x;

	int iy =  I/(d_nt_segx*d_nt_segz);
	int ix =  (I%(d_nt_segx*d_nt_segz))/d_nt_segz;
	int iz =  I%d_nt_segz;
	
	ptype temp;
			
	if(ix < d_nt_segx && iy < d_nt_segy && iz < d_nt_segz) {
		
		temp = d_Wl1[I];   d_Wl1[I] = d_Wl2[I];     d_Wl2[I] = temp;	
		temp = d_Wr1[I];   d_Wr1[I] = d_Wr2[I];     d_Wr2[I] = temp;	
		temp = d_DW1xl[I]; d_DW1xl[I] = d_DW2xl[I]; d_DW2xl[I] = temp;	
		temp = d_DW1yl[I]; d_DW1yl[I] = d_DW2yl[I]; d_DW2yl[I] = temp;	
		temp = d_DW1zl[I]; d_DW1zl[I] = d_DW2zl[I]; d_DW2zl[I] = temp;	
		temp = d_DW1xr[I]; d_DW1xr[I] = d_DW2xr[I]; d_DW2xr[I] = temp;	
		temp = d_DW1yr[I]; d_DW1yr[I] = d_DW2yr[I]; d_DW2yr[I] = temp;	
		temp = d_DW1zr[I]; d_DW1zr[I] = d_DW2zr[I]; d_DW2zr[I] = temp;	
	}	
}

/**-----------------------------------------------------------------------------------------*
 * Global Function : Flips back the Flux
 *------------------------------------------------------------------------------------------*/
__global__ void flipBack_Kernel(ptype *d_F1, ptype *d_F2)
{
	int I = blockIdx.x*(blockDim.x) + threadIdx.x;

	int iy =  I/(d_nt_segx*d_nt_segz);
	int ix =  (I%(d_nt_segx*d_nt_segz))/d_nt_segz;
	int iz =  I%d_nt_segz;
    
    ptype temp;
	
	if(ix < d_nt_segx && iy < d_nt_segy && iz < d_nt_segz) {
		temp = d_F1[I]; d_F1[I] = d_F2[I]; d_F2[I] = temp;
	}
}

/**-----------------------------------------------------------------------------------------*
 * Global Function : Updates the flow field in every iteration
 *------------------------------------------------------------------------------------------*/

__global__ void update_Kernel (ptype *d_W, ptype *d_Fx, ptype *d_Fy, ptype *d_Fz)
{
	int I = blockIdx.x*blockDim.x + threadIdx.x;

	int iy =  I/(d_nt_segx*d_nt_segz);
	int ix =  (I%(d_nt_segx*d_nt_segz))/d_nt_segz;
	int iz =  I%d_nt_segz;

	if(ix < d_nt_segx-3 && iy < d_nt_segy-3 && iz < d_nt_segz-3 && ix > 2 && iy > 2 && iz > 2)
	{
		
		d_W[I] = d_W[I] - (1/d_dx) * ( d_Fx[Id(iy, ix, iz)] - d_Fx[Id(iy, ix-1, iz)] +
                                       d_Fy[Id(iy, ix, iz)] - d_Fy[Id(iy-1, ix, iz)] +
									   d_Fz[Id(iy, ix, iz)] - d_Fz[Id(iy, ix, iz-1)] );		
	}
}

/**-----------------------------------------------------------------------------------------*
 * Global Function : Calculates Flux
 *------------------------------------------------------------------------------------------*/
__global__ void flux(ptype *d_W0, ptype *d_W1, ptype *d_W2, ptype *d_W3, ptype *d_W4,
					 ptype *d_Wl0, ptype *d_Wl1, ptype *d_Wl2, ptype *d_Wl3, ptype *d_Wl4,
					 ptype *d_Wr0, ptype *d_Wr1, ptype *d_Wr2, ptype *d_Wr3, ptype *d_Wr4,
					 ptype *d_DW0xl, ptype *d_DW1xl, ptype *d_DW2xl, ptype *d_DW3xl, ptype *d_DW4xl,
					 ptype *d_DW0xr, ptype *d_DW1xr, ptype *d_DW2xr, ptype *d_DW3xr, ptype *d_DW4xr,
					 ptype *d_DW0yl, ptype *d_DW1yl, ptype *d_DW2yl, ptype *d_DW3yl, ptype *d_DW4yl,
					 ptype *d_DW0yr, ptype *d_DW1yr, ptype *d_DW2yr, ptype *d_DW3yr, ptype *d_DW4yr,
					 ptype *d_DW0zl, ptype *d_DW1zl, ptype *d_DW2zl, ptype *d_DW3zl, ptype *d_DW4zl,
					 ptype *d_DW0zr, ptype *d_DW1zr, ptype *d_DW2zr, ptype *d_DW3zr, ptype *d_DW4zr,
					 ptype *d_F0, ptype *d_F1, ptype *d_F2, ptype *d_F3, ptype *d_F4, int TAG)

{
	int Id = blockIdx.x*blockDim.x + threadIdx.x;

	int iy =  Id/(d_nt_segx*d_nt_segz);
	int ix =  (Id%(d_nt_segx*d_nt_segz))/d_nt_segz;
	int iz =  Id%d_nt_segz;
	
/* ----------------------------------------------------------------------------------------------------------------------------------*
 * Variables Decleration
 * ----------------------------------------------------------------------------------------------------------------------------------*/	
	ptype Pl, denl, laml, Ul[3], Pr, denr, lamr, Ur[3], We[5], Pe, dene, lame, Ue[3], PL, TL,  denL, UL[3];	
	ptype Ie2l, Ie4l, Ie2r, Ie4r, Ipl[3][7], Ifl[3][7], Inr[3][7], Ifr[3][7], Ie2e, Ie4e, Ipe[3][7], Ine[3][7], Ife[3][7];

	ptype Mpgl000[5], Mngr000[5];
	ptype Mfgl100_axl[5], Mfgr100_axr[5], Mfgl010_ayl[5], Mfgr010_ayr[5], Mfgl001_azl[5], Mfgr001_azr[5];

	ptype bxl[5], bxr[5], byl[5], byr[5], bzl[5], bzr[5], bx_l[5], bx_r[5], by_[5], bz_[5], Bl[5], Br[5], B_[5];
	ptype axl[5], axr[5], ayl[5], ayr[5], azl[5], azr[5], ax_l[5], ax_r[5], ay_[5], az_[5], Al[5], Ar[5], A_[5];

	ptype Mpgl000_ayl[5], Mngr000_ayr[5], Mpgl000_azl[5], Mngr000_azr[5];
	ptype tau, w, gm0, gm1, gm2, gm3, gm4, gm5;

	ptype Mfge000[5], Mpge100_ax_l[5], Mnge100_ax_r[5], Mfge010_ay_[5], Mfge001_az_[5], Mpgl100_axl[5], 
          Mngr100_axr[5], Mpgl010_ayl[5], Mngr010_ayr[5], Mpgl001_azl[5], Mngr001_azr[5], Mpgl000_Al[5], Mngr000_Ar[5];

	ptype P, Q, p0, p1, p2, p3, p4, p5;

	ptype Mfge100[5], Mpgl100[5], Mngr100[5], Mpge200_ax_l[5], Mfge100_A_[5], Mnge200_ax_r[5], Mfge110_ay_[5], Mfge101_az_[5], 
	      Mpgl200_axl[5], Mngr200_axr[5], Mpgl110_ayl[5], Mngr110_ayr[5], Mpgl101_azl[5], Mngr101_azr[5], Mpgl100_Al[5], Mngr100_Ar[5];		
	
	ptype WL[5], WR[5];
	ptype Wl[5], Wr[5];
		
	
	bool bound_check;
	if      (TAG == 1) {bound_check = (ix < d_nt_segx-3 && iy < d_nt_segy-3 && iz < d_nt_segz-3 && ix > 1 && iy > 2 && iz > 2);}
	else if (TAG == 2) {bound_check = (ix < d_nt_segx-3 && iy < d_nt_segy-3 && iz < d_nt_segz-3 && ix > 2 && iy > 1 && iz > 2);}
	else if (TAG == 3) {bound_check = (ix < d_nt_segx-3 && iy < d_nt_segy-3 && iz < d_nt_segz-3 && ix > 2 && iy > 2 && iz > 1);}
	
	if (bound_check) {
		// X-Flux
		if (TAG == 1) {
			WL[0] = d_W0[Id(iy,ix,iz)];
			WL[1] = d_W1[Id(iy,ix,iz)];
			WL[2] = d_W2[Id(iy,ix,iz)];
			WL[3] = d_W3[Id(iy,ix,iz)];
			WL[4] = d_W4[Id(iy,ix,iz)];

			WR[0] = d_W0[Id(iy,ix+1,iz)];
			WR[1] = d_W1[Id(iy,ix+1,iz)];
			WR[2] = d_W2[Id(iy,ix+1,iz)];
			WR[3] = d_W3[Id(iy,ix+1,iz)];
			WR[4] = d_W4[Id(iy,ix+1,iz)];

			Wl[0] = d_Wl0[Id(iy,ix,iz)];
			Wl[1] = d_Wl1[Id(iy,ix,iz)];
			Wl[2] = d_Wl2[Id(iy,ix,iz)];
			Wl[3] = d_Wl3[Id(iy,ix,iz)];
			Wl[4] = d_Wl4[Id(iy,ix,iz)];

			Wr[0] = d_Wr0[Id(iy,ix,iz)];
			Wr[1] = d_Wr1[Id(iy,ix,iz)];
			Wr[2] = d_Wr2[Id(iy,ix,iz)];
			Wr[3] = d_Wr3[Id(iy,ix,iz)];
			Wr[4] = d_Wr4[Id(iy,ix,iz)];		

		}
		// Y-Flux
		if (TAG == 2) {
			WL[0] = d_W0[Id(iy,ix,iz)];
			WL[2] = d_W1[Id(iy,ix,iz)];
			WL[1] = d_W2[Id(iy,ix,iz)];
			WL[3] = d_W3[Id(iy,ix,iz)];
			WL[4] = d_W4[Id(iy,ix,iz)];

			WR[0] = d_W0[Id(iy+1,ix,iz)];
			WR[2] = d_W1[Id(iy+1,ix,iz)];
			WR[1] = d_W2[Id(iy+1,ix,iz)];
			WR[3] = d_W3[Id(iy+1,ix,iz)];
			WR[4] = d_W4[Id(iy+1,ix,iz)];

			Wl[0] = d_Wl0[Id(iy,ix,iz)];
			Wl[1] = d_Wl1[Id(iy,ix,iz)];
			Wl[2] = d_Wl2[Id(iy,ix,iz)];
			Wl[3] = d_Wl3[Id(iy,ix,iz)];
			Wl[4] = d_Wl4[Id(iy,ix,iz)];

			Wr[0] = d_Wr0[Id(iy,ix,iz)];
			Wr[1] = d_Wr1[Id(iy,ix,iz)];
			Wr[2] = d_Wr2[Id(iy,ix,iz)];
			Wr[3] = d_Wr3[Id(iy,ix,iz)];
			Wr[4] = d_Wr4[Id(iy,ix,iz)];
		}
		// Z-Flux
		if (TAG == 3) {
			WL[0] = d_W0[Id(iy,ix,iz)];
			WL[3] = d_W1[Id(iy,ix,iz)];
			WL[2] = d_W2[Id(iy,ix,iz)];
			WL[1] = d_W3[Id(iy,ix,iz)];
			WL[4] = d_W4[Id(iy,ix,iz)];

			WR[0] = d_W0[Id(iy,ix,iz+1)];
			WR[3] = d_W1[Id(iy,ix,iz+1)];
			WR[2] = d_W2[Id(iy,ix,iz+1)];
			WR[1] = d_W3[Id(iy,ix,iz+1)];
			WR[4] = d_W4[Id(iy,ix,iz+1)];			

			Wl[0] = d_Wl0[Id(iy,ix,iz)];
			Wl[1] = d_Wl1[Id(iy,ix,iz)];
			Wl[2] = d_Wl2[Id(iy,ix,iz)];
			Wl[3] = d_Wl3[Id(iy,ix,iz)];
			Wl[4] = d_Wl4[Id(iy,ix,iz)];

			Wr[0] = d_Wr0[Id(iy,ix,iz)];
			Wr[1] = d_Wr1[Id(iy,ix,iz)];
			Wr[2] = d_Wr2[Id(iy,ix,iz)];
			Wr[3] = d_Wr3[Id(iy,ix,iz)];
			Wr[4] = d_Wr4[Id(iy,ix,iz)];
		}
		
/* ------------------------------------------------------------------*
 * INTEGRATIONS (MOMENT CALCULATIONS) [l & r]
 * ------------------------------------------------------------------*/
		d_c2p(Wl, denl, Ul, Pl);	d_c2p(Wr, denr, Ur, Pr);
		laml  = 0.5*denl/Pl;	lamr  = 0.5*denr/Pr;
	
		Ie2l = d_K/(2*laml);
		Ie4l = 3*d_K/(4*laml*laml) + d_K*(d_K-1)/(4*laml*laml);
		Ie2r = d_K/(2*lamr);
		Ie4r = 3*d_K/(4*lamr*lamr) + d_K*(d_K-1)/(4*lamr*lamr);
	
		for(int j=0; j<3; j++)
		{
			Ifl[j][0] = 1.0;
			Ifl[j][1] = Ul[j];
			Ifr[j][0] = 1.0;
			Ifr[j][1] = Ur[j];
			Ipl[j][0] = 0.5*(erfc(-sqrt(laml)*Ul[j]));
			Inr[j][0] = 0.5*(erfc(sqrt(lamr)*Ur[j]));
			Ipl[j][1] = Ul[j]*Ipl[j][0] + 0.5*exp(-laml*Ul[j]*Ul[j])/sqrt(laml*pi);
			Inr[j][1] = Ur[j]*Inr[j][0] - 0.5*exp(-lamr*Ur[j]*Ur[j])/sqrt(lamr*pi);

			for(int i=2; i<7; i++)
			{
				Ipl[j][i] = Ul[j]*Ipl[j][i-1] + Ipl[j][i-2]*(i-1)/(2*laml);
				Ifl[j][i] = Ul[j]*Ifl[j][i-1] + Ifl[j][i-2]*(i-1)/(2*laml);
				Inr[j][i] = Ur[j]*Inr[j][i-1] + Inr[j][i-2]*(i-1)/(2*lamr);
				Ifr[j][i] = Ur[j]*Ifr[j][i-1] + Ifr[j][i-2]*(i-1)/(2*lamr);
			}
		}

/* ------------------------------------------------------------------*
 * W0 CALCULATION
 * ------------------------------------------------------------------*/
		d_MCal(Mpgl000, Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0);
		d_MCal(Mngr000, Inr, Ifr, Ie2r, Ie4r, 0, 0, 0);
	
		FOR(q, 5)
			We[q] = denl * Mpgl000[q] + denr * Mngr000[q];
	
		d_c2p(We, dene, Ue, Pe);	lame  = 0.5*dene/Pe;

/* ------------------------------------------------------------------*
 * INTEGRATIONS (MOMENT CALCULATIONS) [e]
 * ------------------------------------------------------------------*/
		Ie2e = d_K/(2*lame);
		Ie4e = 3*d_K/(4*lame*lame) + d_K*(d_K-1)/(4*lame*lame);
	
		for(int j=0; j<3; j++) {
			Ife[j][0] = 1.0;
			Ife[j][1] = Ue[j];
			Ipe[j][0] = 0.5*(erfc(-sqrt(lame)*Ue[j]));
			Ine[j][0] = 0.5*(erfc(sqrt(lame)*Ue[j]));
			Ipe[j][1] = Ue[j]*Ipe[j][0] + 0.5*exp(-lame*Ue[j]*Ue[j])/sqrt(lame*pi);
			Ine[j][1] = Ue[j]*Ine[j][0] - 0.5*exp(-lame*Ue[j]*Ue[j])/sqrt(lame*pi);

			for(int i=2; i<7; i++) {
				Ipe[j][i] = Ue[j]*Ipe[j][i-1] + Ipe[j][i-2]*(i-1)/(2*lame);
				Ife[j][i] = Ue[j]*Ife[j][i-1] + Ife[j][i-2]*(i-1)/(2*lame);
				Ine[j][i] = Ue[j]*Ine[j][i-1] + Ine[j][i-2]*(i-1)/(2*lame);
			}	
		}
/* --------------------------------------------------------------------------*
 * Slope Calculation [phase I]: (axl, axr, ayl, ayr, azl, azr, ax_l, ax_r)
 * --------------------------------------------------------------------------*/
		int x = 0;
			bxl[x] = d_DW0xl[Id]/denl;
			bxr[x] = d_DW0xr[Id]/denr;		
			byl[x] = d_DW0yl[Id]/denl;
			byr[x] = d_DW0yr[Id]/denr;		
			bzl[x] = d_DW0zl[Id]/denl;		
			bzr[x] = d_DW0zr[Id]/denr;
			bx_l[x] = 2*(We[x] - WL[x]) / (dene*d_dx);
			bx_r[x] = 2*(WR[x] - We[x]) / (dene*d_dx);
			
			x = 1;
			bxl[x] = d_DW1xl[Id]/denl;
			bxr[x] = d_DW1xr[Id]/denr;		
			byl[x] = d_DW1yl[Id]/denl;
			byr[x] = d_DW1yr[Id]/denr;		
			bzl[x] = d_DW1zl[Id]/denl;		
			bzr[x] = d_DW1zr[Id]/denr;		
			bx_l[x] = 2*(We[x] - WL[x]) / (dene*d_dx);
			bx_r[x] = 2*(WR[x] - We[x]) / (dene*d_dx);
			
			x = 2;
			bxl[x] = d_DW2xl[Id]/denl;
			bxr[x] = d_DW2xr[Id]/denr;		
			byl[x] = d_DW2yl[Id]/denl;
			byr[x] = d_DW2yr[Id]/denr;		
			bzl[x] = d_DW2zl[Id]/denl;		
			bzr[x] = d_DW2zr[Id]/denr;		
			bx_l[x] = 2*(We[x] - WL[x]) / (dene*d_dx);
			bx_r[x] = 2*(WR[x] - We[x]) / (dene*d_dx);
			
			x = 3;
			bxl[x] = d_DW3xl[Id]/denl;
			bxr[x] = d_DW3xr[Id]/denr;		
			byl[x] = d_DW3yl[Id]/denl;
			byr[x] = d_DW3yr[Id]/denr;		
			bzl[x] = d_DW3zl[Id]/denl;		
			bzr[x] = d_DW3zr[Id]/denr;		
			bx_l[x] = 2*(We[x] - WL[x]) / (dene*d_dx);
			bx_r[x] = 2*(WR[x] - We[x]) / (dene*d_dx);
		
			x = 4;
			bxl[x] = d_DW4xl[Id]/denl;
			bxr[x] = d_DW4xr[Id]/denr;		
			byl[x] = d_DW4yl[Id]/denl;
			byr[x] = d_DW4yr[Id]/denr;		
			bzl[x] = d_DW4zl[Id]/denl;		
			bzr[x] = d_DW4zr[Id]/denr;		
			bx_l[x] = 2*(We[x] - WL[x]) / (dene*d_dx);
			bx_r[x] = 2*(WR[x] - We[x]) / (dene*d_dx);

		d_slopesolver(bxl, Ul, laml, axl);
		d_slopesolver(bxr, Ur, lamr, axr);
		d_slopesolver(byl, Ul, laml, ayl);
		d_slopesolver(byr, Ur, lamr, ayr);
		d_slopesolver(bzl, Ul, laml, azl);
		d_slopesolver(bzr, Ur, lamr, azr);
		d_slopesolver(bx_l, Ue, lame, ax_l);
		d_slopesolver(bx_r, Ue, lame, ax_r);

/* -------------------------------------------------------------------------*
 * Slope Calculation [phase II]: (ay_, az_, Al, Ar)
 * -------------------------------------------------------------------------*/
		d_MCal(Mpgl000_ayl, Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0, ayl);
		d_MCal(Mngr000_ayr, Inr, Ifr, Ie2r, Ie4r, 0, 0, 0, ayr);
		d_MCal(Mpgl000_azl, Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0, azl);
		d_MCal(Mngr000_azr, Inr, Ifr, Ie2r, Ie4r, 0, 0, 0, azr);
		d_MCal(Mfgl100_axl, Ifl, Ifl, Ie2l, Ie4l, 1, 0, 0, axl);
		d_MCal(Mfgr100_axr, Ifr, Ifr, Ie2r, Ie4r, 1, 0, 0, axr);
		d_MCal(Mfgl010_ayl, Ifl, Ifl, Ie2l, Ie4l, 0, 1, 0, ayl);
		d_MCal(Mfgr010_ayr, Ifr, Ifr, Ie2r, Ie4r, 0, 1, 0, ayr);
		d_MCal(Mfgl001_azl, Ifl, Ifl, Ie2l, Ie4l, 0, 0, 1, azl);
		d_MCal(Mfgr001_azr, Ifr, Ifr, Ie2r, Ie4r, 0, 0, 1, azr);
	
		FOR(i, 5) {
			by_[i] = Mpgl000_ayl[i] + Mngr000_ayr[i];
			bz_[i] = Mpgl000_azl[i] + Mngr000_azr[i];	
			Bl[i] = -Mfgl100_axl[i] - Mfgl010_ayl[i] - Mfgl001_azl[i];		 
			Br[i] = -Mfgr100_axr[i] - Mfgr010_ayr[i] - Mfgr001_azr[i];		 
		}
		d_slopesolver(by_, Ue, lame, ay_);
		d_slopesolver(bz_, Ue, lame, az_);
		d_slopesolver(Bl, Ul, laml, Al);
		d_slopesolver(Br, Ur, lamr, Ar);
	
/* -------------------------------------------------------------------------*
 * Collision time scale calculation (tau)
 * -------------------------------------------------------------------------*/
		d_c2p(WL, denL, UL, PL);
		TL   = PL/(R*denL);
		
		tau = d_mu0 * ( pow((TL/d_T0),1.5)*(d_T0 + 110.4)/(TL + 110.4) ) / PL;

		w = abs(denl/laml - denr/lamr)/(abs(denl/laml + denr/lamr));
		tau = tau + d_dt*w;
	
/* -------------------------------------------------------------------------*
 * Slope Calculation [phase III]: (A_)
 * -------------------------------------------------------------------------*/
		gm0 = d_dt - tau*(1-exp(-d_dt/tau));	gm1 = -(1-exp(-d_dt/tau))/gm0;
		gm2 = (-d_dt+2*tau*(1-exp(-d_dt/tau))-d_dt*exp(-d_dt/tau))/gm0;	gm3 = -gm1;
		gm4 = (d_dt*exp(-d_dt/tau) -tau*(1-exp(-d_dt/tau)))/gm0;		gm5 = tau*gm3;
		
		d_MCal(Mfge000, Ife, Ife, Ie2e, Ie4e, 0, 0, 0);
		d_MCal(Mpge100_ax_l, Ipe, Ife, Ie2e, Ie4e, 1, 0, 0, ax_l);	
		d_MCal(Mnge100_ax_r, Ine, Ife, Ie2e, Ie4e, 1, 0, 0, ax_r);	
		d_MCal(Mfge010_ay_,  Ife, Ife, Ie2e, Ie4e, 0, 1, 0, ay_);	
		d_MCal(Mfge001_az_,  Ife, Ife, Ie2e, Ie4e, 0, 0, 1, az_);	
		d_MCal(Mpgl100_axl,  Ipl, Ifl, Ie2l, Ie4l, 1, 0, 0, axl);	
		d_MCal(Mngr100_axr,  Inr, Ifr, Ie2r, Ie4r, 1, 0, 0, axr);	
		d_MCal(Mpgl010_ayl,  Ipl, Ifl, Ie2l, Ie4l, 0, 1, 0, ayl);	
		d_MCal(Mngr010_ayr,  Inr, Ifr, Ie2r, Ie4r, 0, 1, 0, ayr);	
		d_MCal(Mpgl001_azl,  Ipl, Ifl, Ie2l, Ie4l, 0, 0, 1, azl);	
		d_MCal(Mngr001_azr,  Inr, Ifr, Ie2r, Ie4r, 0, 0, 1, azr);	
		d_MCal(Mpgl000_Al,   Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0, Al);	
		d_MCal(Mngr000_Ar,   Inr, Ifr, Ie2r, Ie4r, 0, 0, 0, Ar);
		
		FOR(i, 5) {
			B_[i] = gm1*dene*Mfge000[i] + gm2*dene*(Mpge100_ax_l[i]+Mnge100_ax_r[i]+Mfge010_ay_[i]+Mfge001_az_[i]) + gm3*(denl*Mpgl000[i]+denr*Mngr000[i]) +
					(gm4+gm5)*( denl*(Mpgl100_axl[i]+Mpgl010_ayl[i]+Mpgl001_azl[i]) + denr*(Mngr100_axr[i]+Mngr010_ayr[i]+Mngr001_azr[i]) ) +
					gm5*(denl*Mpgl000_Al[i]+denr*Mngr000_Ar[i]);
		       		
			B_[i] = B_[i] / dene;
		}
		d_slopesolver(B_, Ue, lame, A_);
	
/* ---------------------------------------------------------------------------------------------------------*
 * Flux calculation	
 * ---------------------------------------------------------------------------------------------------------*/
//		Integral dt
		P  = -tau*(exp(-d_dt/tau)-1.0);
		Q  = -tau*d_dt*exp(-d_dt/tau)-tau*tau*(exp(-d_dt/tau)-1.0);
		p0 = (d_dt-P);			p1 = 0.5*d_dt*d_dt - tau*(p0);
		p2 = -tau*(p0) + Q;		p3 = P;
		p4 = -Q - tau*P;		p5 = -tau*P;
	
		d_MCal(Mpgl100, Ipl, Ifl, Ie2l, Ie4l, 1, 0, 0);
		d_MCal(Mngr100, Inr, Ifr, Ie2r, Ie4r, 1, 0, 0);
		d_MCal(Mfge100, Ife, Ife, Ie2e, Ie4e, 1, 0, 0);
		d_MCal(Mfge100_A_,   Ife, Ife, Ie2e, Ie4e, 1, 0, 0, A_);	
		d_MCal(Mpge200_ax_l, Ipe, Ife, Ie2e, Ie4e, 2, 0, 0, ax_l);	
		d_MCal(Mnge200_ax_r, Ine, Ife, Ie2e, Ie4e, 2, 0, 0, ax_r);	
		d_MCal(Mfge110_ay_,  Ife, Ife, Ie2e, Ie4e, 1, 1, 0, ay_);	
		d_MCal(Mfge101_az_,  Ife, Ife, Ie2e, Ie4e, 1, 0, 1, az_);	
		d_MCal(Mpgl200_axl,  Ipl, Ifl, Ie2l, Ie4l, 2, 0, 0, axl);	
		d_MCal(Mngr200_axr,  Inr, Ifr, Ie2r, Ie4r, 2, 0, 0, axr);	
		d_MCal(Mpgl110_ayl,  Ipl, Ifl, Ie2l, Ie4l, 1, 1, 0, ayl);	
		d_MCal(Mngr110_ayr,  Inr, Ifr, Ie2r, Ie4r, 1, 1, 0, ayr);	
		d_MCal(Mpgl101_azl,  Ipl, Ifl, Ie2l, Ie4l, 1, 0, 1, azl);	
		d_MCal(Mngr101_azr,  Inr, Ifr, Ie2r, Ie4r, 1, 0, 1, azr);	
		d_MCal(Mpgl100_Al,   Ipl, Ifl, Ie2l, Ie4l, 1, 0, 0, Al);	
		d_MCal(Mngr100_Ar,   Inr, Ifr, Ie2r, Ie4r, 1, 0, 0, Ar);	
		ptype F[5] = {0};
		
		FOR(i, 5) {
			F[i] = p0*dene*Mfge100[i] + p1*dene*Mfge100_A_[i] + p2*dene*(Mpge200_ax_l[i]+Mnge200_ax_r[i]+Mfge110_ay_[i]+Mfge101_az_[i]) + 
				   p3*(denl*Mpgl100[i]+denr*Mngr100[i]) +  p4*( denl*(Mpgl200_axl[i]+Mpgl110_ayl[i]+Mpgl101_azl[i]) + denr*(Mngr200_axr[i]+Mngr110_ayr[i]+Mngr101_azr[i]) ) + 
				   p5*(denl*Mpgl100_Al[i]+denr*Mngr100_Ar[i]);		
		}
	
		d_F0[Id] = F[0];
		d_F1[Id] = F[1];
		d_F2[Id] = F[2];
		d_F3[Id] = F[3];
		d_F4[Id] = F[4]; 		
		
	} // end of if statement for checking thread out of range access

}

	
/* ----------------------------------------------------------------------*
 *  SLOPE SOLVER
 * ----------------------------------------------------------------------*/

__device__ void d_slopesolver(ptype b[5], ptype U[3], ptype lam, ptype a[5])
{
	ptype R2, R3, R4, R5;
	
	R2 = b[1] - U[0]*b[0];
	R3 = b[2] - U[1]*b[0];
	R4 = b[3] - U[2]*b[0];
	R5 = 2*b[4] - b[0]*(U[0]*U[0]+U[1]*U[1]+U[2]*U[2]+(d_K+3)/(2*lam));
	
	a[4] = (1/PRN)*(R5-2*U[0]*R2-2*U[1]*R3-2*U[2]*R4)*(4*lam*lam)/(d_K+3);
	a[3] = 2*lam*R4 - U[2]*a[4];
	a[2] = 2*lam*R3 - U[1]*a[4];
	a[1] = 2*lam*R2 - U[0]*a[4];
	a[0] = b[0] - a[1]*U[0] - a[2]*U[1] -a[3]*U[2]-.5*a[4]*(U[0]*U[0] +
	       U[1]*U[1] + U[2]*U[2]+(d_K+3)/(2*lam));
}

/**-------------------------------------------------------------------------------------------------*
 * Device Function : Moment Integral Calculator
 *--------------------------------------------------------------------------------------------------*/
/* ---------------------------------------------------------------------------------------------------------------------------------------------------*
 *  Moment Matrix Calculator 
 * ---------------------------------------------------------------------------------------------------------------------------------------------------*/
__device__ void d_MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m, ptype ax[5])
{
	ptype val0, val1, val2, val3;
	 
	val0 = 0.5 * ( I[0][2+k]*If[1][l]*If[2][m] + I[0][k]*If[1][2+l]*If[2][m] + I[0][k]*If[1][l]*If[2][2+m] + I[0][k]*If[1][l]*If[2][m]*Ie2  );
	
	M[0] = ax[0]*(I[0][k]*If[1][l]*If[2][m]) + ax[1]*(I[0][1+k]*If[1][l]*If[2][m]) + ax[2]*(I[0][k]*If[1][1+l]*If[2][m]) +
	       ax[3]*(I[0][k]*If[1][l]*If[2][1+m]) + ax[4]*val0 ;	

				  
	val1 = 0.5 * ( I[0][3+k]*If[1][l]*If[2][m] + I[0][1+k]*If[1][2+l]*If[2][m] + I[0][1+k]*If[1][l]*If[2][2+m] +
	 			   I[0][1+k]*If[1][l]*If[2][m]*Ie2  );
	
	
	M[1] = ax[0]*(I[0][1+k]*If[1][l]*If[2][m]) + ax[1]*(I[0][2+k]*If[1][l]*If[2][m]) + ax[2]*(I[0][1+k]*If[1][1+l]*If[2][m]) +
		   ax[3]*(I[0][1+k]*If[1][l]*If[2][1+m]) + ax[4]*val1;
				 
 
   val2 = 0.5 * ( I[0][2+k]*If[1][1+l]*If[2][m] + I[0][k]*If[1][3+l]*If[2][m] + I[0][k]*If[1][1+l]*If[2][2+m] +
				  I[0][k]*If[1][1+l]*If[2][m]*Ie2  );
	
	
	M[2] = ax[0]*(I[0][k]*If[1][1+l]*If[2][m])   + ax[1]*(I[0][1+k]*If[1][1+l]*If[2][m]) + ax[2]*(I[0][k]*If[1][2+l]*If[2][m]) +
		   ax[3]*(I[0][k]*If[1][1+l]*If[2][1+m]) + ax[4]*val2;
				 
	
	val3 = 0.5 * ( I[0][2+k]*If[1][l]*If[2][1+m] + I[0][k]*If[1][2+l]*If[2][1+m] + I[0][k]*If[1][l]*If[2][3+m] + 
	               I[0][k]*If[1][l]*If[2][1+m]*Ie2  );
	
	
	M[3] = ax[0]*(I[0][k]*If[1][l]*If[2][1+m]) + ax[1]*(I[0][1+k]*If[1][l]*If[2][1+m]) + ax[2]*(I[0][k]*If[1][1+l]*If[2][1+m]) +
	       ax[3]*(I[0][k]*If[1][l]*If[2][2+m]) + ax[4]*val3;
				 
				 
	M[4] = 0.25*ax[4]* ( I[0][4+k]*If[1][l]*If[2][m] + I[0][k]*If[1][4+l]*If[2][m] + I[0][k]*If[1][l]*If[2][4+m] +
	                     I[0][k]*If[1][l]*If[2][m]*Ie4 + 2*I[0][2+k]*If[1][2+l]*If[2][m] + 2*I[0][2+k]*If[1][l]*If[2][2+m] +
	                   2*I[0][2+k]*If[1][l]*If[2][m]*Ie2 + 2*I[0][k]*If[1][2+l]*If[2][2+m] + 2*I[0][k]*If[1][2+l]*If[2][m]*Ie2 +
	                   2*I[0][k]*If[1][l]*If[2][2+m]*Ie2 ) +
		   ax[0]*val0 + ax[1]*val1 + ax[2]*val2 + ax[3]*val3;
}

__device__ void d_MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m)
{
    ptype val0	= 0.5 * ( I[0][2+k]*If[1][l]*If[2][m] + I[0][k]*If[1][2+l]*If[2][m] + I[0][k]*If[1][l]*If[2][2+m] + I[0][k]*If[1][l]*If[2][m]*Ie2  );
		
	M[0] = I[0][k]*If[1][l]*If[2][m];
	M[1] = I[0][1+k]*If[1][l]*If[2][m];
	M[2] = I[0][k]*If[1][1+l]*If[2][m];
	M[3] = I[0][k]*If[1][l]*If[2][1+m];
	M[4] = val0;
}

__device__ void d_c2p(ptype W[5], ptype &den, ptype U[3], ptype &P)
{
	den = W[0];
	U[0]  = W[1]/den;
	U[1]  = W[2]/den;
	U[2]  = W[3]/den;
	P   = (den*(GAM-1))*(W[4]/den - 0.5*(U[0]*U[0]+U[1]*U[1]+U[2]*U[2]));
}

/* ---------------------------------------------------------------------------------------------------------------------------------------------------*
 *  WENO - SYMOO
 * ---------------------------------------------------------------------------------------------------------------------------------------------------*/
__device__ void d_ApplyWENO(ptype WLLL[5], ptype WLL[5], ptype WL[5], ptype WR[5], ptype WRR[5],  ptype WRRR[5], ptype Wl[5], ptype Wr[5])
{	
	ptype P0p, P0n, P1p, P1n, P2p, P2n, P3p, P3n, IS0p, IS0n, IS1p, IS1n, IS2p, IS2n, IS3p, IS3n, 
	      gam0, gam1, gam2, gam3, ALPHA0p, ALPHA0n, ALPHA1p, ALPHA1n, ALPHA2p, ALPHA2n, ALPHA3p, ALPHA3n, sum_ALPHAp, sum_ALPHAn,
	      OMEGA0p, OMEGA0n, OMEGA1p, OMEGA1n, OMEGA2p, OMEGA2n, OMEGA3p, OMEGA3n, ISPMAX, ISNMAX;
	ptype TLLL, TLL, TL, TR, TRR, TRRR;
	gam0 = 1.0/20.0; gam1 = 9.0/20.0; gam2 = 9.0/20.0; gam3 = 1.0/20.0;
	
	FOR(i, 4) {
		P0p =     WLLL[i]/3  - 7*WLL[i]/6  + 11*WL[i]/6;
		P0n =     WRRR[i]/3  - 7*WRR[i]/6  + 11*WR[i]/6;
		P1p =    -WLL[i]/6   + 5*WL[i]/6   + WR[i]/3;
		P1n =    -WRR[i]/6   + 5*WR[i]/6   + WL[i]/3;	
		P2p =     WL[i]/3    + 5*WR[i]/6   - WRR[i]/6;
		P2n =     WR[i]/3    + 5*WL[i]/6   - WLL[i]/6;
		P3p =  11*WR[i]/6    - 7*WRR[i]/6  + WRRR[i]/3;
		P3n =  11*WL[i]/6    - 7*WLL[i]/6  + WLLL[i]/3;
			
		IS0p = 13*pow((WLLL[i] - 2*WLL[i] + WL[i]),2)/12    +  pow(( WLLL[i] - 4*WLL[i] + 3*WL[i]),2)/4;
		IS0n = 13*pow((WRRR[i] - 2*WRR[i] + WR[i]),2)/12    +  pow(( WRRR[i] - 4*WRR[i] + 3*WR[i]),2)/4;
		IS1p = 13*pow((WLL[i]  - 2*WL[i]  + WR[i]),2)/12    +  pow(( -WLL[i] + WR[i]),2)/4;
		IS1n = 13*pow((WRR[i]  - 2*WR[i]  + WL[i]),2)/12    +  pow(( -WRR[i] + WL[i]),2)/4;
		IS2p = 13*pow((WL[i]   - 2*WR[i]  + WRR[i]),2)/12   +  pow((-3*WL[i] + 4*WR[i] - WRR[i]),2)/4;
		IS2n = 13*pow((WR[i]   - 2*WL[i]  + WLL[i]),2)/12   +  pow((-3*WR[i] + 4*WL[i] - WLL[i]),2)/4;
		IS3p = 13*pow((WR[i]   - 2*WRR[i] + WRRR[i]),2)/12  +  pow((-5*WR[i] + 8*WRR[i] - 3*WRRR[i]),2)/4;
		IS3n = 13*pow((WL[i]   - 2*WLL[i] + WLLL[i]),2)/12  +  pow((-5*WL[i] + 8*WLL[i] - 3*WLLL[i]),2)/4;
		
		ISPMAX = max(IS0p,   IS1p);
		ISPMAX = max(ISPMAX, IS2p);
		ISPMAX = max(ISPMAX, IS3p);
		ISNMAX = max(IS0n,   IS1n);
		ISNMAX = max(ISNMAX, IS2n);
		ISNMAX = max(ISNMAX, IS3n);
		IS3p = ISPMAX; 
		IS3n = ISNMAX; 
				
		ALPHA0p = gam0*(1/(eps+IS0p)); 
		ALPHA0n = gam0*(1/(eps+IS0n)); 
		ALPHA1p = gam1*(1/(eps+IS1p)); 
		ALPHA1n = gam1*(1/(eps+IS1n)); 
		ALPHA2p = gam2*(1/(eps+IS2p)); 
		ALPHA2n = gam2*(1/(eps+IS2n)); 
		ALPHA3p = gam3*(1/(eps+IS3p)); 
		ALPHA3n = gam3*(1/(eps+IS3n)); 
	
		sum_ALPHAp = ALPHA0p + ALPHA1p + ALPHA2p + ALPHA3p;
		sum_ALPHAn = ALPHA0n + ALPHA1n + ALPHA2n + ALPHA3n;
	
		OMEGA0p = ALPHA0p/(sum_ALPHAp);
		OMEGA0n = ALPHA0n/(sum_ALPHAn);
		OMEGA1p = ALPHA1p/(sum_ALPHAp);
		OMEGA1n = ALPHA1n/(sum_ALPHAn);
		OMEGA2p = ALPHA2p/(sum_ALPHAp);
		OMEGA2n = ALPHA2n/(sum_ALPHAn);
		OMEGA3p = ALPHA3p/(sum_ALPHAp);
		OMEGA3n = ALPHA3n/(sum_ALPHAn);
	 
		Wl[i] = (OMEGA0p*P0p + OMEGA1p*P1p + OMEGA2p*P2p + OMEGA3p*P3p); 
		Wr[i] = (OMEGA0n*P0n + OMEGA1n*P1n + OMEGA2n*P2n + OMEGA3n*P3n); 
	}

		TL   = (GAM-1) * ( WL[4]/WL[0]     - 0.5*(WL[1]*WL[1] + WL[2]*WL[2] + WL[3]*WL[3])/(WL[0]*WL[0]) ) / R;
		TLL  = (GAM-1) * ( WLL[4]/WLL[0]   - 0.5*(WLL[1]*WLL[1] + WLL[2]*WLL[2] + WLL[3]*WLL[3])/(WLL[0]*WLL[0]) ) / R;
		TLLL = (GAM-1) * ( WLLL[4]/WLLL[0] - 0.5*(WLLL[1]*WLLL[1] + WLLL[2]*WLLL[2] + WLLL[3]*WLLL[3])/(WLLL[0]*WLLL[0]) ) / R;
		TR   = (GAM-1) * ( WR[4]/WR[0]     - 0.5*(WR[1]*WR[1] + WR[2]*WR[2] + WR[3]*WR[3])/(WR[0]*WR[0]) ) / R;
		TRR  = (GAM-1) * ( WRR[4]/WRR[0]   - 0.5*(WRR[1]*WRR[1] + WRR[2]*WRR[2] + WRR[3]*WRR[3])/(WRR[0]*WRR[0]) ) / R;
		TRRR = (GAM-1) * ( WRRR[4]/WRRR[0] - 0.5*(WRRR[1]*WRRR[1] + WRRR[2]*WRRR[2] + WRRR[3]*WRRR[3])/(WRRR[0]*WRRR[0]) ) / R;
		
		P0p =     TLLL/3  - 7*TLL/6  + 11*TL/6;
		P0n =     TRRR/3  - 7*TRR/6  + 11*TR/6;
		P1p =    -TLL/6   + 5*TL/6   + TR/3;
		P1n =    -TRR/6   + 5*TR/6   + TL/3;	
		P2p =     TL/3    + 5*TR/6   - TRR/6;
		P2n =     TR/3    + 5*TL/6   - TLL/6;
		P3p =  11*TR/6    - 7*TRR/6  + TRRR/3;
		P3n =  11*TL/6    - 7*TLL/6  + TLLL/3;
			
		IS0p = 13*pow((TLLL - 2*TLL + TL),2)/12    +  pow(( TLLL - 4*TLL + 3*TL),2)/4;
		IS0n = 13*pow((TRRR - 2*TRR + TR),2)/12    +  pow(( TRRR - 4*TRR + 3*TR),2)/4;
		IS1p = 13*pow((TLL  - 2*TL  + TR),2)/12    +  pow(( -TLL + TR),2)/4;
		IS1n = 13*pow((TRR  - 2*TR  + TL),2)/12    +  pow(( -TRR + TL),2)/4;
		IS2p = 13*pow((TL   - 2*TR  + TRR),2)/12   +  pow((-3*TL + 4*TR - TRR),2)/4;
		IS2n = 13*pow((TR   - 2*TL  + TLL),2)/12   +  pow((-3*TR + 4*TL - TLL),2)/4;
		IS3p = 13*pow((TR   - 2*TRR + TRRR),2)/12  +  pow((-5*TR + 8*TRR - 3*TRRR),2)/4;
		IS3n = 13*pow((TL   - 2*TLL + TLLL),2)/12  +  pow((-5*TL + 8*TLL - 3*TLLL),2)/4;
		
		ISPMAX = max(IS0p,   IS1p);
		ISPMAX = max(ISPMAX, IS2p);
		ISPMAX = max(ISPMAX, IS3p);
		ISNMAX = max(IS0n,   IS1n);
		ISNMAX = max(ISNMAX, IS2n);
		ISNMAX = max(ISNMAX, IS3n);
		IS3p = ISPMAX;
		IS3n = ISNMAX;
				
		ALPHA0p = gam0*(1/(eps+IS0p)); 
		ALPHA0n = gam0*(1/(eps+IS0n)); 
		ALPHA1p = gam1*(1/(eps+IS1p)); 
		ALPHA1n = gam1*(1/(eps+IS1n)); 
		ALPHA2p = gam2*(1/(eps+IS2p)); 
		ALPHA2n = gam2*(1/(eps+IS2n)); 
		ALPHA3p = gam3*(1/(eps+IS3p)); 
		ALPHA3n = gam3*(1/(eps+IS3n));  
	
		sum_ALPHAp = ALPHA0p + ALPHA1p + ALPHA2p + ALPHA3p;
		sum_ALPHAn = ALPHA0n + ALPHA1n + ALPHA2n + ALPHA3n;
	
		OMEGA0p = ALPHA0p/(sum_ALPHAp);
		OMEGA0n = ALPHA0n/(sum_ALPHAn);
		OMEGA1p = ALPHA1p/(sum_ALPHAp);
		OMEGA1n = ALPHA1n/(sum_ALPHAn);
		OMEGA2p = ALPHA2p/(sum_ALPHAp);
		OMEGA2n = ALPHA2n/(sum_ALPHAn);
		OMEGA3p = ALPHA3p/(sum_ALPHAp);
		OMEGA3n = ALPHA3n/(sum_ALPHAn);
	 
		ptype Tl = (OMEGA0p*P0p + OMEGA1p*P1p + OMEGA2p*P2p + OMEGA3p*P3p); 
		ptype Tr = (OMEGA0n*P0n + OMEGA1n*P1n + OMEGA2n*P2n + OMEGA3n*P3n); 
		
		Wl[4] = Wl[0]*R*Tl/(GAM-1.0) + 0.5*(Wl[1]*Wl[1]+Wl[2]*Wl[2]+Wl[3]*Wl[3])/Wl[0];
		Wr[4] = Wr[0]*R*Tr/(GAM-1.0) + 0.5*(Wr[1]*Wr[1]+Wr[2]*Wr[2]+Wr[3]*Wr[3])/Wr[0];	
/*
	FOR(q,5) {
		Wl = 0.5*(WL + WR); Wr = Wl;
	}
*/
}


__global__ void W2T3D(ptype *d_W0, ptype *d_W1, ptype *d_W2, ptype *d_W3, ptype *d_W4, ptype *d_T)
{
	int I = blockIdx.x*(blockDim.x) + threadIdx.x;
	d_T[I] = (GAM-1) * ( d_W4[I]/d_W0[I] - 0.5*(d_W1[I]*d_W1[I] + d_W2[I]*d_W2[I] + d_W3[I]*d_W3[I])/(d_W0[I]*d_W0[I]) ) / R;
}
/**************************************************************************************************************************
 * -----------------------------------------------------------------------------------------------------------------------*
 **************************************************************************************************************************/
