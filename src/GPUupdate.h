#ifndef GPUUPDATEHEADERDEF
#define GPUUPDATEHEADERDEF
#include <iostream>
#include <stdio.h>
#include <iomanip> 
#include <omp.h>
#include "param.h"
#include "GPUAlloc.h"
using namespace std;

/**-------------------------------------------------------------------------------------------------------------*
 * MACROS DEFINITIONS
 * -------------------------------------------------------------------------------------------------------------*/
#define ERR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define ERR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define Bsz3D 64	// KE, PRMS Kernels
#define Bszf 64 	// flux kernel
#define Bszs 64	    // kernels with streams
 
#define CUMAKE(var)  (cudaMalloc((void**) &var, sizeof(ptype)*Nt_seg))
#define CUKILL(var)  (cudaFree(var))

#define HtoD(dv,v) (cudaMemcpy(dv, v, sizeof(ptype)*Nt_seg, cudaMemcpyHostToDevice))
#define DtoH(dv,v) (cudaMemcpy(v, dv, sizeof(ptype)*Nt_seg, cudaMemcpyDeviceToHost))

#define Id(i, j, k)  (i)*d_nt_segx*d_nt_segz + (j)*d_nt_segz + (k)
#define IC(i, j, k)  (i)*d_nc_segx*d_nc_segz + (j)*d_nc_segz + (k)

/**----------------------------------------------------------------------------------------------------------------------*
 * Functions Definitions
 *-----------------------------------------------------------------------------------------------------------------------*/

void evolve(ptype *W[5], DevAlloc *dev, ptype dt);

#endif
