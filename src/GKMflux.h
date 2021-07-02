/****************************************************************************************
 * Header for Flux calculation using the Gas-Kinetic scheme
 * -------------------------------------------------------------------------------------*
 ****************************************************************************************/

#ifndef GKMFLUXHEADERDEF
#define GKMFLUXHEADERDEF

#include "param.h"
#include <iostream>
#include <omp.h>
      
void flux(ptype WL[5], ptype WR[5], ptype Wl[5], ptype Wr[5], ptype DWxl[5], ptype DWxr[5], 
		  ptype DWyl[5], ptype DWyr[5], ptype DWzl[5], ptype DWzr[5], ptype tau, ptype dt, ptype dx, ptype F[5]);

void slopesolver(ptype b[5], ptype U[3], ptype lam, ptype a[5]);

void c2p(ptype W[5], ptype &den, ptype U[3], ptype &P);

void MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m, ptype ax[5]);
void MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m);

#endif

/****************************************************************************************
 * -------------------------------------------------------------------------------------*/
