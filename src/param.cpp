#include "param.h"

int nc, nt, Nc, Nt, nc_segx, nt_segx, nc_segy, nt_segy, nc_segz, nt_segz, numparticles, K,
	Nc_seg, Nt_seg, nprocs, procId[3], sta[3], end[3], procDim[3], myrank_3d;

ptype dx, t0, T0, mu0, den0, p0, Mt, gam, Re;

/* ------------------------------------------------------------------------ */
