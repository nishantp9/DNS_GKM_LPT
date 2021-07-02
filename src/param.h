 /****************************************************************************************
 * Stores the basic problem parameters
 * -------------------------------------------------------------------------------------*
 ****************************************************************************************/

#ifndef PARAMHEADERDEF
#define PARAMHEADERDEF

#include <cmath>

extern int nc, nt, Nc, Nt, nc_segx, nt_segx, nc_segy, nt_segy, nc_segz, numparticles, K,
		   nt_segz, Nc_seg, Nt_seg, nprocs, procId[3], sta[3], end[3], procDim[3], myrank_3d;

#define  procDim_y	2
#define  procDim_x	2
#define  procDim_z  2
#define	 pi 	atan(1.0)*4.0
#define  eps    pow(10.0,-10.0)
#define  PRN    0.70
#define	 cfl    0.10
#define  R      287.0
#define  nomp	5
#define  SAVE_INTERVAL	1000
#define  DHIT_Binary "InitialConditions/Init.dat"

typedef double ptype;	/* precision type  */
extern ptype dx, T0, mu0, den0, p0, Mt, gam, Re;

/* ------------------------------------------*
 * short-hand macros
 * ------------------------------------------*/
#define FOR(i, n) for(int i=0; i<n; i++)
#define For(i, n) for(int i=3; i<n-3; i++)
#define FoR(i, n) for(int i=2; i<n-3; i++)
#define FOr(i, n) for(int i=2; i<n-2; i++)

#define FOR_(i, sta, end) for(int i=sta; i<end; i++)

// we adopt the following index(i,j,k) style for linearization of 3-D array
#define I(i, j, k)   (i)*nt*nt + (j)*nt + (k) 
#define Is(i, j, k)  (i)*nt_segx*nt_segz + (j)*nt_segz + (k) 

// Index def used for reading Initial conditions (ncx x ncz x ncy) 
#define Ii(i, j, k)  (i+3)*nt*nt + (j+3)*nt + (k+3)
#define Iis(i, j, k) (i+3)*nt_segx*nt_segz + (j+3)*nt_segz + (k+3)
#define Ic(i, j, k)  (i)*nc*nc + (j)*nc + (k)
#define Ics(i, j, k) (i)*nc_segx*nc_segz + (j)*nc_segz + (k)
#define Icd(i, j, k) (i)*(nt_segz-4)*(nt_segx-4) + (j)*(nt_segz-4) + (k)

/* -------------------------------------------------------------------------------------*
 * macros for declaring and deleting dynamically allocated memory
 * -------------------------------------------------------------------------------------*/
#define MAKE(var)  var = new ptype[Nt]
#define KILL(var)  delete [] var
#define MAKE5(var) MAKE(var[0]); MAKE(var[1]); MAKE(var[2]); MAKE(var[3]); MAKE(var[4]);
#define KILL5(var) KILL(var[0]); KILL(var[1]); KILL(var[2]); KILL(var[3]); KILL(var[4]); 

#define MAKEs(var)	var = new ptype[Nt_seg];
#define MAKE5s(var)	MAKEs(var[0]); MAKEs(var[1]); MAKEs(var[2]); MAKEs(var[3]); MAKEs(var[4]);

// Save Files (without ghost nodes) for Post-Processing in  Matlab
#define MAKE_(var)  var = new ptype[Nc]
#define KILL_(var)  delete [] var

extern ptype t0;  
#define  tmax  8*t0

// 2) sign template
template <typename T> int sign(T s)
{
	int sign;
	if(s>0)
	sign=1;
	else if(s<0)
	sign=-1;
	else if(s==0)
	sign=0;
	return sign;
}


#endif

/****************************************************************************************
 * -------------------------------------------------------------------------------------*/
