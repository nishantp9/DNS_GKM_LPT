/***************************************************************
 * FlowField Definitions
 ***************************************************************/

#include "FlowField.h"

using namespace std;

void GetParams() {
	ptype t0_;
	ptype den0_, mu0_, p0_, Mt_, RE, gam_;
	
	int n, np;

	ifstream input("InitialConditions/params", ios::in | ios::binary);
	input.read((char*) &n, sizeof(int));
	::nc = n;
	::nt = nc+6;
	::Nc = nc*nc*nc;
	::Nt = nt*nt*nt;

	::nc_segy = nc / procDim[0];
	::nc_segx = nc / procDim[1];
	::nc_segz = nc / procDim[2];
	
	::Nc_seg = nc_segz*nc_segx*nc_segy;
	::nt_segz = nc_segz + 6;
	::nt_segx = nc_segx + 6;
	::nt_segy = nc_segy + 6;
	::Nt_seg  = nt_segz*nt_segx*nt_segy;

	size_t size = sizeof(ptype);
	input.read((char*) &gam_, size);
	input.read((char*) &den0_, size);
	input.read((char*) &Mt_, size);
	input.read((char*) &RE, size);
	input.read((char*) &mu0_, size);	
	input.read((char*) &p0_, size);
	input.read((char*) &t0_, size);
	::gam  = gam_;
	::K    =  int((5.0-3.0*gam)/(gam-1.0));
	::den0 =  den0_;	
	::p0   =  p0_;	
	::t0   =  t0_;
	::dx   =  2.0*pi/nc;	
    ::T0   =  p0/(R*den0); 
	::Mt   =  Mt_;	
	::mu0  =  mu0_;
	::Re   =  RE;
	
	input.close();

	ifstream input2("InitialConditions/particleCount", ios::in | ios::binary);
	input2.read((char*) &np, sizeof(int));
	::numparticles = np;
	input2.close();
}	

void Initialize_DHIT(ptype *W[5]) {
	ptype Vsqr = 0;	
	ptype *u0, *v0, *w0, *pr0, *d0;
	u0  = new ptype[Nc_seg];
	v0  = new ptype[Nc_seg];
	w0  = new ptype[Nc_seg];
	d0  = new ptype[Nc_seg];
	pr0 = new ptype[Nc_seg];
	
    char buffer[26];

	/** 0 -> j; 1 -> i; 2 -> k **/
    int a=procId[0], b=procId[1], c=procId[2];
    
    snprintf(buffer, sizeof(char) * 26, "InitialConditions/Init%i%i%i", a, b, c);
	
	ifstream input(buffer, ios::in | ios::binary);
	size_t size = sizeof(ptype);
/* -------------------------------------------------------------------------------------*
 * storing primitive initial conds to conservative variable
 * -------------------------------------------------------------------------------------*/
		
	input.read((char*) u0,  size*Nc_seg);
	input.read((char*) v0,  size*Nc_seg);
	input.read((char*) w0,  size*Nc_seg);
	input.read((char*) d0,  size*Nc_seg);
	input.read((char*) pr0, size*Nc_seg);
	
	FOR(j, nc_segy)
		FOR(i, nc_segx)
			FOR(k, nc_segz)
			{
				// we adopt the following index(j,i,k) style for linearization
				int I = Iis(j, i, k);
				
				// Matlab style of 3-D array linearization --> our style equiv. index(k,i,j) 
				int J = k*nc_segy*nc_segx + i*nc_segy + j; 	
				
				Vsqr = (u0[J]*u0[J] + v0[J]*v0[J] + w0[J]*w0[J]);
				W[0][I] = d0[J];
				W[1][I] = u0[J]*d0[J];
				W[2][I] = v0[J]*d0[J];
				W[3][I] = w0[J]*d0[J];
				W[4][I] = d0[J]*(pr0[J]/(d0[J]*(gam-1.0)) + 0.5*Vsqr);
			}
			
	delete [] u0;
	delete [] v0;
	delete [] w0;
	delete [] pr0;
	delete [] d0;
				
	input.close();
}


void PeriodicBC(ptype *W[5])
{
/** X-direction Periodic BC **/ 
	int i;

//	# pragma omp parallel for num_threads(nomp)	
		FOR(j, nt)
			FOR(k, nt)
			{
				i=0;
				W[i][I(j, 0, k)]     = 	W[i][I(j, nt-6, k)];
				W[i][I(j, 1, k)]     = 	W[i][I(j, nt-5, k)];
				W[i][I(j, 2, k)]     = 	W[i][I(j, nt-4, k)];
				
				W[i][I(j, nt-3, k)]  = 	W[i][I(j, 3, k)];
				W[i][I(j, nt-2, k)]  = 	W[i][I(j, 4, k)];
				W[i][I(j, nt-1, k)]  = 	W[i][I(j, 5, k)];								
				
				i = 1;
				W[i][I(j, 0, k)]     = 	W[i][I(j, nt-6, k)];
				W[i][I(j, 1, k)]     = 	W[i][I(j, nt-5, k)];
				W[i][I(j, 2, k)]     = 	W[i][I(j, nt-4, k)];
				
				W[i][I(j, nt-3, k)]  = 	W[i][I(j, 3, k)];
				W[i][I(j, nt-2, k)]  = 	W[i][I(j, 4, k)];
				W[i][I(j, nt-1, k)]  = 	W[i][I(j, 5, k)];								

				i = 2;
				W[i][I(j, 0, k)]     = 	W[i][I(j, nt-6, k)];
				W[i][I(j, 1, k)]     = 	W[i][I(j, nt-5, k)];
				W[i][I(j, 2, k)]     = 	W[i][I(j, nt-4, k)];
				
				W[i][I(j, nt-3, k)]  = 	W[i][I(j, 3, k)];
				W[i][I(j, nt-2, k)]  = 	W[i][I(j, 4, k)];
				W[i][I(j, nt-1, k)]  = 	W[i][I(j, 5, k)];								
				
				i = 3;
				W[i][I(j, 0, k)]     = 	W[i][I(j, nt-6, k)];
				W[i][I(j, 1, k)]     = 	W[i][I(j, nt-5, k)];
				W[i][I(j, 2, k)]     = 	W[i][I(j, nt-4, k)];
				
				W[i][I(j, nt-3, k)]  = 	W[i][I(j, 3, k)];
				W[i][I(j, nt-2, k)]  = 	W[i][I(j, 4, k)];
				W[i][I(j, nt-1, k)]  = 	W[i][I(j, 5, k)];								
				
				i = 4;
				W[i][I(j, 0, k)]     = 	W[i][I(j, nt-6, k)];
				W[i][I(j, 1, k)]     = 	W[i][I(j, nt-5, k)];
				W[i][I(j, 2, k)]     = 	W[i][I(j, nt-4, k)];
				
				W[i][I(j, nt-3, k)]  = 	W[i][I(j, 3, k)];
				W[i][I(j, nt-2, k)]  = 	W[i][I(j, 4, k)];
				W[i][I(j, nt-1, k)]  = 	W[i][I(j, 5, k)];								

			}

/** Y-direction Periodic BC **/
		
//	# pragma omp parallel for num_threads(nomp)	
		FOR(j, nt)
			FOR(k, nt)
			{
				i = 0;
				W[i][I(0, j, k)]     = 	W[i][I(nt-6, j, k)];
				W[i][I(1, j, k)]     = 	W[i][I(nt-5, j, k)];
				W[i][I(2, j, k)]     = 	W[i][I(nt-4, j, k)];
				
				W[i][I(nt-3, j, k)]  = 	W[i][I(3, j, k)];
				W[i][I(nt-2, j, k)]  = 	W[i][I(4, j, k)];				
				W[i][I(nt-1, j, k)]  = 	W[i][I(5, j, k)];				

				i = 1;
				W[i][I(0, j, k)]     = 	W[i][I(nt-6, j, k)];
				W[i][I(1, j, k)]     = 	W[i][I(nt-5, j, k)];
				W[i][I(2, j, k)]     = 	W[i][I(nt-4, j, k)];
				
				W[i][I(nt-3, j, k)]  = 	W[i][I(3, j, k)];
				W[i][I(nt-2, j, k)]  = 	W[i][I(4, j, k)];				
				W[i][I(nt-1, j, k)]  = 	W[i][I(5, j, k)];				

				i = 2;
				W[i][I(0, j, k)]     = 	W[i][I(nt-6, j, k)];
				W[i][I(1, j, k)]     = 	W[i][I(nt-5, j, k)];
				W[i][I(2, j, k)]     = 	W[i][I(nt-4, j, k)];
				
				W[i][I(nt-3, j, k)]  = 	W[i][I(3, j, k)];
				W[i][I(nt-2, j, k)]  = 	W[i][I(4, j, k)];				
				W[i][I(nt-1, j, k)]  = 	W[i][I(5, j, k)];				

				i = 3;
				W[i][I(0, j, k)]     = 	W[i][I(nt-6, j, k)];
				W[i][I(1, j, k)]     = 	W[i][I(nt-5, j, k)];
				W[i][I(2, j, k)]     = 	W[i][I(nt-4, j, k)];
				
				W[i][I(nt-3, j, k)]  = 	W[i][I(3, j, k)];
				W[i][I(nt-2, j, k)]  = 	W[i][I(4, j, k)];				
				W[i][I(nt-1, j, k)]  = 	W[i][I(5, j, k)];				

				i = 4;
				W[i][I(0, j, k)]     = 	W[i][I(nt-6, j, k)];
				W[i][I(1, j, k)]     = 	W[i][I(nt-5, j, k)];
				W[i][I(2, j, k)]     = 	W[i][I(nt-4, j, k)];
				
				W[i][I(nt-3, j, k)]  = 	W[i][I(3, j, k)];
				W[i][I(nt-2, j, k)]  = 	W[i][I(4, j, k)];				
				W[i][I(nt-1, j, k)]  = 	W[i][I(5, j, k)];				

			}

/** Z-direction Periodic BC **/
//	# pragma omp parallel for num_threads(nomp)		
		FOR(j, nt)		
			FOR(k, nt)
			{
				i=0;
				W[i][I(j, k, 0)]     = 	W[i][I(j, k, nt-6)];
				W[i][I(j, k, 1)]     = 	W[i][I(j, k, nt-5)];
				W[i][I(j, k, 2)]     = 	W[i][I(j, k, nt-4)];
				
				W[i][I(j, k, nt-3)]  = 	W[i][I(j, k, 3)];
				W[i][I(j, k, nt-2)]  = 	W[i][I(j, k, 4)];				
				W[i][I(j, k, nt-1)]  = 	W[i][I(j, k, 5)];

				i=1;
				W[i][I(j, k, 0)]     = 	W[i][I(j, k, nt-6)];
				W[i][I(j, k, 1)]     = 	W[i][I(j, k, nt-5)];
				W[i][I(j, k, 2)]     = 	W[i][I(j, k, nt-4)];
				
				W[i][I(j, k, nt-3)]  = 	W[i][I(j, k, 3)];
				W[i][I(j, k, nt-2)]  = 	W[i][I(j, k, 4)];				
				W[i][I(j, k, nt-1)]  = 	W[i][I(j, k, 5)];

				i=2;
				W[i][I(j, k, 0)]     = 	W[i][I(j, k, nt-6)];
				W[i][I(j, k, 1)]     = 	W[i][I(j, k, nt-5)];
				W[i][I(j, k, 2)]     = 	W[i][I(j, k, nt-4)];
				
				W[i][I(j, k, nt-3)]  = 	W[i][I(j, k, 3)];
				W[i][I(j, k, nt-2)]  = 	W[i][I(j, k, 4)];				
				W[i][I(j, k, nt-1)]  = 	W[i][I(j, k, 5)];

				i=3;
				W[i][I(j, k, 0)]     = 	W[i][I(j, k, nt-6)];
				W[i][I(j, k, 1)]     = 	W[i][I(j, k, nt-5)];
				W[i][I(j, k, 2)]     = 	W[i][I(j, k, nt-4)];
				
				W[i][I(j, k, nt-3)]  = 	W[i][I(j, k, 3)];
				W[i][I(j, k, nt-2)]  = 	W[i][I(j, k, 4)];				
				W[i][I(j, k, nt-1)]  = 	W[i][I(j, k, 5)];

				i=4;
				W[i][I(j, k, 0)]     = 	W[i][I(j, k, nt-6)];
				W[i][I(j, k, 1)]     = 	W[i][I(j, k, nt-5)];
				W[i][I(j, k, 2)]     = 	W[i][I(j, k, nt-4)];
				
				W[i][I(j, k, nt-3)]  = 	W[i][I(j, k, 3)];
				W[i][I(j, k, nt-2)]  = 	W[i][I(j, k, 4)];				
				W[i][I(j, k, nt-1)]  = 	W[i][I(j, k, 5)];
				
			}	
}


void segment() {

/*  Right now only works for even procDims  */
		
	::sta[0]   = procId[0]*nc_segy;
	::sta[1]   = procId[1]*nc_segx;
	::sta[2]   = procId[2]*nc_segz;
	::end[0]   = (sta[0] + nt_segy);
	::end[1]   = (sta[1] + nt_segx);
	::end[2]   = (sta[2] + nt_segz);
}

void segmentFill(ptype *W[5], ptype *W0[5]) {
	
	For(j, nt_segy)
		For(i, nt_segx)
			For(k, nt_segz) { 
				/** 0 -> j; 1 -> i; 2 -> k **/
				int segId = I(sta[0]+j, sta[1]+i, sta[2]+k);
				int segFillId = Is(j, i, k);
				
				FOR(q, 5){				
					W[q][segFillId] = W0[q][segId];
				}
			}	
}

void PrintParams() {
	
	if (procId[0] + procId[1] + procId[2] == 0) {
		cout << endl << "GKM DNS DHIT" << endl; 
		cout << "Initializing Field ....." <<endl;
		cout << "______________________________" <<endl <<endl;
		cout << "nc            =  " << nc <<endl; 
		cout << "nc_segy       =  " << nc_segy <<endl;
		cout << "nc_segx       =  " << nc_segx <<endl;
		cout << "nc_segz       =  " << nc_segz <<endl;
		cout << "T0            =  " << T0 <<endl;
		cout << "Mt            =  " << Mt <<endl;
		cout << "Re            =  " << Re <<endl;			
		cout << "dx            =  " << dx <<endl;
		cout << "eddy time     =  " << t0 <<endl;
		cout << "p0            =  " << p0 <<endl;
		cout << "den0          =  " << den0 <<endl;	
		cout << "gam           =  " << gam <<endl;	
		cout << "mu0           =  " << mu0 <<endl;
		cout << "K             =  " << K <<endl;
		cout << "CFL           =  " << cfl <<endl;
		cout << "Prandtl No.   =  " << PRN <<endl;		
		cout << "numParticles  =  " << numparticles <<endl;		
		cout << "nprocs    =  " << procDim[0] <<" x "<< procDim[1] <<" x "<< procDim[2] <<endl;		
		cout << endl << "Field Initialized ! " <<endl;
		cout << "______________________________" <<endl;
		cout << endl << "TIME              TKE" <<endl;
		cout << "______________________________" <<endl <<endl;		
	}
}	
