#include "update.h"
using namespace std;

#define InfoSz 8
#define writeInfoSz 42
ptype deni, ui, vi, wi, Ti, Pi, Vsqr, dt, t, KE, part_DISS;
ptype dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, mu;
ptype x, y, z, gradU[3], gradV[3], gradW[3];

int nbrN, nbrS, nbrL, nbrR, nbrF, nbrB, count, totalCount, sendSz, iter;  
int nbr[27];

ptype tau, temp;
ptype F[5],   WL[5],   WR[5];

#ifdef UseCPU
	ptype WLL[5], WRR[5],  WLLL[5], WRRR[5], 
	      WLN[5], WRN[5], WLLN[5], WRRN[5], WLLLN[5], WRRRN[5],		
	      WLS[5], WRS[5], WLLS[5], WRRS[5], WLLLS[5], WRRRS[5],
	      WLF[5], WRF[5], WLLF[5], WRRF[5], WLLLF[5], WRRRF[5],
	      WLB[5], WRB[5], WLLB[5], WRRB[5], WLLLB[5], WRRRB[5],		
	      Wl[5],  Wr[5],  DWxl[5], DWxr[5], DWyl[5], DWyr[5], DWzl[5], DWzr[5],
		  WlN[5], WrN[5], WlS[5],  WrS[5],  WlF[5],  WrF[5],  WlB[5],  WrB[5];

	ptype P0p, P0n, P1p, P1n, P2p, P2n, P3p, P3n, IS0p, IS0n, IS1p, IS1n, IS2p, IS2n, IS3p, 
	      IS3n,  ALPHA0p, ALPHA0n, ALPHA1p, ALPHA1n, ALPHA2p, ALPHA2n, ALPHA3p, ALPHA3n, sum_ALPHAp, sum_ALPHAn, 
	      OMEGA0p, OMEGA0n, OMEGA1p, OMEGA1n, OMEGA2p, OMEGA2n, OMEGA3p, OMEGA3n, ISPMAX, ISNMAX, TL, TLL, TLLL, TR, TRR, TRRR;
#endif

ptype localSijSij, SijSij, part_SijSij, TKE, part_TKE, localTKE, U, localVAL, maxVAL, Rex, valC, localmaxVAL, 
		avg, localDUDX2, localDUDX3, localDUDX4, DUDX2, DUDX3, DUDX4, DVDY4, DWDZ4, SKEWu, part_SKEWu, localDiss, Diss, localDVDY2, localDVDY3, localDVDY4, localDWDZ2, localDWDZ3, localDWDZ4,
		DVDY2, DVDY3, DWDZ2, DWDZ3, SKEWv, SKEWw, localmu, part_SKEWw, KURTu, KURTv, KURTw;

int halosz, halosz5;	
MPI_Comm COMM3D;
/*-----------------------------------------------------------------------------------------------------------------------------
 *----------------------------------------------------------------------------------------------------------------------------*/

void Evolve(ptype *W[5], MPI_Comm comm3d)
{

/** ----------------------------------------------------------------------------
 *  FIND CART NEIGHBORS
 * -----------------------------------------------------------------------------*/
	COMM3D = comm3d;
	
	MPI_Cart_shift(COMM3D, 0, 1, &nbrS, &nbrN);
	MPI_Cart_shift(COMM3D, 1, 1, &nbrL, &nbrR);
	MPI_Cart_shift(COMM3D, 2, 1, &nbrB, &nbrF);	
	findAllNbrs();

/** ----------------------------------------------------------------------------
 *   GPU VARIABLES DECLERATION
 * -----------------------------------------------------------------------------*/	
	#ifdef UseGPU
		DevAlloc *dev = new DevAlloc();
	#endif

/** ----------------------------------------------------------------------------
 *   CPU VARIABLES DECLERATION
 * -----------------------------------------------------------------------------*/
	#ifdef UseCPU  	
		ptype *Fx[5], *Fy[5], *Fz[5];	
		MAKE5s(Fx);	MAKE5s(Fy);	MAKE5s(Fz);
	#endif

	ptype *den = new ptype[Nt_seg], *u = new ptype[Nt_seg],
		  *v   = new ptype[Nt_seg], *w = new ptype[Nt_seg],
		  *p   = new ptype[Nt_seg], *lapU = new ptype[Nt_seg], 
          *lapV = new ptype[Nt_seg], *lapW = new ptype[Nt_seg];
          	
    UBspline_3d_d *splinep[8];
    UBspline_3d_d *splinepn[3];

	halosz  = nt_segy*nt_segz*3;
	halosz5 = 5*halosz;
    
	ptype *send_HaloL = new ptype[halosz5], *recv_HaloL = new ptype[halosz5],
		  *send_HaloR = new ptype[halosz5], *recv_HaloR = new ptype[halosz5];

	halosz  = nt_segx*nt_segz*3;
	halosz5 = 5*halosz;
		  
	ptype *send_HaloN = new ptype[halosz5], *recv_HaloN = new ptype[halosz5],
		  *send_HaloS = new ptype[halosz5], *recv_HaloS = new ptype[halosz5],

	halosz  = nt_segx*nt_segy*3;
	halosz5 = 5*halosz;
	ptype *send_HaloF = new ptype[halosz5], *recv_HaloF = new ptype[halosz5],
		  *send_HaloB = new ptype[halosz5], *recv_HaloB = new ptype[halosz5];

	sendSz = InfoSz*int(numparticles/(nc_segx*nprocs));

	ptype *sendParticle[27];
	ptype *recvParticle[27]; 
	
	FOR(i, 27) {
		sendParticle[i] = new ptype[sendSz];
		recvParticle[i] = new ptype[sendSz];
	}

/** ----------------------------------------------------------------------------
 *   Initializing Particles
 * -----------------------------------------------------------------------------*/
					
	HaloTransfer(W, send_HaloL, recv_HaloL, send_HaloR, recv_HaloR, 
					send_HaloS, recv_HaloS, send_HaloN, recv_HaloN,
					send_HaloB, recv_HaloB, send_HaloF, recv_HaloF);

	generateSplines(W, splinep, den, u, v, w, p, lapU, lapV, lapW);		  		

	timestep(W);
	t = 0; count = 0;
	Particle *particles = new Particle[int(1.5*numparticles/nprocs)];	
	locateParticles(particles, splinep);
	iter = 0;
/*--------------------------------------------------------------------------------------
 *  MAIN SOLVER LOOP BEGINS
 *-------------------------------------------------------------------------------------*/	
	while (t <= tmax) {
        
        generateSplines(W, splinep, den, u, v, w, p, lapU, lapV, lapW);		  		

	    if(iter%SAVE_INTERVAL == 0) {
			writeField(den, u, v, w, p);
			WriteReInit(W);
	    }

		FOR (q, count) {
			particles[q].getParticleProp(splinep);
		}
		GetParticleStats(particles, splinep);

		if(iter%10 == 0) {
			writeParticleData(particles);
			PrintStats(W, splinep);
		}
				
		#ifdef UseCPU
				evolveEulerian(W, Fx, Fy, Fz);
		#endif

		#ifdef UseGPU
				evolveEulerian(W, dev);
		#endif
		
		HaloTransfer(W, send_HaloL, recv_HaloL, send_HaloR, recv_HaloR, 
						send_HaloS, recv_HaloS, send_HaloN, recv_HaloN,
						send_HaloB, recv_HaloB, send_HaloF, recv_HaloF);

		generateSplines(W, splinepn, u, v, w);
		evolveParticles(particles, splinep, splinepn, sendParticle, recvParticle);


		FOR(i, 8)
			destroy_Bspline(splinep[i]);
		
		FOR(i, 3)
			destroy_Bspline(splinepn[i]);

		timestep(W);
		t += dt;
		iter++;		
	
	}
/*--------------------------------------------------------------------------------------
 *  MAIN SOLVER LOOP ENDS
 *-------------------------------------------------------------------------------------*/	


/** NOTE : DELETE ALL VARIABLES 
 */
	delete [] den, u, v , w, p,
			  send_HaloL, recv_HaloL, send_HaloR, recv_HaloR, send_HaloN, 
			  recv_HaloN, send_HaloS, recv_HaloS, send_HaloF, recv_HaloF, send_HaloB, recv_HaloB;

	#ifdef UseCPU	
		FOR(q, 5)
			delete[] Fx[q], Fy[q], Fz[q];
	#endif

	FOR(q, 27)
		delete[] sendParticle[q], recvParticle[q];

}


/** ----------------------------------------------------------------------------
 *   Evolve Field
 * -----------------------------------------------------------------------------*/
#ifdef UseGPU
void evolveEulerian(ptype *W[5], DevAlloc *dev) {
	evolve(W, dev, dt);
}
#endif




#ifdef UseCPU

/** ----------------------------------------------------------------------------
 *   Flux Calculation Routines [CPU]
 * -----------------------------------------------------------------------------*/

void evolveEulerian(ptype *W[5], ptype *Fx[5], ptype *Fy[5], ptype *Fz[5]) {
	derivsX(W, Fx);
	derivsY(W, Fy);
	derivsZ(W, Fz);
	update(W, Fx, Fy, Fz);
}

void derivsX(ptype *W[5], ptype *Fx[5])
{	
//	# pragma omp parallel for num_threads(nomp)
	
	For(j, nt_segy)
		FoR(i, nt_segx)
		{	For(k, nt_segz)
			{
				FOR(q, 5)
				{	
					WL  [q] = W[q][Is(j, i,   k)];
					WLL [q] = W[q][Is(j, i-1, k)];
					WLLL[q] = W[q][Is(j, i-2, k)];
					WR  [q] = W[q][Is(j, i+1, k)];
					WRR [q] = W[q][Is(j, i+2, k)];
					WRRR[q] = W[q][Is(j, i+3, k)];
					
					WLN  [q] = W[q][Is(j+1, i,   k)];
					WLLN [q] = W[q][Is(j+1, i-1, k)];
					WLLLN[q] = W[q][Is(j+1, i-2, k)];
					WRN  [q] = W[q][Is(j+1, i+1, k)];
					WRRN [q] = W[q][Is(j+1, i+2, k)];
					WRRRN[q] = W[q][Is(j+1, i+3, k)];
										
					WLS  [q] = W[q][Is(j-1, i,   k)];
					WLLS [q] = W[q][Is(j-1, i-1, k)];
					WLLLS[q] = W[q][Is(j-1, i-2, k)];
					WRS  [q] = W[q][Is(j-1, i+1, k)];
					WRRS [q] = W[q][Is(j-1, i+2, k)];
					WRRRS[q] = W[q][Is(j-1, i+3, k)];
															
					WLF  [q] = W[q][Is(j, i,   k+1)];
					WLLF [q] = W[q][Is(j, i-1, k+1)];
					WLLLF[q] = W[q][Is(j, i-2, k+1)];
					WRF  [q] = W[q][Is(j, i+1, k+1)];
					WRRF [q] = W[q][Is(j, i+2, k+1)];
					WRRRF[q] = W[q][Is(j, i+3, k+1)];
										
					WLB  [q] = W[q][Is(j, i,   k-1)];
					WLLB [q] = W[q][Is(j, i-1, k-1)];
					WLLLB[q] = W[q][Is(j, i-2, k-1)];
					WRB  [q] = W[q][Is(j, i+1, k-1)];
					WRRB [q] = W[q][Is(j, i+2, k-1)];
					WRRRB[q] = W[q][Is(j, i+3, k-1)];
				}

				ApplyWENO(WLLL, WLL, WL, WR, WRR, WRRR, Wl, Wr);
				ApplyWENO(WLLLN, WLLN, WLN, WRN, WRRN, WRRRN, WlN, WrN);
				ApplyWENO(WLLLS, WLLS, WLS, WRS, WRRS, WRRRS, WlS, WrS);
				ApplyWENO(WLLLF, WLLF, WLF, WRF, WRRF, WRRRF, WlF, WrF);
				ApplyWENO(WLLLB, WLLB, WLB, WRB, WRRB, WRRRB, WlB, WrB);

				FOR(q, 5) {
					DWxl[q] = 2*(Wl[q] - WL[q]) / dx;
					DWxr[q] = 2*(WR[q] - Wr[q]) / dx;
					DWyl[q] = 0.5*(WlN[q] - WlS[q]) / dx;
					DWyr[q] = 0.5*(WrN[q] - WrS[q]) / dx;
					DWzl[q] = 0.5*(WlF[q] - WlB[q]) / dx;
					DWzr[q] = 0.5*(WrF[q] - WrB[q]) / dx;
				}
				
				c2p();
				mu = mu0*pow((Ti/T0),1.5)*(T0 + 110.4)/(Ti + 110.4);
				tau = mu / Pi; 
				
			    flux(WL, WR, Wl, Wr, DWxl, DWxr, DWyl, DWyr, DWzl, DWzr, tau, dt, dx, F);
				//flux(Wl, DWxl, DWyl, DWzl, tau, dt, dx, F);
				FOR(r, 5)
					Fx[r][Is(j, i, k)] = F[r];
			
			}
		}
}

void derivsY(ptype *W[5], ptype *Fy[5])
{	
//	# pragma omp parallel for num_threads(nomp)	
	FoR(j, nt_segy)
		For(i, nt_segx)
			For(k, nt_segz)
			{
				FOR(q, 5)
				{	
					WL  [q] = W[q][Is(j,   i, k)];
					WLL [q] = W[q][Is(j-1, i, k)];
					WLLL[q] = W[q][Is(j-2, i, k)];
					WR  [q] = W[q][Is(j+1, i, k)];					
					WRR [q] = W[q][Is(j+2, i, k)];
					WRRR[q] = W[q][Is(j+3, i, k)];
					
					WLN  [q] = W[q][Is(j,   i-1, k)];
					WLLN [q] = W[q][Is(j-1, i-1, k)];
					WLLLN[q] = W[q][Is(j-2, i-1, k)];
					WRN  [q] = W[q][Is(j+1, i-1, k)];
					WRRN [q] = W[q][Is(j+2, i-1, k)];
					WRRRN[q] = W[q][Is(j+3, i-1, k)];
										
					WLS  [q] = W[q][Is(j,   i+1, k)];
					WLLS [q] = W[q][Is(j-1, i+1, k)];
					WLLLS[q] = W[q][Is(j-2, i+1, k)];
					WRS  [q] = W[q][Is(j+1, i+1, k)];
					WRRS [q] = W[q][Is(j+2, i+1, k)];
					WRRRS[q] = W[q][Is(j+3, i+1, k)];
															
					WLF  [q] = W[q][Is(j,   i, k+1)];
					WLLF [q] = W[q][Is(j-1, i, k+1)];
					WLLLF[q] = W[q][Is(j-2, i, k+1)];
					WRF  [q] = W[q][Is(j+1, i, k+1)];
					WRRF [q] = W[q][Is(j+2, i, k+1)];
					WRRRF[q] = W[q][Is(j+3, i, k+1)];
										
					WLB  [q] = W[q][Is(j,   i, k-1)];
					WLLB [q] = W[q][Is(j-1, i, k-1)];
					WLLLB[q] = W[q][Is(j-2, i, k-1)];
					WRB  [q] = W[q][Is(j+1, i, k-1)];
					WRRB [q] = W[q][Is(j+2, i, k-1)];
					WRRRB[q] = W[q][Is(j+3, i, k-1)];
															
				}
				
				ApplyWENO(WLLL, WLL, WL, WR, WRR, WRRR, Wl, Wr);
				ApplyWENO(WLLLN, WLLN, WLN, WRN, WRRN, WRRRN, WlN, WrN);
				ApplyWENO(WLLLS, WLLS, WLS, WRS, WRRS, WRRRS, WlS, WrS);
				ApplyWENO(WLLLF, WLLF, WLF, WRF, WRRF, WRRRF, WlF, WrF);
				ApplyWENO(WLLLB, WLLB, WLB, WRB, WRRB, WRRRB, WlB, WrB);

				FOR(q, 5) {
					DWxl[q] = 2*(Wl[q] - WL[q]) / dx;
					DWxr[q] = 2*(WR[q] - Wr[q]) / dx;
					DWyl[q] = -0.5*(WlN[q] - WlS[q]) / dx;
					DWyr[q] = -0.5*(WrN[q] - WrS[q]) / dx;
					DWzl[q] =  0.5*(WlF[q] - WlB[q]) / dx;
					DWzr[q] =  0.5*(WrF[q] - WrB[q]) / dx;
				}
								
				c2p();
				mu = mu0*pow((Ti/T0),1.5)*(T0 + 110.4)/(Ti + 110.4);
				tau = mu / Pi; 
				
				temp = WL[2];   WL[2] = WL[1];   WL[1] =  temp;
				temp = WR[2];   WR[2] = WR[1];   WR[1] =  temp;
				temp = Wl[2];   Wl[2] = Wl[1];   Wl[1] =  temp;
				temp = Wr[2];   Wr[2] = Wr[1];   Wr[1] =  temp;

				temp = DWxl[2]; DWxl[2] = DWxl[1]; DWxl[1] =  temp;
				temp = DWxr[2]; DWxr[2] = DWxr[1]; DWxr[1] =  temp;
				temp = DWyl[2]; DWyl[2] = DWyl[1]; DWyl[1] =  temp;
				temp = DWyr[2]; DWyr[2] = DWyr[1]; DWyr[1] =  temp;
				temp = DWzl[2]; DWzl[2] = DWzl[1]; DWzl[1] =  temp;
				temp = DWzr[2]; DWzr[2] = DWzr[1]; DWzr[1] =  temp;
				
			    flux(WL, WR, Wl, Wr, DWxl, DWxr, DWyl, DWyr, DWzl, DWzr, tau, dt, dx, F);
				//flux(Wl, DWxl, DWyl, DWzl, tau, dt, dx, F);
                
				FOR(r, 5)
					Fy[r][Is(j, i, k)] = F[r];
				
				temp  =  Fy[2][Is(j, i, k)]; 
				Fy[2][Is(j, i, k)] =  Fy[1][Is(j, i, k)];
				Fy[1][Is(j, i, k)] =  temp;
			}

}	

void derivsZ(ptype *W[5], ptype *Fz[5])
{	
//	# pragma omp parallel for num_threads(nomp)	
	For(j, nt_segy)
		For(i, nt_segx)
		{	FoR(k, nt_segz)
			{
				FOR(q, 5)
				{	
					WL  [q] = W[q][Is(j, i, k)];
					WLL [q] = W[q][Is(j, i, k-1)];
					WLLL[q] = W[q][Is(j, i, k-2)];
					WR  [q] = W[q][Is(j, i, k+1)];
					WRR [q] = W[q][Is(j, i, k+2)];
					WRRR[q] = W[q][Is(j, i, k+3)];
					
					WLN  [q] = W[q][Is(j, i+1, k)];
					WLLN [q] = W[q][Is(j, i+1, k-1)];
					WLLLN[q] = W[q][Is(j, i+1, k-2)];
					WRN  [q] = W[q][Is(j, i+1, k+1)];
					WRRN [q] = W[q][Is(j, i+1, k+2)];
					WRRRN[q] = W[q][Is(j, i+1, k+3)];
										
					WLS  [q] = W[q][Is(j, i-1, k)];
					WLLS [q] = W[q][Is(j, i-1, k-1)];
					WLLLS[q] = W[q][Is(j, i-1, k-2)];
					WRS  [q] = W[q][Is(j, i-1, k+1)];
					WRRS [q] = W[q][Is(j, i-1, k+2)];
					WRRRS[q] = W[q][Is(j, i-1, k+3)];
										
					WLF  [q] = W[q][Is(j+1, i, k)];
					WLLF [q] = W[q][Is(j+1, i, k-1)];
					WLLLF[q] = W[q][Is(j+1, i, k-2)];
					WRF  [q] = W[q][Is(j+1, i, k+1)];
					WRRF [q] = W[q][Is(j+1, i, k+2)];
					WRRRF[q] = W[q][Is(j+1, i, k+3)];
							
					WLB  [q] = W[q][Is(j-1, i, k)];
					WLLB [q] = W[q][Is(j-1, i, k-1)];
					WLLLB[q] = W[q][Is(j-1, i, k-2)];
					WRB  [q] = W[q][Is(j-1, i, k+1)];
					WRRB [q] = W[q][Is(j-1, i, k+2)];
					WRRRB[q] = W[q][Is(j-1, i, k+3)];
				
				}

				ApplyWENO(WLLL, WLL, WL, WR, WRR, WRRR, Wl, Wr);
				ApplyWENO(WLLLN, WLLN, WLN, WRN, WRRN, WRRRN, WlN, WrN);
				ApplyWENO(WLLLS, WLLS, WLS, WRS, WRRS, WRRRS, WlS, WrS);
				ApplyWENO(WLLLF, WLLF, WLF, WRF, WRRF, WRRRF, WlF, WrF);
				ApplyWENO(WLLLB, WLLB, WLB, WRB, WRRB, WRRRB, WlB, WrB);

				FOR(q, 5) {
					DWxl[q] = 2*(Wl[q] - WL[q]) / dx;
					DWxr[q] = 2*(WR[q] - Wr[q]) / dx;
					DWyl[q] = 0.5*(WlN[q] - WlS[q]) / dx;
					DWyr[q] = 0.5*(WrN[q] - WrS[q]) / dx;
					DWzl[q] = 0.5*(WlF[q] - WlB[q]) / dx;
					DWzr[q] = 0.5*(WrF[q] - WrB[q]) / dx;
				}
								
				c2p();
				mu = mu0*pow((Ti/T0),1.5)*(T0 + 110.4)/(Ti + 110.4);
				tau = mu / Pi; 
				
				temp = WL[3];   WL[3] = WL[1];   WL[1] =  temp;
				temp = WR[3];   WR[3] = WR[1];   WR[1] =  temp;
				temp = Wl[3];   Wl[3] = Wl[1];   Wl[1] =  temp;
				temp = Wr[3];   Wr[3] = Wr[1];   Wr[1] =  temp;

				temp = DWxl[3]; DWxl[3] = DWxl[1]; DWxl[1] =  temp;
				temp = DWxr[3]; DWxr[3] = DWxr[1]; DWxr[1] =  temp;
				temp = DWyl[3]; DWyl[3] = DWyl[1]; DWyl[1] =  temp;
				temp = DWyr[3]; DWyr[3] = DWyr[1]; DWyr[1] =  temp;
				temp = DWzl[3]; DWzl[3] = DWzl[1]; DWzl[1] =  temp;
				temp = DWzr[3]; DWzr[3] = DWzr[1]; DWzr[1] =  temp;

			    flux(WL, WR, Wl, Wr, DWxl, DWxr, DWyl, DWyr, DWzl, DWzr, tau, dt, dx, F);
				//flux(Wl, DWxl, DWyl, DWzl, tau, dt, dx, F);
				
                FOR(r, 5)
					Fz[r][Is(j, i, k)] = F[r];

				temp  =  Fz[3][Is(j, i, k)]; 
				Fz[3][Is(j, i, k)] =  Fz[1][Is(j, i, k)];
				Fz[1][Is(j, i, k)] =  temp;

			}
		}
	
}

void update(ptype *W[5], ptype *Fx[5], ptype *Fy[5], ptype *Fz[5])
{
	For(j, nt_segy)
		For(i, nt_segx)
			For(k, nt_segz)
				FOR(q, 5)
				{					
					W[q][Is(j, i, k)]  -= (1/dx) * (
										  Fx[q][Is(j, i, k)] - Fx[q][Is(j, i-1, k)]
										           +
										  Fy[q][Is(j, i, k)] - Fy[q][Is(j-1, i, k)] 
												   +
										  Fz[q][Is(j, i, k)] - Fz[q][Is(j, i, k-1)] );
				}           
}

void ApplyLimiter(ptype WLL[5], ptype WL[5], ptype WR[5], ptype WRR[5],
					         ptype dx, ptype Wl[5], ptype Wr[5])
{
	double Ll[5];	double Lr[5];	
	double sl[5];	double sr[5];	double srr[5];
			
	for (int i=0;i<5;i++)
	{
		sl[i]=0;	sr[i]=0;	srr[i]=0;	Ll[i]=0;	Lr[i]=0;
		
		sr [i] = (WR [i] - WL [i])/dx;
		sl [i] = (WL [i] - WLL[i])/dx;
		srr[i] = (WRR[i] - WR [i])/dx;
	}
	
/*  Van Leer Limiter    
 */
	for (int i=0;i<5;i++)
	{
		Ll[i] = (sign(sr [i])+sign(sl[i]))*(fabs(sr[i])*fabs(sl[i])) /
				   						   (fabs(sr[i])+fabs(sl[i])+eps);
		Lr[i] = (sign(srr[i])+sign(sr[i]))*(fabs(srr[i])*fabs(sr[i])) /
		             					   (fabs(srr[i])+fabs(sr[i])+eps);
		Wl[i] = WL[i] + Ll[i]*(dx/2);
		Wr[i] = WR[i] - Lr[i]*(dx/2);
	}
}

/**
 *  WENO-SYMOO
 */
void ApplyWENO(ptype WLLL[5], ptype WLL[5], ptype WL[5], ptype WR[5],
		         ptype WRR[5], ptype WRRR[5], ptype Wl[5], ptype Wr[5])
{	
	ptype gam0 = 1.0/20.0, gam1 = 9.0/20.0, gam2 = 9.0/20.0, gam3 = 1.0/20.0;
	
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

	TL   = (gam-1) * ( WL[4]/WL[0]     - 0.5*(WL[1]*WL[1] + WL[2]*WL[2] + WL[3]*WL[3])/(WL[0]*WL[0]) ) / R;
	TLL  = (gam-1) * ( WLL[4]/WLL[0]   - 0.5*(WLL[1]*WLL[1] + WLL[2]*WLL[2] + WLL[3]*WLL[3])/(WLL[0]*WLL[0]) ) / R;
	TLLL = (gam-1) * ( WLLL[4]/WLLL[0] - 0.5*(WLLL[1]*WLLL[1] + WLLL[2]*WLLL[2] + WLLL[3]*WLLL[3])/(WLLL[0]*WLLL[0]) ) / R;
	TR   = (gam-1) * ( WR[4]/WR[0]     - 0.5*(WR[1]*WR[1] + WR[2]*WR[2] + WR[3]*WR[3])/(WR[0]*WR[0]) ) / R;
	TRR  = (gam-1) * ( WRR[4]/WRR[0]   - 0.5*(WRR[1]*WRR[1] + WRR[2]*WRR[2] + WRR[3]*WRR[3])/(WRR[0]*WRR[0]) ) / R;
	TRRR = (gam-1) * ( WRRR[4]/WRRR[0] - 0.5*(WRRR[1]*WRRR[1] + WRRR[2]*WRRR[2] + WRRR[3]*WRRR[3])/(WRRR[0]*WRRR[0]) ) / R;
	
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
	
	Wl[4] = Wl[0]*R*Tl/(gam-1.0) + 0.5*(Wl[1]*Wl[1]+Wl[2]*Wl[2]+Wl[3]*Wl[3])/Wl[0];
	Wr[4] = Wr[0]*R*Tr/(gam-1.0) + 0.5*(Wr[1]*Wr[1]+Wr[2]*Wr[2]+Wr[3]*Wr[3])/Wr[0];	
/*
	FOR(q,5) {
		Wl[q] = 0.5*(WL[q] + WR[q]); Wr[q] = Wl[q];
	}
*/
}


#endif




/** ----------------------------------------------------------------------------
 *   Particle Tracking
 * -----------------------------------------------------------------------------*/


void PrintStats(ptype *W[5], UBspline_3d_d *splinep[8])
{	
	localSijSij = 0.0, SijSij = 0.0, TKE = 0.0,  localTKE = 0.0;
	localDUDX2 = 0.0, localDUDX3 = 0.0, localDUDX4 = 0.0, DUDX2 = 0.0, DUDX3 = 0.0, DUDX4 = 0.0, SKEWu = 0.0, KURTu = 0.0;
	localDVDY2 = 0.0, localDVDY3 = 0.0, localDVDY4 = 0.0, DVDY2 = 0.0, DVDY3 = 0.0, DVDY4 = 0.0, SKEWv = 0.0, KURTv = 0.0;
	localDWDZ2 = 0.0, localDWDZ3 = 0.0, localDWDZ4 = 0.0, DWDZ2 = 0.0, DWDZ3 = 0.0, DWDZ4 = 0.0, SKEWw = 0.0, KURTw = 0.0;
	localDiss = 0.0, Diss = 0;

	ptype localPmean = 0.0, localPrms = 0.0,  localTmean = 0.0, localTrms = 0.0,  localVrms = 0.0,  localVmean = 0.0,
		  localumean = 0.0, localurms = 0.0,  localvmean = 0.0, localvrms = 0.0,  localwmean = 0.0, localwrms = 0.0;
	
	ptype Pmean = 0.0;	ptype Prms = 0.0;
	ptype Tmean = 0.0;	ptype Trms = 0.0;
	ptype Vmean = 0.0;	ptype Vrms = 0.0;
	
	ptype umean = 0.0;	ptype urms = 0.0;
	ptype vmean = 0.0;	ptype vrms = 0.0;
	ptype wmean = 0.0;	ptype wrms = 0.0;
    
	cout.precision(8);
	
    	For(j, nt_segy) 
		For(i, nt_segx)
			For(k, nt_segz)
				{

					FOR(q, 5)
						WL[q] = W[q][Is(j, i, k)];							
					c2p();
					localPmean += Pi;
					localTmean += Ti;
					localVmean += 1/deni;

					y = (sta[0] + j-3)*dx + 0.5*dx;
					x = (sta[1] + i-3)*dx + 0.5*dx;
					z = (sta[2] + k-3)*dx + 0.5*dx;
					
					eval_UBspline_3d_d (splinep[0] , y, x, z, &deni);
					eval_UBspline_3d_d_vg (splinep[1] , y, x, z, &ui, gradU );
					eval_UBspline_3d_d_vg (splinep[2] , y, x, z, &vi, gradV );
					eval_UBspline_3d_d_vg (splinep[3] , y, x, z, &wi, gradW );
					
					eval_UBspline_3d_d (splinep[1] , y, x, z, &ui);
					eval_UBspline_3d_d (splinep[2] , y, x, z, &vi);
					eval_UBspline_3d_d (splinep[3] , y, x, z, &wi);

					dudx = gradU[1];
					dudy = gradU[0];
					dudz = gradU[2];
					dvdx = gradV[1];
					dvdy = gradV[0];
					dvdz = gradV[2];
					dwdx = gradW[1];
					dwdy = gradW[0];
					dwdz = gradW[2];
					
					avg = (dudx + dvdy + dwdz)/3;
					localmu = mu0*pow((Ti/T0),1.5)*(T0 + 110.4)/(Ti + 110.4);
					localDiss  += localmu*( dudx*dudx + dudy*dudy + dudz*dudz + dvdx*dvdx + dvdy*dvdy + dvdz*dvdz + dwdx*dwdx + dwdy*dwdy + dwdz*dwdz);
					localTKE += deni*(ui*ui + vi*vi + wi*wi);				
					localSijSij  += ( dudx*dudx + 0.5*pow(dudy + dvdx, 2) + 0.5*pow(dudz + dwdx, 2) + 0.5*pow(dwdy + dvdz, 2) + dvdy*dvdy + dwdz*dwdz);

					localDUDX2 += dudx*dudx;
					localDUDX3 += dudx*dudx*dudx;
					localDUDX4 += dudx*dudx*dudx*dudx;
					localDVDY2 += dvdy*dvdy;
					localDVDY3 += dvdy*dvdy*dvdy;
					localDVDY4 += dvdy*dvdy*dvdy*dvdy;
					localDWDZ2 += dwdz*dwdz;
					localDWDZ3 += dwdz*dwdz*dwdz;					
					localDWDZ4 += dwdz*dwdz*dwdz*dwdz;					
			}
	
	MPI_Reduce(&localTKE, &TKE, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDiss, &Diss, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localSijSij, &SijSij, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDUDX2, &DUDX2, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDUDX3, &DUDX3, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDUDX4, &DUDX4, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDVDY2, &DVDY2, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDVDY3, &DVDY3, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDVDY4, &DVDY4, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDWDZ2, &DWDZ2, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDWDZ3, &DWDZ3, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDWDZ4, &DWDZ4, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);

	TKE    /= Nc;
	Diss   /= Nc;
	SijSij /= Nc;
	DUDX2  /= Nc;
	DUDX3  /= Nc;
	DUDX4  /= Nc;
	DVDY2  /= Nc;
	DVDY3  /= Nc;
	DVDY4  /= Nc;
	DWDZ2  /= Nc;
	DWDZ3  /= Nc;
	DWDZ4  /= Nc;

	SKEWu = DUDX3 / pow(DUDX2, 1.5); SKEWv = DVDY3 / pow(DVDY2, 1.5); SKEWw = DWDZ3 / pow(DWDZ2, 1.5);
	KURTu = DUDX4 / pow(DUDX2, 2); KURTv = DVDY4 / pow(DVDY2, 2); KURTw = DWDZ4 / pow(DWDZ2, 2);
			
	MPI_Allreduce(&localPmean, &Pmean, 1, MPI_DOUBLE, MPI_SUM, COMM3D);
	MPI_Allreduce(&localTmean, &Tmean, 1, MPI_DOUBLE, MPI_SUM, COMM3D);
	MPI_Allreduce(&localVmean, &Vmean, 1, MPI_DOUBLE, MPI_SUM, COMM3D);
	
	Pmean /= Nc;
	Tmean /= Nc;
	Vmean /= Nc;
		
	For(j, nt_segy) 
		For(i, nt_segx)
			For(k, nt_segz)
				{
					FOR(q, 5)
						WL[q] = W[q][Is(j, i, k)]; 										
					
					c2p();
					localPrms +=  pow((Pmean-Pi),2);
				    localTrms +=  pow((Tmean-Ti),2);
				    localVrms +=  pow((Vmean-1/deni),2);
				}	
	
	MPI_Reduce(&localPrms, &Prms, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localTrms, &Trms, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localVrms, &Vrms, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	
	Prms = sqrt(Prms / Nc);
	Trms = sqrt(Trms / Nc);
	Vrms = sqrt(Vrms / Nc);
	
	ptype c0 = sqrt(gam*R*T0);	
	
	Prms = Prms / (gam*p0*Mt*Mt);
	Trms = Trms / ((gam-1)*T0*Mt*Mt);
	Vrms = Vrms / (Mt*Mt/den0);
		
	if (myrank_3d == 0) {			
		std::cout << scientific << t/t0 << " " << TKE <<  " " << Diss << " " << SijSij << " " << SKEWu << " " << SKEWv << " " << SKEWw << " " << KURTu << " " << KURTv << " " << KURTw << " "  
        << Prms << " " << Trms << " " << " " << part_TKE  << " " << " " << part_SijSij << " " << part_SKEWu << " " <<  part_SKEWw << std::endl;	
	}	

}	


void c2p()
{
	deni = WL[0];
	ui   = WL[1]/deni;
	vi   = WL[2]/deni;
	wi   = WL[3]/deni;
	Vsqr = (ui*ui + vi*vi + wi*wi);
	Pi   = (deni*(gam-1))*(WL[4]/deni - 0.5*Vsqr);
	Ti   = Pi/(R*deni);
}


void HaloTransfer(ptype *W[5], ptype *send_HaloL, ptype *recv_HaloL, ptype *send_HaloR, ptype *recv_HaloR,
							   ptype *send_HaloS, ptype *recv_HaloS, ptype *send_HaloN, ptype *recv_HaloN,
							   ptype *send_HaloB, ptype *recv_HaloB, ptype *send_HaloF, ptype *recv_HaloF)
{
    MPI_Request sendrq[2], recvrq[2], rq[4];
	MPI_Status  sendstatus[2], recvstatus[2]; 	
/* -------------------------------------------------------------------------------------*
 * Halo exchange along Y-direction [S-N]
 * -------------------------------------------------------------------------------------*/
	FOR_(j, 3, 6)
		FOR(i, nt_segx)
			FOR(k, nt_segz) 
				FOR(q, 5) {	
					int haloID = q*nt_segz*nt_segx*3 + (j-3)*nt_segz*nt_segx + (i)*nt_segz + k;
					int segId  = Is(j, i, k);
					send_HaloS[haloID] =  W[q][segId];
				}

	FOR_(j, nt_segy-6, nt_segy-3)
		FOR(i, nt_segx)
			FOR(k, nt_segz) 
				FOR(q, 5) {
					int haloID = q*nt_segz*nt_segx*3 + (j-nt_segy+6)*nt_segz*nt_segx + (i)*nt_segz + k;
					int segId  = Is(j, i, k);
					send_HaloN[haloID] =  W[q][segId];				
				}

	halosz  = nt_segx*nt_segz*3;
    halosz5 = 5*halosz;
/*
	MPI_Isend(send_HaloN, halosz5, MPI_DOUBLE, nbrN, 1, COMM3D, &sendrq[0]);
	MPI_Isend(send_HaloS, halosz5, MPI_DOUBLE, nbrS, 1, COMM3D, &sendrq[1]);
	MPI_Irecv(recv_HaloN, halosz5, MPI_DOUBLE, nbrN, 1, COMM3D, &recvrq[0]);	
	MPI_Irecv(recv_HaloS, halosz5, MPI_DOUBLE, nbrS, 1, COMM3D, &recvrq[1]);
	MPI_Waitall(2, sendrq, MPI_STATUSES_IGNORE);
	MPI_Waitall(2, recvrq, MPI_STATUSES_IGNORE);
*/
	MPI_Sendrecv(send_HaloS, halosz5, MPI_DOUBLE_PRECISION, nbrS, 1, recv_HaloN, halosz5, MPI_DOUBLE_PRECISION, nbrN, 1 , COMM3D, MPI_STATUS_IGNORE);
	MPI_Sendrecv(send_HaloN, halosz5, MPI_DOUBLE_PRECISION, nbrN, 1, recv_HaloS, halosz5, MPI_DOUBLE_PRECISION, nbrS, 1 , COMM3D, MPI_STATUS_IGNORE);

	FOR_(j, nt_segy-3, nt_segy)
		FOR(i, nt_segx)
			FOR(k, nt_segz) 
				FOR(q, 5) {
					int haloID = q*nt_segz*nt_segx*3 + (j-nt_segy+3)*nt_segz*nt_segx + (i)*nt_segz + k;
					int segId  = Is(j, i, k);
					W[q][segId] = recv_HaloN[haloID];
				}

	FOR_(j, 0, 3)
		FOR(i, nt_segx) 
			FOR(k, nt_segz)
				FOR(q, 5) {
					int haloID = q*nt_segx*nt_segz*3 + (j)*nt_segx*nt_segz + (i)*nt_segz + k;
					int segId  = Is(j, i, k);
					W[q][segId] = recv_HaloS[haloID];
				}
			
/* -------------------------------------------------------------------------------------*
 * Halo exchange along X-direction [L-R]
 * -------------------------------------------------------------------------------------*/
	FOR(j, nt_segy)
		FOR_(i, 3, 6)
			FOR(k, nt_segz)
				FOR(q, 5) {
					int haloID = q*nt_segz*nt_segy*3 + (j)*3*nt_segz + (i-3)*nt_segz + k;
					int segId  = Is(j, i, k);
					send_HaloL[haloID] =  W[q][segId];
				}

	FOR(j, nt_segy)
		FOR_(i, nt_segx-6, nt_segx-3)
			FOR(k, nt_segz)
				FOR(q, 5) {
					int haloID = q*nt_segz*nt_segy*3 + (j)*3*nt_segz + (i-nt_segx+6)*nt_segz + k;
					int segId  = Is(j, i, k);
					send_HaloR[haloID] =  W[q][segId];
				}

	halosz  = nt_segy*nt_segz*3;
    halosz5 = 5*halosz;
/*
	MPI_Isend(send_HaloR, halosz5, MPI_DOUBLE, nbrR, 1, COMM3D, &sendrq[0]);
	MPI_Isend(send_HaloL, halosz5, MPI_DOUBLE, nbrL, 1, COMM3D, &sendrq[1]);
	MPI_Irecv(recv_HaloR, halosz5, MPI_DOUBLE, nbrR, 1, COMM3D, &recvrq[0]);	
	MPI_Irecv(recv_HaloL, halosz5, MPI_DOUBLE, nbrL, 1, COMM3D, &recvrq[1]);
	MPI_Waitall(2, sendrq, MPI_STATUSES_IGNORE);
	MPI_Waitall(2, recvrq, MPI_STATUSES_IGNORE);

*/
	MPI_Sendrecv(send_HaloL, halosz5, MPI_DOUBLE_PRECISION, nbrL, 1, recv_HaloR, halosz5, MPI_DOUBLE_PRECISION, nbrR, 1 , COMM3D, MPI_STATUS_IGNORE);
	MPI_Sendrecv(send_HaloR, halosz5, MPI_DOUBLE_PRECISION, nbrR, 1, recv_HaloL, halosz5, MPI_DOUBLE_PRECISION, nbrL, 1 , COMM3D, MPI_STATUS_IGNORE);

	FOR(j, nt_segy)
		FOR_(i, nt_segx-3, nt_segx)
			FOR(k, nt_segz)
				FOR(q, 5) {
					int haloID = q*nt_segz*nt_segy*3 + (j)*3*nt_segz + (i-nt_segx+3)*nt_segz + k;
					int segId  = Is(j, i, k);
					W[q][segId] = recv_HaloR[haloID];				
				}
	FOR(j, nt_segy)
		FOR_(i, 0, 3)
			FOR(k, nt_segz)
				FOR(q, 5) {
					int haloID = q*nt_segz*nt_segy*3 + (j)*3*nt_segz + (i)*nt_segz + k;
					int segId  = Is(j, i, k);
					W[q][segId] = recv_HaloL[haloID];
				}

/* -------------------------------------------------------------------------------------*
 * Halo exchange along Z-direction [B-F]
 * -------------------------------------------------------------------------------------*/		
	FOR(j, nt_segy)
		FOR(i, nt_segx)
			FOR_(k, 3, 6) 
				FOR(q, 5) {
					int haloID = q*nt_segy*nt_segx*3 + (j)*3*nt_segx + (i)*3 + k-3;
					int segId  = Is(j, i, k);
					send_HaloB[haloID] =  W[q][segId];								}
	FOR(j, nt_segy)
		FOR(i, nt_segx)
			FOR_(k, nt_segz-6, nt_segz-3) 
				FOR(q, 5) {
					int haloID = q*nt_segy*nt_segx*3 + (j)*3*nt_segx + (i)*3 + k-nt_segz+6;
					int segId  = Is(j, i, k);
					send_HaloF[haloID] =  W[q][segId];
				}
				
	halosz  = nt_segy*nt_segx*3;
    halosz5 = 5*halosz;
/*	
	MPI_Isend(send_HaloF, halosz5, MPI_DOUBLE, nbrF, 1, COMM3D, &sendrq[0]);
	MPI_Isend(send_HaloB, halosz5, MPI_DOUBLE, nbrB, 1, COMM3D, &sendrq[1]);
	MPI_Irecv(recv_HaloF, halosz5, MPI_DOUBLE, nbrF, 1, COMM3D, &recvrq[0]);	
	MPI_Irecv(recv_HaloB, halosz5, MPI_DOUBLE, nbrB, 1, COMM3D, &recvrq[1]);
	MPI_Waitall(2, sendrq, MPI_STATUSES_IGNORE);
	MPI_Waitall(2, recvrq, MPI_STATUSES_IGNORE);
*/
	MPI_Sendrecv(send_HaloB, halosz5, MPI_DOUBLE_PRECISION, nbrB, 1, recv_HaloF, halosz5, MPI_DOUBLE_PRECISION, nbrF, 1 , COMM3D, MPI_STATUS_IGNORE);
	MPI_Sendrecv(send_HaloF, halosz5, MPI_DOUBLE_PRECISION, nbrF, 1, recv_HaloB, halosz5, MPI_DOUBLE, nbrB, 1 , COMM3D, MPI_STATUS_IGNORE);


	FOR(j, nt_segy)
		FOR(i, nt_segx)
			FOR_(k, nt_segz-3, nt_segz) 
				FOR(q, 5) {
					int haloID = q*nt_segy*nt_segx*3 + (j)*3*nt_segx + (i)*3 + k-nt_segz+3;
					int segId  = Is(j, i, k);
					W[q][segId] = recv_HaloF[haloID];				
				}

	FOR(j, nt_segy)
		FOR(i, nt_segx)
			FOR_(k, 0, 3) 
				FOR(q, 5) {
					int haloID = q*nt_segy*nt_segx*3 + (j)*3*nt_segx + (i)*3 + k;
					int segId  = Is(j, i, k);
					W[q][segId] = recv_HaloB[haloID];
				}
}


void timestep(ptype *W[5])
{	
	U = 0.0, localVAL = 0.0, maxVAL = 0.0, Rex, valC=0, localmaxVAL = 0.0;
	For(j, nt_segy)
		For(i, nt_segx)
			For(k, nt_segz)
			{ 
				FOR(q, 5)
					WL[q] = W[q][Is(j, i, k)];
				c2p();
				U = sqrt(Vsqr);
				
				valC = sqrt(gam*Pi/deni);

				localVAL = (U + valC);
				if (localmaxVAL < localVAL)
					localmaxVAL = localVAL; 
			}
	
	MPI_Allreduce(&localmaxVAL, &maxVAL, 1, MPI_DOUBLE, MPI_MAX, COMM3D);	
	
	dt  = cfl*dx/maxVAL; 	
}


void evolveParticles(Particle *particles, UBspline_3d_d *splinep[8], UBspline_3d_d *splinepn[3], ptype *sendParticle[27], ptype *recvParticle[27]) 
{
	ptype vval;
	int sendCount[27] = {0};
		
	int q = 0;
	while (q < count) {
			
		particles[q].track(splinep, splinepn, dt); 
			
		
		if (particles[q].blockChanged == true) {			
			particles[q].collectInfo();
			
			FOR(ijk, 27) {
						
				if (particles[q].blockId == nbr[ijk]) {
					FOR(i, InfoSz)
						sendParticle[ijk][sendCount[ijk]*InfoSz + i + 1] = particles[q].info[i];
					++sendCount[ijk];
					break;
				}
			}

			popParticle(particles, q);
			--q;
		}
		++q;
	}
	
	FOR(jj, 3)
		FOR(ii, 3)
			FOR(kk, 3) {
				int send_ijk = jj*3*3 + ii*3 + kk;
				int deltaJ = 2*(1 - jj);
				int deltaI = 2*(1 - ii);
				int deltaK = 2*(1 - kk);
				int recv_ijk = (jj+deltaJ)*3*3 + (ii+deltaI)*3 + kk+deltaK;				
				
				if (send_ijk!= 13) {
					sendParticle[send_ijk][0] = sendCount[send_ijk];
					sendParticle[recv_ijk][0] = sendCount[recv_ijk];
					
					MPI_Sendrecv(sendParticle[send_ijk], sendSz, MPI_DOUBLE, nbr[send_ijk], 1, 
								 recvParticle[recv_ijk], sendSz, MPI_DOUBLE, nbr[recv_ijk], 1 , COMM3D, MPI_STATUS_IGNORE);
				
					pushParticle(particles, recvParticle[recv_ijk]);
				}
	}
	
}

void GetParticleStats(Particle *particles, UBspline_3d_d *splinep[8]) {
	totalCount = 0;
	localSijSij = 0.0, part_SijSij = 0.0, localTKE = 0.0, part_TKE = 0.0, localDUDX2 = 0.0, localDUDX3 = 0.0, DUDX2 = 0.0, DUDX3 = 0.0, part_SKEWu = 0.0,
	localDWDZ2 = 0.0, localDWDZ3 = 0.0, DWDZ2 = 0.0, DWDZ3 = 0.0, part_SKEWw = 0.0;
	ptype U[3];
	
	FOR(q, count) {	
		localTKE += (particles[q].U[0]*particles[q].U[0] + particles[q].U[1]*particles[q].U[1] + 
			        particles[q].U[2]*particles[q].U[2]);

		dudx = particles[q].gradU[1];
		dvdx = particles[q].gradV[1];
		dwdx = particles[q].gradW[1];
		dudy = particles[q].gradU[0];
		dvdy = particles[q].gradV[0];
		dwdy = particles[q].gradW[0];
		dudz = particles[q].gradU[2];
		dvdz = particles[q].gradV[2];
		dwdz = particles[q].gradW[2];
	
		localSijSij  += ( dudx*dudx + 0.5*pow(dudy + dvdx, 2) + 0.5*pow(dudz + dwdx, 2) + 0.5*pow(dwdy + dvdz, 2) + dvdy*dvdy + dwdz*dwdz);
		localDUDX2 += dudx*dudx;
		localDUDX3 += dudx*dudx*dudx;
		localDWDZ2 += dwdz*dwdz;
		localDWDZ3 += dwdz*dwdz*dwdz;
	}
	
	MPI_Reduce(&localTKE, &part_TKE, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localSijSij, &part_SijSij, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDUDX2, &DUDX2, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDUDX3, &DUDX3, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDWDZ2, &DWDZ2, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);
	MPI_Reduce(&localDWDZ3, &DWDZ3, 1, MPI_DOUBLE, MPI_SUM, 0, COMM3D);

	part_TKE     /= numparticles;
	part_SijSij  /= numparticles;
	DUDX2  /= numparticles;
	DUDX3  /= numparticles;
	DWDZ2  /= numparticles;
	DWDZ3  /= numparticles;	
	part_SKEWu = DUDX3 / pow(DUDX2, 1.5);	
	part_SKEWw = DWDZ3 / pow(DWDZ2, 1.5);	

	MPI_Reduce(&count, &totalCount, 1, MPI_INT, MPI_SUM, 0, COMM3D);
/*
	if(myrank_3d == 0) {
		assert (totalCount == numparticles); 
	}
*/
}	


void pushParticle(Particle globalParticles, Particle *particles) {
	FOR(q, InfoSz)
		particles[count].info[q] =  globalParticles.info[q];
	particles[count].distributeInfo();
	++count;
}

void pushParticle(Particle *particles, ptype *recv) {
	int n = recv[0];
	FOR(i, n) {
		FOR(q, InfoSz)
			particles[count].info[q] =  recv[i*InfoSz + q + 1];
		particles[count].distributeInfo();		
/**		cout << "particle pushed ...  procId " << myrank_3d << "   " << particles[count].blockId << endl;
*/		
		++count;
	}
}

void popParticle(Particle *particles, int particleId) {
	
/**	cout << "particle popped ...  procId " << myrank_3d << ";  blockId  " << particles[particleId].blockId << endl;
*/	
	--count;
	particles[count].collectInfo();
	FOR(q, InfoSz)
		particles[particleId].info[q] =  particles[count].info[q];
	particles[particleId].distributeInfo();
}

void generateSplines(ptype *W[5], UBspline_3d_d *splinep[8], ptype *den, ptype *u, ptype *v, ptype *w, ptype *p, ptype *lapU, ptype *lapV, ptype *lapW)
{
	FOR(i, Nt_seg) {
		den[i] = W[0][i];
		u[i] = W[1][i]/den[i];
		v[i] = W[2][i]/den[i];
		w[i] = W[3][i]/den[i];
		Vsqr = u[i]*u[i] + v[i]*v[i] + w[i]*w[i];
		p[i] = (den[i]*(gam-1))*(W[4][i]/den[i] - 0.5*Vsqr);
	}

	splinep[0] = get_bsplinep(den);
	splinep[1] = get_bsplinep(u);
	splinep[2] = get_bsplinep(v);
	splinep[3] = get_bsplinep(w);
	splinep[4] = get_bsplinep(p);

    For(j, nt_segy) 
		For(i, nt_segx)
			For(k, nt_segz)
				{
                    y = (sta[0] + j-3)*dx + 0.5*dx;
                    x = (sta[1] + i-3)*dx + 0.5*dx;
                    z = (sta[2] + k-3)*dx + 0.5*dx;
                    eval_UBspline_3d_d_vgl (splinep[1], y, x, z, &temp, gradU, &lapU[Is(j, i, k)]);
                    eval_UBspline_3d_d_vgl (splinep[2], y, x, z, &temp, gradV, &lapV[Is(j, i, k)]);
                    eval_UBspline_3d_d_vgl (splinep[3], y, x, z, &temp, gradW, &lapW[Is(j, i, k)]);
                }
    
    splinep[5]  = get_bsplinep(lapU);
    splinep[6]  = get_bsplinep(lapV);
    splinep[7]  = get_bsplinep(lapW);
}


void generateSplines(ptype *W[5], UBspline_3d_d *splinep[3], ptype *u, ptype *v, ptype *w)
{
	FOR(i, Nt_seg) {
		u[i] = W[1][i]/W[0][i];
		v[i] = W[2][i]/W[0][i];
		w[i] = W[3][i]/W[0][i];
	}

	splinep[0] = get_bsplinep(u);
	splinep[1] = get_bsplinep(v);
	splinep[2] = get_bsplinep(w);
}

void findAllNbrs() {
	
	int j[3], i[3], k[3];
	
	j[0] = nbrS/(procDim[1]*procDim[2]);
	i[1] = (nbrS%(procDim[1]*procDim[2]))/procDim[2];
	k[1] = nbrS%(procDim[2]);

	j[2] = nbrN/(procDim[1]*procDim[2]);
//	i[1] = (nbrN%(procDim[1]*procDim[2]))/procDim[2];
//	k[1] = nbrN%(procDim[2]);

	j[1] = nbrL/(procDim[1]*procDim[2]);
	i[0] = (nbrL%(procDim[1]*procDim[2]))/procDim[2];
//	k[1] = nbrL%(procDim[2]);

//	j[1] = nbrR/(procDim[1]*procDim[2]);
	i[2] = (nbrR%(procDim[1]*procDim[2]))/procDim[2];
//	k[1] = nbrR%(procDim[2]);

//	j[1] = nbrB/(procDim[1]*procDim[2]);
//	i[1] = (nbrB%(procDim[1]*procDim[2]))/procDim[2];
	k[0] = nbrB%(procDim[2]);

//	j[1] = nbrF/(procDim[1]*procDim[2]);
//	i[1] = (nbrF%(procDim[1]*procDim[2]))/procDim[2];
	k[2] = nbrF%(procDim[2]);
	
	FOR(jj, 3)
		FOR(ii, 3)
			FOR(kk, 3)			
				nbr[jj*3*3 + ii*3 + kk] = j[jj]*procDim[1]*procDim[2] + i[ii]*procDim[2] + k[kk]; 	
}


void locateParticles(Particle *particles, UBspline_3d_d *splinep[8]) {

	Particle *globalParticles = new Particle[numparticles];

	ptype *x0 = new ptype[numparticles];
	ptype *y0 = new ptype[numparticles];
	ptype *z0 = new ptype[numparticles];
    
    char buffer[30];
    snprintf(buffer, sizeof(char) * 30, "InitialConditions/unirand");	
	ifstream input(buffer, ios::in | ios::binary);
	size_t size = sizeof(ptype);
	input.read((char*) x0, size*numparticles);
	input.read((char*) y0, size*numparticles);
	input.read((char*) z0, size*numparticles);
	input.close();		

	FOR(q, numparticles) {
		if(myrank_3d == 0) {
			globalParticles[q].initializeParticle(x0[q], y0[q], z0[q], q);
			globalParticles[q].collectInfo();
		}
		
		MPI_Bcast(&globalParticles[q].info, InfoSz, MPI_DOUBLE, 0, COMM3D);
		globalParticles[q].distributeInfo();		

		if(myrank_3d == globalParticles[q].blockId)
			pushParticle(globalParticles[q], particles);		
	}
	
	delete [] x0, y0, z0;		
	
	delete[] globalParticles;
	
}

void writeField(ptype *den, ptype *u, ptype *v, ptype *w, ptype *p) {
	FILE * pFile;
	char Field[50];
	snprintf(Field, sizeof(char)*50, "Results/Field/Field_%i_%i", myrank_3d, iter);
	pFile = fopen (Field,"w");

	FOR(j, nc_segy)
		FOR(i, nc_segx)
			FOR(k, nc_segz)
			{
				int I = Iis(j, i, k);	
				fprintf(pFile, "%d %d %d %.15e %.15e %.15e %.15e %.15e\n", sta[1]+i, sta[0]+j, sta[2]+k, den[I], u[I], v[I], w[I], p[I]);
			}
	fclose(pFile);
}

void writeParticleData(Particle *particles) {
	
	FILE * pFile;
	char PInfo[50];
	snprintf(PInfo, sizeof(char)*50, "Results/Particles/P_%i_%i", myrank_3d, iter);
	pFile = fopen (PInfo,"w");
	
	FOR(i, count) {
		fprintf(pFile, "%d ", particles[i].tagId);
		fprintf(pFile, "%.15e %.15e %.15e ", particles[i].pos[1], particles[i].pos[0], particles[i].pos[2]);
		fprintf(pFile, "%.15e ", particles[i].DEN);
		fprintf(pFile, "%.15e %.15e %.15e ", particles[i].U[0], particles[i].U[1], particles[i].U[2]);
		fprintf(pFile, "%.15e ", particles[i].P);
		fprintf(pFile, "%.15e %.15e %.15e ",  particles[i].gradD[1], particles[i].gradD[0], particles[i].gradD[2]);
		fprintf(pFile, "%.15e %.15e %.15e ",  particles[i].gradU[1], particles[i].gradU[0], particles[i].gradU[2]);
		fprintf(pFile, "%.15e %.15e %.15e ",  particles[i].gradV[1], particles[i].gradV[0], particles[i].gradV[2]);
		fprintf(pFile, "%.15e %.15e %.15e ",  particles[i].gradW[1], particles[i].gradW[0], particles[i].gradW[2]);
		fprintf(pFile, "%.15e %.15e %.15e ", particles[i].gradP[1], particles[i].gradP[0], particles[i].gradP[2]);
        fprintf(pFile, "%.15e %.15e %.15e ", particles[i].gradLapU[1], particles[i].gradLapU[0], particles[i].gradLapU[2]);
        fprintf(pFile, "%.15e %.15e %.15e ", particles[i].gradLapV[1], particles[i].gradLapV[0], particles[i].gradLapV[2]);
        fprintf(pFile, "%.15e %.15e %.15e ", particles[i].gradLapW[1], particles[i].gradLapW[0], particles[i].gradLapW[2]);
        fprintf(pFile, "%.15e %.15e %.15e ",  particles[i].hessP[4], particles[i].hessP[3], particles[i].hessP[5]);
        fprintf(pFile, "%.15e %.15e %.15e ",  particles[i].hessP[1], particles[i].hessP[0], particles[i].hessP[2]);
        fprintf(pFile, "%.15e %.15e %.15e\n", particles[i].hessP[7], particles[i].hessP[6], particles[i].hessP[8]);
	}
	fclose(pFile);

	if (myrank_3d == 0) {
		FILE * TFile;
		snprintf(PInfo, sizeof(char)*50, "Results/Time/T_%i", iter);
		TFile = fopen (PInfo,"w");

		fprintf(TFile, "%.15e %.15e %.15e", t, dt, t0);
		fclose(TFile);
	}
}

void WriteReInit(ptype *W[5]) {

	ptype *u0 = new ptype[Nc_seg];
	ptype *v0 = new ptype[Nc_seg];
	ptype *w0 = new ptype[Nc_seg];
	ptype *d0 = new ptype[Nc_seg];
	ptype *pr0 = new ptype[Nc_seg];
	FOR(j, nc_segy)
		FOR(i, nc_segx)
			FOR(k, nc_segz)
			{
				// we adopt the following index(j,i,k) style for linearization
				int I = Iis(j, i, k);
				
				// Matlab style of 3-D array linearization --> our style equiv. index(k,i,j) 
				int J = k*nc_segx*nc_segy + i*nc_segy + j;

				FOR(q, 5)
					WL[q] = W[q][I];
				c2p(); 	
				u0[J] = ui;
				v0[J] = vi;
				w0[J] = wi;
				d0[J] = deni;	
				pr0[J] = (deni*(gam-1))*(WL[4]/deni - 0.5*Vsqr);
			}	
	char data[50];
	snprintf(data, sizeof(char) * 50, "InitialCondition/Init%i%i%i", procId[0], procId[1], procId[2]);	
	fstream U_writer(data, std::ios::out | std::ios::binary);
	
	size_t size = sizeof(ptype)*Nc_seg; 
	U_writer.write((char*) u0, size);	
	U_writer.write((char*) v0, size);	
	U_writer.write((char*) w0, size);	
	U_writer.write((char*) d0, size);	
	U_writer.write((char*) pr0, size);	
	U_writer.flush();	
	U_writer.close();
	delete[] u0, v0, w0, d0, pr0;
}


void writeVelGradTensor(UBspline_3d_d *splinep[8]) {
	FILE * pFile;
	char Field[50];
	snprintf(Field, sizeof(char)*50, "Results/Aij/A_%i_%i", myrank_3d, iter);
	pFile = fopen (Field,"w");
	
	For(j, nt_segy) 
		For(i, nt_segx)
			For(k, nt_segz) {
				y = (sta[0] + j-3)*dx + 0.5*dx;
				x = (sta[1] + i-3)*dx + 0.5*dx;
				z = (sta[2] + k-3)*dx + 0.5*dx;
				
				eval_UBspline_3d_d_vg (splinep[1] , y, x, z, &ui, gradU );
				eval_UBspline_3d_d_vg (splinep[2] , y, x, z, &vi, gradV );
				eval_UBspline_3d_d_vg (splinep[3] , y, x, z, &wi, gradW );
				
				fprintf(pFile, "%.15e %.15e %.15e ",  ui, vi, wi);
				fprintf(pFile, "%.15e %.15e %.15e ",  gradU[1], gradU[0], gradU[2]);				
				fprintf(pFile, "%.15e %.15e %.15e ",  gradV[1], gradV[0], gradV[2]);				
				fprintf(pFile, "%.15e %.15e %.15e\n", gradW[1], gradW[0], gradW[2]);				
			}

	fclose(pFile);
	
}

/**
 *  WENO JS
 * 
void ApplyWENO(ptype WLLL[5], ptype WLL[5], ptype WL[5], ptype WR[5],
		         ptype WRR[5], ptype WRRR[5], ptype Wl[5], ptype Wr[5])
{	
	gam0 = 0.1, gam1 = 0.6, gam2 = 0.3;
	FOR(i, 4) {
		P0p = WLLL[i]/3 - 7*WLL[i]/6 + 11*WL[i]/6;
		P0n = 11*WR[i]/6 - 7*WRR[i]/6 + WRRR[i]/3;
		P1p = -WLL[i]/6 + 5*WL[i]/6 + WR[i]/3;
		P1n = WL[i]/3 + 5*WR[i]/6 - WRR[i]/6;
	
		P2p = WL[i]/3 + 5*WR[i]/6 - WRR[i]/6;
		P2n = -WLL[i]/6 + 5*WL[i]/6 + WR[i]/3;
			
		IS0p = 13*pow((WLLL[i] - 2*WLL[i] + WL[i]),2)/12 + pow((WLLL[i] - 4*WLL[i] + 3*WL[i]),2)/4;
		IS0n = 13*pow((WRRR[i] - 2*WRR[i] + WR[i]),2)/12 + pow((WRRR[i] - 4*WRR[i] + 3*WR[i]),2)/4;

		IS1p = 13*pow((WLL[i] - 2*WL[i] + WR[i]),2)/12  + pow((WLL[i] - WR[i]),2)/4;
		IS1n = 13*pow((WL[i]  - 2*WR[i] + WRR[i]),2)/12 + pow((WL[i] - WRR[i]),2)/4;
		IS2p = 13*pow((WL[i]  - 2*WR[i] + WRR[i]),2)/12 + pow((3*WL[i] - 4*WR[i] + WRR[i]),2)/4;
		IS2n = 13*pow((WLL[i] - 2*WL[i] + WR[i]),2)/12  + pow((WLL[i] - 4*WL[i] + 3*WR[i]),2)/4;
	
		ALPHA0p = gam0*pow(1/(eps+IS0p), 2); 
		ALPHA0n = gam0*pow(1/(eps+IS0n), 2); 
		ALPHA1p = gam1*pow(1/(eps+IS1p), 2); 
		ALPHA1n = gam1*pow(1/(eps+IS1n), 2); 
		ALPHA2p = gam2*pow(1/(eps+IS2p), 2); 
		ALPHA2n = gam2*pow(1/(eps+IS2n), 2); 
	
		sum_ALPHAp = ALPHA0p + ALPHA1p + ALPHA2p;
		sum_ALPHAn = ALPHA0n + ALPHA1n + ALPHA2n;
	
		OMEGA0p = ALPHA0p/(sum_ALPHAp);
		OMEGA0n = ALPHA0n/(sum_ALPHAn);
		OMEGA1p = ALPHA1p/(sum_ALPHAp);
		OMEGA1n = ALPHA1n/(sum_ALPHAn);
		OMEGA2p = ALPHA2p/(sum_ALPHAp);
		OMEGA2n = ALPHA2n/(sum_ALPHAn);
	 
		Wl[i] = (OMEGA0p*P0p + OMEGA1p*P1p + OMEGA2p*P2p); 
		Wr[i] = (OMEGA0n*P0n + OMEGA1n*P1n + OMEGA2n*P2n); 
	}
		
		TL   = (gam-1) * ( WL[4]/WL[0] - 0.5*(WL[1]*WL[1]+WL[2]*WL[2]+WL[3]*WL[3])/(WL[0]*WL[0]) ) / R;
		TLL  = (gam-1) * ( WLL[4]/WLL[0] - 0.5*(WLL[1]*WLL[1]+WLL[2]*WLL[2]+WLL[3]*WLL[3])/(WLL[0]*WLL[0]) ) / R;
		TLLL = (gam-1) * ( WLLL[4]/WLLL[0] - 0.5*(WLLL[1]*WLLL[1]+WLLL[2]*WLLL[2]+WLLL[3]*WLLL[3])/(WLLL[0]*WLLL[0]) ) / R;
		
		TR   = (gam-1) * ( WR[4]/WR[0] - 0.5*(WR[1]*WR[1]+WR[2]*WR[2]+WR[3]*WR[3])/(WR[0]*WR[0]) ) / R;
		TRR  = (gam-1) * ( WRR[4]/WRR[0] - 0.5*(WRR[1]*WRR[1]+WRR[2]*WRR[2]+WRR[3]*WRR[3])/(WRR[0]*WRR[0]) ) / R;
		TRRR = (gam-1) * ( WRRR[4]/WRRR[0] - 0.5*(WRRR[1]*WRRR[1]+WRRR[2]*WRRR[2]+WRRR[3]*WRRR[3])/(WRRR[0]*WRRR[0]) ) / R;

		P0p = TLLL/3 - 7*TLL/6 + 11*TL/6;
		P0n = 11*TR/6 - 7*TRR/6 + TRRR/3;
		P1p = -TLL/6 + 5*TL/6 + TR/3;
		P1n = TL/3 + 5*TR/6 - TRR/6;
	
		P2p = TL/3 + 5*TR/6 - TRR/6;
		P2n = -TLL/6 + 5*TL/6 + TR/3;
		
		IS0p = 13*pow((TLLL - 2*TLL + TL),2)/12 + pow((TLLL - 4*TLL + 3*TL),2)/4;
		IS0n = 13*pow((TRRR - 2*TRR + TR),2)/12 + pow((TRRR - 4*TRR + 3*TR),2)/4;

		IS1p = 13*pow((TLL - 2*TL + TR),2)/12  + pow((TLL - TR),2)/4;
		IS1n = 13*pow((TL  - 2*TR + TRR),2)/12 + pow((TL - TRR),2)/4;
		IS2p = 13*pow((TL  - 2*TR + TRR),2)/12 + pow((3*TL - 4*TR + TRR),2)/4;
		IS2n = 13*pow((TLL - 2*TL + TR),2)/12  + pow((TLL - 4*TL + 3*TR),2)/4;
	
		ALPHA0p = gam0*pow(1/(eps+IS0p), 2); 
		ALPHA0n = gam0*pow(1/(eps+IS0n), 2); 
		ALPHA1p = gam1*pow(1/(eps+IS1p), 2); 
		ALPHA1n = gam1*pow(1/(eps+IS1n), 2); 
		ALPHA2p = gam2*pow(1/(eps+IS2p), 2); 
		ALPHA2n = gam2*pow(1/(eps+IS2n), 2); 
	
		sum_ALPHAp = ALPHA0p + ALPHA1p + ALPHA2p;
		sum_ALPHAn = ALPHA0n + ALPHA1n + ALPHA2n;
	
		OMEGA0p = ALPHA0p/(sum_ALPHAp);
		OMEGA0n = ALPHA0n/(sum_ALPHAn);
		OMEGA1p = ALPHA1p/(sum_ALPHAp);
		OMEGA1n = ALPHA1n/(sum_ALPHAn);
		OMEGA2p = ALPHA2p/(sum_ALPHAp);
		OMEGA2n = ALPHA2n/(sum_ALPHAn);
	 
		ptype Tl = (OMEGA0p*P0p + OMEGA1p*P1p + OMEGA2p*P2p); 
		ptype Tr = (OMEGA0n*P0n + OMEGA1n*P1n + OMEGA2n*P2n);
		
		Wl[4] = Wl[0]*R*Tl/(gam-1.0) + 0.5*(Wl[1]*Wl[1]+Wl[2]*Wl[2]+Wl[3]*Wl[3])/Wl[0];
		Wr[4] = Wr[0]*R*Tr/(gam-1.0) + 0.5*(Wr[1]*Wr[1]+Wr[2]*Wr[2]+Wr[3]*Wr[3])/Wr[0];	
}
**/

