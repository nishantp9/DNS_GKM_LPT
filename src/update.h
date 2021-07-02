#ifndef UPDATEHEADERDEF
#define UPDATEHEADERDEF

#include <mpi.h>
#include "param.h"
#include "ParticleTrack.h"
#include <bspline.h>
#include "Interpolation.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cassert>

#ifdef UseCPU
	#include "GKMflux.h"
	void evolveEulerian(ptype *W[5], ptype *Fx[5], ptype *Fy[5], ptype *Fz[5]);
	void derivsX(ptype *W[5], ptype *Fx[5]);
	void derivsY(ptype *W[5], ptype *Fy[5]);
	void derivsZ(ptype *W[5], ptype *Fz[5]);
	
	void ApplyWENO(ptype WLLL[5], ptype WLL[5], ptype WL[5], ptype WR[5],
					 ptype WRR[5], ptype WRRR[5], ptype Wl[5], ptype Wr[5]);

	void ApplyLimiter(ptype WLL[5], ptype WL[5], ptype WR[5], ptype WRR[5],
					  ptype dx, ptype Wl[5], ptype Wr[5]);
	
	void update(ptype *W[5], ptype *Fx[5], ptype *Fy[5], ptype *Fz[5]);

#endif

#ifdef UseGPU
	#include "GPUupdate.h"
	#include "GPUAlloc.h"
	void evolveEulerian(ptype *W[5], DevAlloc *dev);
#endif		

	void Evolve(ptype *W[5], MPI_Comm comm3d);
	
	void HaloTransfer(ptype *W[5], ptype *send_HaloL, ptype *recv_HaloL, ptype *send_HaloR, ptype *recv_HaloR,
								   ptype *send_HaloS, ptype *recv_HaloS, ptype *send_HaloN, ptype *recv_HaloN,
								   ptype *send_HaloB, ptype *recv_HaloB, ptype *send_HaloF, ptype *recv_HaloF);
	
	void PrintStats(ptype *W[5], UBspline_3d_d *splinep[14]);

	void c2p();

	void timestep(ptype *W[5]);
	
	void WriteField(ptype *W[5], int iter);
		
	void evolveParticles(Particle *particles, UBspline_3d_d *splinep[8], UBspline_3d_d *splinepn[3], ptype *sendParticle[27], ptype *recvParticle[27]);
	
	void printTrajectory(Particle *particles);	

	void printTrajectory(Particle *particles, ptype *W[5]);	
	
	void GetParticleStats(Particle *particles, UBspline_3d_d *splinep[8]);

	void pushParticle(Particle globalParticles, Particle *particles);

	void pushParticle(Particle *particles, ptype *recv);

	void popParticle(Particle *particles, int particleId);

	void findAllNbrs();

	void HaloTransfer(ptype *W[5]);

    void generateSplines(ptype *W[5], UBspline_3d_d *splinep[8], ptype *den, ptype *u, ptype *v, ptype *w, ptype *p, ptype *lapU, ptype *lapV, ptype *lapW); 

    void generateSplines(ptype *W[5], UBspline_3d_d *splinep[3], ptype *u, ptype *v, ptype *w); 
						 
	void locateParticles(Particle *particles, UBspline_3d_d *splinep[8]);

	void writeParticleData(Particle *particles);

	void writeField(ptype *den, ptype *u, ptype *v, ptype *w, ptype *p);

	void writeVelGradTensor(UBspline_3d_d *splinep[8]);

	void WriteReInit(ptype *W[5]);

#endif


/**	
	int arrSz = int(1.1*numparticles/nprocs);
	 
	size_t bufferSz = arrSz*writeInfoSz*sizeof(ptype);
	MPI_Offset offset = bufferSz*myrank_3d;
	MPI_Request req;
	ptype *collection = new ptype[bufferSz];
	
	collection[0] = count;
	
	FOR(i, count) {
		collection[(i+1)*25 + 0]  = particles[i].tagId;
		collection[(i+1)*25 + 1]  = particles[i].blockId;
		collection[(i+1)*25 + 2]  = particles[i].pos[1];
		collection[(i+1)*25 + 3]  = particles[i].pos[0];
		collection[(i+1)*25 + 4]  = particles[i].pos[2];
		collection[(i+1)*25 + 5]  = particles[i].DEN;
		collection[(i+1)*25 + 6]  = particles[i].U[0];
		collection[(i+1)*25 + 7]  = particles[i].U[1];
		collection[(i+1)*25 + 8]  = particles[i].U[2];
		collection[(i+1)*25 + 9]  = particles[i].P;			
		collection[(i+1)*25 + 10] = particles[i].gradD[1];
		collection[(i+1)*25 + 11] = particles[i].gradD[0];
		collection[(i+1)*25 + 12] = particles[i].gradD[2];
		collection[(i+1)*25 + 13] = particles[i].gradU[1];
		collection[(i+1)*25 + 14] = particles[i].gradU[0];
		collection[(i+1)*25 + 15] = particles[i].gradU[2];
		collection[(i+1)*25 + 16] = particles[i].gradV[1];
		collection[(i+1)*25 + 17] = particles[i].gradV[0];
		collection[(i+1)*25 + 18] = particles[i].gradV[2];
		collection[(i+1)*25 + 19] = particles[i].gradW[1];
		collection[(i+1)*25 + 20] = particles[i].gradW[0];
		collection[(i+1)*25 + 21] = particles[i].gradW[2];
		collection[(i+1)*25 + 22] = particles[i].gradP[1];
		collection[(i+1)*25 + 23] = particles[i].gradP[0];
		collection[(i+1)*25 + 24] = particles[i].gradP[2];
	}
	char data[40];
	snprintf(data, sizeof(char) * 40, "Results/Particles/Part_%i", iter);
	MPI_File file;
	MPI_File_open(COMM3D, data, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
	MPI_File_seek(file, offset, MPI_SEEK_SET);	
	
	MPI_File_iwrite(file, collection, bufferSz, MPI_DOUBLE, &req);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_File_sync(file);
	MPI_File_close(&file);

	if(myrank_3d == 0) {
		FILE * pFile;
		char timeInfo[40];
		snprintf(timeInfo, sizeof(char) * 40, "Results/Time/time%i", iter);
		pFile = fopen (timeInfo,"w");
		fprintf(pFile, "%.15f\n", t);
		fclose(pFile);
	}
	delete [] collection;
*/
