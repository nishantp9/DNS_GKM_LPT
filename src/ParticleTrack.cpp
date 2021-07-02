#include "ParticleTrack.h"
#include <iostream>
using namespace std; 


Particle::Particle() {
	infoSz = 8;
	FOR(q, infoSz)
		info[q] = 0;
}

void Particle::initializeParticle(ptype x0, ptype y0, ptype z0, int tag) {
	tagId  = tag;
	pos[1] = x0;
	pos[0] = y0;
	pos[2] = z0;
	identifyBlockId();
}

void Particle::identifyBlock() {
	
	FOR(q, 3) 
		block[q] = int(pos[q]*procDim[q]/(2*pi));
	
	blockChanged = false;
	if (block[0]*procDim[1]*procDim[2] + block[1]*procDim[2] + block[2] != myrank_3d)
		blockChanged = true;

}

void Particle::identifyBlockId() {
	identifyBlock();
	blockId  = block[0]*procDim[1]*procDim[2] + block[1]*procDim[2] + block[2];
	if (blockId >= nprocs || blockId < 0)	{
		cout << " Block-Index breach found; BLOCK = " << blockId<< ", " << myrank_3d <<endl;
	}	
}

void Particle::track(UBspline_3d_d *splinep[8], UBspline_3d_d *splinepn[3], ptype dt) {
	updatePosition(splinep, splinepn, dt);
	identifyBlockId();
}

void Particle::updatePosition(UBspline_3d_d *splinep[8], UBspline_3d_d *splinepn[3], ptype dt) {

/**	RK-2 (HEUN)
 */ 
	ptype U2[3];
	eval_UBspline_3d_d (splinepn[0], pos[0] + U[1]*dt, pos[1] + U[0]*dt, pos[2] + U[2]*dt, &U2[0]);
	eval_UBspline_3d_d (splinepn[1], pos[0] + U[1]*dt, pos[1] + U[0]*dt, pos[2] + U[2]*dt, &U2[1]);
	eval_UBspline_3d_d (splinepn[2], pos[0] + U[1]*dt, pos[1] + U[0]*dt, pos[2] + U[2]*dt, &U2[2]);
	
	pos[0] +=  0.5*(U[1] + U2[1])*dt;
	pos[1] +=  0.5*(U[0] + U2[0])*dt;
	pos[2] +=  0.5*(U[2] + U2[2])*dt;

	FOR(q, 3) {
		if (pos[q] > 2*pi) {
			pos[q] = pos[q] - 2*pi;
		}
		if (pos[q] < 0) {
			pos[q] = 2*pi + pos[q];
		}
	}
}

void Particle::collectInfo() {
	info[0]  = tagId;		
	info[1]  = blockId;		
	info[2]  = pos[0];
	info[3]  = pos[1];
	info[4]  = pos[2];
	info[5]  = U[0];
	info[6]  = U[1];
	info[7]  = U[2];
}

void Particle::distributeInfo() {
	tagId    = info[0];		
	blockId  = info[1];		
	pos[0]   = info[2];
	pos[1]   = info[3];
	pos[2]   = info[4];
	U[0]     = info[5];
	U[1]     = info[6];
	U[2]     = info[7];
	
}

void Particle::getParticleProp(UBspline_3d_d *splinep[8]) {
    ptype temp; ptype tempV[3];
     
	eval_UBspline_3d_d_vg (splinep[0], pos[0], pos[1], pos[2], &DEN , gradD);
	eval_UBspline_3d_d_vg (splinep[1], pos[0], pos[1], pos[2], &U[0], gradU);
	eval_UBspline_3d_d_vg (splinep[2], pos[0], pos[1], pos[2], &U[1], gradV);
	eval_UBspline_3d_d_vg (splinep[3], pos[0], pos[1], pos[2], &U[2], gradW);
	eval_UBspline_3d_d_vgh (splinep[4], pos[0], pos[1], pos[2], &P, gradP, hessP);
    
    eval_UBspline_3d_d_vg (splinep[5], pos[0], pos[1], pos[2], &lapU, gradLapU);
    eval_UBspline_3d_d_vg (splinep[6], pos[0], pos[1], pos[2], &lapV, gradLapV);
    eval_UBspline_3d_d_vg (splinep[7], pos[0], pos[1], pos[2], &lapW, gradLapW); 
}
