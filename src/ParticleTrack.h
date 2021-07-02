/***************************************************************
 * Tracks the Particles
 **************************************************************/
#ifndef PARTICLETRACKHEADERDEF
#define PARTICLETRACKHEADERDEF

#include <fstream>
#include <iostream>
#include "param.h"
#include <bspline.h>
#include "Interpolation.h"

using namespace std;
class Particle
{
public:
	Particle();
	int infoSz, tagId; ptype info[8];
	bool blockChanged;
	ptype pos[3], DEN, U[3], P, gradD[3], gradU[3], gradV[3], gradW[3], gradP[3], lapU, lapV, lapW, hessP[9], 
          gradLapU[3], gradLapV[3], gradLapW[3];
	
	int  blockId, iter, block[3];
	void initializeParticle(ptype x0, ptype y0, ptype z0, int tag);
	void track(UBspline_3d_d *splinep[8], UBspline_3d_d *splinepn[3], ptype dt);
	void identifyBlock();
	void identifyBlockId();
	
	void updatePosition(UBspline_3d_d *splinep[8], UBspline_3d_d *splinepn[3], ptype dt);	
	void collectInfo();
	void distributeInfo();
	void getParticleProp(UBspline_3d_d *splinep[8]);

private:
	
};

#endif
