#include <iostream>
#include <mpi.h>
#include <omp.h>
#include "param.h"
#include "FlowField.h"
#include "update.h"

using namespace std;

int main(int argc, char* argv[]) {

	int myrank;
	ptype *W[5];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
/**	dim = int (cbrt(nprocs));
*/	
	procDim[0] = procDim_y;
	procDim[1] = procDim_x;
	procDim[2] = procDim_z;
	
    int periodic[3] = {true, true, true};
    
	MPI_Comm comm3d;
	MPI_Cart_create(MPI_COMM_WORLD, 3, procDim, periodic, true, &comm3d);
	MPI_Comm_rank(comm3d, &myrank_3d);
	MPI_Cart_coords(comm3d, myrank_3d, 3, procId);

	GetParams();
	PrintParams();

	segment();	 
	MAKE5s(W);

/**---------------------------------------------------------------------------
 * --------------------------------------------------------------------------*/
	Initialize_DHIT(W);
	Evolve(W, comm3d);
/**---------------------------------------------------------------------------
 * --------------------------------------------------------------------------*/
	
	MPI_Finalize();

	FOR(q, 5)
		delete [] W[q];
}
