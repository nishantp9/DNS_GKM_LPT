# Direct numerical simulation code for simulating Decaying Isotropic Flow and Lagrangian Particle Tracking
## Key Features
* Solves 3D compressible turbulent flow in a cubic domain with periodic Boundary Conditions
* Simulates Decaying compressible isotropic turbulent flow
* Can be extended to simulate few other canonical flows as well, e.g. Flat Plate Boundary Layer, Channel Flow, Isotropic Stationary Turbulence etc. by changing the boundary conditions.
* By adding a seperate module to add forcing term, the code can be extended to solve Stationary Isotropic Turbulence.
* The codebase is writtin in C++ and parallelized using MPI and CUDA.

## Packages required
* CUDA compliers
  - `module load compiler/cuda/7.0/compilervars`
* Intel parallelStudio
  - `module load suite/intel/parallelStudio`
* Install einspline library `http://einspline.sourceforge.net/` in the `$HOME` folder

### Einspline installation
* To install einspline library with the latest gcc compilers use the modified repo:
```shell
  git clone https://github.com/nishantp9/einspline-0.9.2
  cd einspline-0.9.2
  chmod +777 configure
  ./configure --prefix=$HOME/einspline
  make
  make install
```

## Instructions to run the code on IITD HPC cluster `https://supercomputing.iitd.ac.in/`
### Generate initial conditions
* Open the script `init_DNS.m` from `InitialConditions` directory and specify turbulent Mach number, Reynolds number and grid dimension
* Change parameters: `(procDim_y, procDim_x, procDim_z)` defining the number of virtual 3D topology of processors
* Total number of cpu-processors required is then `procDim_y x procDim_x x procDim_z`
* The data is distributed into cubical chunks decided by the virtual topology of the processors
* Open `params.h` header file in <src> directory and specify `(procDim_y, procDim_x, procDim_z)` values that are specified in `init_DNS.m`
* Running the script `init_DNS.m` will generate the initial conditions for the flow

### Generate random initial locations of particles to be tracked with the flow
* Open the script `Part_Gen.m` from `InitialConditions` directory and specify the number of particles to be tracked
* Running the script `Part_Gen.m` will generate random particle locations

### Generate required directories
  - `source getFolders.sh`

### Compling the code for running on distribued nodes with multi-GPUs (MPI and CUDA)
* Recommended for running high resolution simulations >= `512^3`
  - `make -f makefileGPU`

### Compling the code for running on distributed multi-core CPUs only (MPI)
* Recommended for running lower resolution simulations <= `256^3`
  - `make -f makefileCPU`

### Submitting Job on HPC
* Open the bash file `pbsbatch.sh` and specify project name, log-files and required compute resource
* Number of processors for mpirun command should be equal to `procDim_y x procDim_x x procDim_z`
* Submit the job following instructions from: `https://supercomputing.iitd.ac.in/?pbs`
### Testing and compiling on local linux machine
```shell
make -f makefileCPU_local
mpirun -np <num_processors> ./Main.out > log_dns
```
