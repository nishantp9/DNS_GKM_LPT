# Direct numerical simulation code for simulating Decaying Isotropic Flow and Lagrangian Particle Tracking

## Key Features
* Solves 3D compressilbe turbulent flow in a cubic domain with periodic Boundary Conditions
* Writtin in C++ and parallelized using MPI and CUDA

## Packages required
* module load compiler/cuda/7.0/compilervars
* module load suite/intel/parallelStudio
* Install einspline library http://einspline.sourceforge.net/ in the HPC $HOME folder

## Instructions to run the code on IITD HPC cluster https://supercomputing.iitd.ac.in/
### Generate initial conditions
* Open the script "init_DNS.m" from <InitialConditions> directory and specify turbulent Mach number, Reynolds number and grid dimension
* Change parameters: (procDim_y, procDim_x, procDim_z) defining the number of virtual 3D topology of processors
* The data is distributed into cubical chunks decided by the virtual topology of the processors
* Open "params.h" header file in <src> directory and specify (procDim_y, procDim_x, procDim_z) values that are specified in "init_DNS.m"
* Running the script "init_DNS.m" will generate the initial conditions for the flow

### Generate random initial locations of particles to be tracked with the flow
* Open the script "Part_Gen.m" from <InitialConditions> directory and specify the number of particles to be tracked
* Running the script "Part_Gen.m" will generate random particle locations

### Generate required directories
* source getFolders.sh

### Compling the code for running on distribued nodes with multi-GPUs (MPI and CUDA)
* Recommended for running high resolution simulations >= 512^3
* make -f makefileGPU

### Compling the code for running on distributed multi-core CPUs only (MPI)
* Recommended for running lower resolution simulations <= 256^3
* make -f makefileCPU

### Submitting Job on HPC
* Open the bash file "pbsbatch.sh" and specify project name, log-files and required compute resource
* Submit the job following instructions from: https://supercomputing.iitd.ac.in/?pbs
                                                              
