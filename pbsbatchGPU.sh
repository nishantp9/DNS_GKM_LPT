#!/bin/sh
### Set the job name
#PBS -N Mpt7R400_2
### Set the project name, your department dc by default
#PBS -P am
#PBS -q standard
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M amz168166@iitd.ac.in
####
#PBS -l place=scatter
#PBS -l select=32:ngpus=1:ncpus=2
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=168:00:00
#PBS -e stderr_file
#### Get environment variables from submitting shell
####PBS -V
#PBS -l software=GKM 
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
cd $PBS_O_WORKDIR
#job
module load compiler/cuda/7.0/compilervars
module load suite/intel/parallelStudio
time mpirun -np 64 ./Main.out > DNSMpt7R400_2
#NOTE
# The job line is an example : users need to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
