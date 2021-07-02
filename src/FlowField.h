/***************************************************************
 * Flow field initialization and BC implementation
 **************************************************************/
#ifndef FLOWFIELDHEADERDEF
#define FLOWFIELDHEADERDEF
 
#include "param.h"
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include <iostream>

void GetParams();
void Initialize_DHIT(ptype *W[5]);
void PeriodicBC(ptype *W[5]);
void segment();
void segmentFill(ptype *W[5], ptype *W0[5]);
void PrintParams();

#endif
