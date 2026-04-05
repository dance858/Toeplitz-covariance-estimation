#ifndef NML_SOLVE
#define NML_SOLVE
#include "nml/NML_work.h"
#include <complex.h>

int NML(double complex *Z_data, int n, int K, NML_out *output, double tol,
        double beta, double alpha, int verbose, int max_iter);
void NML_free_output(NML_out *output);

#endif
