#ifndef NML_SOLVE
#define NML_SOLVE
#include "NML_work.h"
#include <complex.h>
 
int NML(double complex *Z_data, const int n, const int K, NML_out *output,
        double tol, double beta, double alpha, int verbose, int max_iter);
void init_guess(double complex *Z_data, NML_work *w, NML_out *output);

#endif
