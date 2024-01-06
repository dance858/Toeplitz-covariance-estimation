#include <complex.h>
#include <fftw3.h>
#include "diff.h"
#include "utils.h"
#include <string.h>
#include <cblas.h>

void compute_derivatives_packed(NML_work *w){

    /* DFT of chol_toep. */
    pad_with_zeros(w->full_chol_toep, w->R_DFT, w->n_plus_one, w->n_plus_one, w->N);
    fftw_execute_dft(w->plan_R_DFT, w->R_DFT, w->R_DFT);
    double complex one = 1;

    /* compute A = R*(R'*L), where R is lower triangular. Store it in W->RHL. */
    cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                w->n_plus_one, w->n_plus_one, &one, w->full_chol_toep, 
                w->n_plus_one, w->RHL, w->n_plus_one);

    /* compute DFT of A */
    pad_with_zeros(w->RHL, w->A_DFT, w->n_plus_one, w->n_plus_one, w->N);
    fftw_execute_dft(w->plan_A_DFT, w->A_DFT, w->A_DFT);

    /* compute quantity that should be IFFT:ed to obtain the gradient */
    int i, k;
    double complex Rik, Aik;  // register variable?
    memset(w->grad_help, 0, w->N*sizeof(double complex));
    for(k = 0; k < w->n_plus_one; k++){
        for(i=0; i<w->N; i++){
            Rik = w->R_DFT[i+k*(w->N)];
            Aik = w->A_DFT[i+k*(w->N)];
            /* at this point grad_help is real but it will later become complex */
            w->grad_help[i] += (creal(Rik)*creal(Rik) + cimag(Rik)*cimag(Rik) - 
                                creal(Aik)*creal(Aik) - cimag(Aik)*cimag(Aik));
        }
    }

    /* IFFT (not normalized) */
    fftw_execute(w->plan_grad_help);

    /* correct scaling */
    complex double alpha = 2.0/(w->N);
    cblas_zscal(w->N, &alpha, w->grad_help, 1);

    /* parse grad_help to obtain the true gradient */
    w->grad[0] = creal(w->grad_help[0]);
    for(i=1; i<w->n_plus_one; i++){
        w->grad[i] = creal(w->grad_help[i]);
        w->grad[i+w->n] = -cimag(w->grad_help[i]);
    }

    /* F = R_DFT*R_DFT^H, complex-valued Hermitian matrix stored as lower triangular */
    cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, w->N, w->n_plus_one, 1,
                w->R_DFT, w->N, 0, w->F, w->N);

    /* G = A_DFT*A_DFT^H, complex-valued Hermitian matrix stored as lower triangular */
    cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, w->N, w->n_plus_one, 1,
                w->A_DFT, w->N, 0, w->G, w->N);

    /* hess_help = F.*G^T + F^T.*G - F.*F^T. */
      for(k=0; k<w->N; k++){
        for(i=k; i<w->N; i++){
            w->hess_help[i + k*w->N] = 
                2 * (creal(w->F[i + k*w->N])*creal(w->G[i + k*w->N])  +
                     cimag(w->F[i + k*w->N])*cimag(w->G[i + k*w->N])) -
                     creal(w->F[i + k*w->N])*creal(w->F[i + k*w->N])  -
                     cimag(w->F[i + k*w->N])*cimag(w->F[i + k*w->N]);
            w->hess_help[k + i*w->N] = w->hess_help[i + k*w->N];
        }
    }

    /* hess_help =  (F.*G^T + F^T.*G - F.*F^T)*W, 
       but size N x N instead of the mathematically correct size that is
       N x (n+1).  */
    fftw_execute_dft(w->plan_hess_help, w->hess_help, w->hess_help);
    hermitian_conj(w->hess_help, w->N); /* can we get rid of this one? should be possible */
    

    /* hess_help =  W^H*(F.*G^T + F^T.*G - F.*F^T)*W, but size N x N instead 
       of the mathematically correct size that is (n+1) x (n+1). */ 
    fftw_execute_dft(w->plan_hess_help, w->hess_help, w->hess_help);

    /* correct scaling, hess_help =  4/N^2*W^H*(F.*G^T + F^T.*G - F.*F^T)*W.
       Note that it has size N x N instead of (n+1) x (n+1). */
    alpha = 4.0/(w->N*w->N);
    cblas_zscal(w->N*w->N, &alpha, w->hess_help, 1);

    /* build the Hessian. Loop index k represents column number, loop index i 
       represents row number. */
    for(k = 0; k < w->n_plus_one; k++){ /* xx-block */
        for(i = k; i < w->n_plus_one; i++){
            w->hess_packed[i + k * w->two_n_plus_one - k * (k + 1) / 2] = 
                0.5*(creal(w->hess_help[i + k*(w->N)]) 
                   + creal(w->hess_help[(k + 1)*(w->N) - i]));
        }
    }
    w->hess_packed[0] = creal(w->hess_help[0]);

    for(k = 0; k < w->n_plus_one; k++){ /* yx-block */
        for(i = w->n_plus_one; i < w->two_n_plus_one; i++){
            w->hess_packed[i + k * w->two_n_plus_one - k * (k + 1) / 2] =  
            0.5*(cimag(w->hess_help[(k + 1)*(w->N) - i + (w->n)]) 
               - cimag(w->hess_help[i + k*(w->N) - (w->n)]));
        }
    }

     for(k = w->n_plus_one; k < w->two_n_plus_one; k++){ /* yy-block */
        for(i = k; i < w->two_n_plus_one; i++){
            w->hess_packed[i + k * w->two_n_plus_one - k * (k + 1) / 2] =
             0.5*(creal(w->hess_help[i + (k - w->n)*(w->N) - (w->n)]) 
                - creal(w->hess_help[(k + 1 - w->n)*(w->N) - i + (w->n)]));
        }
    }
}
