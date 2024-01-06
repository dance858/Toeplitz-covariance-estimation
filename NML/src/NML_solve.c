#include "NML_solve.h"
#include "levinson_durbin.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <lapacke.h>           
#include <cblas.h>
#include <utils.h>


/* allocates workspace for NML */
NML_work *init_work(const int n, const int K, double tol, double beta, double alpha,
                    int verbose, int max_iter){
    NML_work *w = malloc(sizeof(*w));
    w->n = n;
    w->N = 2 * (n + 1);
    w->n_plus_one = n + 1;
    w->two_n_plus_one = 2*n + 1;
    w->K = K;

    /* Algorithmic parameters */
    w->tol = tol;
    w->beta = beta;
    w->alpha = alpha;
    w->verbose = verbose;
    w->max_iter = max_iter;

    /* allocate memory for iterate */
    w->xy = malloc(sizeof(*w->xy) * (2 * n + 1));
    w->z = malloc(sizeof(*w->z) * (n + 1));
    
    /* allocate memory for derivatives */
    w->grad_help = (double complex *)fftw_malloc(sizeof(double complex) * w->N);
    w->hess_help = (double complex *)fftw_malloc(sizeof(double complex) * w->N * w->N);
    w->grad = malloc(sizeof(*w->grad) * (2 * n + 1));
    w->F = malloc(sizeof(*w->F) * w->N * w->N);
    w->G = malloc(sizeof(*w->G) * w->N * w->N);
    //w->hess = malloc(sizeof(*w->hess) * (2* n + 1) * (2 * n + 1)); 
    w->hess_packed = malloc(sizeof(*w->hess_packed) * (n + 1) * (2 * n + 1));
    w->chol_hess_packed = malloc(sizeof(*w->chol_hess_packed) * (n + 1) * (2 * n + 1));
    w->hess_evals = malloc(sizeof(*w->hess_evals) * (2 * n + 1));

    /* allocate memory for Cholesky factor of inverse Toeplitz matrix and
       Cholesky factor of Hessian */
    w->full_chol_toep = malloc(sizeof(*w->full_chol_toep) * (n + 1) * (n + 1));  
    w->chol_toep = malloc(sizeof(*w->chol_toep) * (n + 1) * (n + 2) / 2);
    w->sigma2 = malloc(sizeof(*w->sigma2) * (n + 1));
    w->RHL = malloc(sizeof(*w->RHL) * (n + 1) * (n + 1));
    w->L_full = malloc(sizeof(*w->L_full) * (n + 1) * (n + 1));
    w->sample_cov = malloc(sizeof(*w->sample_cov) * (n + 1) * (n + 1));

    /* allocate memory for search direction */
    w->neg_dir = malloc(sizeof(*w->neg_dir) * (2 * n + 1));
    
    /* allocate memory for DFTs, both R_DFT and A_DFT are N x (n+1). */
    w->R_DFT = (double complex *)fftw_malloc(sizeof(double complex) * w->N * (n + 1));
    w->A_DFT = (double complex *)fftw_malloc(sizeof(double complex) * w->N * (n + 1));

    /* build plans for FFT */
    w->plan_R_DFT = fftw_plan_many_dft(1, &(w->N), n+1, w->R_DFT, &(w->N), 1, w->N, 
                     w->R_DFT, &(w->N), 1, w->N, FFTW_FORWARD, FFTW_ESTIMATE);
    w->plan_A_DFT = fftw_plan_many_dft(1, &(w->N), n+1, w->A_DFT, &(w->N), 1, w->N, 
                     w->A_DFT, &(w->N), 1, w->N, FFTW_FORWARD, FFTW_ESTIMATE);
    w->plan_grad_help = fftw_plan_dft_1d(w->N, w->grad_help, w->grad_help, 
                                         FFTW_BACKWARD, FFTW_ESTIMATE);                
    w->plan_hess_help = fftw_plan_many_dft(1, &(w->N), w->N, w->A_DFT, &(w->N), 1, w->N, 
                         w->A_DFT, &(w->N), 1, w->N, FFTW_BACKWARD, FFTW_ESTIMATE);
    return w;
}

void finish_work(NML_work *w){

    /* deallocate iterates */
    free(w->xy);
    free(w->z);

    fftw_free(w->grad_help);
    fftw_free(w->hess_help);
    free(w->grad);
    free(w->F);
    free(w->G);
    free(w->hess_packed);
    free(w->chol_hess_packed);
    free(w->hess_evals);

    /* deallocate memory for Cholesky factors */
    free(w->full_chol_toep);
    free(w->chol_toep);
    free(w->sigma2);
    free(w->RHL);
    free(w->L_full);
    free(w->sample_cov);

    /* deallocate memory for search direction */
    free(w->neg_dir);

    /* destroy FFT plans */
    fftw_destroy_plan(w->plan_R_DFT);
    fftw_destroy_plan(w->plan_A_DFT);
    fftw_destroy_plan(w->plan_grad_help);
    fftw_destroy_plan(w->plan_hess_help);

    /* deallocate memory for DFTs */
    fftw_free(w->R_DFT);
    fftw_free(w->A_DFT);

    /* deallocate workspace */
    free(w);

}
/* Assume that output = malloc(sizeof(*output)) has been executed */
void init_output(NML_out *output, int n){
    output->x_sol = malloc(sizeof(double) * (n + 1));
    output->y_sol = malloc(sizeof(double) * n);
    output->diag_init_succeded = 1;
    output->num_of_hess_chol_fails = 0;
}

/* Computes an initial guess as described in the paper.
    IN: Z: data matrix, size (n+1) x K stored in column-major order
   
    NOTE: After exit, w->chol_toep has been modified so 
          T(x, y)^{-1} = w->chol_toep * w->chol_toep^H.
*/
void init_guess(double complex *Z_data, NML_work *w, NML_out *output){
    int i, status;
    ;
    /* Compute sample covariance. The lower triangular part is stored in a 
       full matrix of size (n+1) x (n+1) */
       memset(w->sample_cov, 0, sizeof(double complex) * w->n_plus_one * w->n_plus_one);
       cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, w->n_plus_one, w->K,
                  1.0 / w->K, Z_data, w->n_plus_one, 0.0, w->sample_cov, w->n_plus_one);

    /* Cholesky factorization of sample covariance. If the factorization fails
       because of numerical issues we add a small regularization. */     
    status = 1;
    while (status != 0){
        memcpy(w->L_full, w->sample_cov, sizeof(double complex) * w->n_plus_one * w->n_plus_one);
        status = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'L', w->n_plus_one, w->L_full, w->n_plus_one);

        /* if (status != 0){
            printf("Adding small regularization to sample covariance. ");
        } */
      
        /* add small regularization to diagonal elements of sample cov */
        for (i = 0; i < w->n_plus_one; i++){
            w->sample_cov[i + i*w->n_plus_one] += 1e-8;
        } 
    }
     
    /* Diagonal averaging of sample covariance */
    diagonal_averaging_full(w->z, w->sample_cov, w->n+1);
    
    /* Check if preliminary initial guess belongs to domain */
    status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, w->n);


    /* if status is equal to 1 it means that the initial guess does not belong 
       to the domain. In this case we should compute an initial guess based on 
       the circulant ML estimate. */
    if (status == 1){
        output->diag_init_succeded = 0;

        /* IFFT of data points, scaled correctly. Stored in Z_data since we no longer
           need Z_data. */
        fftw_plan plan_Z_data =  fftw_plan_many_dft(1, &(w->n_plus_one), w->K,
         Z_data, &(w->n_plus_one), 1, w->n_plus_one, Z_data, &(w->n_plus_one), 1,
         w->n_plus_one, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute_dft(plan_Z_data, Z_data, Z_data);

        /* compute q of size (n+1) x 1. It is stored in z. We omit multiplicative factors.  */
        memset(w->z, 0, sizeof(double complex) * (w->n_plus_one));
        for(int k = 0; k < w->K; k++){
            for (i = 0; i < w->n_plus_one; i++){
                w->z[i] += (creal(Z_data[i + k*(w->n_plus_one)]) * creal(Z_data[i + k*(w->n_plus_one)]) 
                         +  cimag(Z_data[i + k*(w->n_plus_one)]) * cimag(Z_data[i + k*(w->n_plus_one)]));
            }
        }
    
        /*  compute FFT of q */
        fftw_plan plan_q = fftw_plan_dft_1d(w->n_plus_one, w->z, w->z, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan_q);
        
        /*  scale with 1/(K*(n+1)^2). After this operation, w->z stores the first 
            column of the circulant ML estimate. */
        complex double my_alpha = 1.0/(w->K * w->n_plus_one * w->n_plus_one);
        cblas_zscal(w->n_plus_one, &my_alpha, w->z, 1);
    
        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, w->n);
        fftw_destroy_plan(plan_q);
        fftw_destroy_plan(plan_Z_data);
    }    
     /* scale w->chol_toep so T(x, y)^{-1} = (w->chol_toep)*(w->chol_toep)^H */
        lower_tri_diag_isqrt_mult(w->n+1, w->sigma2, w->chol_toep);

        (w->xy)[0] = creal((w->z)[0])/2.0;
        for(i = 1; i<w->n_plus_one; i++){
            (w->xy)[i] = creal((w->z)[i]);
            (w->xy)[w->n + i] = -cimag((w->z)[i]); /* minus sign follows from definition of T(x, y) */
        }
}

/* Where should output be deallocated? */
int NML(double complex *Z_data, const int n, const int K, NML_out *output,
        double tol, double beta, double alpha, int verbose, int max_iter){

    struct timeval stop_total_time, start_total_time;
    gettimeofday(&start_total_time, NULL);

    /* set up workspace */
    NML_work *w = init_work(n, K, tol, beta, alpha, verbose, max_iter); 
    init_output(output, n);
    
    /* compute initial point */
    init_guess(Z_data, w, output);
    
    /* Newton's method */
    //printf("Starting Newton's method");
    Newton(w, output);
    //printf("Finished Newton's method");
    
    /* deallocate workspace */
    finish_work(w);
  
    gettimeofday(&stop_total_time, NULL);
    output->total_time =  (stop_total_time.tv_sec - start_total_time.tv_sec)
                        + (stop_total_time.tv_usec - start_total_time.tv_usec) / 1000000.0;
    
    return 0;
}
