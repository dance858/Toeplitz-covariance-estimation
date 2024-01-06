#include "Newton.h"
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <linalg.h>
#include "diff.h"
#include <string.h>
#include <time.h>
#include "utils.h"

/* This macro should never be called wiith arguments like MIN(++a, ++b) */
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* Evaluates the objective function 
   f(x, y) = log det T(x, y) + Tr( T(x, y)^{-1} S).

   NOTE: Before this function is called, chol_toep and sigma2 must have been 
   computed:
            T(x, y)^{-1} = chol_toep*diag(sigma2)^{-1}*chol_toep^H.
*/
double compute_obj(NML_work *w){
    double obj = 0;
    int ii;
    double complex one = 1;
    
    /* compute w->RHL = R^H*L, where R is lower triangular and L is treated as a 
       full matrix (despite L being lower triangular).
       
       I could not find a BLAS routine for matrix-matrix mult A*B where A
       is a triangular matrix stored in PACKED format. However, I did find 
       routines for packed triangular matrix-vector multiplication (CTPMV).
       Maybe we can refactor these three lines into one single function with
       for loops? If such a refactorization is made, note that it is necessary
       to run tri_to_full before "pad_with_zeros" inside compute_derivatives.
    */
    memcpy(w->RHL, w->L_full, sizeof(double complex) * (w->n_plus_one) * (w->n_plus_one));
    tri_to_full(w->full_chol_toep, w->chol_toep, w->n_plus_one);
    cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit,
                w->n_plus_one, w->n_plus_one, &one, w->full_chol_toep, 
                w->n_plus_one, w->RHL, w->n_plus_one);
    
    /* compute Tr(T(x, y)^{-1} S) */
    obj += cblas_dznrm2((w->n_plus_one)*(w->n_plus_one), w->RHL, 1);
    obj *= obj;

    /* compute log det T(x, y) */
   for(ii = 0; ii < w->n_plus_one; ii++){
        obj += log((w->sigma2)[ii]);
    }
    return obj;
} 

/* Convergence if ||grad||_2 < tol */
int has_converged(NML_work *w){
    w->grad_norm = cblas_dnrm2(w->two_n_plus_one, w->grad, 1);

    if (w->verbose){
        printf("grad_norm/obj: %.6e \t %.6f \n", w->grad_norm, w->obj);
    }

    return (w->grad_norm < w->tol);
}


/* Convergence if Newton decrement < tol */
int has_converged_new(NML_work *w){
    double newton_dec = cblas_ddot(w->two_n_plus_one, w->grad, 1, w->neg_dir, 1);
    if (w->verbose){
        w->grad_norm = cblas_dnrm2(w->two_n_plus_one, w->grad, 1);
        printf("grad_norm/obj: %.6e \t %.6f \n", w->grad_norm, w->obj);
    }

    return (newton_dec < w->tol);
}

/* Computes the step size.

    NOTE: When this function is called, w->neg_dir must contain the NEGATIVE
          search direction.
*/
void compute_stepsize(NML_work *w){
    w->step_size = 1/(w->beta);           /* corresponds to initial step size 1 */
    int status, i;

    /* backtrack to ensure that the new iterate is in dom f */
    do {
        w->step_size *= w->beta;

        /* w-> stores the first column of T(x, y) */
        w->z[0] = 2*(w->xy[0] - w->step_size*w->neg_dir[0]);
        for(i = 1; i < w->n_plus_one; i++){
            w->z[i] = (w->xy[i] - w->step_size*w->neg_dir[i])
                    - (w->xy[w->n + i] - w->step_size*w->neg_dir[w->n + i])*I;
        }

        /* status equal to 1 indicates that the factorization failed */
        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, w->n);
       
    } while(status == 1);
    
    /* scale w->chol_toep so T(x, y)^{-1} = (w->chol_toep)*(w->chol_toep)^H */
    lower_tri_diag_isqrt_mult(w->n_plus_one, w->sigma2, w->chol_toep);

    /* backtrack until descent condition is satisfied */
    double dir_der;                                        /* directional derivative */
    dir_der = -cblas_ddot(w->two_n_plus_one, w->grad, 1, w->neg_dir, 1);
    
    while (1){
        w->new_obj = compute_obj(w);  
        
        if (w->new_obj < w->obj + w->alpha * w->step_size * dir_der){
            break;
        }

        w->step_size *= w->beta;
        w->z[0] = 2*(w->xy[0] - w->step_size * w->neg_dir[0]);
        for(i = 1; i < w->n_plus_one; i++){
            w->z[i] = (w->xy[i] - w->step_size * w->neg_dir[i])
                    - (w->xy[w->n + i] - w->step_size * w->neg_dir[w->n + i])*I;
        }
        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, w->n);
        lower_tri_diag_isqrt_mult(w->n_plus_one, w->sigma2, w->chol_toep);
    }
    w->obj = w->new_obj;
}

/* NOTE:  Assumes that w->chol_toep has been computed. */
void Newton(NML_work *w, NML_out *output){
    int i, k, info;
    int num_of_found_evals;
    int not_needed_1;
    double not_needed_2;
   
    /* Compute objective value */
    w->obj = compute_obj(w);
    
    for(i = 0; i<w->max_iter; i++){
        /* compute gradient and Hessian */
        compute_derivatives_packed(w);
        
        /* check termination criteria */
        //if (has_converged(w)){
        //    break;
        //}
        
        /* Cholesky factorization of Hessian, modify it if necessary */
        while (1){
            memcpy(w->chol_hess_packed, w->hess_packed, sizeof(double) * w->n_plus_one * w->two_n_plus_one);
            info = LAPACKE_dpptrf(LAPACK_COL_MAJOR, 'L', w->two_n_plus_one, w->chol_hess_packed);
            if (info == 0){
                break;
            }
            
            /* factorization failed */
            if (w->verbose){
                printf("iter %i. Hessian not PD. Adding multiple of the identity. \n ", i);
            }
            
            output->num_of_hess_chol_fails += 1;
            /* compute eigenvalues of packed matrix. Some arguments are not needed. 
            Overwrites so must first copy.  */
            memcpy(w->chol_hess_packed, w->hess_packed, sizeof(double) * w->n_plus_one * w->two_n_plus_one);
            info = LAPACKE_dspevx(LAPACK_COL_MAJOR, 'N', 'A', 'L', w->two_n_plus_one,
                                    w->chol_hess_packed, 0.0, 0.0, 0.0, 0.0, -1.0,
                                    &num_of_found_evals, w->hess_evals, &not_needed_2,  1.0, &not_needed_1); 
            
            w->hess_evals[0] = MIN(-0.1, w->hess_evals[0])*1.05;
            /* add multiple of the identity to Hessian */
            for (k = 0; k < w->two_n_plus_one; k++){
                    w->hess_packed[k * (w->two_n_plus_one + 1) - k * (k + 1) / 2] -= w->hess_evals[0];
            }    
        }
        
        /* compute Newton direction. After this step the NEGATIVE of the
           direction is stored in  w->neg_dir. */
        memcpy(w->neg_dir, w->grad, sizeof(double) * w->two_n_plus_one);
        cblas_dtpsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, w->two_n_plus_one, 
                    w->chol_hess_packed, w->neg_dir, 1);
        cblas_dtpsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, w->two_n_plus_one, 
                    w->chol_hess_packed, w->neg_dir, 1);  
        
        if (has_converged_new(w)) {
            break;
        }
        
        /* compute step size. chol_toep and w->obj are updated. */
        compute_stepsize(w);
        
        /* update iterate */
        cblas_daxpy(w->two_n_plus_one, -w->step_size, w->neg_dir, 1, w->xy, 1);
    }

     /* store output */
    output->obj = w->obj;
    output->grad_norm = w->grad_norm;
    output->iter = i;
    memcpy(output->x_sol, w->xy, sizeof(double) * (w->n + 1));
    memcpy(output->y_sol, w->xy + w->n + 1, sizeof(double) * w->n);   
}
