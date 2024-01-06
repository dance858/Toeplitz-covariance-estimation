#include "utils.h"

/* reverses a real-valued array of size N */
void reverse_real(double *y, const int N){
    double c;
    int i, j;
    for(i = 0, j = N-1; i < j; i++, j--){
        c = y[i];
        y[i] = y[j];
        y[j] = c;
    }
}

/* reverses a complex-valued array of size N */
void reverse_complex(double complex *y, const int N){
    double complex c;
    int i, j;
    for(i = 0, j = N-1; i < j; i++, j--){
        c = y[i];
        y[i] = y[j];
        y[j] = c;
    }
}

void print_my_vector_real(double *x, const int n){
    for(int i=0; i < n; i++)
        printf("%f \t", x[i]);
    printf("\n");
}

void print_my_vector_complex(double complex *x, const int n){
    for(int i=0; i < n; i++)
        printf("%f + %fi \t", creal(x[i]), cimag(x[i]));
    printf("\n");
}

/* Assumes column-major order */
void print_complex_lower_triangular_matrix(double complex *L, int nrows){
    int i, j, index;

    for(i = 0; i<nrows; i++){    
        index = i;
        for(j = 0; j<=i; j++){
             printf("%f + %fi \t", creal(L[index]), cimag(L[index]));
             index += (nrows - 1 - j);
        }
        for(j=0; j<nrows-1-i; j++)
             printf("%f + %fi \t", 0.0, 0.0);
        printf("\n");
    }
}
/* Assumes column-major order */
void print_real_lower_triangular_matrix(double *L, int nrows){
    int i, j, index;

    for(i = 0; i<nrows; i++){    
        index = i;
        for(j = 0; j<=i; j++){
             printf("%f \t", L[index]);
             index += (nrows - 1 - j);
        }
        for(j=0; j<nrows-1-i; j++)
             printf("%f \t", 0.0);
        printf("\n");
    }
}


/* Assumes column-major order */
void print_complex_matrix(double complex *A, int nrows, int ncols){
    int i, j;
    for(i = 0; i<nrows; i++){
        for(j = 0; j <ncols; j++){
             printf("%.2f + %.2fi \t", creal(A[i+j*nrows]), cimag(A[i+j*nrows]));
        }
        printf("\n");
    }
}

/* Assumes column-major order */
void print_real_matrix(double *A, int nrows, int ncols){
    int i, j;
    for(i = 0; i<nrows; i++){
        for(j = 0; j <ncols; j++){
             printf("%.2f \t", A[i+j*nrows]);
        }
        printf("\n");
    }
}


