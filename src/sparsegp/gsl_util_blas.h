#ifndef __GSL_UTIL_BLAS_H__
#define __GSL_UTIL_BLAS_H__

#include <assert.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline double
gslu_blas_vTmv (const gsl_vector *x, const gsl_matrix *A, const gsl_vector *y)
{
    assert (x->size==A->size1 && A->size2==y->size);
    double s = 0.0;
    for (size_t i=0; i<A->size1; i++) {
        double xi = gsl_vector_get (x, i);
        for (size_t j=0; j<A->size2; j++)
            s += xi * gsl_matrix_get (A, i, j) * gsl_vector_get (y, j);
    }
    return s;
}

#ifdef __cplusplus
}
#endif

#endif //__GSL_UTIL_BLAS_H__
