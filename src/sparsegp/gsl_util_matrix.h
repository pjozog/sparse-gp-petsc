#ifndef __GSL_UTIL_MATRIX_H__
#define __GSL_UTIL_MATRIX_H__

#include <math.h>
#include <assert.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "gsl_util_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Macro for creating a gsl_matrix_view object and data on the stack.
 * Use gslu macros to create matrix elements on the stack with matrix
 * views.  var will have subfields .data and .matrix.  The advantage
 * is that elements created on the stack do not have to be manually
 * freed by the programmer, they automatically have limited scope
 */
#define GSLU_MATRIX_VIEW(var,i,j,...)                                   \
    struct {                                                            \
        double data[i*j];                                               \
        gsl_matrix matrix;                                              \
    } var = {__VA_ARGS__};                                              \
    {   /* _view_ has local scope */                                    \
        gsl_matrix_view _view_ = gsl_matrix_view_array (var.data, i, j); \
        var.matrix = _view_.matrix;                                     \
    }

/**
 * Macro for type conversion, e.g. the following command would
 * convert gsl_matrix_float *f, to gsl_matrix *d
 * GSLU_MATRIX_TYPEA_TO_TYPEB (gsl_matrix_float, f, gsl_matrix, d);
 */
#define GSLU_MATRIX_TYPEA_TO_TYPEB(typeA, A, typeB, B) {                \
        const typeA *_AA_ = A;                                          \
        const typeB *_BB_ = B;                                          \
        assert (_AA_->size1 == _BB_->size1 && _AA_->size2 == _BB_->size2); \
        for (size_t _i_=0; _i_<_AA_->size1; _i_++) {                    \
            for (size_t _j_=0; _j_<_AA_->size2; _j_++) {                \
                typeB ## _set (B, _i_, _j_, typeA ## _get (A, _i_, _j_)); \
            }                                                           \
        };                                                              \
    }


/** 
 * prints the contents of a matrix to stdout.  each element is formatted
 * using the printf-style format specifier fmt.  
 *
 * @param fmt if it is NULL, then it defaults to "%f"
 * @param trans one of either CblasNoTrans, CblasTrans, CblasConjTrans
 */
void
gslu_matrix_printfc (const gsl_matrix *m, const char *name, 
                     const char *fmt, CBLAS_TRANSPOSE_t trans);

/**
 * prints the contents of a matrix to stdout. e.g. gslu_matrix_printf (A, "A");
 */
static inline void
gslu_matrix_printf (const gsl_matrix *A, const char *name)
{
    gslu_matrix_printfc (A, name, NULL, CblasNoTrans);
}

#ifdef __cplusplus
}
#endif

#endif //__GSL_UTIL_MATRIX_H__
