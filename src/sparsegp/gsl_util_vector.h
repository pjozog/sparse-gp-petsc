#ifndef __GSL_UTIL_VECTOR_H__
#define __GSL_UTIL_VECTOR_H__

#include <assert.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "gsl_util_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Macro for creating a gsl_vector_view object and data on the stack.
 * Use gslu macros to create vector elements on the stack with vector
 * views.  var will have subfields .data and .vector.  The advantage
 * is that elements created on the stack do not have to be manually
 * freed by the programmer, they automatically have limited scope
 */
#define GSLU_VECTOR_VIEW(var,i,...)                                     \
    struct {                                                            \
        double data[i];                                                 \
        gsl_vector vector;                                              \
    } var = {__VA_ARGS__};                                              \
    {   /* _view_ has local scope */                                    \
        gsl_vector_view _view_ = gsl_vector_view_array (var.data, i);   \
        var.vector = _view_.vector;                                     \
    }

void
gslu_vector_printf (const gsl_vector *v, const char *name);

/**
 * prints the contents of a vector to stdout.  each element is formatted
 * using the printf-style format specifier fmt.  
 *
 * @param fmt if it is NULL, then it defaults to "%f"
 * @param trans one of either CblasNoTrans, CblasTrans, CblasConjTrans
 */
void
gslu_vector_printfc (const gsl_vector *v, const char *name,
                     const char *fmt, CBLAS_TRANSPOSE_t trans);

#ifdef __cplusplus
}
#endif

#endif //__GSL_UTIL_VECTOR_H__
