#ifndef __HAS_KERNELS_H__
#define __HAS_KERNELS_H__

#include <stdint.h>

#include <gsl/gsl_vector_double.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum AcfrKernType {
    ACFR_KERN_TYPE_1 = 1,
    ACFR_KERN_TYPE_2 = 2,
} AcfrKernType;

typedef struct AcfrKern {
    double beta;
    double sigma0;
    gsl_vector *lengths;
    AcfrKernType type;
} AcfrKern;

double
AcfrKern_eval (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2);

double
AcfrKern_KGradSigma0 (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2);

double
AcfrKern_KGradLength (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2, size_t ind);

void
AcfrKern_free (AcfrKern *kern);

#ifdef __cplusplus
}
#endif

#endif  /* __HAS_KERNELS_H__ */
