#ifndef __HAS_SPARSEGP_H__
#define __HAS_SPARSEGP_H__

#include "sparsekernels.h"

#include <petscksp.h>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#ifdef _cplusplus
extern "C" {
#endif

#define GP_RANK_ROOT 0

typedef struct SparseGp {
    int rank;
    AcfrKern *kern;

    gsl_matrix *trainObs;
    gsl_vector *trainLabels;
    gsl_matrix *testObs;
    gsl_vector *testLabels;

    Vec _trainLabels;
    Mat _K;
    Mat _KGradient;

    int rstart;
    int rend;
    int nlocal;

    KSP ksp;
    PC pc;
} SparseGp;

typedef enum HyperParam {
    GP_HYPERPARAM_SIGMA0,
    GP_HYPERPARAM_LENGTH_I,
} HyperParam;

SparseGp *
SparseGp_create (AcfrKern *kern, int rank, gsl_matrix *trainObs, 
                 gsl_vector *trainLabels, gsl_matrix *testObs, 
                 gsl_vector *testLabels);

int
SparseGp_destroy (SparseGp *gp);

int
SparseGp_train (SparseGp *gp);

int
SparseGp_learn (SparseGp *gp);

int
SparseGp_KGradient (SparseGp *gp, HyperParam hp, int lengthInd);

int
SparseGp_logLikeGrad (SparseGp *gp, HyperParam hp, int lengthInd, double *logLikeGrad);

double
SparseGp_logLikelihood (SparseGp *gp);

int
SparseGp_solve (SparseGp *gp, Vec rhs, Vec *result);

int
SparseGp_setupKsp (SparseGp *gp);

double
SparseGp_percentNonzero (SparseGp *gp);

int
SparseGp_computeMuAt (SparseGp *gp, gsl_vector *mu, const gsl_matrix *x);

int
SparseGp_computeVarAt (SparseGp *gp, gsl_vector *var, const gsl_matrix *x);

#ifdef _cplusplus
}
#endif

#endif  /* __HAS_SPARSEGP_H__ */
