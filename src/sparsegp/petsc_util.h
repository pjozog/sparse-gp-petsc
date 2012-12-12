#ifndef __HAS_PETSC_UTIL_H__
#define __HAS_PETSC_UTIL_H__

#include <petscvec.h>
#include <petscmat.h>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#include "sparsekernels.h"

#ifdef __cplusplus
extern "C" {
#endif

int
petsc_util_fillVec (const gsl_vector *gslVec, Vec *petscVec, int startRow, int endRow);

int
petsc_util_fillVecFromKernel (const AcfrKern *kern, const gsl_matrix *X, const gsl_vector *y, Vec *vec, int startRow, int endRow);

int
petsc_util_fillMatFromKernel (const AcfrKern *kern, const gsl_matrix *X, Mat *mat, int startRow, int endRow, int startCol, int endCol);

int
petsc_util_createVec (Vec *vec, int size, int length);

int
petsc_util_fillLenghtGradMatFromKernel (const AcfrKern *kern, const gsl_matrix *X, 
                                        Mat *mat, int startRow, int endRow, int startCol, int endCol, int lengthInd);

int
petsc_util_fillSigma0GradMatFromKernel (const AcfrKern *kern, const gsl_matrix *X, 
                                        Mat *mat, int startRow, int endRow, int startCol, int endCol);

#ifdef __cplusplus
}
#endif

#endif  /* __HAS_PETSC_UTIL_H__ */
