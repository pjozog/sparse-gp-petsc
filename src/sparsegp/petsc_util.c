#include "petsc_util.h"
#include "gsl_util_vector.h"

int
petsc_util_fillVec (const gsl_vector *gslVec, Vec *petscVec, int startRow, int endRow)
{
    PetscErrorCode ierr;

    for (int i=startRow; i<endRow; i++) {
        double val = gsl_vector_get (gslVec, i);
        if (fabs (val) > 0) {
            ierr = VecSetValue (*petscVec, i, val, INSERT_VALUES);
            CHKERRQ (ierr);
        }
    }

    ierr = VecAssemblyBegin (*petscVec); CHKERRQ (ierr);
    ierr = VecAssemblyEnd (*petscVec); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
petsc_util_fillVecFromKernel (const AcfrKern *kern, const gsl_matrix *X, const gsl_vector *y, Vec *vec, int startRow, int endRow)
{
    PetscErrorCode ierr;

    for (int i=startRow; i<endRow; i++) { 
        gsl_vector_const_view Xi = gsl_matrix_const_row (X, i);
        printf ("%d --- %ld, %ld\n", i, Xi.vector.size, y->size);
        double kernValue = AcfrKern_eval (kern, &Xi.vector, y);

        if (fabs (kernValue) > 0) {
            ierr = VecSetValue (*vec, i, kernValue, INSERT_VALUES); 
            CHKERRQ (ierr);
        }
    }

    ierr = VecAssemblyBegin (*vec); CHKERRQ (ierr);
    ierr = VecAssemblyEnd (*vec); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
petsc_util_fillMatFromKernel (const AcfrKern *kern, const gsl_matrix *X, 
                              Mat *mat, int startRow, int endRow, int startCol, int endCol)
{
    PetscErrorCode ierr;

    for (int i=startRow; i<endRow; i++) {
        for (int j=startCol; j<endCol; j++) {
            gsl_vector_const_view Xi = gsl_matrix_const_column (X, i);
            gsl_vector_const_view Xj = gsl_matrix_const_column (X, j);
            double kernValue = AcfrKern_eval (kern, &Xi.vector, &Xj.vector);

            if (i == j)
                kernValue += kern->beta;

            if (fabs (kernValue) > 0) {
                ierr = MatSetValue (*mat, i, j, kernValue, INSERT_VALUES);
                CHKERRQ (ierr);
            }
        }
    }

    ierr = MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);
    ierr = MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
petsc_util_fillSigma0GradMatFromKernel (const AcfrKern *kern, const gsl_matrix *X, 
                                        Mat *mat, int startRow, int endRow, int startCol, int endCol)
{
    PetscErrorCode ierr;

    for (int i=startRow; i<endRow; i++) {
        for (int j=startCol; j<endCol; j++) {
            gsl_vector_const_view Xi = gsl_matrix_const_column (X, i);
            gsl_vector_const_view Xj = gsl_matrix_const_column (X, j);

            double kernValue = AcfrKern_KGradSigma0 (kern, &Xi.vector, &Xj.vector);

            if (fabs (kernValue) > 0) {
                ierr = MatSetValue (*mat, i, j, kernValue, INSERT_VALUES);
                CHKERRQ (ierr);
            }
        }
    }

    ierr = MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);
    ierr = MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
petsc_util_fillLenghtGradMatFromKernel (const AcfrKern *kern, const gsl_matrix *X, 
                                        Mat *mat, int startRow, int endRow, int startCol, int endCol, int lengthInd)
{
    PetscErrorCode ierr;

    for (int i=startRow; i<endRow; i++) {
        for (int j=startCol; j<endCol; j++) {
            gsl_vector_const_view Xi = gsl_matrix_const_column (X, i);
            gsl_vector_const_view Xj = gsl_matrix_const_column (X, j);

            double kernValue = AcfrKern_KGradLength (kern, &Xi.vector, &Xj.vector, lengthInd);

            if (fabs (kernValue) > 0) {
                ierr = MatSetValue (*mat, i, j, kernValue, INSERT_VALUES);
                CHKERRQ (ierr);
            }
        }
    }

    ierr = MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);
    ierr = MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
petsc_util_createVec (Vec *vec, int size, int length)
{
    PetscErrorCode ierr;

    ierr = VecCreate (PETSC_COMM_WORLD, vec);
    ierr = VecSetSizes (*vec, size, length); CHKERRQ (ierr);
    ierr = VecSetFromOptions (*vec); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}
