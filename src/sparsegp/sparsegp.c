#include "sparsegp.h"

#include "petsc_util.h"
#include "io_util.h"

SparseGp *
SparseGp_create (AcfrKern *kern, int rank, gsl_matrix *trainObs, 
                 gsl_vector *trainLabels, gsl_matrix *testObs, 
                 gsl_vector *testLabels)
{
    PetscErrorCode ierr;        /* not used here - not returning int */
    (void) ierr;

    SparseGp *gp = calloc (1, sizeof (*gp));

    gp->kern = kern;

    gp->trainObs = trainObs;
    gp->trainLabels = trainLabels;
    gp->testObs = testObs;
    gp->testLabels = testLabels;

    PetscInt N = gp->trainLabels->size;
    ierr = VecCreate (PETSC_COMM_WORLD, &(gp->_trainLabels)); /* CHKERRQ (ierr); */
    ierr = VecSetSizes (gp->_trainLabels, PETSC_DECIDE, N); /* CHKERRQ (ierr); */
    ierr = VecSetFromOptions (gp->_trainLabels); /* CHKERRQ (ierr); */

    ierr = VecGetOwnershipRange (gp->_trainLabels, &(gp->rstart), &(gp->rend)); /* CHKERRQ (ierr); */
    ierr = VecGetLocalSize (gp->_trainLabels, &(gp->nlocal)); /* CHKERRQ (ierr); */
    petsc_util_fillVec (gp->trainLabels, &gp->_trainLabels, gp->rstart, gp->rend);

    gp->rank = rank;

    return gp;
}

int
SparseGp_train (SparseGp *gp)
{
    PetscErrorCode ierr;

    int N = gp->trainObs->size2;

    /* create sparse K matrix */
    ierr = MatCreate (PETSC_COMM_WORLD, &(gp->_K)); CHKERRQ(ierr);
    ierr = MatSetSizes (gp->_K, gp->nlocal, gp->nlocal, N, N); CHKERRQ(ierr);
    ierr = MatSetFromOptions(gp->_K); CHKERRQ(ierr);
    ierr = MatSetUp (gp->_K); CHKERRQ(ierr);

    /* assemble matrix */
    petsc_util_fillMatFromKernel (gp->kern, gp->trainObs, &(gp->_K), 0, N, gp->rstart, gp->rend);
    ierr = MatAssemblyBegin(gp->_K, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);
    ierr = MatAssemblyEnd(gp->_K, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
SparseGp_learn (SparseGp *gp)
{
    PetscErrorCode ierr;
    (void) ierr;

    HyperParam hp = GP_HYPERPARAM_SIGMA0;
    SparseGp_KGradient (gp, hp, -1);
    /* MatView (gp->_KGradient, PETSC_VIEWER_STDOUT_WORLD); */
    MatDestroy (&gp->_KGradient);

    /* compute length_i gradient */
    hp = GP_HYPERPARAM_LENGTH_I;
    for (int i=0; i<gp->kern->lengths->size; i++) {
        SparseGp_KGradient (gp, hp, i);
        /* MatView (gp->_KGradient, PETSC_VIEWER_STDOUT_WORLD); */
        MatDestroy (&gp->_KGradient);
    }

    return EXIT_SUCCESS;
}

int
SparseGp_KGradient (SparseGp *gp, HyperParam hp, int lengthInd)
{
    PetscErrorCode ierr;
    (void) ierr;

    int N = gp->trainObs->size2;

    /* create sparse K matrix */
    ierr = MatCreate (PETSC_COMM_WORLD, &(gp->_KGradient)); CHKERRQ(ierr);
    ierr = MatSetSizes (gp->_KGradient, gp->nlocal, gp->nlocal, N, N); CHKERRQ(ierr);
    ierr = MatSetFromOptions(gp->_KGradient); CHKERRQ(ierr);
    ierr = MatSetUp (gp->_KGradient); CHKERRQ(ierr);

    /* assemble matrix */
    switch (hp) {
    case GP_HYPERPARAM_SIGMA0: {
        petsc_util_fillSigma0GradMatFromKernel (gp->kern, gp->trainObs, &(gp->_KGradient), 0, N, gp->rstart, gp->rend);
        break;
    }
    case GP_HYPERPARAM_LENGTH_I: {
        petsc_util_fillLenghtGradMatFromKernel (gp->kern, gp->trainObs, &(gp->_KGradient), 0, N, gp->rstart, gp->rend, lengthInd);
        break;
    }
    default:
        printf ("Unknown hyperparameter type\n");
        exit (EXIT_FAILURE);
    }

    ierr = MatAssemblyBegin(gp->_KGradient, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);
    ierr = MatAssemblyEnd(gp->_KGradient, MAT_FINAL_ASSEMBLY); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
SparseGp_solve (SparseGp *gp, Vec rhs, Vec *result)
{
    PetscErrorCode ierr;

    ierr = KSPSolve (gp->ksp, rhs, *result); CHKERRQ (ierr);
    
    return EXIT_SUCCESS;
}

int
SparseGp_logLikeGrad (SparseGp *gp, HyperParam hp, int lengthInd, double *logLikeGrad)
{
    PetscErrorCode ierr;
    (void) ierr;
    /* compute t' inv(K) (dKdt inv(K) t) */
    /* 1. solve inv(K) t */
    PetscInt N = gp->trainLabels->size;
    Vec invKt;
    ierr = petsc_util_createVec (&invKt, gp->nlocal, N);
    SparseGp_solve (gp, gp->_trainLabels, &invKt);

    /* 2. multiply dKdt by invKt */
    Vec dKdtInvKt;
    ierr = petsc_util_createVec (&dKdtInvKt, gp->nlocal, N);
    SparseGp_KGradient (gp, hp, lengthInd);
    MatMult (gp->_KGradient, invKt, dKdtInvKt);

    /* 3. solve invK (vector from step 2.) */
    Vec invKDkdtInvKt;
    ierr = petsc_util_createVec (&invKDkdtInvKt, gp->nlocal, N);
    SparseGp_solve (gp, dKdtInvKt, &invKDkdtInvKt);

    /* 4. compute inner product */
    double dotProd;
    VecDot (gp->_trainLabels, invKDkdtInvKt, &dotProd);

    /* compute trace (invK dKdt) */
    /* 1. for each column in dKdt, solve */
    double myTrace = 0;

    Vec col, solution;
    petsc_util_createVec (&col, gp->nlocal, N);
    petsc_util_createVec (&solution, gp->nlocal, N);

    for (int i=0; i<N; i++) {
        /* IOU_ROOT_PRINT ("%d --- %d\n", i, N); */
        int row = i;

        ierr = MatGetColumnVector (gp->_KGradient, col, row);

        /* IOU_ROOT_PRINT ("solving...\n"); */
        SparseGp_solve (gp, col, &solution);
        /* IOU_ROOT_PRINT ("solved\n"); */

        if (row < gp->rend && row >= gp->rstart) {
            double ii;
            VecGetValues (solution, 1, &row, &ii);
            myTrace += ii;
        }

    }

    /* gather all trace terms */
    int numProcs;
    MPI_Comm_size (PETSC_COMM_WORLD, &numProcs);
    double diag[numProcs];

    int ret = MPI_Allgather (&myTrace, 1, MPI_DOUBLE, diag, 1, MPI_DOUBLE, PETSC_COMM_WORLD);
    (void) ret;

    double trace = 0;
    for (int i=0; i<numProcs; i++)
        trace += diag[i];

    /* IOU_ROOT_PRINT ("Cleaning up...\n"); */
    /* clean up */
    VecDestroy (&invKt);
    VecDestroy (&dKdtInvKt);
    VecDestroy (&invKDkdtInvKt);
    VecDestroy (&col);
    VecDestroy (&solution);
    /* IOU_ROOT_PRINT ("Done cleaning up...\n"); */

    *logLikeGrad = 0.5*dotProd + 0.5*trace;

    return EXIT_SUCCESS;
}

double
SparseGp_logLikelihood (SparseGp *gp)
{
    return -1;
}

int
SparseGp_setupKsp (SparseGp *gp)
{
    PetscErrorCode ierr;

    ierr = KSPCreate(PETSC_COMM_WORLD,&(gp->ksp)); CHKERRQ (ierr);

    ierr = KSPSetOperators(gp->ksp,gp->_K,gp->_K,DIFFERENT_NONZERO_PATTERN); CHKERRQ (ierr);

    ierr = KSPGetPC(gp->ksp,&gp->pc); CHKERRQ (ierr);
    ierr = PCSetType(gp->pc,PCJACOBI); CHKERRQ (ierr);
    ierr = KSPSetTolerances(gp->ksp,1.e-2,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ (ierr);

    ierr = KSPSetFromOptions(gp->ksp); CHKERRQ (ierr);

    return EXIT_SUCCESS;
}

int
SparseGp_destroy (SparseGp *gp)
{
    PetscErrorCode ierr;

    ierr = MatDestroy (&gp->_K); CHKERRQ (ierr);
    ierr = MatDestroy (&gp->_KGradient); CHKERRQ (ierr);
    ierr = VecDestroy (&gp->_trainLabels); CHKERRQ (ierr);
    ierr = KSPDestroy(&gp->ksp); CHKERRQ (ierr);

    gsl_matrix_free (gp->trainObs);
    gsl_vector_free (gp->trainLabels);

    gsl_matrix_free (gp->testObs);
    gsl_vector_free (gp->testLabels);

    return EXIT_SUCCESS;
}

double
SparseGp_percentNonzero (SparseGp *gp)
{
    MatInfo info;
    MatGetInfo (gp->_K, MAT_GLOBAL_SUM, &info);

    PetscLogDouble nzUsed = info.nz_used;
    PetscInt N = gp->trainLabels->size;
    
    return (nzUsed / (N*N)) * 100;
}

int
SparseGp_computeMuAt (SparseGp *gp, gsl_vector *mu, const gsl_matrix *x)
{
    PetscErrorCode ierr;

    PetscInt N = gp->trainLabels->size;
    Vec invKt;
    ierr = petsc_util_createVec (&invKt, gp->nlocal, N);
    ierr = SparseGp_solve (gp, gp->_trainLabels, &invKt);

    for (int i=0; i<x->size2; i++) {
        gsl_vector *k = gsl_vector_calloc (gp->trainObs->size2);
        gsl_vector_const_view xi = gsl_matrix_const_column (x, i);

        for (int j=0; j<k->size; j++) {
            gsl_vector_const_view xj = gsl_matrix_const_column (gp->trainObs, j);
            gsl_vector_set (k, j, AcfrKern_eval (gp->kern, &xj.vector, &xi.vector));
        }

        Vec _k;
        ierr = petsc_util_createVec (&_k, gp->nlocal, N);
        petsc_util_fillVec (k, &_k, gp->rstart, gp->rend);

        double val;
        ierr = VecDot (_k, invKt, &val);
        gsl_vector_set (mu, i, val);

        /* clean up */
        gsl_vector_free (k);
        VecDestroy (&_k);
    }

    /* clean up */
    VecDestroy (&invKt);
    
    return EXIT_SUCCESS;
}

int
SparseGp_computeVarAt (SparseGp *gp, gsl_vector *var, const gsl_matrix *x)
{
    PetscErrorCode ierr;

    PetscInt N = gp->trainLabels->size;
    Vec invKk;
    ierr = petsc_util_createVec (&invKk, gp->nlocal, N);

    for (int i=0; i<x->size2; i++) {
        gsl_vector *k = gsl_vector_calloc (gp->trainObs->size2);
        gsl_vector_const_view xi = gsl_matrix_const_column (x, i);

        for (int j=0; j<k->size; j++) {
            gsl_vector_const_view xj = gsl_matrix_const_column (gp->trainObs, j);
            gsl_vector_set (k, j, AcfrKern_eval (gp->kern, &xj.vector, &xi.vector));
        }

        Vec _k;
        ierr = petsc_util_createVec (&_k, gp->nlocal, N);
        petsc_util_fillVec (k, &_k, gp->rstart, gp->rend);

        SparseGp_solve (gp, _k, &invKk);

        double val;
        ierr = VecDot (_k, invKk, &val);

        double c;
        c = AcfrKern_eval (gp->kern, &xi.vector, &xi.vector) + gp->kern->beta;
        
        gsl_vector_set (var, i, c - val);

        /* clean up */
        gsl_vector_free (k);
        VecDestroy (&_k);
    }

    ierr = VecDestroy (&invKk);

    return EXIT_SUCCESS;
}
