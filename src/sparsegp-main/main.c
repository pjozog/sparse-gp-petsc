#include "sparsegp/sparsegp.h"
#include "sparsegp/io_util.h"
#include "sparsegp/gsl_util_matrix.h"
#include "sparsegp/gsl_util_vector.h"
#include "sparsegp/petsc_util.h"

static char help[] = "Main routine for 587 final project.\n\n";

#include <petscksp.h>

#define DEFAULT_OBS_FILE "../../data/obs.txt"
#define DEFAULT_LABELS_FILE "../../data/labels.txt"

#define DEFAULT_TEST_OBS_FILE "../../data/testObs.txt"
#define DEFAULT_TEST_LABELS_FILE "../../data/testLabels.txt"

#define DO_TEST

typedef struct Self 
{
    int rank;
    AcfrKern *kern;

    gsl_matrix *obs;
    gsl_vector *labels;
    
    gsl_matrix *testObs;
    gsl_vector *testLabels;
    
    char *obsFile;
    char *labelsFile;

    char *testObsFile;
    char *testLabelsFile;
} Self;

static Self *
_buildSelf ()
{
    PetscErrorCode ierr;
    (void) ierr;

    Self *self = calloc (1, sizeof (*self));
    self->kern = calloc (1, sizeof (*(self->kern)));
    size_t len = 1024;
    self->obsFile = calloc (len, sizeof (*(self->obsFile)));
    self->labelsFile = calloc (len, sizeof (*(self->labelsFile)));
    self->testObsFile = calloc (len, sizeof (*(self->obsFile)));
    self->testLabelsFile = calloc (len, sizeof (*(self->labelsFile)));
    self->kern->type = ACFR_KERN_TYPE_1;
    self->kern->sigma0 = 2;
    self->kern->beta = .05;
    PetscReal l0 = 0.3;

    strcpy (self->obsFile, DEFAULT_OBS_FILE);
    strcpy (self->labelsFile, DEFAULT_LABELS_FILE);
    strcpy (self->testObsFile, DEFAULT_TEST_OBS_FILE);
    strcpy (self->testLabelsFile, DEFAULT_TEST_LABELS_FILE);

    /* get options */
    ierr = PetscOptionsGetInt (PETSC_NULL, "-t", (PetscInt *)&(self->kern->type), PETSC_NULL);
    ierr = PetscOptionsGetReal (PETSC_NULL, "-sigma0", &(self->kern->sigma0), PETSC_NULL);
    ierr = PetscOptionsGetReal (PETSC_NULL, "-length0", &l0, PETSC_NULL);
    ierr = PetscOptionsGetString (PETSC_NULL, "-obs", self->obsFile, len, PETSC_NULL);
    ierr = PetscOptionsGetString (PETSC_NULL, "-labels", self->labelsFile, len, PETSC_NULL);
    ierr = PetscOptionsGetString (PETSC_NULL, "-testObs", self->testObsFile, len, PETSC_NULL);
    ierr = PetscOptionsGetString (PETSC_NULL, "-testLabels", self->testLabelsFile, len, PETSC_NULL);


    self->obs = io_util_readMatFromTxt (self->obsFile);
    if (!self->obs) {
        printf ("Couldn't load training data inputs\n");
        exit (EXIT_FAILURE);
    }
    self->labels = io_util_readVecFromTxt (self->labelsFile);
    if (!self->labels) {
        printf ("Couldn't load training data labels\n");
        exit (EXIT_FAILURE);
    }

#ifdef DO_TEST
    self->testObs = io_util_readMatFromTxt (self->testObsFile);
    if (!self->testObs) {
        printf ("Couldn't load test data inputs\n");
        exit (EXIT_FAILURE);
    }
    self->testLabels = io_util_readVecFromTxt (self->testLabelsFile);
    if (!self->testLabels) {
        printf ("Couldn't load test data labels\n");
        exit (EXIT_FAILURE);
    }
#endif

    self->kern->lengths = gsl_vector_calloc (self->obs->size1);
    gsl_vector_set_all (self->kern->lengths, l0);

    MPI_Comm_rank (PETSC_COMM_WORLD, &(self->rank));

    return self;
}

static void
_freeSelf (Self *self)
{
    /* These are freed in SparseGp_destroy() */
    /* gsl_matrix_free (self->obs); */
    /* gsl_vector_free (self->labels); */
    AcfrKern_free (self->kern);
    free (self->obsFile);
    free (self->labelsFile);
    free (self);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
    PetscErrorCode ierr;
    double startTime, endTime;

    PetscInitialize(&argc,&args,(char *)0,help);

    Self *self = _buildSelf ();

    SparseGp *gp = SparseGp_create (self->kern, self->rank, 
                                    self->obs, self->labels, 
                                    self->testObs, self->testLabels);
    
    /* First thing's first!  Need to compute K(X,X) */
    startTime = MPI_Wtime ();
    ierr = SparseGp_train (gp); CHKERRQ (ierr);
    endTime = MPI_Wtime ();

    IOU_ROOT_PRINT ("Computing K(X,X) took %f seconds\n", endTime - startTime);

    /* setup the solver */
    ierr = SparseGp_setupKsp (gp); CHKERRQ (ierr);

    /* compute the gradients */
    startTime = MPI_Wtime ();
    MPI_Barrier (PETSC_COMM_WORLD);
    double gradSigma0, gradLength0;
    ierr = SparseGp_logLikeGrad (gp, GP_HYPERPARAM_SIGMA0, -1, &gradSigma0); CHKERRQ (ierr);
    ierr = SparseGp_logLikeGrad (gp, GP_HYPERPARAM_LENGTH_I, 0, &gradLength0); CHKERRQ (ierr);
    MPI_Barrier (PETSC_COMM_WORLD);
    endTime = MPI_Wtime ();

    IOU_ROOT_PRINT ("Log-likelihood gradients: %f %f\n", gradSigma0, gradLength0);
    IOU_ROOT_PRINT ("Finished in %f seconds\n", endTime - startTime);
    double percentNonzero = SparseGp_percentNonzero (gp);
    IOU_ROOT_PRINT ("Percent non-zero: %f\n", percentNonzero);

#ifdef DO_TEST
    gsl_vector *mu = gsl_vector_calloc (self->testObs->size2);
    SparseGp_computeMuAt (gp, mu, self->testObs);
    if (self->rank == GP_RANK_ROOT)
        gslu_vector_printf (mu, "mu");
    gsl_vector_free (mu);

    gsl_vector *var = gsl_vector_calloc (self->testObs->size2);
    SparseGp_computeVarAt (gp, var, self->testObs);
    if (self->rank == GP_RANK_ROOT)
        gslu_vector_printf (var, "var");
    gsl_vector_free (var);
#endif

    ierr = PetscFinalize(); CHKERRQ (ierr);

    _freeSelf (self);

    return 0;
}
