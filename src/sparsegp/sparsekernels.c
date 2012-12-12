#include "sparsekernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>

#include "gsl_util_blas.h"

/* begin static functions */

static double
_mahaDist (const gsl_vector *x1, const gsl_vector *x2, const gsl_vector *charLengths)
{
    gsl_vector *invCharLengths = gsl_vector_calloc (charLengths->size);
    for (int i=0; i<invCharLengths->size; i++)
        gsl_vector_set (invCharLengths, i, 1 / pow (gsl_vector_get (charLengths, i), 2));
    gsl_matrix *Omega = gsl_matrix_calloc (x1->size, x2->size);
    gsl_vector_view OmegaDiag = gsl_matrix_diagonal (Omega);
    gsl_vector_memcpy (&OmegaDiag.vector, invCharLengths);

    gsl_vector *x1MinX2 = gsl_vector_calloc (x1->size);
    gsl_vector_memcpy (x1MinX2, x1);
    gsl_vector_sub (x1MinX2, x2);

    double r = gslu_blas_vTmv (x1MinX2, Omega, x1MinX2);

    /* clean up */
    gsl_vector_free (invCharLengths);
    gsl_matrix_free (Omega);
    gsl_vector_free (x1MinX2);

    return sqrt (r);
}

static double
_eval_type1_single (double x1, double x2, double l, double sigma0)
{
    double val = 0;
    double d = fabs (x1 - x2);

    if (d < l)
        val = sigma0 * (((2+cos (2*M_PI*d/l)) / 3) * (1-d/l) + 1/(2*M_PI) * sin (2*M_PI*d/l));

    return val;
}

static double
_eval_type2_single (double r, double sigma0)
{
    double val = 0;
    
    if (r < 1)
        val = sigma0 * (((2+cos (2*M_PI*r)) / 3) * (1-r) + 1/(2*M_PI) * sin (2*M_PI*r));

    return val;
}

static double
_eval_type1 (const AcfrKern* kern, const gsl_vector *x1, const gsl_vector *x2)
{
    double prod = 1.0;
    double sigma0 = kern->sigma0;

    for (int i=0; i<kern->lengths->size; i++)
        prod *= _eval_type1_single (gsl_vector_get (x1, i), 
                                    gsl_vector_get (x2, i), 
                                    gsl_vector_get (kern->lengths, i), 1);
    
    return prod * sigma0;
}

static double
_eval_type2 (const AcfrKern* kern, const gsl_vector *x1, const gsl_vector *x2)
{
    double sigma0 = kern->sigma0;
    double r = _mahaDist (x1, x2, kern->lengths);
    return _eval_type2_single (r, sigma0);
}

static double
_KGradSigma0_type1 (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2)
{
    return 1/kern->sigma0 * AcfrKern_eval (kern, x1, x2);
}

static double
_KGradSigma0_type2 (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2)
{
    double r  = _mahaDist (x1, x2, kern->lengths);
    if (r >= 0 && r < 1)
        return (2+cos(2*M_PI*r))/3*(1-r) + (1/(2*M_PI)) * sin(2*M_PI*r);
    else
        return 0;
}

static double
_KGradLength_type1 (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2, size_t ind)
{
    double sigma0 = kern->sigma0;
    double di = fabs (gsl_vector_get (x1, ind) - gsl_vector_get (x2, 0));
    double li = gsl_vector_get (kern->lengths, ind);

    if (di >= li)
        return 0;

    double term1 = (4*sigma0)/3 * di/(li*li);
    double term2 = (M_PI*(1-di/li)*cos (M_PI*di/li) + sin (M_PI*di/li)) * sin (M_PI*di/li);

    return term1 * term2;
}

static double
_KGradLength_type2 (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2, size_t ind)
{
    double sigma0 = kern->sigma0;
    double r = _mahaDist (x1, x2, kern->lengths);
    if (r > 0 && r < 1) {
        double li = gsl_vector_get (kern->lengths, ind);
        double x1i = gsl_vector_get (x1, ind);
        double x2i = gsl_vector_get (x2, ind);
        double term1 = 4*sigma0/3 * (M_PI*(1-r)*cos(M_PI*r) + sin(M_PI*r));
        double term2 = (sin(M_PI*r)/r) * (1/li) * pow((x1i - x2i)/li, 2);
        return term1 * term2;
    }
    else
        return 0;
}

/* end static functions */

double
AcfrKern_eval (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2)
{
    assert (x1->size == x2->size && x1->size == kern->lengths->size);

    switch (kern->type) {
    case ACFR_KERN_TYPE_1:
        return _eval_type1 (kern, x1, x2);
    case ACFR_KERN_TYPE_2:
        return _eval_type2 (kern, x1, x2);
    default:
        errno = EINVAL;
        perror ("AcfrKern_eval");
        exit (EXIT_FAILURE);
    }

    return 0;
}

double
AcfrKern_KGradSigma0 (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2)
{
    assert (x1->size == x2->size && x1->size == kern->lengths->size);

    switch (kern->type) {
    case ACFR_KERN_TYPE_1:
        return _KGradSigma0_type1 (kern, x1, x2);
    case ACFR_KERN_TYPE_2:
        return _KGradSigma0_type2 (kern, x1, x2);
    default:
        errno = EINVAL;
        perror ("AcfrKern_KGradSigma0");
        exit (EXIT_FAILURE);
    }

    return 0;
}

double
AcfrKern_KGradLength (const AcfrKern *kern, const gsl_vector *x1, const gsl_vector *x2, size_t ind)
{
    assert (x1->size == x2->size && x1->size == kern->lengths->size);

    switch (kern->type) {
    case ACFR_KERN_TYPE_1:
        return _KGradLength_type1 (kern, x1, x2, ind);
    case ACFR_KERN_TYPE_2:
        return _KGradLength_type2 (kern, x1, x2, ind);
    default:
        errno = EINVAL;
        perror ("AcfrKern_KGradSigma0");
        exit (EXIT_FAILURE);
    }

    return 0;
}

void
AcfrKern_free (AcfrKern *kern)
{
    gsl_vector_free (kern->lengths);
    free (kern);
}
