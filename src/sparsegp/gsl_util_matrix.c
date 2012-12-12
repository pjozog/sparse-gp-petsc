#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <sys/ioctl.h>

#include <gsl/gsl_min.h>
#include <gsl/gsl_matrix.h>

#include "gsl_util_matrix.h"

void
gslu_matrix_printfc (const gsl_matrix *A, const char *name, const char *fmt, CBLAS_TRANSPOSE_t Trans)
{

    // figure out our terminal window size
    int ncols = 0;
    struct winsize w;
    if (ioctl (0, TIOCGWINSZ, &w))
        ncols = 80;
    else
        ncols = round (w.ws_col / 12.0);

    if (ncols == 0)
        ncols = 80;

    if (name != NULL && Trans==CblasNoTrans)
        printf ("%s =\n", name);
    else
        printf ("%s' =\n", name);

    // default printf format
    int __default__ = 1;
    if (fmt == NULL)
        fmt = "%10.4f";
    else
        __default__ = 0;

    size_t size1, size2;
    if (Trans == CblasNoTrans) {     
        size1 = A->size1;
        size2 = A->size2;
    } else {
        size1 = A->size2;
        size2 = A->size1;
    }

    // emulate matlab scientific format
    bool scifmt = 0;
    const int scimin = -3;
    const int scimax =  4;
    const double ZERO = DBL_EPSILON;
    double fabsmax = fabs (gsl_matrix_max (A));
    int sci;
    if (fabsmax > ZERO)
        sci = floor (log10 (fabsmax));
    else
        sci = 0;

    if (sci < scimin) {
        sci++;
        scifmt = 1;
    } 
    else if (sci > scimax) {
        sci--;
        scifmt = 1;
    }
    const double tens = pow (10, sci);
    if (scifmt)
        printf ("   1.0e%+03d *\n", sci);

    // print matrix
    for (size_t n=0; n < size2; n+=ncols) {
        if (ncols < size2) {
            if (n == (size2 - 1))
                printf ("    Column %zd\n", n);
            else
                printf ("    Columns %zd through %zd\n", n, GSL_MIN(n+ncols, size2)-1);
        }
        for (size_t i=0; i<size1; i++) {
            for (size_t j=GSL_MIN(n, size2); j<GSL_MIN(n+ncols, size2); j++) {
                double v;
                if (Trans == CblasNoTrans)
                    v = gsl_matrix_get (A, i, j);
                else
                    v = gsl_matrix_get (A, j, i);

                if (scifmt)
                    v /= tens;

                if (__default__ && fabs (v) < ZERO)
                    printf ("%10.4g", 0.0);
                else
                    printf (fmt, v);
                printf (" ");
            }
            printf ("\n");
        }
    }
}
