#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/ioctl.h>

// external linking req'd
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "gsl_util_matrix.h"
#include "gsl_util_vector.h"

void
gslu_vector_printf (const gsl_vector *a, const char *name)
{
    gslu_vector_printfc (a, name, NULL, CblasNoTrans);
}

void
gslu_vector_printfc (const gsl_vector *a, const char *name, const char *fmt, CBLAS_TRANSPOSE_t Trans)
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
        size1 = a->size;
        size2 = 1;
    } else {
        size1 = 1;
        size2 = a->size;
    }

    // emulate matlab scientific format
    bool scifmt = 0;
    const int scimin = -3;
    const int scimax =  4;
    const double ZERO = DBL_EPSILON;
    double fabsmax = fabs (gsl_vector_max (a));
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
                    v = gsl_vector_get (a, i);
                else
                    v = gsl_vector_get (a, j);

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
