#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#include "io_util.h"

size_t
io_util_numLinesInFile (FILE *fp)
{
    fseek (fp, 0, SEEK_SET);
    
    size_t numLines = 0;
    char c;
    while ((c = fgetc (fp)) != EOF) {
        if (c == '\n')
            numLines++;
    }
    return numLines;

    fseek (fp, 0, SEEK_SET);
}

size_t
io_util_numFeatures (FILE *fp)
{
    fseek (fp, 0, SEEK_SET);
    size_t numFeatures = 0;
    size_t len = 1024;
    char line[len];
    fgets (line, len, fp);

    for (int i=0; i<strlen (line); i++)
        if (line[i] == ' ')
            numFeatures++;

    fseek (fp, 0, SEEK_SET);

    return numFeatures + 1;
}

static void
_loadLineToVec (gsl_vector *col, const char *line, size_t M)
{
    float _colData[M];

    switch (M) {
    case 1: {
        sscanf (line, "%f\n", _colData);
        break;
    }
    case 2: {
        sscanf (line, "%f %f\n", _colData, _colData+1);
        break;
    }
    case 3: {
        sscanf (line, "%f %f %f\n", _colData, _colData+1, _colData+2);
        break;
    }
    default:
        fprintf (stderr, "Unimplemented feature size %ld\n", M);
        exit (EXIT_FAILURE);
    }

    for (int i=0; i<M; i++)
        gsl_vector_set (col, i, _colData[i]);
}

gsl_matrix *
io_util_readMatFromTxt (const char *filename)
{
    FILE *fp = fopen (filename, "r");
    gsl_matrix *mat = NULL;
    if (!fp) {
        perror ("fopen");
        return mat;
    }

    size_t N = 0, M = 0;
    N = io_util_numLinesInFile (fp);
    M = io_util_numFeatures (fp);
    mat = gsl_matrix_calloc (M, N);

    size_t len = 1024;
    char line[len];
    int i = 0;
    while (fgets (line, len, fp)) {
        gsl_vector_view col = gsl_matrix_column (mat, i);
        _loadLineToVec (&col.vector, line, M);
        i++;
    }

    /* clean up */
    fclose (fp);

    return mat;
}

gsl_vector *
io_util_readVecFromTxt (const char *filename)
{
    gsl_matrix *mat = io_util_readMatFromTxt (filename);
    assert (mat->size1 == 1);

    gsl_vector *vec = gsl_vector_calloc (mat->size2);
    gsl_vector_const_view row = gsl_matrix_const_row (mat, 0);
    gsl_vector_memcpy (vec, &row.vector);

    /* clean up */
    gsl_matrix_free (mat);

    return vec;
}
