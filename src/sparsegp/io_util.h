#ifndef __HAS_IO_UTIL_H__
#define __HAS_IO_UTIL_H__

#include <stdio.h>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#ifdef __cplusplus
extern "C" {
#endif

int __io_rank;

#define IOU_ROOT_PRINT(format, ...) \
    MPI_Comm_rank (PETSC_COMM_WORLD, &__io_rank); if (__io_rank == GP_RANK_ROOT) fprintf (stdout, "[sparsegp]\t" format, ## __VA_ARGS__)


size_t
io_util_numLinesInFile (FILE *fp);

size_t
io_util_numFeatures (FILE *fp);

gsl_matrix *
io_util_readMatFromTxt (const char *filename);

gsl_vector *
io_util_readVecFromTxt (const char *filename);

#ifdef __cplusplus
}
#endif

#endif  /* __HAS_IO_UTIL_H__ */
