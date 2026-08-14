#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    printf("Running matrix size: %d x %d\n", N, N);

    float *A, *B, *C;

    A = (float*)malloc(N*N*sizeof(float));
    B = (float*)malloc(N*N*sizeof(float));
    C = (float*)malloc(N*N*sizeof(float));

    char fileAname[256], fileBname[256];
    sprintf(fileAname, "../datasets/matrixA_%d.txt", N);
    sprintf(fileBname, "../datasets/matrixB_%d.txt", N);
    printf("Loading dataset: %s\n", fileAname);
    printf("Loading dataset: %s\n", fileBname);

    FILE *fileA = fopen(fileAname, "r");
    FILE *fileB = fopen(fileBname, "r");

    if (!fileA || !fileB)
    {
        printf("Error opening dataset files.\n");
        return 1;
    }

    for(int i = 0; i < N*N; i++)
    {
        fscanf(fileA, "%f", &A[i]);
        fscanf(fileB, "%f", &B[i]);
    }

    fclose(fileA);
    fclose(fileB);

    double start = omp_get_wtime();

    #pragma omp parallel for
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            float sum = 0;

            for(int k=0;k<N;k++)
            {
                sum += A[i*N+k]*B[k*N+j];
            }

            C[i*N+j] = sum;
        }
    }

    double end = omp_get_wtime();

    printf("OpenMP CPU time: %f seconds\n", end-start);

    // Decide logging based on thread mode
    char *threads_env = getenv("OMP_NUM_THREADS");

    // ---- NORMAL SIZE BENCHMARK ----
    if (threads_env == NULL || atoi(threads_env) <= 1)
    {
        FILE *fp_check = fopen("../results/openmp_results.csv", "r");
        if (fp_check == NULL)
        {
            FILE *fp_init = fopen("../results/openmp_results.csv", "w");
            if (fp_init)
            {
                fprintf(fp_init, "size,time\n");
                fclose(fp_init);
            }
        }
        else
        {
            fclose(fp_check);
        }

        FILE *fp = fopen("../results/openmp_results.csv", "a");
        if (fp)
        {
            fprintf(fp, "%d,%f\n", N, end - start);
            fclose(fp);
        }
        else
        {
            perror("Error opening OpenMP results file");
        }
    }

    // Thread scaling logging
    // reuse previously read OMP_NUM_THREADS
    if (threads_env != NULL)
    {
        FILE *fp2 = fopen("../results/thread_scaling_results.csv", "a");

        // ensure header
        FILE *fp2_check = fopen("../results/thread_scaling_results.csv", "r");
        if (fp2_check == NULL)
        {
            FILE *fp2_init = fopen("../results/thread_scaling_results.csv", "w");
            if (fp2_init)
            {
                fprintf(fp2_init, "threads,time\n");
                fclose(fp2_init);
            }
        }
        else
        {
            fclose(fp2_check);
        }

        if (fp2 != NULL)
        {
            fprintf(fp2, "%s,%f\n", threads_env, end-start);
            fclose(fp2);
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}