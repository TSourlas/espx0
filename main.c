#include <stdio.h>
#include <stdlib.h>
#include "knn.h"
#include <time.h>
#include <pthread.h>
#include <unistd.h>  // για sleep
#include <float.h>

#define PARALLEL
#define NUM_THREADS 1

typedef struct {
    const double *C;
    const double *Q;
    int N;
    int M;
    int d;
    int k;
    int start;
    int end;
    int *idx;
    double *dst;
} ThreadArgs;

void *thread_worker(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;

    // Δείκτης αρχής για το υποσύνολο των query points
    const double *Q_block = args->Q + args->start * args->d;
    double *dst_block = args->dst + args->start * (args->k + 1);
    int *idx_block = args->idx + args->start * (args->k + 1);

    // Initialize memory for the current thread's block
    int num_queries = args->end - args->start;
    printf("Thread is processing points from index %d to %d\n", args->start, args->end - 1);

    // for (int i = args->start; i < args->end; ++i) {
    //     printf("Point %d: ", i);
    //     for (int j = 0; j < args->d; ++j) {
    //         printf("%.4f ", Q_block[(i - args->start) * args->d + j]);
    //     }
    //     printf("\n");
    // }

    // Call knnsearch for this block of queries
    knnsearch(
        args->C,
        Q_block,
        args->N,
        num_queries,
        args->d,
        args->k,
        idx_block,
        dst_block
    );
    free(arg);
    return NULL;
}

int main() {
    int N = 10;   //number of points
    int d = 4;      //dimensions
    int k = 4;      //nearest neighbours
    double *C = malloc(N * d * sizeof(double));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            double val = (i * (j + 1)) % 23 + (j * j) % 7;
            C[i * d + j] = val + 0.3 * sin(i + j * 3.14);
        }
    }

    int *idx = malloc(2*N * k * sizeof(int));
    double *dst = malloc(2*N * k * sizeof(double));

    for (int i = 0; i < N * (k + 1); ++i) {
        idx[i] = -1;
        dst[i] = FLT_MAX;
    }

#ifdef SERIAL
    printf("[INFO] Running Serial Implementation...\n");

    knnsearch(C, C, N, N, d, k, idx, dst);

    for (int i = 0; i < N; ++i) {
        printf("Point %d:\n", i);
        int printed = 0;
        int base = i * k;
        for (int j = 0; j < k; ++j) {
            printf("  Neighbor %d -> Index: %d, Distance: %.4f\n",
                   printed + 1, idx[base + j], dst[base + j]);
            printed++;
        }
    }
#endif

#ifdef PARALLEL
    printf("[INFO] Running Pthreads Implementation...\n");

    pthread_t threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    int points_per_thread = N / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        ThreadArgs *arg = malloc(sizeof(ThreadArgs));
        arg->C = C;
        arg->Q = C;
        arg->N = N;
        arg->M = N;
        arg->d = d;
        arg->k = k;
        arg->start = i * points_per_thread;
        arg->end = (i == NUM_THREADS - 1) ? N : (i + 1) * points_per_thread;
        arg->idx = idx;
        arg->dst = dst;

        pthread_create(&threads[i], NULL, thread_worker, arg);

        sleep(1);
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        int ret = pthread_join(threads[i], NULL);
        if (ret != 0) {
            // Handle thread join error
            printf("Error joining thread %d\n", i);
            exit(1);
        }
    }
    // Εκτύπωση των idx και dst μετά την ολοκλήρωση των νημάτων
    printf("[DEBUG] idx:\n");
    for (int i = 0; i < N; ++i) {
        printf("Query %d: ", i);
        for (int j = 0; j < k; ++j) {
            printf("%d ", idx[i * k + j]);
        }
        printf("\n");
    }

    printf("[DEBUG] dst:\n");
    for (int i = 0; i < N; ++i) {
        printf("Query %d: ", i);
        for (int j = 0; j < k; ++j) {
            printf("%.4f ", dst[i * k + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < N; ++i) {
        printf("Point %d:\n", i);
        int printed = 0;
        int base = i * k;
        for (int j = 0; j < k; ++j) {
            printf("  Neighbor %d -> Index: %d, Distance: %.4f\n",
                   printed + 1, idx[base + j], dst[base + j]);
            printed++;
        }
    }
#endif

    free(idx);
    free(dst);
    free(C);
    return 0;
}
