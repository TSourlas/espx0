//
// Created by Thanasis on 5/4/2025.
//

#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "knn.h"


static void swap(double *a, double *b) {
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

static void swap_int(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

static int partition(double *dist, int *indices, int left, int right, int pivot_index) {
    double pivot_value = dist[pivot_index];
    swap(&dist[pivot_index], &dist[right]);
    swap_int(&indices[pivot_index], &indices[right]);

    int store_index = left;

    for (int i = left; i < right; i++) {
        if (dist[i] < pivot_value) {
            swap(&dist[store_index], &dist[i]);
            swap_int(&indices[store_index], &indices[i]);
            store_index++;
        }
    }

    swap(&dist[right], &dist[store_index]);
    swap_int(&indices[right], &indices[store_index]);

    return store_index;
}

void quickselect_k(double *dist, int *indices, int n, int k) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int pivot_index = left + rand() % (right - left + 1);
        int pivot_new = partition(dist, indices, left, right, pivot_index);

        if (pivot_new == k)
            return;  // Έχουμε τους k μικρότερους
        else if (pivot_new < k)
            left = pivot_new + 1;
        else
            right = pivot_new - 1;
    }
}

void knnsearch(
    const double *C, const double *Q,
    int N, int M, int d, int k,
    int *idx, double *dst
) {
    // Υπολογισμός C^2 (σταθερό για όλα τα query)
    double *C_sq = malloc(N * sizeof(double));
    for (int i = 0; i < N; ++i) {
        C_sq[i] = 0.0;
        for (int j = 0; j < d; ++j) {
            double val = C[i * d + j];
            C_sq[i] += val * val;
        }
    }

    // Υπολογισμός Q^2
    double *Q_sq = malloc(M * sizeof(double));
    for (int i = 0; i < M; ++i) {
        Q_sq[i] = 0.0;
        for (int j = 0; j < d; ++j) {
            double val = Q[i * d + j];
            Q_sq[i] += val * val;
        }
    }

    // Υπολογισμός dot products
    double *dot_products = malloc(M * N * sizeof(double));
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        M, N, d,
        -2.0, Q, d, C, d,
        0.0, dot_products, N
    );

    // Υπολογισμός αποστάσεων
    double *D = malloc(M * N * sizeof(double));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double val = Q_sq[i] + C_sq[j] + dot_products[i * N + j];
            D[i * N + j] = sqrt(fmax(val, 0.0));
        }
    }

    // Quickselect για κάθε query
    for (int i = 0; i < M; ++i) {
        double *dist_row = D + i * N;

        double *tmp_dist = malloc(2 * N * sizeof(double));
        int *tmp_idx = malloc(2* N * sizeof(int));
        for (int j = 0; j < N; ++j) {
            tmp_dist[j] = dist_row[j];
            tmp_idx[j] = j;
        }

        quickselect_k(tmp_dist, tmp_idx, N, k);

        for (int j = 0; j < k; ++j) {
            dst[i * k + j] = tmp_dist[j];
            idx[i * k + j] = tmp_idx[j];
        }

        free(tmp_dist);
        free(tmp_idx);
    }

    free(C_sq);
    free(Q_sq);
    free(dot_products);
    free(D);
}
