//
// Created by Thanasis on 6/4/2025.
//

#ifndef KNN_H
#define KNN_H

void quickselect_k(double *dist, int *indices, int n, int k);
void knnsearch(
    const double *C, const double *Q,
    int N, int M, int d, int k,
    int *idx, double *dst
);

#endif
