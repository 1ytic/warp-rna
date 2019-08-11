#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#ifndef __CUDACC__
unsigned int atomicAdd(unsigned int *address, unsigned int value);
float __shfl_up_sync(unsigned int mask, float var, unsigned int delta, int width=warpSize);
unsigned int __activemask();
int __popc(unsigned int x);
void __threadfence();
#endif

#define W 32
#define G 1024
#define B 256

__forceinline__ __device__ static int idx2(int n, int u, int U1) {
    return n * U1 + u;
}

__forceinline__ __device__ static int idx3(int n, int t, int u, int T, int U) {
    return n * (T * U) + t * U + u;
}

__forceinline__ __device__ static int idx4(int n, int t, int u, int v, int T, int U, int V) {
    return n * (T * U * V) + t * (U * V) + u * V + v;
}

__forceinline__ __device__ static float log_sum_exp(float a, float b) {
    float maximum, diff;
    if (a > b) {
        maximum = a;
        diff = b-a;
    } else {
        maximum = b;
        diff = a-b;
    }
    //if (diff > -42) {
        maximum += log1pf(expf(diff));
    //}
    return maximum;
}

__device__
void kernel_warp_alphas(unsigned int *counts, float *alphas, const int *labels, const float *log_probs,
                        const int *xn, const int *yn, int T, int S, int U, int V, int blank) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    assert (d < W);
    assert (u <= U);
    assert (gridDim.y == U);
    assert (blockDim.x == W);

    int actual_u = yn[n]+1;
    int actual_s = xn[n]-actual_u+2;

    if (t > actual_s || u > actual_u)
        return;

    unsigned int *lock = counts + n * U * 2 + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        alphas[idx3(n, 0, 0, S, U)] = 0;
    }

    if (blockIdx.x > 0) {
        // Wait previous row
        do {} while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0) {
        // Wait previous column
        do {} while (atomicAdd(lock-1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u) {

        // Compute initial row value

        unsigned int l = labels[idx2(n, u-1, U-1)];

        float a = alphas[idx3(n, 0, u-1, S, U)];
        float b = log_probs[idx4(n, u-1, u-1, l, T, U, V)];

        alphas[idx3(n, 0, u, S, U)] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_s) {

        // Compute initial column with local scan algorithm

        float a;
        float b = log_probs[idx4(n, t-1, 0, blank, T, U, V)];

#pragma unroll
        for(unsigned int i = 1; i < W; i *= 2) {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d) {
                b += a;
            }
        }

        a = alphas[idx3(n, p, 0, S, U)];

        alphas[idx3(n, t, 0, S, U)] = a + b;
    }

    if (t < actual_s && u < actual_u) {

        // Ready to compute alphas[t, u]

        unsigned int l = labels[idx2(n, u-1, U-1)];

        float bias = log_probs[idx4(n, t+u-1, u, blank, T, U, V)];
        float skip = alphas[idx3(n, p, u, S, U)] + bias;
        float emit = alphas[idx3(n, t, u-1, S, U)] + log_probs[idx4(n, t+u-1, u-1, l, T, U, V)];

        float r = log_sum_exp(skip, emit);
        float output = r;

        for(unsigned int i = 1; i < W; i++) {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d) {
                r = log_sum_exp(r + bias, emit);
                output = r;
            }
        }

        alphas[idx3(n, t, u, S, U)] = output;
    }

    unsigned int mask = __activemask();
    int w = __popc(mask) - 1;

    if (d == w) {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__device__
void kernel_warp_betas(unsigned int *counts, float *betas, const int *labels, const float *log_probs,
                       const int *xn, const int *yn, int T, int S, int U, int V, int blank) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    assert (d < W);
    assert (u <= U);
    assert (gridDim.y == U);
    assert (blockDim.x == W);

    int actual_t = xn[n];
    int actual_u = yn[n]+1;
    int actual_s = actual_t-actual_u+2;

    if (t > actual_s || u > actual_u)
        return;

    int S1 = actual_s - 1;
    int U1 = actual_u - 1;

    unsigned int *lock = counts + n * U * 2 + U + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        betas[idx3(n, S1, U1, S, U)] = 0;
    }

    if (blockIdx.x > 0) {
        // Wait previous row
        do {} while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0) {
        // Wait previous column
        do {} while (atomicAdd(lock-1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u) {

        // Compute last row value

        unsigned int l = labels[idx2(n, U1-u, U-1)];

        float a = betas[idx3(n, S1, U1-u+1, S, U)];
        float b = log_probs[idx4(n, actual_t-u, U1-u, l, T, U, V)];

        betas[idx3(n, S1, U1-u, S, U)] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_s) {

        // Compute last column with local scan algorithm

        float a;
        float b = log_probs[idx4(n, actual_t-t, U1, blank, T, U, V)];

#pragma unroll
        for(unsigned int i = 1; i < W; i *= 2) {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d) {
                b += a;
            }
        }

        a = betas[idx3(n, S1-p, U1, S, U)];

        betas[idx3(n, S1-t, U1, S, U)] = a + b;
    }

    if (t < actual_s && u < actual_u) {

        // Ready to compute betas[S1-t, U1-u]

        unsigned int l = labels[idx2(n, U1-u, U-1)];

        float bias = log_probs[idx4(n, actual_t-t-u, U1-u, blank, T, U, V)];
        float skip = betas[idx3(n, S1-p, U1-u, S, U)] + bias;
        float emit = betas[idx3(n, S1-t, U1-u+1, S, U)] + log_probs[idx4(n, actual_t-t-u, U1-u, l, T, U, V)];

        float r = log_sum_exp(skip, emit);
        float output = r;

        for(unsigned int i = 1; i < W; i++) {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d) {
                r = log_sum_exp(r + bias, emit);
                output = r;
            }
        }

        betas[idx3(n, S1-t, U1-u, S, U)] = output;
    }

    unsigned int mask = __activemask();
    int w = __popc(mask) - 1;

    if (d == w) {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__global__
void kernel_warp(unsigned int *counts, float *alphas, float *betas, const int *labels, const float *log_probs,
                       const int *xn, const int *yn, int T, int S, int U, int V, int blank) {
    if (threadIdx.y == 0) {
        kernel_warp_alphas(counts, alphas, labels, log_probs, xn, yn, T, S, U, V, blank);
    }
    else if (threadIdx.y == 1) {
        kernel_warp_betas(counts, betas, labels, log_probs, xn, yn, T, S, U, V, blank);
    }
}

__global__
void kernel_grads_blank(float *grads, const float *alphas, const float *betas, const float *log_probs,
                        const int *xn, const int *yn, int T, int S, int U, int V, int blank) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = g * G + d;

    assert (u < U);
    assert (d < G);
    assert (blockDim.x == G);
    assert (gridDim.y == U);

    int actual_u = yn[n]+1;
    int actual_s = xn[n]-actual_u+2;

    if (t >= actual_s-1 || u >= actual_u)
        return;

    float a = alphas[idx3(n, t, u, S, U)];

    float b = betas[idx3(n, t+1, u, S, U)];

    float log_like = betas[idx3(n, 0, 0, S, U)];

    unsigned int index = idx4(n, t+u, u, blank, T, U, V);

    grads[index] = -expf(a + b + log_probs[index] - log_like);
}

__global__
void kernel_grads_label(float *grads, const float *alphas, const float *betas,
                        const int *labels, const float *log_probs,
                        const int *xn, const int *yn, int T, int S, int U, int V) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = g * G + d;

    assert (u < U - 1);
    assert (d < G);
    assert (blockDim.x == G);
    assert (gridDim.y == U - 1);

    int actual_u = yn[n]+1;
    int actual_s = xn[n]-actual_u+2;

    if (t >= actual_s || u >= actual_u-1)
        return;

    float a = alphas[idx3(n, t, u, S, U)];

    float b = betas[idx3(n, t, u+1, S, U)];

    unsigned int l = labels[idx2(n, u, U-1)];

    float log_like = betas[idx3(n, 0, 0, S, U)];

    unsigned int index = idx4(n, t+u, u, l, T, U, V);

    grads[index] = -expf(a + b + log_probs[index] - log_like);
}

__global__
void kernel_fill_costs(float *costs, float *grads, const float *alphas, const float *betas,
                       const int *xn, const int *yn, int N, int T, int S, int U, int V) {

    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N)
        return;

    int u = yn[n];
    int t = xn[n]-u;

    float a = alphas[idx3(n, t, u, S, U)];
    float b = betas[idx3(n, 0, 0, S, U)];

    float ratio = fabsf(a - b) / fabsf(fmaxf(a, b));

    if (ratio > 0.001) {

        printf("\nWARNING: sample %d [%d, %d] has a forward/backward mismatch %f / %f\n",
                n, t, u, a, b);

        float *g = grads + idx4(n, 0, 0, 0, T, U, V);

        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < U; ++j) {
                for (int v = 0; v < V; ++v, ++g) {
                    *g = 0;
                }
            }
        }

        b = (a + b) / 2.0f;
    }

    costs[n] = -b;
}

rnaStatus_t run_warp_rna(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                         const int *labels, const float *log_probs, float *grads, float *costs,
                         const int *xn, const int *yn, int N, int T, int S, int U, int V, int blank) {

    dim3 threads1(W, 2);
    dim3 blocks1((S + W - 1) / W, U, N);
    kernel_warp <<<blocks1, threads1, 0, stream>>> (counts, alphas, betas, labels, log_probs, xn, yn, T, S, U, V, blank);
    if (cudaGetLastError() != cudaSuccess)
        return RNA_STATUS_WARP_FAILED;

    if (S > 1) {

        dim3 blocks2((S - 1 + G - 1) / G, U, N);
        kernel_grads_blank <<<blocks2, G, 0, stream>>> (grads, alphas, betas, log_probs, xn, yn, T, S, U, V, blank);
        if (cudaGetLastError() != cudaSuccess)
            return RNA_STATUS_GRADS_BLANK_FAILED;
    }

    if (U > 1) {

        dim3 blocks3((S + G - 1) / G, U - 1, N);
        kernel_grads_label <<<blocks3, G, 0, stream>>> (grads, alphas, betas, labels, log_probs, xn, yn, T, S, U, V);
        if (cudaGetLastError() != cudaSuccess)
            return RNA_STATUS_GRADS_LABEL_FAILED;
    }

    dim3 blocks4((N + B - 1) / B, 1, 1);
    kernel_fill_costs <<<blocks4, B, 0, stream>>> (costs, grads, alphas, betas, xn, yn, N, T, S, U, V);
    if (cudaGetLastError() != cudaSuccess)
        return RNA_STATUS_COSTS_FAILED;

    return RNA_STATUS_SUCCESS;
}