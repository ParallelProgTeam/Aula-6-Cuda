/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>
#include <cuda_runtime.h>
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

StopWatchInterface *timer = NULL;

// includes, kernels
//#include <test1_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel template for flops test
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata) 
{
    float result=1;
    // read two values
    float val1 = g_idata[0];
    float val2 = g_idata[1];
 
    // place loop/unrolled loop here to do a bunch of multiply add ops
    // make sure you use results, so compiler does not optomize out
    result = val2 + (result * val1);

     g_odata[0] = result;
}

void cleanup(void)
{
    sdkDeleteTimer(&timer);
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{

    cudaDeviceProp deviceProps;
    float elapsedTimeInMs = 0.0f;
    cudaEvent_t start, stop;
    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    //unsigned int timer = 0;
    int devID = findCudaDevice(argc, (const char **)argv);

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n",
           deviceProps.name, deviceProps.multiProcessorCount);


//    CUT_SAFE_CALL( cutCreateTimer( &timer));
//    CUT_SAFE_CALL( cutStartTimer( timer));

    // adjust number of threads here
    unsigned int num_threads = 2;
    unsigned int mem_size = sizeof( float) * num_threads;

    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));
    // allocate host memory
    float* h_idata = (float*) malloc( mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float* d_idata;
    cudaMalloc( (void**) &d_idata, mem_size);
    // copy host memory to device
    checkCudaErrors(cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice)) ;

    // allocate device memory for result
    float* d_odata;
    checkCudaErrors(cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    // adjust thread block sizes here
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);

    // check if kernel execution generated and error

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy( h_odata, d_odata, sizeof( float) * num_threads,
                                cudaMemcpyDeviceToHost)) ;

    checkCudaErrors(cudaEventRecord(stop, 0));
    //Since device to device memory copies are non-blocking,
    //cudaDeviceSynchronize() is required in order to get
    //proper timing.
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
    printf( "Processing time: %f (ms)\n", elapsedTimeInMs);

    sdkDeleteTimer(&timer);
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));
    // cleanup memory
    free( h_idata);
    free( h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
}
