#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

namespace StreamCompaction {
    namespace Naive 
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveParallelScan(int n, int* odata, const int* idata, int level) 
        {
            int index =  blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
      
            int levelOffset = 1 << (level - 1);
            int valueToAdd = (index >= levelOffset) ? idata[index - levelOffset] : 0;
            odata[index] = valueToAdd + idata[index];
        }

            /**
             * Performs prefix-sum (aka scan) on idata, storing the result into odata.
             */
        void scan(int n, int* odata, const int* idata)
        {
            int* dev_a;
            int* dev_b;
            cudaMalloc((void**)&dev_a, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_a failed!");
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_b failed!");
            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int numLevels = ilog2ceil(n);

            timer().startGpuTimer();
            for (int i = 1; i <= numLevels; ++i)
            {
                naiveParallelScan<<<fullBlocksPerGrid, blockSize >>>(n, dev_b, dev_a, i);
                std::swap(dev_a, dev_b);
            }

            timer().endGpuTimer();

            //cudaDeviceSynchronize();
            odata[0] = 0;
            cudaMemcpy(odata+1, dev_a, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}