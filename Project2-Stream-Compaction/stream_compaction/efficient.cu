#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void upSweep(int n, int* inplacedata, int level)
        {
            int index =  blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            
            int frontOffset = (1 << level+1);
            int backOffset = (1 << level);
            int bound = frontOffset - 1;
            if ((index + frontOffset - 1) >= n) return;
            if((index%frontOffset) == bound) inplacedata[index + frontOffset - 1] += inplacedata[index + backOffset - 1];
        }

        __global__ void downSweep(int n, int* inplacedata, int level)
        {
            int index =  blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
      
            int frontOffset = 1 << (level+1);
            int backOffset = 1 << level;
            int bound = backOffset - 1;
            if ((index + frontOffset - 1) >= n) return;
            if ((index + frontOffset-1) % backOffset == bound)
            {
                int temp = inplacedata[index + backOffset -1];
                inplacedata[index + backOffset - 1] = inplacedata[index + frontOffset - 1];
                inplacedata[index + frontOffset - 1] += temp;
            }
        }
        


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int numLevels = ilog2ceil(n);

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d < numLevels; ++d)
            {
                upSweep<<<fullBlocksPerGrid, blockSize >>>(n, dev_idata, d);
            }
            cudaMemset(&dev_idata[n - 1], 0, sizeof(int));

            for (int d = numLevels - 1; d >= 0; --d)
            {
                downSweep<<<fullBlocksPerGrid, blockSize >>>(n, dev_idata, d);
            }
         
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
