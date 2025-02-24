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
            
            int frontOffset = 1 << (level+1);
            int backOffset = (1 << level);
            int bound = frontOffset - 1;
            if((index%frontOffset) == bound) inplacedata[index] += inplacedata[index - backOffset];
        }

        __global__ void downSweep(int n, int* inplacedata, int level)
        {
            int index =  blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
      
            int frontOffset = 1 << (level+1);
            int backOffset = 1 << level;
            int bound = frontOffset - 1;
            if (index % frontOffset == bound)
            {
                int temp = inplacedata[index - backOffset];
                inplacedata[index - backOffset] = inplacedata[index];
                inplacedata[index] += temp;
            }
        }
        


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            int* dev_idata;
            int numLevels = ilog2ceil(n);
            int extended = pow(2, numLevels );
            int arrSize = n % 2 == 0 ? n : extended;
            cudaMalloc((void**)&dev_idata, arrSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d < numLevels; ++d)
            {
                upSweep<<<fullBlocksPerGrid, blockSize >>>(n, dev_idata, d);
            }
            
            cudaMemset(&dev_idata[arrSize - 1], 0, sizeof(int));


            for (int d = numLevels - 1; d >= 0; --d)
            {
                downSweep<<<fullBlocksPerGrid, blockSize >>>(arrSize, dev_idata, d);
            }
            
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);


        }

        void indeviceScan(int n, int numLevels,  int* odata, const int* idata)
        {
            int* dev_idata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy Device to Device idata  failed!");

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            
            // TODO     
            for (int d = 0; d < numLevels; ++d)
            {
                upSweep<<<fullBlocksPerGrid, blockSize >>>(n, dev_idata, d);
                checkCUDAError("upSweep %d failed!", d);
            }
            
            cudaMemset(&dev_idata[n - 1], 0, sizeof(int));


            for (int d = numLevels - 1; d >= 0; --d)
            {
                downSweep<<<fullBlocksPerGrid, blockSize >>>(n, dev_idata, d);
                checkCUDAError("downSweep %d failed!", d);
            }

            cudaMemcpy(odata, dev_idata, n* sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy Device to Device dev_idata  failed!");
            cudaFree(dev_idata);
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
            int* dev_idata;
            int* dev_bools;
            int* dev_indicies;
            int* dev_scatter;
            int numLevels = ilog2ceil(n);
            int extended = pow(2, numLevels );
            int arrSize = n % 2 == 0 ? n : extended;
            int t;
            int t2;
         
            cudaMalloc((void**)&dev_idata, arrSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, arrSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_indicies, arrSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_indicies failed!");
            cudaMalloc((void**)&dev_scatter, arrSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_scatter failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(&dev_idata[n], 0, (arrSize-n) * sizeof(int));

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(arrSize, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");
            indeviceScan(arrSize,numLevels, dev_indicies, dev_bools);
            
            Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(arrSize, dev_scatter, dev_idata, dev_bools, dev_indicies);
            checkCUDAError("kernScatter failed!");
            timer().endGpuTimer();
            

            
            cudaMemcpy(&t,  &dev_indicies[arrSize-1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&t2, &dev_bools[arrSize-1], sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_indicies failed!");

            cudaMemcpy(odata, dev_scatter, (t+t2) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indicies);
            cudaFree(dev_scatter);
            return t;
        }
    }
}
