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
            int offset = 1 << (level + 1);
            if (index < (n / offset))
            {
     
                int backOffset = (1 << level);
                int k = index * offset;
                inplacedata[k + offset -1] += inplacedata[k + backOffset -1];
            }
        }

        __global__ void downSweep(int n, int* inplacedata, int level)
        {
            int index =  blockIdx.x * blockDim.x + threadIdx.x;
            int offset = 1 << (level + 1);
            if (index < (n / offset))
            {
                int backOffset = 1 << level;
                int k = index * offset;
                int temp = inplacedata[k + backOffset -1];
                inplacedata[k + backOffset-1] = inplacedata[k+offset -1];
                inplacedata[k + offset -1] += temp;

            }
            
        }

        /*
        __global__ void scanSM(int n, int* idata, int* odata)
        {
            int thid = threadIdx.x;
            extern __shared__ float temp[];
            temp[2 * thid] = idata[2 * thid];
            int offset = 1;
            for (int d = n >> 1; d > 0; d >>= 1)
            {
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }
            if (thid == 0) temp[n - 1] = 0;
            for (int d = 1; d < n; d *= 2)
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    float t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            odata[2 * thid] = temp[2 * thid];
            odata[2 * thid + 1] = temp[2 * thid + 1];
        }
        */


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
            //dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d < numLevels; ++d)
            {
                int numThreads = extended / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                upSweep<<<fullBlocksPerGrid, blockSize >>>(n, dev_idata, d);
            }
            cudaDeviceSynchronize();

            cudaMemset(&dev_idata[arrSize - 1], 0, sizeof(int));


            for (int d = numLevels - 1; d >= 0; --d)
            {
                int numThreads = extended / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                downSweep<<<fullBlocksPerGrid, blockSize >>>(arrSize, dev_idata, d);
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);


        }
        /*
        void scanSharedMem(int n, int* odata, const int* idata)
        {
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            scanSM<<<fullBlocksPerGrid, blockSize >>>(n, dev_idata, dev_odata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

        }
        */
        void indeviceScan(int n, int numLevels,  int* odata, const int* idata)
        {
            //int* dev_idata;
            int extended = pow(2, numLevels);
            //cudaMalloc((void**)&dev_idata, n * sizeof(int));
            //checkCUDAError("cudaMalloc dev_idata failed!");
            //cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            //checkCUDAError("cudaMemcpy Device to Device idata  failed!");

            cudaMemcpy(odata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            int blockSize = 256;
            //dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            
            // TODO     
            for (int d = 0; d < numLevels; ++d)
            {
                int numThreads = extended / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                upSweep<<<fullBlocksPerGrid, blockSize >>>(n, odata, d);
                checkCUDAError("upSweep %d failed!", d);
            }
            
            cudaMemset(&odata[n - 1], 0, sizeof(int));


            for (int d = numLevels - 1; d >= 0; --d)
            {
                int numThreads = extended / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                downSweep<<<fullBlocksPerGrid, blockSize >>>(n, odata, d);
                checkCUDAError("downSweep %d failed!", d);
            }

            //cudaMemcpy(odata, dev_idata, n* sizeof(int), cudaMemcpyDeviceToDevice);
            //checkCUDAError("cudaMemcpy Device to Device dev_idata  failed!");
            //cudaFree(dev_idata);
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
