#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        // exclusive prefix sum
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i-1] + idata[i-1];
            }
            timer().endCpuTimer();
        }

        void scanWithoutTimer(int n, int* odata, const int* idata)
        {
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i-1] + idata[i-1];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        //remove 0s from array
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] == 0) continue;
                odata[count] = idata[i];
                count++;

            }
            timer().endCpuTimer();
            return count;
        }

        void criteria(int n, int* odata, const int* idata)
        {
            for (int i = 0; i < n; i++)
            {
                if (idata[i] == 0) odata[i] = 0;
                else odata[i] = 1;

            }
         
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *temp = new int[n];
            int *tempScan = new int[n];
            int id = 0;
            
            
            timer().startCpuTimer();
            // TODO
            criteria(n, temp, idata);
            scanWithoutTimer(n, tempScan, temp);
            for (int i = 0; i < n; i++)
            {
                id = tempScan[i];
                if (temp[i]) odata[id] = idata[i];
            }
            timer().endCpuTimer();

            delete[] temp;
            delete[] tempScan;
            return id;
        }
    }
}
