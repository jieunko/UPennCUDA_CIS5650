#pragma once

#include "common.h"


namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scanWithoutTimer(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        void criteria(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);
    }
}
