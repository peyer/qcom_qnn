//==============================================================================
//
// Copyright (c) 2018,2021 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef PERF_TIMING_H
#define PERF_TIMING_H 1

#include <stdint.h>

#pragma GCC visibility push(default)

class PcyclePoint {
  public:
    PcyclePoint(bool enable);
    void stop();
    uint64_t get_total() const { return end > start ? (end - start) : 0; }
    uint64_t get_start() const { return start; }
    uint64_t get_end() const { return end; }
    //private:
    uint64_t start;
    uint64_t end;
};

#pragma GCC visibility pop

#endif //PERF_TIMING_H
