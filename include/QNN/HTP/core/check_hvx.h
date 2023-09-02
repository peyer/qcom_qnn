//==============================================================================
//
// Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "cc_pp.h"

#ifndef CHECK_HVX_H
#define CHECK_HVX_H 1

EXTERN_C_BEGIN

//
// This makes sure that we have an HVX context (or not).  Does nothing on H2 or
// QuRT, but on x86, makes use of a TLS variable to do the check.
//

#ifdef __hexagon__

static inline void check_hvx() {}
static inline void check_not_hvx() {}

#else

__attribute__((visibility("default"))) void check_hvx();
__attribute__((visibility("default"))) void check_not_hvx();

#endif

EXTERN_C_END

#endif // CHECK_HVX_H
