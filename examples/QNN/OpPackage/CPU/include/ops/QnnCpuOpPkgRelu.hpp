//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once
#include <cmath>

#include "CPU/QnnCpuOpPackage.h"
#include "QnnCpuMacro.hpp"
#include "ops/QnnCpuOpPkgOpBase.hpp"

typedef struct {
  float beta;
} QnnCpuReluParams_t;

class QnnCpuOpPkgRelu final : public QnnCpuOpPkgOpBase {
 public:
  QnnCpuOpPkgRelu() {}
  QnnCpuOpPkgRelu(QnnCpuOpPackage_Node_t* node) : QnnCpuOpPkgOpBase(node->name, node->typeName) {}

  Qnn_ErrorHandle_t finalize();

  void execute(const float* in, const int inputSize, float* out);

  Qnn_ErrorHandle_t execute();

  Qnn_ErrorHandle_t setOpNode(QnnCpuOpPackage_Node_t* node);
};
