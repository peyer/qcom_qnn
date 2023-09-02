//=============================================================================
//
//  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "ops/QnnCpuOpPkgRelu.hpp"

Qnn_ErrorHandle_t QnnCpuOpPkgRelu::finalize() {
  QNN_CPU_BE_ENSURE_EQ(numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CPU_BE_ENSURE_EQ(numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  auto input  = getInput(0);
  auto output = getOutput(0);
  QNN_CPU_BE_ENSURE_EQ(input->dataType, output->dataType, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  // Supporting upto 4D input tensor
  const int numInDims  = numTensorDim(input);
  const int numOutDims = numTensorDim(output);
  QNN_CPU_BE_ENSURE(numInDims == numOutDims, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CPU_BE_ENSURE(numInDims >= 1 && numInDims <= 4, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  setIsFinalize(true);

  return QNN_SUCCESS;
}

void QnnCpuOpPkgRelu::execute(const float* in, const int inputSize, float* out) {
  for (int32_t s = 0; s < inputSize; ++s) {
    const float f = *in;
    if (f < 0) {
      *out = 0;
    } else {
      *out = f;
    }
    in++;
    out++;
  }
}

Qnn_ErrorHandle_t QnnCpuOpPkgRelu::execute() {
  QNN_CPU_BE_ENSURE(getIsFinalize(), QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED);
  auto input  = getInput(0);
  auto output = getOutput(0);

  execute((const float*)input->data, nunTensorSize(input), (float*)output->data);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnCpuOpPkgRelu::setOpNode(QnnCpuOpPackage_Node_t* node) {
  // Add input
  for (uint32_t i = 0; i < node->numOfInputs; i++) {
    addInput(node->inputs[i]);
  }

  // Add output
  for (uint32_t i = 0; i < node->numOfOutputs; i++) {
    addOutput(node->outputs[i]);
  }

  return QNN_SUCCESS;
}