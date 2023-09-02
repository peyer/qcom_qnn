//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

template <typename T_Ttype>
void evaluate(const T_Ttype* in, uint32_t inputSize, T_Ttype* out);

namespace relu {

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  const auto input  = operation->getInput(0);
  const auto output = operation->getOutput(0);

  QNN_CUSTOM_BE_ENSURE_EQ(
      input->dataType, output->dataType, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  // Supporting only 1D and 2D input tensor
  QNN_CUSTOM_BE_ENSURE(numDimensions(input) >= 1 && numDimensions(input) <= 2,
                       QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  return QNN_SUCCESS;
}

template <typename T_Ttype>
void evaluate(const T_Ttype* in, uint32_t inputSize, T_Ttype* out) {
  for (uint32_t s = 0; s < inputSize; ++s) {
    const T_Ttype f = *in;
    if (f < 0) {
      *out = 0;
    } else {
      *out = f;
    }
    in++;
    out++;
  }
}

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  auto input  = operation->getInput(0);
  auto output = operation->getOutput(0);

  evaluate((float*)input->data, numTensorSize(input), (float*)output->data);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Relu"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT);

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  return QNN_SUCCESS;
}
}  // namespace relu

CustomOpRegistration_t* register_ReluCustomOp() {
  static CustomOpRegistration_t reluRegister = {
      relu::execute, relu::finalize, nullptr, relu::validateOpConfig, relu::populateFromNode};
  return &reluRegister;
}

REGISTER_OP(Relu, register_ReluCustomOp);
