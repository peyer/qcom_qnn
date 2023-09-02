#=============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

import os

# Executable and library paths
SNPE = 'SNPE'
QNN = 'QNN'
UNQUANTIZED = 'unquantized'
NET_RUN_OUTPUT_DIR = 'output'
X86_LINUX_CLANG = 'x86_64-linux-clang'
BIN_PATH_IN_SDK = os.path.join('bin', X86_LINUX_CLANG)
LIB_PATH_IN_SDK = os.path.join('lib', X86_LINUX_CLANG)
PYTHONPATH = os.path.join('lib', 'python')
CONFIG_PATH = '/qti/aisw/quantization_checker/configs'

QNN_TF_CONVERTER_BIN_NAME = 'qnn-tensorflow-converter'
QNN_TFLITE_CONVERTER_BIN_NAME = 'qnn-tflite-converter'
QNN_ONNX_CONVERTER_BIN_NAME = 'qnn-onnx-converter'
QNN_MODEL_LIB_GENERATOR_BIN_NAME = 'qnn-model-lib-generator'
QNN_NET_RUN_BIN_NAME = 'qnn-net-run'

SNPE_TF_CONVERTER_BIN_NAME = 'snpe-tensorflow-to-dlc'
SNPE_TFLITE_CONVERTER_BIN_NAME = 'snpe-tflite-to-dlc'
SNPE_ONNX_CONVERTER_BIN_NAME = 'snpe-onnx-to-dlc'
SNPE_QUANTIZER_BIN_NAME = 'snpe-dlc-quantize'
SNPE_NET_RUN_BIN_NAME = 'snpe-net-run'
SNPE_UDO_ROOT = os.path.join('share', 'SNPE', 'SnpeUdo')

BACKEND_LIB_NAME = 'libQnnCpu.so'

MODEL_SO_OUTPUT_PATH = X86_LINUX_CLANG

TENSORFLOW = 'TENSORFLOW'
ONNX = 'ONNX'
TFLITE = 'TFLITE'
# Environment paths
# TENSORFLOW
TF_DISTRIBUTE = 'distribute'
TF_PYTHON_PATH = os.path.join('dependencies', 'python')
# TFLITE
TFLITE_DISTRIBUTE = 'distribute'
TFLITE_PYTHON_PATH = os.path.join('dependencies', 'python')
# ONNX
ONNX_DISTRIBUTE = 'distribute'
ONNX_PYTHON_PATH = os.path.join('dependencies', 'python')
