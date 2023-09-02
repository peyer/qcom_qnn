##############################################################################
#
# Copyright (c) 2022, 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import os

def getEnvironment(configParams, sdkDir, pythonPath=None, mlFramework=None):
    environ = dict()
    environ['PYTHONPATH'] = os.path.join(sdkDir, 'lib/python')
    if pythonPath:
        environ['PYTHONPATH'] = pythonPath + ':' + environ['PYTHONPATH']
    environ['QNN_SDK_ROOT'] = sdkDir
    if "ANDROID_NDK_PATH" not in configParams:
        print("ERROR: Please provide ANDROID_NDK PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["ANDROID_NDK_PATH"]
    if "CLANG_PATH" not in configParams:
        print("ERROR: Please provide CLANG PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["CLANG_PATH"] + ':' + environ['PATH']
    if "BASH_PATH" not in configParams:
        print("ERROR: Please provide BASH PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["BASH_PATH"] + ':' + environ['PATH']
    if "PY3_PATH" not in configParams:
        print("ERROR: Please provide Python3 environment bin PATH to config_file.", flush=True)
        exit(-1)
    if "BIN_PATH" not in configParams:
        print("ERROR: Please provide BIN PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams['PY3_PATH'] + ':' + environ['PATH'] + ':' + configParams["BIN_PATH"]
    environ['PATH'] = os.path.join(sdkDir, 'bin/x86_64-linux-clang') + ':' + environ['PATH']
    environ['LD_LIBRARY_PATH'] = os.path.join(sdkDir, 'lib/x86_64-linux-clang')
    if 'TENSORFLOW_HOME' in configParams:
        environ['TENSORFLOW_HOME'] = configParams['TENSORFLOW_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + environ['TENSORFLOW_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TENSORFLOW_HOME'], 'distribute')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TENSORFLOW_HOME'], 'dependencies/python')
    if 'TFLITE_HOME' in configParams:
        environ['TFLITE_HOME'] = configParams['TFLITE_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + environ['TFLITE_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TFLITE_HOME'], 'distribute')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TFLITE_HOME'], 'dependencies/python')
    if 'ONNX_HOME' in configParams:
        environ['ONNX_HOME'] = configParams['ONNX_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + environ['ONNX_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['ONNX_HOME'], 'distribute')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['ONNX_HOME'], 'dependencies/python')

    return environ
