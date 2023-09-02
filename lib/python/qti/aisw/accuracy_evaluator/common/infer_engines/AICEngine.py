##############################################################################
#
# Copyright (c) 2022 Qualcomm Technologies, Inc.
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

import logging
import onnx
import os
import random
import shutil
import psutil
import sys
import time
from abc import ABC

from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
import qti.aisw.accuracy_evaluator.common.defaults as df
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.defaults import qaic_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.infer_engine import InferenceEngine
from qti.aisw.accuracy_evaluator.common.infer_engines.executors import LocalExecutor

defaults = df.Defaults.getInstance()

class AicExecutors:
    __instance = None

    def __init__(self):
        if AicExecutors.__instance is not None:
            raise ce.InferenceEngineException('instance of AicExecutors already exists')
        else:
            AicExecutors.__instance = self
        self.server = None
        self.username = None
        self.password = None
        self.compile_inst, self.infer_inst = self.get_aic_executor()

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = AicExecutors()
        return cls.__instance

    def get_aic_compile_executor(self):
        return self.compile_inst

    def get_aic_infer_executor(self):
        return self.infer_inst

    def close(self):
        self.infer_inst.close()

    def get_aic_executor(self):
        """
        This method returns the appropriate aic excecutor for compilation and execution.
        """
        local_ins = LocalExecutor()
        return local_ins, local_ins