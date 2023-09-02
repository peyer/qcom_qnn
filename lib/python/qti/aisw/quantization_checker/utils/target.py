#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
from qti.aisw.quantization_checker.utils.Logger import PrintOptions
from qti.aisw.quantization_checker.utils.FileUtils import FileUtils
from qti.aisw.quantization_checker.utils import utils
import qti.aisw.quantization_checker.EnvironmentManager as em
import qti.aisw.quantization_checker.utils.Constants as Constants

# contains general information about the target execution environment
class target:
    def __init__(self, quantizationVariation, inputList, sdkDir, sdkType, outputDir, configParams, logger):
        self.netRunCmdAndArgs = ''
        self.quantizationVariation = quantizationVariation
        self.inputList = inputList
        self.sdkDir = sdkDir
        self.sdkType = sdkType
        self.outputDir = outputDir
        self.netRunOutputDir = os.path.join(self.outputDir, Constants.NET_RUN_OUTPUT_DIR, self.quantizationVariation)
        self.configParams = configParams
        self.logger = logger
        self.fileHelper = FileUtils(self.logger)

# executes locally on x86 arch
class x86_64(target):
    def __init__(self, quantizationVariation, inputList, sdkDir, sdkType, outputDir, configParams, logger):
        super().__init__(quantizationVariation, inputList, sdkDir, sdkType, outputDir, configParams, logger)

    def buildNetRunArgs(self):
        if self.sdkType.upper() == Constants.QNN:
            qnnNetRunArgs = '--backend ' + os.path.join(self.sdkDir, Constants.LIB_PATH_IN_SDK, Constants.BACKEND_LIB_NAME) + ' --output_dir ' + self.netRunOutputDir + ' --input_list ' + self.inputList + ' --input_data_type float --output_data_type float_only --debug'
            qnnNetRunCommand = os.path.join(self.sdkDir, Constants.BIN_PATH_IN_SDK, Constants.QNN_NET_RUN_BIN_NAME)
            pathToCompiledModel = os.path.join(self.outputDir, Constants.MODEL_SO_OUTPUT_PATH, 'lib' + self.quantizationVariation + '.so')

            self.netRunCmdAndArgs = qnnNetRunCommand + ' --model ' + pathToCompiledModel
            self.netRunCmdAndArgs += ' ' + qnnNetRunArgs
        else:
            self.netRunCmdAndArgs = os.path.join(self.sdkDir, Constants.BIN_PATH_IN_SDK, Constants.SNPE_NET_RUN_BIN_NAME) + ' --container ' + os.path.join(self.outputDir, self.quantizationVariation, 'unquantized.dlc') + ' --output_dir ' + self.netRunOutputDir + ' --input_list ' + self.inputList + ' --debug'
        self.logger.print('net-run command: ' + self.netRunCmdAndArgs, PrintOptions.LOGFILE)

    def runModel(self):
        if self.fileHelper.deleteDirAndContents(self.netRunOutputDir):
            self.logger.print('Unable to clear contents from previous net-run output, please verify current results do not include previous results.', PrintOptions.LOGFILE)
        if self.sdkType.upper() == Constants.QNN:
            environment = em.getEnvironment(self.configParams, self.sdkDir, Constants.QNN)
            environment['LD_LIBRARY_PATH'] = environment['LD_LIBRARY_PATH'] + ':' + os.path.join(self.sdkDir, Constants.LIB_PATH_IN_SDK)
        else:
            environment = em.getEnvironment(self.configParams, self.sdkDir, Constants.SNPE)
        return utils.issueCommandAndWait(self.netRunCmdAndArgs, self.logger, environment, False, True)
